#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:52:09 2023

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from scipy.stats import mode
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline, interp1d
import rawpy
from time import time
from skimage.filters import gaussian, gabor
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk, remove_small_holes
import imageio
from tqdm import tqdm
from skimage.segmentation import felzenszwalb
from skimage.restoration import unwrap_phase as unwrap
#%%
def guass(x,mu,si):
    return 1/ np.sqrt(2*np.pi* si**2) * np.exp(-1/2* ((x-mu)/si)**2)
        
def mrot(th):
    mat = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    return mat

def rot(y,z,mat):
    n1,n2 = np.shape(y)
    yzf = np.zeros((n1*n2,2))
    yzf[:,0], yzf[:,1] = y.flatten(), z.flatten()
    yzr = np.dot(yzf,mat.T)
    yr,zr = yzr[:,0].reshape(n1,n2), yzr[:,1].reshape(n1,n2)
    return yr,zr

def lin(x,m,o):
    return m*x+o

def patron(x,y,cx,cy,a=2.5,k=0.2):
    # print(cx,cy)
    return a + a*np.cos(k*np.sqrt( (x-x[cx,cx])**2 + (y-y[cy,cy])**2) )

def alt(dp,Lp,D,w0):
    return dp * Lp / (dp - D*w0 )

#%%
def calibrate_params(tod,phi, limtar, initial, inlim=[0.002,20,20] ,showerr=True):
    '''
    Calibrates the parameter d, L, D, and w by using least_squares and a fit to the pattern. Real scale is mm
    tod: is used to get the size of the image
    phi: phase of the target
   limtar: limits of the image that define the the position of the target (in px)
   initial: initial guess for d, cx,cy (cx,cy are the center of the target) (inside the window defined by limtar)
   inlim: bounds of the initial guess
   showerr: print and graph the error
    '''
    def valores(v0): # d,cx,cy
        ny,nx = np.shape(tod[0])

        xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * v0[0], (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * v0[0]
        xr,yr = np.meshgrid(xr,yr)

        pary,cov = curve_fit(lin, yr[:,200], phi[:,200])
        m2 = mrot( -np.arctan(pary[0]) )
        yro,phiy = rot(yr,phi,m2)
        parx,cov = curve_fit(lin, xr[200], phiy[200,:])
        m2 = mrot( -np.arctan(parx[0]) )
        xro, fase = rot(xr,phiy,m2)

        pat = patron(xro,yro,int(v0[1]),int(v0[2]))

        fase -= np.mean(fase)
        pat -= np.mean(pat)

        para, cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(1000,350,1), bounds=( (300,150,0.1),(5500,1000,10.0) ) ) 

        haj = alt(fase,*para) 
        error = (pat-haj)[50:-50,50:-50].flatten()
        return error

    calib = least_squares( valores, [initial[0],initial[1],initial[2]], diff_step=(0.0001,1,1), \
                          bounds=( [initial[0]-inlim[0],initial[1]-inlim[1],initial[2]-inlim[2]] ,\
                                   [initial[0]+inlim[0],initial[1]+inlim[1],initial[2]+inlim[2]] )  )     
        
    d, cx, cy = calib.x[0], int(calib.x[1]), int(calib.x[2])
    
    ny,nx = np.shape(tod[0])

    xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
    xr,yr = np.meshgrid(xr,yr)

    pary,cov = curve_fit(lin, yr[:,100], phi[:,100])
    m2 = mrot( -np.arctan(pary[0]) )
    yro,phiy = rot(yr,phi,m2)
    parx,cov = curve_fit(lin, xr[100], phiy[100,:])
    m2 = mrot( -np.arctan(parx[0]) )
    xro, fase = rot(xr,phiy,m2)

    pat = patron(xro,yro,cx,cy)

    fase -= np.mean(fase)
    pat -= np.mean(pat)

    (L,D,w), cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(2500,560,0.2), bounds=( (300,150,0.01),(5500,1000,10.0) ) ) 

    haj = alt(fase,L,D,w) 
    err = (pat-haj)[50:-50,50:-50].flatten()
     
    if showerr:
        print( np.mean(err), np.std(err) )
        xerr=np.linspace(-2,2,1000)
        plt.figure()
        plt.hist( err, bins=200, density=True)
        plt.plot(xerr,guass(xerr,np.mean(err),np.std(err)))
        plt.show()
    
    return d, L, D, w


def detect_phases(file,fpass,barl,fline,difl,limbl,limgr, npa=0):
    '''
    Detects frames to use for psp.
    '''
    with imageio.get_reader(file,mode='I') as vid:
            
        bars = np.zeros((fpass, barl[1]-barl[0] , barl[3]-barl[2] ))
        for j,i in enumerate(range(fpass*npa, fpass*(npa+1))):
            barr = vid.get_data(i)[barl[0]:barl[1],barl[2]:barl[3]]
            bars[j] = barr 
        bars = bars / np.max(bars)
        
        
    vals = np.array([ fline - difl * i + j for i in range(10) for j in range(-2,3)])
    x,y = np.arange(barl[3]-barl[2]), np.arange(barl[1]-barl[0])
    xx,yy = np.meshgrid(x,y)
    
    filt = np.isin(yy,vals ) 
    
    barf = np.mean( -( bars * filt + (-1*filt+1) ), axis=2 )
    
    lpn, lpg = [],[]
    for i in range(0,fpass):
        pen = find_peaks(barf[i], height=-limbl, distance=10 )[0]
        peg = find_peaks(barf[i], height=(-limgr[1],-limgr[0]), distance=10 )[0]
        lpn.append(len(pen))
        lpg.append(len(peg))
        
    lpn, lpg = np.array(lpn), np.array(lpg)
    # lpb = lpn * (lpg == 0) + (lpg==1) * (lpn==1) * 1
    lpb = lpn * (lpg == 0) + (lpg>=1) * (lpn==1) * 1
    
    grad = lpn[4:] * lpn[:-4] / 10 #np.gradient(lpn)
    maxis = np.where( grad == 10 )[0]
    ind = np.argmax( maxis[1:]-maxis[:-1] ) 
    pind,uind = maxis[ind] + 4, maxis[ind+1] + 4
    
    frames = np.arange(pind,uind)
    frm = lpb[pind:uind]
    
    frames, frv = frames[frm>0], frm[frm>0]
    frames = frames[ np.abs(frv - np.roll(frv,1)) > 0 ]
    return frames


def all_frames(file,frames, refr=2):
    with imageio.get_reader(file,mode='I') as vid:
        ref = vid.get_data(frames[0]-refr)    
    
        nx,ny = np.shape(ref)
        
        tod = np.zeros((len(frames),nx,ny))
        
        for i in range(len(frames)):
            ima = vid.get_data(frames[i])
            tod[i] = ima*1. - ref*1. 
    return tod, ref

def st_psm(tod, frecline=200, initfrec=2, reduce=False, xlims=[0,1000], sumA = -1):
    nt,ny,nx = np.shape(tod)
    #step 1: PSP to get A and B
    A, B = 0,0
    for i in range(nt):
        A += tod[i]
        B += tod[i] * np.exp(-1j * 2*np.pi*i/nt)
    A, B = np.abs(A)/nt, np.abs(B) * 2/nt

    tod_b = (tod + sumA * A) / (B + 0.000001) 
    # step 2: sample Moire for all images
    Am,BPm = 0,0

    im = tod_b[0]

    if not reduce:
        fto = np.fft.fft( im[frecline] ) [initfrec:int(nx/2)]
        kfo = np.fft.fftfreq(nx)
        ind = np.argmax(np.abs(fto))
        ne = int(1/kfo[ind + initfrec])
    if reduce:
        numx = xlims[1] - xlims[0]
        fto = np.fft.fft( im[frecline,xlims[0]:xlims[1]] ) [initfrec:int(numx/2)]
        kfo = np.fft.fftfreq(numx)
        ind = np.argmax(np.abs(fto))
        ne = int(1/kfo[ind + initfrec])

    for t in range(nt):
        im = tod_b[t]
        
        x = np.arange(nx)
        for e in range(ne):
            imt = im[:,e::ne]
            
            # cs = CubicSpline(x[e::ne], imt,axis=1 )#, bc_type='not-a-knot')
            # imc = cs(x) 
            
            cs = make_interp_spline(x[e::ne],imt,1,axis=1) #3rd input could be 1 or 3 (spline degree/ polinomial order)
            imc = cs(x) 
            
    # step 3: calculate the variables
            Am += imc
            BPm += imc * np.exp(-1j * 2*np.pi*e/ne) * np.exp(-1j * 2*np.pi*t/nt) 

    Am = np.abs(Am) / (nt*ne)
    Bm = np.abs(BPm) * 2/(nt*ne)

    Pm = np.angle(BPm)
    # Pm = np.unwrap(np.unwrap(Pm),axis=0)
    
    corr = ((np.cumsum( np.ones_like(Pm),axis=1) - 1)/nx) * 2*np.pi/ne # correction described in paper
    psr = Pm + corr

    return psr, ne, Am, Bm

#%%
t1 = time()

file = './Melting/23-04-19_Calibration/Calibration_C001H001S0004.tif'
nframes = 560 # total number of frames
fpass = 140 # number of frames for each pass
barl = [195,870,270,370] # [up, bottom, left, right] -> phase counting bar
fline, difl =  670,73 # where line starts and line separation -> for filter
limbl = 0.3 # max limit of black values
limgr = [0.2,0.6] # [min,max] limit of gray values
limtar = [150,860,480,1200] # [up, bottom, left, right] -> target limits

frames = detect_phases(file, fpass, barl, fline, difl, limbl, limgr)
tod, ref = all_frames(file, frames)
phi, ne, _,_ = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]],initfrec=2)

t2 = time()
print(t2 - t1)
#%%
phi = np.unwrap(phi,axis=1)
t1 = time()
d,L,D,w = calibrate_params(tod, phi, limtar, [0.1881,355,360])
t2 = time()
print(d,L,D,w)
print(t2-t1)
#%%

#%%
n = 4

# phi, ne, A,B = st_psm(tod[:,150:860,480:1200],initfrec=2)
phid, _,_,_ = st_psm(tod, initfrec=10, reduce=True, xlims=[500,1200])
#%%
t1 = time()

nframes = 819 # total number of frames in tif  (623 for the last one)
fpass = 140 # number of frames for each pass
barl = [190,870,270,370] # [up, bottom, left, right] -> phase counting bar
fline, difl =  675,74 # where line starts and line separation -> for filter
limbl = 0.3 # max limit of black values
limgr = [0.25,0.5] # [min,max] limit of gray values
limtar = [110,860,520,1030] # [up, bottom, left, right] -> target or ice limits

ff = 0
file = './Melting/23-04-19_Vertical_block/Vertical_block_C001H001S0002-0'+str(ff)+'.tif'

# with imageio.get_reader(file,mode='I') as vid:
        
#     bars = np.zeros((fpass, barl[1]-barl[0] , barl[3]-barl[2] ))
#     for i in range(fpass):
#         barr = vid.get_data(i)[barl[0]:barl[1],barl[2]:barl[3]]
#         bars[i] = barr 
#     bars = bars / np.max(bars)
    
    
# vals = np.array([ fline - difl * i + j for i in range(10) for j in range(-2,3)])
# x,y = np.arange(barl[3]-barl[2]), np.arange(barl[1]-barl[0])
# xx,yy = np.meshgrid(x,y)

# filt = np.isin(yy,vals ) 

# barf = np.mean( -( bars * filt + (-1*filt+1) ), axis=2 )

# lpn, lpg = [],[]
# for i in range(0,fpass):
#     pen = find_peaks(barf[i], height=-limbl, distance=10 )[0]
#     peg = find_peaks(barf[i], height=(-limgr[1],-limgr[0]), distance=10 )[0]
#     lpn.append(len(pen))
#     lpg.append(len(peg))
    
# lpn, lpg = np.array(lpn), np.array(lpg)
# lpb = lpn * (lpg == 0) + (lpg>=1) * (lpn==1) * 1

# grad = lpn[4:] * lpn[:-4] / 10 #np.gradient(lpn)
# maxis = np.where( grad == 10 )[0]
# ind = np.argmax( maxis[1:]-maxis[:-1] ) 
# pind,uind = maxis[ind] + 4, maxis[ind+1] + 4

# frames = np.arange(pind,uind)
# frm = lpb[pind:uind]

# frames, frv = frames[frm>0], frm[frm>0]
# frames = frames[ np.abs(frv - np.roll(frv,1)) > 0 ]

# phases = np.zeros((5,limtar[1]-limtar[0], limtar[3]-limtar[2]))
phases = []
for i in range(5):
    frames = detect_phases(file, fpass, barl, fline, difl, limbl, limgr, npa=i)
    tod, ref = all_frames(file, frames,refr=3)
    phiic, ne, _,_ = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]], initfrec=10)
    # phases[i] = phiic
    phases.append(phiic)

phases = np.array(phases)

t2 = time()
print(t2 - t1)

#%%
plt.figure()
plt.imshow(hice)
plt.show()
#%%
ny,nx = np.shape(phiic)
x,y = np.arange(nx),np.arange(ny)
xx,yy = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection='3d')

for ni in [0,1]:
    ax.contour3D(xx, yy, phiic, 100)#, cmap='jet')

ax.set_box_aspect([3,3,1])
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
ax.set_zlabel('phase (rad)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(30, -60,vertical_axis='z')
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()
#%%
pp = np.unwrap(phases, axis=0)

plt.figure()
# for ni in range(10):
#     plt.plot(phases[ni,:,100],label=ni)
#     # plt.plot(pp[ni,:,100],'--',label=ni)
plt.plot(phases[:,50,50],'.-')
plt.plot(pp[:,50,50],'.-')
# plt.legend()
plt.grid()


#%%
pp = np.unwrap(phases, axis=0)


ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

for ni in range(5):
    if ni%2 == 0:
        ax.contour3D(xr, pp[ni], yr, 200)#, cmap='jet')
    if ni%2 == 1:
        ax.contour3D(xr, pp[ni], yr, 200, cmap='jet')

# ni = 0
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(30, -60,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()


#%%
n = 73
plt.figure()
plt.plot(barf[n])
plt.show()
plt.figure()
plt.imshow(bars[n])
plt.show()

#%%
for n in range(7):
    plt.figure()
    plt.imshow(tod[n], cmap='gray')
    plt.show()
#%%
plt.figure()
plt.imshow(ref[110:860,520:1030], cmap='gray', vmax=8000)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tod[1,110:860,520:1030], cmap='gray')
plt.show()
#%%

totim, imfile = 3080, 819
file = ['./Melting/23-04-19_Vertical_block/Vertical_block_C001H001S0002-0','.tif']

t1 = time()

allims = np.array([ [int(i/imfile),i%imfile] for i in range(totim) ])

# n = 5
for n in [0,5]: # range(22):
    imsn = allims[n*140 : (n+1)*140 ]
    
    nfil = [ imsn[0,0], imsn[-1,0] ]
    
    cam = np.where( ( imsn[:,0] - np.roll(imsn[:,0],1) ) == 1  )[-1]
    
    impos = []
    if len(cam) > 0:
        impos.append( imsn[:cam[0],1] )
        impos.append( imsn[cam[0]:,1] )
    else:
        impos.append( imsn[:,1] )
        
    for i,j in enumerate( range(nfil[0], nfil[1]+1 ) ):        
        filn = file[0]+str(j)+file[1] 
        with imageio.get_reader(filn,mode='I') as vid:        
            for k in impos[i]:
                barr = vid.get_data(k)
                # print(n, j, k)

    

t2 = time()
print(t2 - t1)




#%%
# =============================================================================
# Bueno?
# =============================================================================


#%%
def detect_phases_2(file,fpass,barl,fline,difl,limbl,limgr, totim, imfile, npa=0):
    '''
    Detects frames to use for psp.
    '''
    allims = np.array([ [int(i/imfile),i%imfile] for i in range(totim) ])
    
    imsn = allims[npa*140 : (npa+1)*140 ]
    
    nfil = [ imsn[0,0], imsn[-1,0] ]
    
    cam = np.where( ( imsn[:,0] - np.roll(imsn[:,0],1) ) == 1  )[-1]
    
    impos = []
    if len(cam) > 0:
        impos.append( imsn[:cam[0],1] )
        impos.append( imsn[cam[0]:,1] )
    else:
        impos.append( imsn[:,1] )
        
    bars = np.zeros((fpass, barl[1]-barl[0] , barl[3]-barl[2] ))
    cont = 0
    for k,l in enumerate( range(nfil[0], nfil[1]+1 ) ):        
        filn = file[0]+str(l)+file[1] 
        
        with imageio.get_reader(filn,mode='I') as vid:
                
            for j in impos[k]:
                barr = vid.get_data(j)[barl[0]:barl[1],barl[2]:barl[3]]
                bars[cont] = barr
                cont += 1
    bars = bars / np.max(bars)
        
        
    vals = np.array([ fline - difl * i + j for i in range(10) for j in range(-2,3)])
    x,y = np.arange(barl[3]-barl[2]), np.arange(barl[1]-barl[0])
    xx,yy = np.meshgrid(x,y)
    
    filt = np.isin(yy,vals ) 
    
    barf = np.mean( -( bars * filt + (-1*filt+1) ), axis=2 )
    
    lpn, lpg = [],[]
    for i in range(0,fpass):
        pen = find_peaks(barf[i], height=-limbl, distance=10 )[0]
        peg = find_peaks(barf[i], height=(-limgr[1],-limgr[0]), distance=10 )[0]
        lpn.append(len(pen))
        lpg.append(len(peg))
        
    lpn, lpg = np.array(lpn), np.array(lpg)
    # lpb = lpn * (lpg == 0) + (lpg==1) * (lpn==1) * 1
    lpb = lpn * (lpg == 0) + (lpg>=1) * (lpn==1) * 1
    
    grad = lpn[4:] * lpn[:-4] / 10 #np.gradient(lpn)
    maxis = np.where( grad == 10 )[0]
    ind = np.argmax( maxis[1:]-maxis[:-1] ) 
    pind,uind = maxis[ind] + 4, maxis[ind+1] + 4
    
    frames = np.arange(pind,uind)
    frm = lpb[pind:uind]
    
    frames, frv = frames[frm>0], frm[frm>0]
    frames = frames[ np.abs(frv - np.roll(frv,1)) > 0 ]
    return imsn[frames]


def all_frames_2(file,frames, refr=2):
    filr = file[0]+str(frames[0,0])+file[1] 
    with imageio.get_reader(filr,mode='I') as vid:
        ref = vid.get_data(frames[0,1]-refr)    
    
        nx,ny = np.shape(ref)
        
    tod = np.zeros((len(frames),nx,ny))
    
    nfil = [frames[0,0] , frames[-1,0] ]
    
    cam = np.where( ( frames[:,0] - np.roll(frames[:,0],1) ) == 1  )[-1]
    
    impos = []
    if len(cam) > 0:
        impos.append( frames[:cam[0],1] )
        impos.append( frames[cam[0]:,1] )
    else:
        impos.append( frames[:,1] )
    
    cont = 0
    for k,l in enumerate( range(nfil[0], nfil[1]+1 ) ):        
        filn = file[0]+str(l)+file[1] 
        
        with imageio.get_reader(filn,mode='I') as vid:
            
            for i in impos[k]:
                ima = vid.get_data(i)
                tod[cont] = ima*1. - ref*1. 
                cont += 1
                
    return tod, ref




#%%
d,L,D,w = 0.18670285960255195, 690.1128021406407, 488.2446364317459, 0.6273367397624637

nframes = 819 # total number of frames in tif  (623 for the last one)
fpass = 140 # number of frames for each pass
barl = [190,870,270,370] # [up, bottom, left, right] -> phase counting bar
fline, difl =  675,74 # where line starts and line separation -> for filter
limbl = 0.3 # max limit of black values
limgr = [0.25,0.5] # [min,max] limit of gray values
limtar = [350,700,600,950] # [up, bottom, left, right] -> target or ice limits

totim, imfile = 3080, 819
file2 = ['./Melting/23-04-19_Vertical_block/Vertical_block_C001H001S0002-0','.tif']


t1 = time()

phases = []
refs = []
# tods = [] 
for npa in tqdm(range(18)):
    frames2 = detect_phases_2(file2,fpass,barl,fline,difl,limbl,limgr, totim, imfile, npa=npa)

    refr = 2
    if npa == 3: refr = 3

    tod2, ref2 = all_frames_2(file2,frames2, refr=refr)
    
    todc = tod2[:,limtar[0]:limtar[1],limtar[2]:limtar[3]]
    vm = np.mean( todc, axis=(1,2) )
    vs = np.std( todc, axis=(1,2) )
    todn = ((vm - todc.T) / vs).T

    phiic, ne, _,_ = st_psm(todn, initfrec=10)
    # phases[i] = phiic
    phases.append(phiic)
    refs.append(ref2)
    # tods.append(tod2)

phases = np.array(phases)

t2 = time()
print(t2-t1)
#%%
#solo el primero

limtar = [100,870,520,1030] 

phases = []
refs = []
# tods = [] 
for npa in tqdm(range(1)):
    frames2 = detect_phases_2(file2,fpass,barl,fline,difl,limbl,limgr, totim, imfile, npa=npa)

    refr = 2
    if npa == 3: refr = 3

    tod2, ref2 = all_frames_2(file2,frames2, refr=refr)
    
    todc = tod2[:,limtar[0]:limtar[1],limtar[2]:limtar[3]]
    vm = np.mean( todc, axis=(1,2) )
    vs = np.std( todc, axis=(1,2) )
    todn = ((vm - todc.T) / vs).T

    phiic, ne, _,_ = st_psm(todn, initfrec=10)
    # phases[i] = phiic
    phases.append(phiic)
    refs.append(ref2)
    # tods.append(tod2)

phases = np.array(phases)
pp = np.unwrap(phases,axis=1)




#%%
pp = np.unwrap( np.unwrap(phases, axis=0), axis =2)

n = 0
plt.figure()
plt.imshow(phases[n])
plt.show()

plt.figure()
plt.imshow(pp[n])
plt.show()
#%%
npa = 17

# vm = np.mean( tods[npa], axis=(1,2) )
# vs = np.std( tods[npa], axis=(1,2) )
# vma = np.max( tods[npa], axis=(1,2) )

# todn = ((vm - tods[npa].T) / vs).T 

# for n in range(20):
plt.figure()
# plt.imshow(tods[npa][n] , cmap = 'gray', vmax =10000, vmin =1000)
# plt.imshow(todn[n,limtar[0]:limtar[1],limtar[2]:limtar[3]] , cmap = 'gray', vmax =2, vmin =-2)
# plt.imshow(refs[npa][limtar[0]:limtar[1],limtar[2]:limtar[3]] , cmap = 'gray')
plt.imshow(refs[npa] , cmap = 'gray', vmax=10000)
plt.colorbar()
plt.title(str(npa)+ ' ' +str(n))
plt.show()

plt.figure()
plt.imshow( alts[npa]  )
plt.show()
#%%
# pp = np.unwrap(phases, axis=0)

altsd = alt(phases, L, D, w) 
alts = alt(pp, L, D, w)


plt.figure()
# plt.plot(altsd[:,100,130],'.--')
# plt.plot(alts[:,100,130],'.-')

# plt.plot(altsd[:,100,150],'.--')
# plt.plot(alts[:,100,150],'.-')

plt.plot( np.mean(altsd, axis = (1,2)) ,'.--')
plt.plot( np.mean(alts, axis = (1,2)) ,'.-')

# plt.plot(altsd[7,100,:],'-')
# plt.plot(alts[7,100,:],'-')

plt.grid()
plt.show()
#%%

ny,nx = np.shape(tod2[0])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

# for ni in [0]:
#     if ni%2 == 0:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200)#, cmap='jet')
#     if ni%2 == 1:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200, cmap='jet')

ni = 0
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')
ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 300) #, cmap='jet')

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(17, 120,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()
#%%
# pp = np.unwrap( np.unwrap(phases, axis=0), axis =2)
pp = np.unwrap(np.unwrap( np.unwrap(phases, axis=0),axis=1),axis=2)

d,L,D,w = 0.23990251906028134, 5499.870962858253, 814.7594549835866, 2.549666044116526 #for inclined block
hices = alt(pp,L,D,w )

from matplotlib.colors import LightSource, Normalize

ny,nx = np.shape(tod2[17])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)

# fase,xro,yro = rotar_caram(haj, xr, yr)
hice = hices[39]
# hice = alt(pp[0],L,D,w )

 
# hicf = np.fft.fft2(hice)
# kx,ky = np.fft.fftshift(np.fft.fftfreq(len(hicf))), np.fft.fftshift(np.fft.fftfreq(np.shape(hicf)[1]))
# kkx,kky = np.meshgrid(ky,kx)
# kk = np.sqrt( kkx**2 + kky**2 )
# sig = 0.05
# ga = np.fft.fftshift( np.exp( -(kk / sig)**2 ) )
# hicesu = np.real(np.fft.ifft2( hicf * ga ))

az,al = 270 , 10
ver,fra = 20. , 0.5

ls = LightSource(azdeg=az,altdeg=al)

# shade data, creating an rgb array.
rgb1 = ls.shade(hice,plt.cm.Blues) #, blend_mode='soft')
# rgb = ls.hillshade(hice,vert_exag=ver,fraction=fra , dx = 1, dy =1)

rgb = ls.shade_rgb(rgb1, hice, vert_exag=ver,fraction=fra)




color_dimension = rgb # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xr,hice,yr, facecolors=fcolors, rcount=250,ccount=250) #for 1057
# ax.plot_surface(xr[70:-70,70:-70],hice[70:-70,70:-70],yr[70:-70,70:-70], facecolors=fcolors, rcount=250,ccount=250) #for 1057

# ax.plot_surface(xro[:1100],yro[:1100],fase[:1100], facecolors=fcolors, rcount=150,ccount=150) #for 1057
# ax.plot_surface(xr,yr,haj, facecolors=fcolors) #for 1048

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.set_title('3D contour')
# ax.invert_zaxis()
ax.invert_xaxis()
# ax.invert_zaxis()
ax.set_ylim(-100,17)
# ax.set_yticks([-20,-10,0,10])
# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
ax.view_init(17,120)
# plt.savefig('imageni2.png',bbox_inches='tight',dpi=1000)
plt.show()
#%%
plt.figure()
plt.plot(hice[100])
plt.plot(hice[:,100])
plt.show()

#%%
# pp = np.unwrap( np.unwrap(phases, axis=0), axis =2)
pp = np.unwrap(np.unwrap( np.unwrap(phases, axis=0),axis=1),axis=2)
hices = alt(pp,L,D,w )

ny,nx = np.shape(tod2[17])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)

# az,al = 90 , 0
# ver,fra = 20. , 0.1
az,al = 270 , 10
ver,fra = 20. , 0.1


lista_im = []
t_spl = np.linspace(0,1,1000)
for num in tqdm(range(40)): #1251

    plt.close('all')
    plt.ioff()
    
    hice = hices[num]
    # hicf = np.fft.fft2(hice)
    # kx = np.fft.fftshift(np.fft.fftfreq(len(hicf)))
    # kkx,kky = np.meshgrid(kx,kx)
    # kk = np.sqrt( kkx**2 + kky**2 )
    # sig = 0.1
    # ga = np.fft.fftshift( np.exp( -(kk / sig)**2 ) )
    # hicesu = np.real(np.fft.ifft2( hicf * ga ))

    ls = LightSource(azdeg=az,altdeg=al)

    # shade data, creating an rgb array.
    rgb1 = ls.shade(hice,plt.cm.Blues)
    # rgb = ls.hillshade(hice,vert_exag=ver,fraction=fra)
    rgb = ls.shade_rgb(rgb1, hice, vert_exag=ver,fraction=fra)

    color_dimension = rgb # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(xr,hice,yr, facecolors=fcolors, rcount=250,ccount=250) #for 1057
    # ax.plot_surface(xr[50:-50,50:-50],hice[50:-50,50:-50],yr[50:-50,50:-50], facecolors=fcolors, rcount=250,ccount=250) #for 1057
    # ax.plot_surface(xro[:1100],yro[:1100],fase[:1100], facecolors=fcolors, rcount=150,ccount=150) #for 1057
    # ax.plot_surface(xr,yr,haj, facecolors=fcolors) #for 1048

    ax.set_box_aspect([3,2,3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    # ax.set_title('3D contour')
    # ax.invert_zaxis()
    ax.invert_xaxis()
    
    # ax.set_ylim(-23,17)
    ax.set_ylim(-100,17)
    
    # ax.set_yticks([-20,-10,0,10])
    ax.set_yticks([0,-25,-50,-75,-100])
    
    # ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
    ax.view_init(17,120)

    plt.savefig('imageni.png',bbox_inches='tight',dpi=500)
    lista_im.append(imageio.imread('imageni.png'))
#%%
# [350,700,600,950]

lista_im = []
t_spl = np.linspace(0,1,1000)
for num in tqdm(range(40)): #1251

    plt.close('all')
    plt.ioff()
    
    plt.figure()
    plt.imshow( refs[num], cmap='gray') # , vmax = 10000, vmin= 100)
    plt.plot( [limtar[2],limtar[3],limtar[3],limtar[2],limtar[2]] , [limtar[0],limtar[0],limtar[1],limtar[1],limtar[0]], 'r-' )
    
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

    # plt.xlim(410,1080)
    # plt.ylim(930,60)
    plt.xlim(250,800)
    plt.ylim(750,100)
    
    plt.savefig('imageni.png',bbox_inches='tight',dpi=500)
    lista_im.append(imageio.imread('imageni.png'))
#%%
imageio.mimsave('./Melting/inc_block2.gif',lista_im,fps=24)

#%%
fil = './Melting/23-04-19_Vertical_block/Vertical_block_C001H001S0002-00.tif'
lista_im = []

for num in tqdm(range(72,138)): #1251

    plt.close('all')
    plt.ioff()
    
    plt.figure()
    with imageio.get_reader(fil,mode='I') as vid:
        ref = vid.get_data(69)
        ref = (ref - np.mean(ref)) / np.std(ref)
        
        im = vid.get_data(num)
        im = (im - np.mean(im)) / np.std(im)
        
        plt.imshow( im , cmap='gray', vmax = 5, vmin = -2) #, vmax = 50000, vmin= 50)
        # plt.plot( [limtar[2],limtar[3],limtar[3],limtar[2],limtar[2]] , [limtar[0],limtar[0],limtar[1],limtar[1],limtar[0]], 'r-' )
        
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.xlim(410,1080)
    plt.ylim(930,60)

    plt.savefig('imageni.png',bbox_inches='tight',dpi=300)
    lista_im.append(imageio.imread('imageni.png'))


#%%
# imageio.mimsave('./Melting/pattern_block.gif',lista_im,fps=20)

plt.figure()
plt.imshow(im , cmap='gray', vmax = 5, vmin = -2)
# plt.imshow(im-ref , cmap='gray', vmax = 2, vmin = -1)
plt.colorbar()
# plt.xlim(410,1080)
# plt.ylim(930,60)
# plt.xlim(520,1030)
# plt.ylim(870,100)
plt.show()












#%%

a = np.array([[1,1,1,1],
              [2,2,2,2]])
b = np.array([1,1])
# c = np.array([[1],[1]])
print(np.shape(a), np.shape(b) )
print(a)
print( (a.T - b).T )


#%%
def all_steps(file, barl, fpass, imfile, totim, bhei = 0.6, npa=0, dist=10):
    
    allims = np.array([ [int(i/imfile),i%imfile] for i in range(totim) ])
    imsn = allims[npa*fpass : (npa+1)*fpass ]

    ims = [0 for i in range(21)]
    cuenti = [0 for i in range(21)]

    for j in range(fpass):
        with imageio.get_reader(file[0]+ str( imsn[j,0] ) +file[1],mode='I') as vid:
            
            im = vid.get_data( imsn[j,1] )
            bar = im[barl[0]:barl[1],barl[2]:barl[3]]    
            bar = bar #/ np.max(bar)
            
            peak = find_peaks( -np.mean(bar,axis=1), height=-bhei, distance=dist )[0]
            paso = len(peak)
            
            # print(len(cuenti),paso)
            cuenti[paso] += 1
            ims[paso] = ims[paso] *  (cuenti[paso]-1) / (cuenti[paso]) + im / (cuenti[paso])
                
    ims = np.array(ims)
    ref = ims[0]
    tod = ims[1:] - ref
    
    return tod, ref






#%%
# =============================================================================
# Detectar con lineas
# =============================================================================

totim, imfile = 230, 230 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [165,730,100,160] # [up, bottom, left, right] -> phase counting bar
limtar = [100,650,310,870] # [up, bottom, left, right] -> target or ice limits

npa = 0
file = ['./Melting/23-04-25_Calibration_1/Calibration(1)_C001H001S0002-','.tif']

t1 = time()

tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=0.4)
phi, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]])
phi = np.unwrap(np.unwrap(phi, axis=0),axis=1)

d,L,D,w = calibrate_params(tod,phi, limtar, [0.243,280,275], inlim=[0.002,20,20] ,showerr=True)

t2 = time()
print(t2-t1)
print(d,L,D,w)
#%%
d, L, D, w = 0.24249687717704455, 5496.216937072147, 825.8193972643186, 2.59840574825902
t1 = time()

totim, imfile = 4830, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [165,730,100,160] # [up, bottom, left, right] -> phase counting bar
limtar = [250,550,400,650] # [up, bottom, left, right] -> target or ice limits

file = ['./Melting/23-04-25_Ice_block_0/Ice_block_0_C001H001S0001-0','.tif']

phases = []
for npa in tqdm(range(40)):
    tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=0.6)
    phase, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]])
    phases.append(phase)
    
phases = np.array(phases)
# problemas con varias fases en 4 y 10

t2 = time()
print(t2-t1)
#%%
pp = np.unwrap(phases,axis=1)

n = 17
plt.figure()
plt.imshow(pp[n])
plt.show()
plt.figure()
plt.plot(phases[:,100,100],'.-')
plt.plot(pp[:,100,100],'.--')
plt.show()
#%%
# pp = np.unwrap(np.unwrap(phases,axis=2),axis=0)
pp = np.unwrap(phases,axis=2)
hice = alt(pp, L,D,w)


ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

for ni in [17]:
    if ni%2 == 0:
        ax.contour3D(xr, hice[ni], yr, 200)#, cmap='jet')
    if ni%2 == 1:
        ax.contour3D(xr, hice[ni] , yr, 200, cmap='jet')

# ni = 0
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')
# ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200) #, cmap='jet')

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(30, -60,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()





#%%
t1 = time()
totim, imfile = 230, 230 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [130,690,280,850] # [up, bottom, left, right] -> target or ice limits

npa = 0
file = ['./Melting/23-04-26_Calibration_3/Calibration_3_C001H001S0001-','.tif']

tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=23000,dist=15)
phi, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]], sumA = 1)
phi = np.unwrap(np.unwrap(phi, axis=0),axis=1)

d,L,D,w = calibrate_params(tod,phi, limtar, [0.241,280,285], inlim=[0.004,40,40] ,showerr=True)


t2= time()
print(t2-t1)
print(d,L,D,w)
#%%
t1 = time()

totim, imfile = 4945, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [280,600,400,650] # [up, bottom, left, right] -> target or ice limits

file = ['./Melting/23-04-26_Ice_block_45/Ice_block_45(2)_C001H001S0001-0','.tif']

phases = []
refs = []
backs = []
for npa in tqdm(range(40)):
    tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000, dist=16)
    phase, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]], sumA=1)
    phases.append(phase)
    refs.append(ref)
    backs.append(np.sum(tod,axis=0))
    
phases = np.array(phases)
# problemas con varias fases en 4 y 10

# npa = 0
# tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000,dist=15)
# phase, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]])


t2 = time()
print(t2-t1)


#%%
pp = np.unwrap(np.unwrap( np.unwrap(phases, axis=0),axis=1),axis=2)
n = 39
# vn = tod[n,limtar[0]:limtar[1],limtar[2]:limtar[3]] - np.sum(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]],axis=0)/20

# plt.figure()
# # plt.imshow(refs[n],cmap='gray')
# plt.imshow(tod[0],cmap='gray')
# # plt.plot(vn[300])
# plt.show()

plt.figure()
plt.imshow(refs[n],cmap='gray')
plt.plot( [limtar[2],limtar[3],limtar[3],limtar[2],limtar[2]] , [limtar[0],limtar[0],limtar[1],limtar[1],limtar[0]], 'r-' )
# plt.plot(pp[0])
plt.colorbar()
plt.xlim(250,800)
plt.ylim(750,100)
plt.show()
#%%
ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

# for ni in [0]:
#     if ni%2 == 0:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200)#, cmap='jet')
#     if ni%2 == 1:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200, cmap='jet')

ni = 39
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')
ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 300) #, cmap='jet')

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(17, 120,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()


#%%
plt.figure()
plt.plot(alt(pp[ni],L,D,w )[0])
plt.show()




#%%
bhei = 30000
dist = 16

npa = 1

allims = np.array([ [int(i/imfile),i%imfile] for i in range(totim) ])
imsn = allims[npa*fpass : (npa+1)*fpass ]

ims = [0 for i in range(21)]
cuenti = [0 for i in range(21)]

plt.figure()

for j in range(fpass):
    with imageio.get_reader(file[0]+ str( imsn[j,0] ) +file[1],mode='I') as vid:
        
        im = vid.get_data( imsn[j,1] )
        bar = im[barl[0]:barl[1],barl[2]:barl[3]]    
        bar = bar #/ np.median(bar)
        
        peak = find_peaks( -np.mean(bar,axis=1), height=-bhei, distance=dist, prominence=0.0005 )[0]
        paso = len(peak)
        
        x = np.arange(len(bar))
        plt.plot( -np.mean(bar,axis=1) )
        plt.plot(x[peak], -np.mean(bar,axis=1)[peak],'r.')
        
        if paso > 20: print(paso, j)
plt.show()
#%%
x = np.arange(len(bar))
plt.figure()
# plt.imshow(bar)
plt.plot(-np.mean(bar,axis=1))
plt.plot(x[peak], -np.mean(bar,axis=1)[peak],'r.')
plt.show()


#%%
# =============================================================================
# FTP
# =============================================================================

def dphase(defo,thx,thy,ns,inde=2,y_orientation=True):
    # defino un par de parametros
    # inde: para que no cuente el pico central
    # thx,thy: controlan ancho del filtro. ns: controla la pendiente del filtro

    ny,nx = np.shape(defo)
    # transformo en fourier y encuentro el maximo
    fY0 = np.fft.fftshift( np.fft.fft2(defo) )
    
    if y_orientation: cor = np.abs(fY0[:,0:int(np.floor(nx/2))-inde])
    else: cor = np.abs(fY0[0:int(np.floor(ny/2))-inde,:])
        
    imax = np.unravel_index(cor.argmax(), cor.shape)
    ifmax_y, ifmax_x = imax[0], imax[1]

    # defino el filtro (distribucion tukey)
    HW_x, HW_y = np.round(ifmax_x*thx), np.round(ifmax_y*thy)
    W_x, W_y = 2*HW_x, 2*HW_y
    win_x, win_y = signal.tukey(int(W_x),ns), signal.tukey(int(W_y),ns)
    wxx, wyy = np.meshgrid(win_x,win_y)
    win = wxx * wyy

    gaussfilt1D = np.zeros((ny,nx))
    gaussfilt1D[int(ifmax_y-HW_y):int(ifmax_y+HW_y), int(ifmax_x-HW_x):int(ifmax_x+HW_x)] = win[:,:]

    # multiplico las transformadas por el filtro y antitransformo 
    Nfy0 = np.fft.ifftshift(fY0*gaussfilt1D)

    Ny0 = np.fft.ifft2(Nfy0)

    # calculo las fases, hago unwrap, y encuentro la diferencia
    phase0 = np.angle(Ny0)
    
    k_x,k_y = np.fft.fftshift( np.fft.fftfreq(nx) ), np.fft.fftshift( np.fft.fftfreq(ny) )
    wx,wy = 2*np.pi* k_x[ifmax_x], 2*np.pi* k_y[ifmax_y]
    
    return phase0, np.array([wx,wy]), fY0, gaussfilt1D, [ifmax_x,ifmax_y]
#%%
totim, imfile = 4945, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [280,600,390,600] # [up, bottom, left, right] -> target or ice limits

thx,thy, ns = 0.05, 0.04, 0.8

file = ['./Melting/23-04-26_Ice_block_45/Ice_block_45(2)_C001H001S0001-0','.tif']

# phases = []
# refs = []
# for npa in tqdm(range(40)):
#     tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000, dist=16)
#     phase, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]])
#     phases.append(phase)
#     refs.append(ref)
    
# phases = np.array(phases)
# problemas con varias fases en 4 y 10

npa = 30
tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000,dist=15)
dp, _,fy0,ga,ifm = dphase( (tod[0]-ref)[limtar[0]:limtar[1],limtar[2]:limtar[3]] ,thx,thy,ns,inde=2,y_orientation=True)
#%%
d,L,D,w = 0.23990251906028134, 5499.870962858253, 814.7594549835866, 2.549666044116526
pp = np.unwrap(np.unwrap(dp,axis=0),axis=1)

# plt.figure()
# plt.imshow(tod[0] - ref, cmap='gray')
# plt.show()

plt.figure()
plt.imshow( pp )
plt.show()
#%%
plt.figure()
plt.imshow( np.log(np.abs(fy0)) )
plt.show()

plt.figure()
plt.imshow( ga )
plt.show()
#%%
ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

# for ni in [0]:
#     if ni%2 == 0:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200)#, cmap='jet')
#     if ni%2 == 1:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200, cmap='jet')

ni = 0
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')
ax.contour3D(xr, alt(pp,L,D,w ), yr, 300) #, cmap='jet')

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(17, 120,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()
#%%
# =============================================================================
# Mas pruebas
# =============================================================================

totim, imfile = 4945, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [280,600,390,600] # [up, bottom, left, right] -> target or ice limits

file = ['./Melting/23-04-26_Ice_block_45/Ice_block_45(2)_C001H001S0001-0','.tif']
npa = 30
tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000,dist=15)

tod = tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]]
nt,ny,nx = np.shape(tod)
#step 1: PSP to get A and B
A, B = 0,0
for i in range(nt):
    A += tod[i]
    B += tod[i] * np.exp(-1j * 2*np.pi*i/nt)
A, B = np.abs(A)/nt, np.abs(B) * 2/nt

tod_b = (tod + A) / (B + 0.000001) 
# step 2: sample Moire for all images
Am,BPm = 0,0

im = tod_b[0]

# if not reduce:
#     fto = np.fft.fft( im[frecline] ) [initfrec:int(nx/2)]
#     kfo = np.fft.fftfreq(nx)
#     ind = np.argmax(np.abs(fto))
#     ne = int(1/kfo[ind + initfrec])
# if reduce:
#     numx = xlims[1] - xlims[0]
#     fto = np.fft.fft( im[frecline,xlims[0]:xlims[1]] ) [initfrec:int(numx/2)]
#     kfo = np.fft.fftfreq(numx)
#     ind = np.argmax(np.abs(fto))
#     ne = int(1/kfo[ind + initfrec])

# for t in range(nt):
#     im = tod_b[t]
#%%
# plt.figure()
# plt.imshow(A)
# plt.show()
# plt.figure()
# plt.imshow(B)
# plt.show()

n = 3

plt.figure()
plt.imshow(tod[n])
plt.show()
plt.figure()
plt.imshow(tod_b[n])
plt.show()







#%%
# =============================================================================
# roconocer hielo
# =============================================================================
from skimage.filters import gaussian, gabor, laplace, frangi, sato, roberts, sobel, hessian
from skimage.filters.rank import enhance_contrast 
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects, binary_closing, disk, remove_small_holes, binary_erosion, binary_opening
from skimage.util import img_as_ubyte
from skimage.feature import canny
#%%
limtar = [100,750,270,800] # [up, bottom, left, right] -> target or ice limits
n = 39
# plt.figure()
# plt.imshow(refs[n][limtar[0]:limtar[1],limtar[2]:limtar[3]], cmap='gray')
# plt.show()
# plt.figure()
# plt.imshow(refs[n] - backs[n], cmap='gray')
# plt.show()

refe = refs[n][limtar[0]:limtar[1],limtar[2]:limtar[3]]
# ga = gabor(refs[n],0.6)
# la = roberts(refe)
# binb = la > 2e3
# br = remove_small_objects(binb, min_size=2000)

# def cua(v0): # v0=[lxl,lyt,lxr,lyb]
#     bry,brx = np.where(br)[0],np.where(br)[1]
#     hb,ht = np.abs(bry - v0[3]), np.abs(bry - v0[1])
#     vl,vr = np.abs(brx - v0[0]), np.abs(brx - v0[2])
#     lins = np.vstack((hb,ht,vl,vr))
#     return np.sum(np.amin(lins,axis=0))

# vv = least_squares( cua, [30,15,550,500], diff_step=(1,1,1,1), bounds= ([1,1,1,1],[1000,1000,1000,1000]), max_nfev=100000 )
# print(vv)


saa = binary_closing(sato(refe) > 1.3e3, disk(11))
la = label( ~ saa )
pro = regionprops(la)
aldis = []
for i in range(np.max(la)):
    cy,cx = pro[i].centroid
    dis = (cy - (limtar[1]-limtar[0])/2 )**2 + (cx - (limtar[3]-limtar[2])/2 )**2
    aldis.append(dis)
per, bbox = pro[np.argmin(aldis)].perimeter, pro[np.argmin(aldis)].bbox #left,up,right,bottom
block = la == (np.argmin(aldis)+1)


# plt.figure()
# plt.imshow(binb)
# plt.show()
# plt.figure()
# plt.imshow( block )
# plt.plot( [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]], [bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]], 'r-' )
# plt.plot( [bbox[1]+30,bbox[1]+30,bbox[3]-30,bbox[3]-30,bbox[1]+30], \
#           [bbox[0]+30,bbox[2]-30,bbox[2]-30,bbox[0]+30,bbox[0]+30], 'g-' )
# plt.plot( [bbox[1]+50,bbox[1]+50,bbox[3]-50,bbox[3]-50,bbox[1]+50], \
#           [bbox[0]+50,bbox[2]-50,bbox[2]-50,bbox[0]+50,bbox[0]+50], 'b-' )
# plt.show()
# plt.figure()
# plt.imshow(ga[1])
# plt.show()

plt.figure()
plt.imshow(refs[n][limtar[0]:limtar[1],limtar[2]:limtar[3]], cmap='gray')
plt.plot( [bbox[1]+50,bbox[1]+50,bbox[3]-50,bbox[3]-50,bbox[1]+50], \
          [bbox[0]+50,bbox[2]-50,bbox[2]-50,bbox[0]+50,bbox[0]+50], 'b-' )
plt.show()
    
#%%
print(cua([30,15,550,500]))
# bry,brx = np.where(br)[0],np.where(br)[1]
# # hb,ht = np.abs(bry - v0[3]), np.abs(bry - v0[1])
# # vl,vr = np.abs(brx - v0[0]), np.abs(brx - v0[2])

# plt.figure()
# plt.plot(bry)
# plt.plot(brx)
plt.show()
#%%
a = [1,2,3,4,5]
b = [2,3,3,2,1]
d = [5,5,1,1,3]
c = np.vstack((a,b,d))

print(c)
print( np.amin(c,axis=0) )
#%%

nt,ny,nx = np.shape(tod)
#step 1: PSP to get A and B
A, B = 0,0
for i in range(nt):
    A += tod[i]
    B += tod[i] * np.exp(-1j * 2*np.pi*i/nt)
A, B = np.abs(A)/nt, np.abs(B) * 2/nt

tod_b = (tod - A) / (B + 0.000001)

plt.figure()
plt.imshow(A)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(backs[39])
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(refs[39])
plt.colorbar()
plt.show()
#%%
n = 1
hii, bed = np.histogram(refs[n], bins=100)
pe = find_peaks(hii,height=1000,prominence=10000, distance=10)[0]
va = find_peaks(-hii,height=-30000,prominence=5000)[0]

vv = np.array(va)
v1 = np.argmin( np.abs(vv-pe[1]) )
vv[v1] = 1e4
v2 = np.argmin( np.abs( vv-pe[1]) )

print(pe[1],va,v1,v2 )
print(bed[pe[1]], bed[va[v1]], bed[va[v2]])

plt.figure()
plt.imshow(refs[n])
plt.colorbar()
plt.show()

plt.figure()
plt.bar(bed[:-1],hii, width=bed[1]-bed[0], align='edge' )
plt.plot(bed[pe],hii[pe],'r.')
plt.plot(bed[va],hii[va],'g.')
plt.show()
#%%
# print(pe[1])
# print(va)

# v1 = np.argmin( va-pe[1] )
# va[v1] = 1e4
# v2 = np.argmin( va-pe[1] )

#%%
n = 0
difv = np.abs(bed[va[v1]] - bed[va[v2]])/2
mult = 0.8
l1,l2 = bed[pe[1]] - mult*difv, bed[pe[1]] + mult*difv


ahi = (refs[n] > l1) * (refs[n] < l2)
ah = binary_opening(ahi,disk(5))
ah2 = remove_small_objects(ah,min_size=30000)
la = label(ah2)
if np.max(la) >= 2:
    ah2 = la==2
hiel = binary_closing(ah2,disk(50))

# ah = remove_small_holes(ahi,area_threshold=1e4 )

plt.figure()
plt.imshow(refs[n])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow( hiel )
plt.show()

dx =100

# plt.figure()
# plt.imshow( binary_erosion(ah, np.array([[0]*dx,
#                                          [1]*dx,
#                                          [0]*dx]) ) )
# plt.imshow( binary_opening(ah,disk(25)) )
# plt.show()
#%%
n =0
plt.figure()
plt.imshow(refs[n])
plt.show()
plt.figure()
plt.imshow(refs[n] * hiel)
plt.show()
#%%
def recognice(refs,n):
    hii, bed = np.histogram(refs[n], bins=100)
    pe = find_peaks(hii,height=1000,prominence=10000, distance=10)[0]
    va = find_peaks(-hii,height=-30000,prominence=5000)[0]

    vv = np.array(va)
    v1 = np.argmin( np.abs(vv-pe[1]) )
    vv[v1] = 1e4
    v2 = np.argmin( np.abs( vv-pe[1]) )
    
    difv = np.abs(bed[va[v1]] - bed[va[v2]])/2
    mult = 0.8
    l1,l2 = bed[pe[1]] - mult*difv, bed[pe[1]] + mult*difv


    ahi = (refs[n] > l1) * (refs[n] < l2)
    ah = binary_opening(ahi,disk(5))
    ah2 = remove_small_objects(ah,min_size=30000)
    la = label(ah2)
    if np.max(la) >= 2:
        ah2 = la==2
    hiel = binary_closing(ah2,disk(50))
    return hiel
#%%
hils = []
for n in tqdm(range(40)):
    hiel = recognice(refs, n)
    hils.append(hiel)
#%%
n = 0
plt.figure()
# plt.imshow( gaussian(hessian(refs[n], sigmas=range(1,2,2) )**10,sigma=10) )
# plt.imshow( gaussian( frangi(refs[n], sigmas=range(1, 2, 2)),sigma=5 ))

cont = sato(refs[n], sigmas=range(1, 2, 5))
econ = enhance_contrast( img_as_ubyte(cont/np.max(cont)) ,disk(5))
ere = remove_small_objects(econ >12, min_size=100)

ll = 10
ker = np.array([[0]*ll,
                [1]*ll,
                [0]*ll])
erod = binary_erosion(ere, ker)

plt.imshow( erod )
plt.show()

#%%
n = 0

refe = refs[n][100:750,250:800]

contours = find_contours(refe, 3e4)
contus = find_contours(refe, 1.6e4)

plt.figure()
# plt.imshow( (canny(refs[n],sigma=2)) )
plt.imshow( refe, cmap='gray' )
# plt.imshow(  -refs[n]+np.max(refs[n]), cmap='gray' )
for contour in contours:
    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
for contour in contus:
    plt.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=2)
plt.show()
#%%
from skimage.segmentation import (watershed, morphological_geodesic_active_contour, inverse_gaussian_gradient,
                                    morphological_chan_vese, felzenszwalb, slic, quickshift, mark_boundaries)
from skimage.color import rgb2gray
#%%
n = 3
refe = refs[n][100:750,250:800]

t1 = time()
divs = felzenszwalb(refe, scale=0.9e7, sigma=0.9, min_size=20) # funca bastante bien, por ahora lo mejor que tengo
lab = int(np.median(divs[300:400,200:300]))
hiel = binary_closing(divs==lab, disk(15))

t2 = time()
print(t2-t1)
print(lab)

plt.figure()
plt.imshow( mark_boundaries(refe/np.max(refe),divs) )
plt.show()
plt.figure()
plt.imshow( divs )
plt.show()
plt.figure()
plt.imshow( hiel )
plt.show()
#%%
def recognice(refs,n,lims):
    refe = refs[n][lims[0]:lims[1],lims[2]:lims[3]]  
    divs = felzenszwalb(refe, scale=1e7, sigma=0.9, min_size=20) # funca bastante bien, por ahora lo mejor que tengo
    lab = int(np.median(divs[200:300,200:300]))
    hiel = binary_closing(divs==lab, disk(15))
    hieli = remove_small_holes(hiel,area_threshold=1e4)
    
    mask = np.zeros_like(refs[n])
    mask[lims[0]:lims[1],lims[2]:lims[3]] += hieli
    return mask


#%%
t1 = time()
lims = [100,750,250,800]
hils = []
for n in tqdm(range(40)):
    hiel = recognice(refs, n, lims)
    hils.append(hiel)
    
t2 = time()
print('\n',t2-t1)
#%%
n = 5
plt.figure()
plt.imshow(hils[n])
plt.show()


#%%
t1 = time()

totim, imfile = 4945, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [280,600,400,650] # [up, bottom, left, right] -> target or ice limits


file = ['./Melting/23-04-26_Ice_block_45/Ice_block_45(2)_C001H001S0001-0','.tif']

phases = []
refs = []
backs = []
for npa in tqdm(range(40)):
    tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000, dist=16)
    # phase, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]], sumA=1)
    # phases.append(phase)
    refs.append(ref)
    # backs.append(np.sum(tod,axis=0))
    
phases = np.array(phases)

t2 = time()
print(t2-t1)
#%%
t1 = time()
todm = tod * hils[39]
phase, ne, A,B = st_psm(todm[:], frecline=400, sumA=1)
t2 = time()
print(t2-t1)
#%%
# pp = np.unwrap(np.unwrap(phase * hils[39],axis=1) ,axis=1) #* hils[39]
# pp = np.unwrap(phase * hils[39],axis=1) #* hils[39]

phm = np.zeros_like(phase)
phm[ hils[39] == 0] = np.nan
phm += phase #* hils[39]

# pp = np.unwrap(phm,axis=1)
ppp = unwrap(phase)
pp = unwrap(phase*hils[39])

# a[~np.isnan(a)] = np.unwrap(a[~np.isnan(a)])

n = 3
plt.figure()
plt.imshow(phase)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ppp)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(pp)
plt.colorbar()
plt.show()





#%%
# =============================================================================
# Complete
# =============================================================================
def recognice(ref,lims):
    refe = ref[lims[0]:lims[1],lims[2]:lims[3]]  
    divs = felzenszwalb(refe, scale=0.9e7, sigma=0.9, min_size=20) # funca bastante bien, por ahora lo mejor que tengo
    lab = int(np.median(divs[300:400,200:300]))
    hiel = binary_closing(divs==lab, disk(15))
    hieli = remove_small_holes(hiel,area_threshold=1e4)
    
    mask = np.zeros_like(ref)
    mask[lims[0]:lims[1],lims[2]:lims[3]] += hieli
    return mask
#%%
t1 = time()

totim, imfile = 4945, 1024 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
# limtar = [280,600,400,650] # [up, bottom, left, right] -> target or ice limits
lims = [100,750,250,800] # lims to help with the masking

file = ['./Melting/23-04-26_Ice_block_45/Ice_block_45(2)_C001H001S0001-0','.tif']

phases = []
masks = []
for npa in tqdm(range(40)):
    tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=30000, dist=16)
    mask = recognice(ref, lims)
    phase, ne, A,B = st_psm(tod * mask, frecline=400, sumA=1)
    phases.append(phase)
    masks.append(mask)
    
phases = np.array(phases)
masks = np.array(masks)

t2 = time()
print(t2-t1)

#%%
d,L,D,w = 0.23990251906028134, 5499.870962858253, 814.7594549835866, 2.549666044116526 #for inclined block
t1 = time()

pp = []
for n in range(40):
    pp.append( unwrap(phases[n] * masks[n]) )
pp = np.array(pp)

pp[ masks==0 ] = np.nan

pp = np.unwrap(pp,axis=0)
hices = alt(pp,L,D,w )

t2 = time()
print(t2-t1)

#%%
n = 14
plt.figure()
plt.imshow(hices[n])
plt.show()

plt.figure()
plt.plot(pp[:,400,400])
plt.show()

# plt.figure()
# plt.imshow(masks[n])
# plt.show()
#%%
ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


fig = plt.figure()
ax = plt.axes(projection='3d')

# for ni in [0]:
#     if ni%2 == 0:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200)#, cmap='jet')
#     if ni%2 == 1:
#         ax.contour3D(xr, alt(pp[ni],L,D,w ), yr, 200, cmap='jet')

ni = 0
# ax.contour3D(xr, alt(phases[ni],L,D,w ), yr, 200, cmap='jet')
# ax.contour3D(xr, hices[ni] , yr, 300) #, cmap='jet')
ax.plot_surface(xr, hices[ni] , yr, cstride=10, rstride=10, shade=True)

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')

ax.set_zlim(-60,100)

# ax.invert_yaxis()
# ax.invert_zaxis()
ax.view_init(17, 120,vertical_axis='z')
# plt.legend()
# plt.savefig('target3d.png',bbox_inches='tight', transparent=True)
plt.show()

#%%
n = 3
hice = hices[n]

from matplotlib.colors import LightSource, Normalize

ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
xr,yr = np.meshgrid(xr,yr)


az,al = 90 , 90
ver,fra = 0.01 , 5.

ls = LightSource(azdeg=az,altdeg=al)

# shade data, creating an rgb array.
rgb1 = ls.shade(hice,plt.cm.Blues) #, blend_mode='soft')
# rgb = ls.hillshade(hice,vert_exag=ver,fraction=fra , dx = 1, dy =1)

rgb = ls.shade_rgb(rgb1, hice, vert_exag=ver,fraction=fra)


color_dimension = rgb # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.plot_surface(xr,hice,yr, facecolors=fcolors, rcount=400,ccount=400) 

ax.plot_surface(xr,hice,yr, rcount=250,ccount=250, shade=True, lightsource=ls, )  

# ax.plot_surface(xr[70:-70,70:-70],hice[70:-70,70:-70],yr[70:-70,70:-70], facecolors=fcolors, rcount=250,ccount=250) 

# ax.plot_surface(xro[:1100],yro[:1100],fase[:1100], facecolors=fcolors, rcount=150,ccount=150)
# ax.plot_surface(xr,yr,haj, facecolors=fcolors) 

ax.set_box_aspect([3,2,3])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.set_title('3D contour')
# ax.invert_zaxis()
ax.invert_xaxis()

ax.set_zlim(-60,100)
# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
ax.view_init(17,120)
# plt.savefig('imageni2.png',bbox_inches='tight',dpi=1000)
plt.show()
#%%
from matplotlib.colors import LightSource, Normalize
for n in range(30,40):
    hice = hices[n]
    
    ny,nx = np.shape(tod[0])
    
    xr,yr = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
    xr,yr = np.meshgrid(xr,yr)
    
    
    az,al = 90 , 90
    ver,fra = 0.01 , 5.
    
    blue = np.array([1., 1., 1.])
    rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ls = LightSource()
    illuminated_surface = ls.shade_rgb(rgb, hice)
    
    ax.plot_surface(xr, hice, yr, ccount=300, rcount=300,
                    antialiased=True,
                    facecolors=illuminated_surface)
    
    ax.set_box_aspect([3,2,3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    # ax.set_title('3D contour')
    # ax.invert_zaxis()
    ax.invert_xaxis()
    
    ax.set_zlim(-60,100)
    ax.set_xlim(70,-70)
    ax.set_ylim(-30,120)
    
    # ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
    ax.view_init(17,120)
    # plt.savefig('imageni2.png',bbox_inches='tight',dpi=1000)
    plt.show()
#%%
ny,nx = np.shape(tod[0])

xr,yr = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
xr,yr = np.meshgrid(xr,yr)

az,al = 90 , 90
ver,fra = 0.01 , 5.


lista_im = []
t_spl = np.linspace(0,1,1000)
for num in tqdm(range(40)): #1251

    plt.close('all')
    plt.ioff()
    
    hice = hices[num]
    
    blue = np.array([1., 1., 1.])
    rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ls = LightSource()
    illuminated_surface = ls.shade_rgb(rgb, hice)

    ax.plot_surface(xr, hice, yr, ccount=300, rcount=300,
                    antialiased=True,
                    facecolors=illuminated_surface)

    ax.set_box_aspect([3,2,3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    # ax.set_title('3D contour')
    # ax.invert_zaxis()
    ax.invert_xaxis()

    ax.set_zlim(-60,100)
    ax.set_xlim(70,-70)
    ax.set_ylim(-30,120)
    

    # ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
    ax.view_init(17,120)
    # plt.savefig('imageni2.png',bbox_inches='tight',dpi=1000)

    plt.savefig('imageni.png',bbox_inches='tight',dpi=500)
    lista_im.append(imageio.imread('imageni.png'))
#%%
imageio.mimsave('./Melting/inc_block_3.gif',lista_im,fps=12)
#%%
# =============================================================================
# Pruebas showing
# =============================================================================
import pyvista as pv
from pyvista import examples
#%%
ny,nx = np.shape(tod[0])
xr,yr = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
xr,yr = np.meshgrid(xr,yr)

n = 30
grid = pv.StructuredGrid(xr, hices[n], yr)


# surf = grid.extract_geometry().clean(tolerance=1e-6)

# mu = 0.02
# campos = [-5/mu,20/mu,4/mu]

# surf = grid.extract_geometry()
# # smooth = surf.smooth()

# p = pv.Plotter(shape=(1, 1), border=False) #, lighting='three lights')

# # With eye-dome lighting
# p.subplot(0, 0)
# # p.add_mesh(grid)
# p.add_mesh(grid.gaussian_smooth(std_dev=2.0))
# p.show_grid()
# p.camera.zoom(1.)
# p.camera.azimuth = 83
# p.camera.elevation = -44
# # p.enable_eye_dome_lighting()

# p.show()

# p.camera.position = campos
# p.cpos(campos1)

# p.enable_eye_dome_lighting()

# grid.plot( show_grid=True, smooth_shading=True) 
# surf.plot( show_grid=True, smooth_shading=True) 


# grid.enable_eye_dome_lighting()
# smooth.plot( show_grid=True)

# grid.plot_curvature(clim=[-1, 1])


# grid = pv.UniformGrid(
#     dimensions=(nx, 32, ny), spacing=(0.5, 0.5, 0.5)
# )

# grid['data'] = np.linalg.norm(grid.center - grid.points, axis=1)
# grid['data'] = np.abs(grid['data'] - grid['data'].max()) ** 3
# grid.plot(volume=True)

# nefertiti.plot(eye_dome_lighting=True, cpos=[-1, -1, 0.2], color=True)
# print( np.shape(grid.points) )



#%%

