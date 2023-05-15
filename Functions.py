#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:52:09 2023

@author: tomasferreyrahauchar
"""
#%%
# =============================================================================
# Cosas en limpio, antes de remover todo lo de arriba
# =============================================================================
import numpy as np

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
# from scipy.ndimage import rotate
# from scipy.stats import mode
from scipy.interpolate import make_interp_spline

# import rawpy
import imageio
from tqdm import tqdm

from time import time
# from skimage.filters import gaussian, gabor
# from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes #, remove_small_objects
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.restoration import unwrap_phase as unwrap
#%%

def guass(x,mu,si):
    """
    Returns a guassian distribution funtion.
    
    x: array_like
    mu: float, mean
    si: float, standard deviation
    """
    return 1/ np.sqrt(2*np.pi* si**2) * np.exp(-1/2* ((x-mu)/si)**2)
        
def mrot(th):
    """
    Return a 2x2 rotation matrix given the angle th.
    
    th: float (in radians)
    """
    mat = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]])
    return mat

def rot(y,z,mat):
    """
    Returns the rotated 2d arrays y and z using the rotation matrix from mrot().
    
    y: 2d-array
    z: 2d-array
    mat: 2x2 matrix from mrot()
    """
    n1,n2 = np.shape(y)
    yzf = np.zeros((n1*n2,2))
    yzf[:,0], yzf[:,1] = y.flatten(), z.flatten()
    yzr = np.dot(yzf,mat.T)
    yr,zr = yzr[:,0].reshape(n1,n2), yzr[:,1].reshape(n1,n2)
    return yr,zr

def lin(x,m,o):
    """
    Linear function.
    
    x: array_like
    m: float, slope
    o: float, origin
    """
    return m*x+o

def patron(x,y,cx,cy,a=2.5,k=0.2):
    """
    Calibration pattern
    
    x,y: 2d-array
    cx,cy: int, center of the pattern
    a: float, amplitud of the pattern (in mm)
    k: float, frequency 
    """
    # print(cx,cy)
    return a + a*np.cos(k*np.sqrt( (x-x[cx,cx])**2 + (y-y[cy,cy])**2) )

def alt(dp,Lp,D,w0):
    """
    Returns the heigth profile.
    
    dp: 2d -array, phase difference
    Lp: float, distance from camera/projector to object (in mm)
    D: float, distance between camera and projector (in mm)
    w0: float, frequency of the pattern (in 1/mm)
    """
    return dp * Lp / (dp - D*w0 )


def calibrate_params(tod,phi, limtar, initial, inlim=[0.002,20,20] ,showerr=True):
    '''
    Calibrates the parameter d, L, D, and w by using least_squares and a fit to the pattern. Real scale is mm
    tod: is used to get the size of the image. d is the transformaion value from px to mm (in mm/px). See alt() 
    for L, D and w (also Lp, D, w0).
    
    tod: 3d-array, images of the pattern (returned from all_frames())
    phi: 2d-array, phase of the target
    limtar: list, limits of the image that define the position of the target [top, bottom, left, right] (in px)
    initial: initial guess for d, cx,cy (cx,cy are the center of the target) (inside the window defined by limtar)
    inlim: tuple, bounds of the initial guess
    showerr: boolean, print and graph the error
    '''
    def valores(v0): # d,cx,cy
        ny,nx = np.shape(tod[0])

        xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * v0[0],\
            (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * v0[0]
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

        para, cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(1000,350,1), \
                              bounds=( (300,150,0.1),(5500,1000,10.0) ) ) 

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

    (L,D,w), cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(2500,560,0.2),\
                             bounds=( (300,150,0.01),(5500,1000,10.0) ) ) 

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


def st_psm(tod, frecline=200, initfrec=2, sumA = -1):
    """
    Returns phase diference of the object from the images of the patterns. 
    
    tod: 3d-array, images of the pattern (returned from all_frames())
    frecline: int, row where we calculate the frequency of the pattern
    initfrec: int, how many frequency t skip before looking for maximum
    sumA: 1 or -1, whether to add or substract A for normalization
    """
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

    fto = np.fft.fft( im[frecline] ) [initfrec:int(nx/2)]
    kfo = np.fft.fftfreq(nx)
    ind = np.argmax(np.abs(fto))
    ne = int(1/kfo[ind + initfrec])

    for t in range(nt):
        im = tod_b[t]
        
        x = np.arange(nx)
        for e in range(ne):
            imt = im[:,e::ne]
            
            # cs = CubicSpline(x[e::ne], imt,axis=1 )#, bc_type='not-a-knot')
            # imc = cs(x) 
            
            cs = make_interp_spline(x[e::ne],imt,1,axis=1) 
            #3rd input could be 1 or 3 (spline degree/ polinomial order)
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


def all_steps(file, barl, fpass, imfile, totim, bhei = 0.6, npa=0, dist=10):
    """
    Returns all the phase shift and the reference image from a set of pattern images. Each experiment 
    consists of various .tif files, each file containing more than one image
    
    file: list, [location + name of the file (not including the count number of the file), extension] 
    barl: list, [top, bottom, left, right] px of the phase counting lines
    fpass: int, number of frames for each pass/set of pattern images. 
    imfile: int, total images (one file)
    totim: int, total images (all files)
    bhei: float, height from where to start counting peaks
    npa: int, iteration where I want the patterns images
    dist: int, distance between peaks for recognizing phase shift
    """
    
    allims = np.array([ [int(i/imfile),i%imfile] for i in range(totim) ])
    imsn = allims[npa*fpass : (npa+1)*fpass ]

    ims = [0 for i in range(21)]
    cuenti = [0 for i in range(21)]

    # for j in range(fpass):
    #     with imageio.get_reader(file[0]+ str( imsn[j,0] ) +file[1],mode='I') as vid:
            
    #         im = vid.get_data(0)[imsn[j,1]]
    
    ind = np.where( (imsn[:,0] - np.roll(imsn[:,0],1)) == 1 )[0]

    imsn = [imsn]
    for vin in ind:
        imsn = np.split(imsn[0], [vin], axis=0)


    for j in range(len(imsn)):
        k = imsn[j][0,0]
        with imageio.get_reader(file[0]+ str( k ) +file[1], mode='I') as vid:  
            imss = vid.get_data(0)
            for l in range(len(imsn[j])):
                im = imss[imsn[j][l,1]]
    
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

def recognice(ref,lims, scale=0.9e7, sigma=0.9, min_size=20, loci=[300,400,200,300]):
    """
    Return binary 2d-array (mask) that defines the location of the ice
    
    ref: 2d-array, reference image (returned from all_frames())
    lims: list, [top, bottom, left, right] px of where I could find the ice  
    scale: float, argument for the felzenszwalb function
    sigma: float, argument for the felzenszwalb function
    min_size: int, argument for the felzenszwalb function
    loci: list, [top, bottom, left, right] px for location where I can always find ice (used for finding the label of the ice)
    """
        
    refe = ref[lims[0]:lims[1],lims[2]:lims[3]]  
    divs = felzenszwalb(refe, scale=0.9e7, sigma=0.9, min_size=20) # funca bastante bien, por ahora lo mejor que tengo
    
    lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
    
    hiel = binary_closing(divs==lab, disk(15))
    hieli = remove_small_holes(hiel,area_threshold=1e4)
    
    mask = np.zeros_like(ref)
    mask[lims[0]:lims[1],lims[2]:lims[3]] += hieli
    return mask


#%%
# =============================================================================
# Example of how to use
# =============================================================================
# Calibration

t1 = time()
totim, imfile = 230, 230 #total images (all files), total images (one file)
fpass = 115 # number of frames for each pass
barl = [250,800,80,140] # [up, bottom, left, right] -> phase counting bar
limtar = [130,690,280,850] # [up, bottom, left, right] -> target or ice limits

npa = 0
file = ['./Melting/23-04-26_Calibration_3/Calibration_3_C001H001S0001-','.tif']

tod, ref = all_steps(file, barl, fpass, imfile, totim, npa=npa, bhei=23000,dist=15)
phi, ne, A,B = st_psm(tod[:,limtar[0]:limtar[1],limtar[2]:limtar[3]], sumA = -1)
phi = np.unwrap(np.unwrap(phi, axis=0),axis=1)

d,L,D,w = calibrate_params(tod,phi, limtar, [0.241,280,285], inlim=[0.004,40,40] ,showerr=True)


t2= time()
print(t2-t1)
print('d,L,D,w =', d,',',L,',',D,',',w )
#%%
d,L,D,w = 0.23990257662890377 , 5499.86657262652 , 814.6376201885718 , 2.5500455007016094

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

np.save('./Height profiles/ice_block_45-0.npy',hices)

#%%
# =============================================================================
# Example of showing height profile
# =============================================================================

n = 0
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
