#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:30:43 2023

@author: tomasferreyrahauchar
"""
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, correlate, fftconvolve, peak_prominences
import scipy.ndimage as snd # from scipy.ndimage import rotate
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline


# import rawpy
import imageio
from tqdm import tqdm

from time import time
from skimage.filters import gaussian #, gabor
from skimage.filters.rank import enhance_contrast
# from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion #, remove_small_objects
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

        para, cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(2200,600,0.5), \
                              bounds=( (1900,400,0.1),(2500,800,10.0) ) ) 

        haj = alt(fase,*para) 
        error = (pat-haj)[50:-50,50:-50].flatten()
        return error

    calib = least_squares( valores, [initial[0],initial[1],initial[2]], diff_step=(0.001,1,1), \
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

    (L,D,w), cur = curve_fit( alt, fase.ravel(), pat.ravel(), p0=(2200,600,0.5), \
                          bounds=( (1900,400,0.1),(2500,800,10.0) ) ) 

    haj = alt(fase,L,D,w) 
    err = (pat-haj)[50:-50,50:-50].flatten()
     
    if showerr:
        print('Mean = ', np.mean(err), ' Std = ', np.std(err) )
        xerr=np.linspace(-2,2,1000)
        plt.figure()
        plt.hist( err, bins=200, density=True)
        plt.plot(xerr,guass(xerr,np.mean(err),np.std(err)))
        plt.show()
    
    return d, L, D, w

def all_steps(file, barl, fpass, imfile, totim, bhei = 0.6, npa=0, dist=10, prominence=100):
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
                # bar = bar #/ np.max(bar)
                
                peak = find_peaks( -np.mean(bar,axis=1), height=-bhei, distance=dist, prominence=prominence )[0]
                paso = len(peak)
                
                # print(imsn[j][l,1],paso)
                
                # print(len(cuenti),paso)
                cuenti[paso] += 1
                ims[paso] = ims[paso] *  (cuenti[paso]-1) / (cuenti[paso]) + im / (cuenti[paso])
                
    ims = np.array(ims)
    ref = ims[0]
    tod = ims[1:] - ref
    
    return tod, ref


def all_steps2(file, barl, fpass, npa=0, bhei = 0.6, dist=10, prominence=100, zfill=6):
    """
    Returns all the phase shift and the reference image from a set of pattern images. 
    
    file: list, [location + name of the file (not including the count number of the file), extension] 
    barl: list, [top, bottom, left, right] px of the phase counting lines
    fpass: int, number of frames for each pass/set of pattern images. 
    bhei: float, height from where to start counting peaks
    npa: int, iteration where I want the patterns images
    dist: int, distance between peaks for recognizing phase shift
    """
    
    imsn = list(range(npa*fpass, (npa+1)*fpass) )  

    ims = [0 for i in range(21)]
    cuenti = [0 for i in range(21)]


    for j in imsn:
        with imageio.get_reader(file[0]+ str(j+1).zfill(zfill) +file[1], mode='I') as vid:
                im = vid.get_data(0)
    
                bar = im[barl[0]:barl[1],barl[2]:barl[3]]    
                # bar = bar #/ np.max(bar)
                
                peak = find_peaks( -np.mean(bar,axis=1), height=-bhei, distance=dist, prominence=prominence )[0]
                paso = len(peak)
                
                # print(imsn[j][l,1],paso)
                
                # print(len(cuenti),paso)
                cuenti[paso] += 1
                ims[paso] = ims[paso] *  (cuenti[paso]-1) / (cuenti[paso]) + im / (cuenti[paso])
                
    ims = np.array(ims)
    ref = ims[0]
    tod = ims[1:] #- ref
    
    return tod, ref



def all_steps3(file, barl, fpass, npa=0, bhei = 0.6, dist=10, prominence=100, zfill=6):
    """
    Returns all the phase shift and the reference image from a set of pattern images. Each experiment 
    consists of various .tif files, each file containing more than one image
    
    file: list, [location + name of the file (not including the count number of the file), extension] 
    barl: list, [top, bottom, left, right] px of the phase counting lines
    fpass: int, number of frames for each pass/set of pattern images. 
    bhei: float, height from where to start counting peaks
    npa: int, iteration where I want the patterns images
    dist: int, distance between peaks for recognizing phase shift
    """
    
    imsn = list(range(npa*fpass, (npa+1)*fpass) )  

    ims = [0 for i in range(21)]
    cuenti = [0 for i in range(21)]


    for j in imsn:
        with imageio.get_reader(file[0]+ str(j+1).zfill(zfill) +file[1], mode='I') as vid:
                im = vid.get_data(0)
    
                # bar = im[barl[0]:barl[1],barl[2]:barl[3]]    
                # # bar = bar #/ np.max(bar)
                
                # peak = find_peaks( -np.mean(bar,axis=1), height=-bhei, distance=dist, prominence=prominence )[0]
                # paso = len(peak)
                
                # ibar = im[720:920,945:955]
                barli =  -np.mean( gaussian(im[barl[0]:barl[1],barl[2]:barl[3]],sigma=1), axis=1 )  #-np.mean( im[720:920,925:950], axis=1 )
                # xb = np.arange(len(barl))
                
                # vli, cc = curve_fit(lin, xb, barl)
                bar = barli/np.max(-barli) #- lin(xb, 0.4,0) 
                
                peak = find_peaks( bar, height=-100, distance=6, prominence=0.06, wlen=9)[0]  #height=-600 (para 1ro)
        
                paso = len(peak)

                
                # print(imsn[j][l,1],paso)
                
                # print(len(cuenti),paso)
                cuenti[paso] += 1
                ims[paso] = ims[paso] *  (cuenti[paso]-1) / (cuenti[paso]) + im / (cuenti[paso])
                
    ims = np.array(ims)
    ref = ims[0]
    tod = ims[1:] #- ref
    
    return tod, ref

def st_psm_orto(tod, frecline=200, initfrec=2, sumA = -1, nef=0):
    """
    Returns phase diference of the object from the images of the patterns. Using 2d grating, considering both directions same frequency 
    
    tod: 3d-array, images of the pattern (returned from all_frames())
    frecline: int, row where we calculate the frequency of the pattern
    initfrec: int, how many frequency t skip before looking for maximum
    sumA: 1 or -1, whether to add or substract A for normalization
    """
    nt,ny,nx = np.shape(tod)
    N = int(np.sqrt(nt))
    
    A, B = 0,0
    for i in range(nt):
        A += tod[i]
        B += tod[i] * np.exp(-1j * 2*np.pi* (i) / N)
    A, B = np.abs(A)/nt, np.abs(B) * 2/nt
    
    tod_b = (tod + sumA * A) / (B + 0.000001) 
    
    Am,Bpmx,Bpmy = 0,0,0
    
    im = tod_b[0]
    
    fto = np.fft.fft( im[:,frecline] ) [initfrec:int(ny/2)]
    kfo = np.fft.fftfreq(ny)
    ind = np.argmax(np.abs(fto))
    
    if nef == 0: ne = int(1/kfo[ind + initfrec]) 
    else: ne = nef
    
    print(ne)
    
    imts = []
    for i in range(N):
        for j in range(N):
            im = tod_b[i*N+j]
            
            x = np.arange(nx)
            y = np.arange(ny)
            
            for e in range(ne):
                for f in range(ne):
    
                    imtr = im[f::ne,e::ne]
                    
                    csx = make_interp_spline(x[e::ne],imtr,1,axis=1)
                    csy = make_interp_spline(y[f::ne],csx(x),1,axis=0)
                    imc = csy(y)
                    imts.append(imc)
    
                    
                    Bpmx += imc * np.exp(-1j * 2*np.pi*e/ne) * np.exp(-1j * 2*np.pi*i/N) 
                    
                    Bpmy += imc * np.exp(-1j * 2*np.pi*f/ne) * np.exp(-1j * 2*np.pi*j/N) 
    
                    Am += imc
                    # BPm += imc * np.exp(-1j * 2*np.pi*e/ne) * np.exp(-1j * 2*np.pi*t/nt) 
                    
                    
            
    Am = np.abs(Am) / (N**2*ne**2)
    # Bm = np.abs(BPm) * 2/(nt*ne)
    
    Pmx = np.angle(Bpmx)
    Pmy = np.angle(Bpmy)

    return Pmx,Pmy, ne

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
    refi = enhance_contrast(refe/np.max(refe),disk(7))
    divs = felzenszwalb(refi, scale=scale, sigma=sigma, min_size=min_size) # funca bastante bien, por ahora lo mejor que tengo
    
    lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
    
    hiel = binary_closing(divs==lab, disk(15))
    hieli = remove_small_holes(hiel,area_threshold=1e4)
    
    mask = np.zeros_like(ref)
    mask[lims[0]:lims[1],lims[2]:lims[3]] += hieli
    return mask


def st_psm(tod, frecline=200, initfrec=2, sumA = -1, nef=0):
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
    
    if nef == 0: ne = int(1/kfo[ind + initfrec]) 
    else: ne = nef

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


def st_psm2(tod, maskr, frecline=200, initfrec=2, sumA = -1):
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


    phs = []
    for t in range(nt):
        im = tod_b[t]
        
        Bpm = 0
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
            # BPm += imc * np.exp(-1j * 2*np.pi*e/ne) * np.exp(-1j * 2*np.pi*t/nt) 
            Bpm += imc * np.exp(-1j * 2*np.pi*e/ne) #* np.exp(-1j * 2*np.pi*t/nt) 
        
        phs.append( np.angle(Bpm) )

    Am = np.abs(Am) / (nt*ne)
    # Bm = np.abs(BPm) * 2/(nt*ne)

    uprs = []
    
    for n in range(0,20):
        pr = ma.masked_array( phs[n], mask= -maskr+1 )
        uprs.append( unwrap( pr ))
    uprs = np.array(uprs)
    uprs = np.array( [ uprs[i] - uprs[i,300,600] for i in range(20)] )  + uprs[5,300,600]
    Pm = np.mean(uprs,axis=0)
    # Pm = uprs[5]
    
    # Pm = np.unwrap(np.unwrap(Pm),axis=0)
    
    corr = ((np.cumsum( np.ones_like(Pm),axis=1) - 1)/nx) * 2*np.pi/ne # correction described in paper
    psr = Pm + corr

    return psr, ne, Am #, Bm


def alt(dp,Lp,D,w0):
    """
    Returns the heigth profile.
    
    dp: 2d -array, phase difference
    Lp: float, distance from camera/projector to object (in mm)
    D: float, distance between camera and projector (in mm)
    w0: float, frequency of the pattern (in 1/mm)
    """
    return dp * Lp / (dp - D*w0 )

def calibrate_tilted(phi,tod, initial, inlim, limtar, init1=(2100,600,0.2)):
    def valldw(v1):
        global hal, d, cx,cy, Lm
        hal = alt(phi, v1[0], v1[1], v1[2] )
        Lm = v1[0]
        
        calib2 = least_squares(valdc, [initial[0],initial[1],initial[2]], diff_step=(0.001,1,1), \
                              bounds=( [initial[0]-inlim[0],initial[1]-inlim[1],initial[2]-inlim[2] ] ,\
                                       [initial[0]+inlim[0],initial[1]+inlim[1],initial[2]+inlim[2] ]) )
        
        d,cx,cy = calib2.x
        # print(d)
        error = calib2.cost
        return error 
    
    def valdc(v0):
        global hal, d, fase
        ny,nx = np.shape(tod[0])
    
        xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * v0[0],\
            (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * v0[0]
        xr,yr = np.meshgrid(xr,yr)
        
        xrp = xr - hal/Lm * xr
        yrp = yr - hal/Lm * yr
        
        prfo = hal
        rrx, rry = xrp, yrp
        A = np.array([rrx.flatten()*0+1,rrx.flatten(),rry.flatten()]).T
        finan = ~np.isnan(prfo.flatten())
        coeff, r, rank, s = np.linalg.lstsq(A[finan] , (prfo.flatten())[finan], rcond=None)
        # plane = coeff[0] + coeff[1] * rrx + coeff[2] * rry
        m2 = mrot( -np.arctan(coeff[1]) )
        xro, prot = rot(rrx,prfo,m2)
        pary,cov = curve_fit(lin, (rry.flatten())[finan], (prot.flatten())[finan])
        m2 = mrot( -np.arctan(pary[0]) )
        yro, fase = rot(rry,prot,m2)

    
        pat = patron(xro,yro,int(v0[1]),int(v0[2]))
    
        pat -= np.mean(pat)
        fase -= np.mean(fase)
        
        error = np.sum(np.abs(fase - pat))
        return error
    
    calib = least_squares(valldw, [init1[0],init1[1],init1[2]], diff_step=(0.1,0.1,0.1) ,bounds=( [1700,400,0.1] , [2500,800,10.0] )  )
    L,D,w = calib.x
    
    ny,nx = np.shape(tod[0])

    xr,yr = (np.arange(0.5,nx+0.5)[limtar[2]:limtar[3]] - nx/2) * d, (-np.arange(0.5,ny+0.5)[limtar[0]:limtar[1]] + ny/2) * d
    xr,yr = np.meshgrid(xr,yr)
    
    hal = alt(phi, L, D, w )
    
    xrp = xr - hal/Lm * xr
    yrp = yr - hal/Lm * yr

    prfo = hal
    rrx, rry = xrp, yrp
    A = np.array([rrx.flatten()*0+1,rrx.flatten(),rry.flatten()]).T
    finan = ~np.isnan(prfo.flatten())
    coeff, r, rank, s = np.linalg.lstsq(A[finan] , (prfo.flatten())[finan], rcond=None)
    # plane = coeff[0] + coeff[1] * rrx + coeff[2] * rry
    m2 = mrot( -np.arctan(coeff[1]) )
    xro, prot = rot(rrx,prfo,m2)
    pary,cov = curve_fit(lin, (rry.flatten())[finan], (prot.flatten())[finan])
    m2 = mrot( -np.arctan(pary[0]) )
    yro, fase = rot(rry,prot,m2)

    pat = patron(xro,yro,int(cx),int(cy))
    # print( int(cx),int(cy) )    

    fase -= np.mean(fase)
    pat -= np.mean(pat)

    err = (pat-fase)[50:-50,50:-50].flatten()    
    
    print('Mean = ', np.mean(err), ' Std = ', np.std(err) )
    xerr=np.linspace(-2,2,1000)
    plt.figure()
    plt.hist( err, bins=200, density=True)
    plt.plot(xerr,guass(xerr,np.mean(err),np.std(err)))
    plt.show()
    
    # print('Angulos', np.arctan(pary[0]) *180/np.pi, np.arctan(parx[0]) *180/np.pi )
    print('Angulos', np.arctan(coeff[1]) *180/np.pi, np.arctan(pary[0]) *180/np.pi )
    return d, L, D, w  #, fase, pat, xr, yr, xro,yro, xrp, yrp

#%%
# =============================================================================
# Find limits for phase recognition
# =============================================================================
# barl = [642,862,775,795] # for _ref and _cal,, [644,865,768,790] # for the ice,, (for (8))
# barl = [740,960,885,915] #[740,960,885,915] # for ref and cal #[740,960,865,892] # for the ice (for (9))
# barl = [750,975,865,880] #for all (10), except some frames
# barl = [755,982,810,830] #for all (12)
# barl = [775,975,890,905] #all (13)
# barl = [740,965,897,920] #for ref (14), [750,972,897,915] #for cal/ice (14)
# barl = [750,975,885,900] # for all (15)
# barl = [763,978,914,930] # for all (16)
# barl = [756,975,910,925] # for all (17)
# barl = [756,975,885,900] # for all (18)
# barl = [758,976,892,910] # for all (19)
# barl = [755,973,877,891] # for all (20)
# barl = [764,983,900,917] # for all (21)
# barl = [760,976,900,913] # for all (22)
# barl = [760,980,895,917] # for all (23)
# barl = [763,975,914,933] # for all (24)
# barl = [765,980,895,907] # for all 30(0)

file = ['/Volumes/Ice blocks/Ice_block_30(0)/Ice_block_30(0)','.tif']
barl = [765,980,895,907] # for all 30(0)

# file = ['/Volumes/Ice blocks/mano1/mano1','.tif']
# barl = [675,887,960,990] # for all 30(0)

pasos = []
barms = []
for n in [0]: # tqdm(range(90*70,90*79)):
    with imageio.get_reader(file[0]+ str(n+1).zfill(6) +file[1], mode='I') as vid:
        im = vid.get_data(0)
    
        bar = im[barl[0]:barl[1],barl[2]:barl[3]]  
        barm = -np.mean(bar,axis=1)
        
        peak = find_peaks( barm, height=-600, distance=7, prominence=100. )[0]  #height: 1500 cal,  ref, h: 500 (a40), 1000 (a70), 600 (a79)
            
        # if n in range(90*64,90*65): #range(5759,5778): 
        #     bar = im[730:975,800:900]  
        #     barm = -np.mean(bar,axis=1)        
        #     peak = find_peaks( barm, height=-1200, distance=7, prominence=100.0 )[0]   
        
        paso = len(peak)
        pasos.append(paso)
        
        # if n in range(5670+90+10,5670+90+20):
        #     x= np.arange(len(barm))
        #     plt.figure()
        #     plt.plot(barm)
        #     plt.plot(x[peak],barm[peak],'k.')
        #     plt.grid()
        #     plt.show()    

plt.figure()
plt.imshow(im,cmap='gray') #, vmax=8e2, vmin=4e2)
plt.show()

plt.figure()
plt.imshow(im[100:988,240:810],cmap='gray') #, vmax=8e2, vmin=4e2)
plt.show()

# plt.figure()
# # plt.imshow(bar,cmap='gray')
# plt.plot(barm)
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(pasos,'.-')
# plt.grid()
# plt.show()

#%%
t1 = time()

fpass = 90
filer = ['/Volumes/Ice blocks/Ice_block_30(0)_ref/Ice_block_30(0)','.tif']
barl = [765,980,895,907] # for _ref 
lims = [100,988,240,810]

todr, refr = all_steps2(filer, barl, fpass, bhei = 1200, dist=7, prominence=100.)
maskr = np.zeros_like(todr[0])
maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 
phar, ne, A,B = st_psm(todr*maskr, frecline=400, initfrec=5, sumA = -1)
ppr =  unwrap(phar*maskr)

print(ne)

filec = ['/Volumes/Ice blocks/Ice_block_30(0)_cal/Ice_block_30(0)_cal','.tif']
lims = [525,800,330,685]

todc, refc = all_steps2(filec, barl, fpass, bhei = 1200, dist=7, prominence=100.)
maskc = np.zeros_like(todc[0])
maskc[lims[0]:lims[1],lims[2]:lims[3]] = 1. 
phac, ne, A,B = st_psm(todc*maskc, frecline=600, initfrec=5, sumA = -1)
ppc = unwrap(phac*maskc)

phcal = (ppr-ppc)[lims[0]:lims[1],lims[2]:lims[3]]

# d,L,D,w = calibrate_params(todc,phcal, lims, [0.383,173,137], inlim=[0.5,30,30] ,showerr=True)
d, L, D, w = calibrate_tilted(phcal, todc, [0.383,173,137], [0.05,30,30], lims, init1=(2200,600,0.5))
print('d = ',d,'\nL =',L,'\nD = ',D,'\nw = ',w)

t2 = time()
print(t2-t1)
#%%


#%%
# d,L,D,w = 0.37566904425667014, 2499.999999932132, 609.0871574258465, 0.9389265438674103 #for (8)
# d,L,D,w =  0.3980130399774895, 2499.9999887761237, 621.1111520306822, 0.8431245290816022 #for (9)
# d,L,D,w =  0.3957774907299554, 1900.0000000000023, 593.352331815572, 0.6760494622850699 #for (10)
# d,L,D,w =  0.39250081308793844, 2499.9998858815215, 608.9698382219664, 0.8415913885599839 # for (12)
# d,L,D,w = 0.3779326761033847, 1900.000000000452, 595.3599432631457, 0.8001854350978813 #for (13)
# d,L,D,w = 0.37940313057152375, 2499.999999773106, 616.4550176521002, 0.9014385246832246 # for (14)
# d,L,D,w =  0.37917605048206754, 2499.999999905409, 608.9334436745069, 0.9216659970300204 # for (15)
# d,L,D,w = 0.3780874065675317, 1900.000065431102, 590.598215058353, 0.7501207000178731 #for (16)
# d,L,D,w = 0.3773925197773744, 2499.9999998625904, 609.1629945522117, 0.9400966965894046 # for (17)
# d,L,D,w =  0.37804894658912125, 1900.0019814700518, 603.2782092108954, 0.7247724325836093 #for (18)
# d,L,D,w =  0.37799748108152204, 2499.999999995713, 622.819756163178, 0.9313900832636329 #for (19)
# d,L,D,w =  0.37903057740207474, 2499.9999997968257, 609.1642474117855, 0.9335198064091149 #for (20)
# d,L,D,w =  0.3788519951979505, 2499.9999999787165, 608.939644473253, 0.9359136030520501 #for (21)
# d,L,D,w =  0.3795787182927243, 1900.0000917628493,  590.2214894913317, 0.7402513080313529 #for (22)
# d,L,D,w =  0.3782083129763212, 1900.0000000000182, 585.4933773165832, 0.7459332502126153 #for (23)
# d,L,D,w =  0.3783574601007215, 2499.999998376778, 610.4975328127313, 0.9983736262657915 #for (24)
d,L,D,w =  0.37990734146147753, 2199.984663102545, 599.9788548057287, 0.907462683712167  #for 30(0)

t1 = time()

fpass = 90
file = ['/Volumes/Ice blocks/Ice_block_30(0)/Ice_block_30(0)','.tif']
barl = [765,980,895,907]
lims = [100,1000,240,810]


masks = []
refes = []
pps, nes = [], []
for npa in tqdm(range(0,80)):
    
    if npa < 40: tod, ref = all_steps2(file, barl, fpass, bhei = 500, dist=7, prominence=100., npa=npa)
    elif npa < 70: tod, ref = all_steps2(file, barl, fpass, bhei = 1000, dist=7, prominence=100., npa=npa)
    else: tod, ref = all_steps2(file, barl, fpass, bhei = 600, dist=7, prominence=100., npa=npa)
    
    if npa == 49: mask = recognice(ref, lims, scale=0.08e4, sigma=0.9, min_size=100)
    elif npa < 60: mask = recognice(ref, lims, scale=0.05e4, sigma=0.9, min_size=100)
    elif npa in [67,72,75,76,77,78,79]:  mask = recognice(ref, lims, scale=0.04e4, sigma=0.5, min_size=10000)
    elif npa in [61,62,63,64,65,68,69,70,71,73,74]: mask = recognice(ref, lims, scale=0.5e4, sigma=0.5, min_size=50000)
    elif npa in [66]: mask = recognice(ref, lims, scale=0.6e4, sigma=0.9, min_size=100)
    
    # if npa > 91:
    #     mask[:110 + 153 + 2*(npa%92),:] = False
    
    # # mask[990:,:] = False
    # if npa == 72: mask = binary_erosion(masks[-1],disk(3))
    
    masks.append(mask)
    refes.append(ref)

# maskr = np.zeros_like(todr[0])
# maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 
    

    phar, ne, A,B = st_psm(tod * mask, frecline=600, initfrec=20, sumA = -1, nef=13)  
    
    pps.append(phar)
    nes.append(ne)
    

# ppr =  unwrap(phar*maskr)

t2 = time()
print(t2-t1)
#%%

for n in range(60,80,1):
    plt.figure()
    # plt.imshow( mark_boundaries(refes[n]/np.max(refes[n]),masks[n]>0,color=(1,0,1)), cmap='gray')
    plt.imshow(pps[n])
    # plt.imshow(refes[n], cmap='gray')
    # plt.imshow(refes[n] > 900)
    
#     plt.title(n)
#     plt.show()
    
# plt.figure()
# plt.plot(nes,'.-')
# plt.grid()
# plt.show()

    
# plt.figure()
# plt.plot(ppr,'.-')
# plt.grid()
# plt.imshow(ppr)
# plt.plot(ppr[:,400])
# plt.plot(ppr[400])
# plt.show()

#%%

# np.shape(refes)
# mer = np.median(refes, axis=(0))

#%%
i = 30
plt.figure()
plt.imshow(refes[i], cmap='gray')
plt.show()
plt.figure()
plt.imshow(refes[i+1], cmap='gray')

plt.show()
plt.figure()
plt.imshow((refes[i+1] - refes[i]) < -100, cmap='gray')
plt.show()

#%%
# =============================================================================
# Test for parameters for recognice
# =============================================================================
hielis = []
for n in range(66,67):
# n = 0

    # plt.figure()
    # plt.imshow( refes[n][lims[0]:lims[1],lims[2]:lims[3]]  )
    # plt.show()

    lims = [100,1000,240,810]
    
    refe = refes[n][lims[0]:lims[1],lims[2]:lims[3]]  
    refi = enhance_contrast(refe/np.max(refe),disk(7))
    
    # if n<40: divs = felzenszwalb(refi, scale=0.9e3, sigma=1.0, min_size=500)
    divs = felzenszwalb(refi, scale=0.6e4, sigma=0.9, min_size=100) #scale 0.05e4, sigma 0.9, min 100 (hasta 60), excepto s: 0.08e4 (a 49), s: (a 61) 
    # scale 0.5e4, sigma 0.5, min 50000 (61-74), no 67/72,  scale=0.04e4, sigma=0.5, min_size=10000 (67,72,75-79)
    
    loci = [300,400,200,300]
    lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
    hiel = binary_closing(divs==lab, disk(15))
    hieli = remove_small_holes(hiel,area_threshold=1e4)
    
    # if n > 91:
    #     hieli[:153 + 2*(n%92),:] = False
    
    # hieli[990-100:,:] = False
    
    # plt.figure()
    # plt.imshow( mark_boundaries(refe/np.max(refe),divs,color=(1,0,1)), cmap='gray' )
    # plt.title(n)
    # plt.show()
    
    # plt.figure()
    # plt.imshow( hieli )
    # plt.title(n)
    # plt.show()
    
    plt.figure()
    plt.imshow( mark_boundaries(refe/np.max(refe),hieli,color=(1,0,1)), cmap='gray' )
    plt.title(n)
    plt.show()
#%%

plt.figure()
plt.imshow( refes[92] )
plt.show()
plt.figure()
plt.imshow( refes[93]-refes[92] )
plt.show()
#%%
# for n in range(60,70):
#     plt.figure()
#     # plt.imshow(masks[n])
#     plt.imshow(pps[n])
#     plt.show()

#     plt.figure()
#     # plt.imshow(masks[n])
#     plt.imshow(pps[n])
#     plt.show()


n = 40
pps, masks = np.array(pps), np.array(masks)

t1 = time()
pf = ma.masked_array( pps[:n], mask= -masks[:n]+1 )
up = unwrap( pf )
t2 = time()
print(t2-t1)
#%%
# for i in range(20,40):
#     plt.figure()
#     plt.imshow(up[i])
#     plt.colorbar()
#     plt.show()

lims = [220,970,180,650]

ppr_c = ppr[lims[0]:lims[1],lims[2]:lims[3]]

x,y = np.meshgrid( np.arange(lims[2],lims[3]),np.arange(lims[0],lims[1]) )
def fitplane(v):
    err = np.std(v[0] + v[1]*x + v[2]*y - ppr_c )
    su = np.sum( v[0] + v[1]*x + v[2]*y - ppr_c )
    return err+su  #np.sum( (v[0] + v[1]*x + v[2]*y - ppr_c )**1 ) 
    
les = least_squares(fitplane, (13,-0.02115,-0.001119), bounds=((12,-0.03,-0.003),(13.5,-0.009,-0.0009))  ) 
dats = les.x
# def fip2(v):
#     return np.sum(v[0] + dats[1]*x + dats[2]*y - ppr_c)

x,y = np.meshgrid(np.arange(1024),np.arange(1024))
plane = dats[0] + dats[1]*x + dats[2]*y

plt.figure()
plt.imshow(ppr)
# plt.imshow(y)
plt.show()


line = 300

plt.figure()
# plt.imshow(ppr)

plt.plot(ppr[:,line],'b-')
# plt.plot((y*-0.001)[:,400],'b--')
plt.plot(plane[:,line],'b--')

plt.plot(ppr[line],'g-')
# plt.plot((x*-0.01)[400],'g--')
plt.plot(plane[line],'g--')
plt.show()


# altus = alt(-(up-ppr), L, D, w)

# me = np.mean(altus,axis=(1,2)).data
# sd = np.std(altus,axis=(1,2)).data

#%%
n = 80

masks = np.array(masks)
pps = np.array(pps)

t1 = time()
pf = ma.masked_array( pps[:n], mask= -masks[:n]+1 )
up = unwrap( pf )
t2 = time()
print(t2-t1)


# altus = alt(-(up-plane), L, D, w)
altus = alt(-(up-ppr), L, D, w)
# altus = alt( -up, L,D,w)

me = np.mean(altus,axis=(1,2)).data
sd = np.std(altus,axis=(1,2)).data

#%%
plt.figure()
# plt.plot(me,'.-')
# plt.plot(sd,'.-')
plt.plot(up[:,450,450],'.-')
plt.grid()
plt.show()

# # # n = 4
# for n in range(40,80,2):
#     plt.figure()
#     plt.imshow(altus[n])
#     plt.colorbar()
#     plt.title(n)
#     plt.show()

    
# plt.figure()
# plt.plot(altus[0,400,:])
# # plt.plot(altus[0,:,500])
# # plt.plot(altus[0,:,600])
# plt.grid()
# plt.show()

# plt.figure()
# plt.imshow(pps[n])
# plt.show()
# plt.figure()
# plt.imshow( np.unwrap(np.unwrap(pps[n],axis=0),axis=1)  )
# plt.show()

# plt.figure()
# plt.imshow( (altus.mask)[10] )
# plt.show()
# plt.figure()
# plt.imshow( (altus.data)[n] )
# plt.show()

# plt.figure()
# plt.plot( (altus.data)[n,:,400])
# plt.plot( (altus.data)[n,400])
# plt.show()
#%%
alts = altus.data
alts[ altus.mask ] = np.nan

# np.savetxt('./Height profiles/ice_block_0_5.csv',alts, delimiter=';')
print('Are you sure to save? (y/n):') #to make sure I don't overwrite old things
x = input()
if x == 'y': 
    print('Saved')
    np.save('./Documents/Height profiles/ice_block_30_11',alts)
else:
    print('Not saved')

#%%
d,L,D,w =  0.3980130399774895, 2499.9999887761237, 621.1111520306822, 0.8431245290816022 #for (9)

t1 = time()

fpass = 90
file = ['/Volumes/Ice blocks/Ice_block_0(9)/Ice_block_0(9)','.tif']
barl = [740,960,865,892]
lims = [120,975,173,650]

maskss = []
refess = []
ppss = []
for npa in tqdm(range(67,72)):
    tod, ref = all_steps2(file, barl, fpass, bhei = 3000, dist=7, prominence=0.001, npa=npa)
    
    if npa<60: mask = recognice(ref, lims, scale=0.1e5, sigma=0.6, min_size=5000)
    # elif npa<60: mask = recognice(ref, lims, scale=0.7e3, sigma=0.6, min_size=5000)
    # # elif npa == 70 or npa ==71: mask = recognice(ref, lims, scale=0.7e3, sigma=0.5, min_size=5000)
    else: mask = recognice(ref, lims, scale=0.02e5, sigma=0.6, min_size=5000)
    
    if npa == 18: mask = binary_erosion(masks[17],disk(3))
    elif npa in [26,27]: mask = binary_erosion(masks[25],disk(3))
    elif npa == 28: mask = binary_erosion(masks[25],disk(5))
    elif npa == 51: mask = binary_erosion(masks[50],disk(3))
    
    maskss.append(mask)
    refess.append(ref)

# maskr = np.zeros_like(todr[0])
# maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 

    # plt.figure()
    # plt.imshow( (tod*mask)[0])
    # plt.title(npa)
    # plt.show()

    if npa in [68,70]: phar, ne, A,B = st_psm(tod * mask, frecline=400, initfrec=20, sumA = -1)
    else: phar, ne, A,B = st_psm(tod * mask, frecline=600, initfrec=20, sumA = -1)
         
    print(npa,ne)
    
    ppss.append(phar)

# ppr =  unwrap(phar*maskr)

t2 = time()
print(t2-t1)

#%%
for n in range(5):
    plt.figure()
    # plt.imshow( mark_boundaries(refess[n]/np.max(refess[n]), maskss[n]>0,color=(1,0,1)))
    plt.imshow( ppss[n] )
    plt.show()



# frecline = 600
# initfrec = 20

# nt,ny,nx = np.shape(tod)
# #step 1: PSP to get A and B
# A, B = 0,0
# for i in range(nt):
#     A += (tod[i] / ref) * mask
#     B += (tod[i] / ref) * mask * np.exp(-1j * 2*np.pi*i/nt)
# A, B = np.abs(A)/nt, np.abs(B) * 2/nt

# tod_b = ( (tod/ref) * mask + -1 * A) / (B + 0.000001) 
# # step 2: sample Moire for all images
# Am,BPm = 0,0

# im = tod_b[0]

# fto = np.fft.fft( im[frecline] ) #[initfrec:int(nx/2)]
# kfo = np.fft.fftfreq(nx)
# ind = np.argmax(np.abs(fto)[initfrec:int(nx/2)])
# ne = int(1/kfo[ind + initfrec]) 

# print(ind, ne)

# # plt.figure()
# # plt.imshow((tod[0]/ref)*mask)
# # # plt.imshow(A)
# # plt.colorbar()
# # plt.show()

# plt.figure()
# # plt.plot(im[frecline])
# plt.plot(np.abs(fto))
# plt.grid()
# plt.show()

# plt.figure()
# plt.imshow(tod_b[0], vmax=1, vmin=-5)
# # plt.imshow(B)
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow(phar)
# plt.show()


# ups = [] 

# for i in range(10):
    
#     pf = ma.masked_array( ppss[i], mask= -maskss[i]+1 )
#     up = unwrap( pf )
#     ups.append(up)
#     # plt.figure()
#     # plt.imshow(up)
#     # plt.show()

# ups = np.array(ups)
# upn = np.unwrap(ups,axis=0)

# pf = ma.masked_array(ppss, mask=-np.array(maskss)+1)
# ups = unwrap(pf)

# for i in range(10):
#     plt.figure()
#     plt.imshow(ups[i])
#     plt.show()


# plt.figure()
# plt.plot(ups[:,600,400])
# plt.show()
#%%

# for ice_block_0(9)
d,L,D,w =  0.3980130399774895, 2499.9999887761237, 621.1111520306822, 0.8431245290816022 #for (9)

t1 = time()

fpass = 90
file = ['/Volumes/Ice blocks/Ice_block_0(9)/Ice_block_0(9)','.tif']
barl = [740,960,865,892]
lims = [120,975,173,650]

masks = []
refes = []
pps, nes = [], []
for npa in tqdm(range(85)):
    tod, ref = all_steps2(file, barl, fpass, bhei = 3000, dist=7, prominence=0.001, npa=npa)
    
    if npa<60: mask = recognice(ref, lims, scale=0.1e5, sigma=0.6, min_size=5000)
    # elif npa<60: mask = recognice(ref, lims, scale=0.7e3, sigma=0.6, min_size=5000)
    # # elif npa == 70 or npa ==71: mask = recognice(ref, lims, scale=0.7e3, sigma=0.5, min_size=5000)
    else: mask = recognice(ref, lims, scale=0.02e5, sigma=0.6, min_size=5000)
    
    if npa == 18: mask = binary_erosion(masks[17],disk(3))
    elif npa in [26,27]: mask = binary_erosion(masks[25],disk(3))
    elif npa == 28: mask = binary_erosion(masks[25],disk(5))
    elif npa == 51: mask = binary_erosion(masks[50],disk(3))
    
    masks.append(mask)
    refes.append(ref)

# maskr = np.zeros_like(todr[0])
# maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 

    if npa in [68,70]: phar, ne, A,B = st_psm(tod * mask, frecline=400, initfrec=20, sumA = -1)
    else: phar, ne, A,B = st_psm(tod * mask, frecline=600, initfrec=20, sumA = -1)
    
    pps.append(phar)
    nes.append(ne)
    

# ppr =  unwrap(phar*maskr)

t2 = time()
print(t2-t1)
#%%

fpass = 90
filer = ['/Volumes/Ice blocks/mano_ref/mano_ref','.tif']
barl = [675,887,960,990] # for _ref 
# lims = [110,472,500,850]
lims = [50,450,200,600]

todr, refr = all_steps2(filer, barl, fpass, bhei = 1500, dist=7, prominence=100.)
maskr = np.zeros_like(todr[0])
maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 
phar, ne, A,B = st_psm(todr*maskr, frecline=400, initfrec=5, sumA = -1)
ppr =  unwrap(phar*maskr)[lims[0]:lims[1],lims[2]:lims[3]]


filer = ['/Volumes/Ice blocks/mano1/mano1','.tif']

todr, refes = all_steps2(filer, barl, fpass, bhei = 1500, dist=7, prominence=100., npa=2)

# refe = refes[lims[0]:lims[1],lims[2]:lims[3]]  
# refi = enhance_contrast(refe/np.max(refe),disk(7))
# divs = felzenszwalb(refi, scale=0.6e5, sigma=0.2, min_size=10) #scale 0.1e5, sigma 0.7, min 10 (hasta 80),  s: 0.07e5 (a 112), s: 
# loci=[300,400,200,300]
# lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
# hieli = remove_small_holes(divs==lab,area_threshold=1e4)
# mask = np.zeros_like(refes)
# mask[lims[0]:lims[1],lims[2]:lims[3]] += hieli
# # maskr = np.zeros_like(todr[0])
# # maskr[lims[0]:lims[1],lims[2]:lims[3]] = 1. 

# phar, ne, A,B = st_psm(todr*mask, frecline=400, initfrec=5, sumA = -1)
# ppc =  unwrap(phar*mask)[lims[0]:lims[1],lims[2]:lims[3]]

#%%

#%%
from skimage.measure import label, regionprops
lims = [50,450,200,600]


refe = refes[lims[0]:lims[1],lims[2]:lims[3]]  
refi = enhance_contrast(refe/np.max(refe),disk(7))
# refb = (refi > 10) * (refi < 200)
# refl = label(refb)
# loci=[150,250,200,300]
# # lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
# # hiel = binary_closing(divs==lab, disk(15))
# hieli = remove_small_holes(refl==4,area_threshold=1e4)

# plt.show()
# plt.imshow( refb )
# plt.show()

# # if n<40: divs = felzenszwalb(refi, scale=0.9e3, sigma=1.0, min_size=500)
divs = felzenszwalb(refi, scale=0.6e5, sigma=0.2, min_size=10) #scale 0.1e5, sigma 0.7, min 10 (hasta 80),  s: 0.07e5 (a 112), s: 
    

loci=[300,400,200,300]
lab = int(np.median(divs[loci[0]:loci[1],loci[2]:loci[3]])) 
# hiel = binary_closing(divs==lab, disk(15))
hieli = remove_small_holes(divs==lab,area_threshold=1e4)

# # if n > 91:
# #     hieli[:153 + 2*(n%92),:] = False

# # hieli[990-100:,:] = False

plt.figure()
# plt.imshow( mark_boundaries(refe/np.max(refe),hieli,color=(1,0,1)), cmap='gray' )
plt.imshow( mark_boundaries(refe/np.max(refe),divs,color=(1,0,1)), cmap='gray' )
plt.title(n)
plt.show()

#%%
hieli2 = binary_erosion(hieli, disk(3))
ppc[hieli2 == 0] = np.nan


plt.figure()
plt.imshow(ppr)
plt.show()
plt.figure()
plt.imshow(ppc)
plt.show()
#%%



ys, xs = np.arange(390-28), np.arange(350) 
xs,ys = np.meshgrid(xs,ys)

blue = np.array([1., 1., 1.])
rgb = np.tile(blue, (ppc.shape[0], ppc.shape[1], 1))

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

ls = LightSource(azdeg=10,    altdeg=20)
illuminated_surface = ls.shade_rgb(rgb, ppc) #, vert_exag=10)

ax.plot_surface(xs, ys , ppc, ccount=300, rcount=300, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
                antialiased=True,
                facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

# ax.plot_wireframe(xs, ys , ppc, ccount=50, rcount=50, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
#                 antialiased=True )
                # facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

# ax.set_xlabel('x (mm)')
# ax.set_ylabel('z (mm)')
# ax.set_zlabel('y (mm)')
# ax.invert_zaxis()
ax.invert_xaxis()
# ax.invert_yaxis()

lxd, lxi = np.nanmax(xs), np.nanmin(xs)
lzd, lzi = np.nanmin(ys), np.nanmax(ys)
lyi, lyd = np.nanmin(ppc-ppr), np.nanmax(ppc)

print( lzd,lzi )
print( lxd,lxi )
print( lyd,lyi )

# ax.set_box_aspect([2,1.6,4])
# ax.set_zlim(-180,140)
# ax.set_xlim(110,-70)
# ax.set_ylim(-60,0)

# ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)])
# ax.set
# ax.set_zlim(lzd-5,lzi+5)
# ax.set_xlim(lxd+5,lxi-5)
# ax.set_ylim(lyi-5,lyd+5)

plt.locator_params(axis='y', nbins=1)
# ax.text(30,40,180, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title( 't = '+str(0.5*n)+'s', fontsize=12)

# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
ax.view_init(85,85)
# plt.savefig('./Documents/blocksurface.png',dpi=400, bbox_inches='tight')
plt.show()
#%%


file = ['/Volumes/Ice blocks/Ice_block_0(12)/Ice_block_0(12)','.tif']
barl = [765,980,895,907] # for all 30(0)

frames = [90*5 + i for i in range(72)] + [90*5 + i for i in range(80,90)]  

lista_im = []
for n in frames: # tqdm(range(90*0,90*1)):
    with imageio.get_reader(file[0]+ str(n+1).zfill(6) +file[1], mode='I') as vid:
        im = vid.get_data(0)

    # print(n)
    plt.close('all')
    plt.ioff()
        
    plt.figure()
    plt.imshow( im[150:990,150:670] , cmap='gray', vmin=0, vmax=1800)
    # plt.imshow( im , cmap='gray', vmin=500)
    plt.colorbar()
    # plt.show()
    # print(n)

    plt.savefig('imgifi.png',dpi=400, bbox_inches='tight')
    lista_im.append(imageio.imread('imgifi.png')[:,:,0])
imageio.mimsave('./Documents/patronmov.gif', lista_im, fps=15, format='gif')


