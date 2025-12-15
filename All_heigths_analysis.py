#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:17:46 2024

@author: tomasferreyrahauchar
"""

import h5py
from PIL import Image, ImageDraw
import io
import cv2

import numpy as np
import numpy.ma as ma

# from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, correlate, fftconvolve, peak_prominences, correlate2d, peak_prominences, savgol_filter, convolve, convolve2d
import scipy.ndimage as snd # from scipy.ndimage import rotate from scipy.ndimage import convolve
from scipy.stats import linregress, skew, kurtosis
from scipy.interpolate import make_interp_spline, Rbf, griddata, splrep, splev
from scipy.ndimage import maximum_filter

# import rawpy
import imageio
from tqdm import tqdm
from time import time

from skimage.filters import gaussian, frangi, sato, hessian, meijering, roberts, sobel #, gabor
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion, thin, skeletonize, binary_dilation
from skimage.morphology import remove_small_objects, binary_opening, dilation
from skimage.segmentation import felzenszwalb, mark_boundaries, watershed
from skimage.restoration import unwrap_phase as unwrap
from skimage.feature import peak_local_max

import uncertainties as un
from uncertainties import unumpy

import os 

import itertools

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.collections import LineCollection

# plt.rcParams.update({'font.size':12})

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plt.rcParams.update({
    # 'axes.edgecolor': 'black',
    "text.usetex": True,
    "font.family": "serif",
    'font.size':12,
})
# mpl.use('Agg') 


def nangauss(altus, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):    
        V = altus.copy()
        V[np.isnan(altus)] = 0
        W = 0 * altus.copy() + 1
        W[np.isnan(altus)] = 0
        VV,WW = snd.gaussian_filter(V, sigma), snd.gaussian_filter(W, sigma)
        gdp = VV/WW
        gdp[np.isnan(altus)] = np.nan
        return gdp

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

def area_interest(halg,t,x,y, sob=4000, sho=4000):
    gt,gy,gx = np.gradient(halg, t,y,x)
    filtr = []
    for i in tqdm(range(len(t))):
        ff3 = remove_small_objects( remove_small_holes( (1.*gy[i]>0) * (1.*np.abs(gx[i])<1) > 0, area_threshold=sho ), min_size=sob )
        filtr.append( ff3 )
    filtr = np.array(filtr) * 1.
    filtr[ filtr == 0] = np.nan
    return filtr

def area_interest_0a(halg,t,x,y, sob=4000, sho=4000):
    gt,gy,gx = np.gradient(halg, t,y,x)
    filtr = []
    for i in tqdm(range(len(t))):
        ff3 = remove_small_objects( remove_small_holes( (1.*np.abs(gx[i])<1) > 0, area_threshold=sho ), min_size=sob )
        filtr.append( ff3 )
    filtr = np.array(filtr) * 1.
    filtr[ filtr == 0] = np.nan
    return filtr

def inter_pol(halgt, xrt,yrt):
    dy, dx = yrt[:,1:,:]-yrt[:,:-1,:], xrt[:,:,1:] - xrt[:,:,:-1]
    xn = np.arange(np.nanmin(xrt),np.nanmax(xrt), np.abs(np.nanmean(dx)) ) 
    yn = np.arange(np.nanmin(yrt),np.nanmax(yrt), np.abs(np.nanmean(dy)) )[::-1]
    xn,yn = np.meshgrid(xn,yn)

    hint = []
    for n in tqdm(range(len(t))):

        finan = ~np.isnan(yrt[n].flatten())
        points = np.array( [ (xrt[n].flatten())[finan], (yrt[n].flatten())[finan] ] )
        values = (halgt[n].flatten())[finan]
        
        hint.append( griddata(points.T, values, (xn,yn), method='linear') )
        
    return np.array(hint), xn, yn


def av_melt(hint,t,xn, timt=15):
    me = np.zeros_like(xn) * np.nan
    meer = np.zeros_like(xn) * np.nan
    lens = np.zeros_like(xn)
    
    ny,nx = np.shape(xn)
    for i in range(ny):
        for j in range(nx):
            linea = hint[:,i,j]
            filt = ~np.isnan(linea)
            
            # lre = linregress(t,linea)
            if np.sum(filt)>timt: 
                lre = linregress(t[filt],linea[filt],alternative='less')
            else: continue
            me[i,j], meer[i,j] = lre[0], lre[4]
            lens[i,j] = np.sum(filt)

    return me, meer, lens
    

def untilt( halg, mmm, xr, yr):
    nt,nx,ny = np.shape(halg)
    
    n = 0
    prfo = halg[n] * mmm[n]
    rrx, rry = xr[n], yr[n]
    
    A = np.array([rrx.flatten()*0+1,rrx.flatten(),rry.flatten()]).T
    finan = ~np.isnan(prfo.flatten())

    coeff, r, rank, s = np.linalg.lstsq(A[finan] , (prfo.flatten())[finan], rcond=None)
    plane = coeff[0] + coeff[1] * rrx + coeff[2] * rry
    plane = plane * mmm[n]

    m2 = mrot( -np.arctan(coeff[1]) )
    xrot, prot = rot(rrx,prfo,m2)
    
    pary,cov = curve_fit(lin, (rry.flatten())[finan], (prot.flatten())[finan])
    m2 = mrot( -np.arctan(pary[0]) )
    yrot, prot = rot(rry,prot,m2)
    
    uihalg = [ prot ]
    yrots, xrots = [yrot], [xrot]

    for n in range(1,nt):        
        prfo = halg[n] * mmm[n]
        rrx, rry = xr[n], yr[n]
            
        m2 = mrot( -np.arctan(coeff[1]) )
        xrot, prot = rot(rrx,prfo,m2)
        
        m2 = mrot( -np.arctan(pary[0]) )
        yrot, prot = rot(rry,prot,m2)
        
        uihalg.append( prot )
        yrots.append( yrot )
        xrots.append( xrot )
        
    return np.array(uihalg), np.array(xrots), np.array(yrots), np.arctan(coeff[1]) * 180 / np.pi , np.arctan(pary[0]) * 180 / np.pi

def water_dens(T,Si):
    S = Si/1000
    a1,a2,a3,a4,a5 = 9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8
    b1,b2,b3,b4,b5 = 8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5
    rho = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4 + b1*S + b2*S*T + b3*S * T**2 + b4*S * T**3 + b5 * S**2 * T**2    
    return rho

def polyfit(n,i,hints,xns,yns):
    haltura = hints[n][i]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xns[n][~np.isnan(haltura)], yns[n][~np.isnan(haltura)]
    A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol2(coeff,xn,yn):
    return coeff[0] + coeff[1] * xn + coeff[2] * yn + coeff[3] * xn**2 + coeff[4] * xn*yn + coeff[5] * yn**2

def polyfitn(n,i,hints,xns,yns,order=4):
    haltura = hints[n][i]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xns[n][~np.isnan(haltura)], yns[n][~np.isnan(haltura)]

    expos = list(dict.fromkeys( [expo for expo in itertools.permutations(list(range(order+1))*2,2) if sum(expo) < order+1] ))
    A = np.array( [xfit**e1 * yfit**e2 for (e1,e2) in expos ] ).T
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def poln(coeff,xn,yn,order=4):
    expos = list(dict.fromkeys( [expo for expo in itertools.permutations(list(range(order+1))*2,2) if sum(expo) < order+1] ))
    terms = np.array( [xn**e1 * yn**e2 for (e1,e2) in expos ] )
    coso = 0 
    for k in range(len(coeff)):
        coso += terms[k] * coeff[k]
    return coso


def track2d(mxs,mys,tts, delim=5, dtm=1):
    partsx, partsy = [],[]
    tiemps = []
    
    for i in range(len(mxs)):
        
        pacx,pacy,tac = mxs[i],mys[i],tts[i]
            
        for k in range(len(partsx)):

            ptrx, ptry, ttr = partsx[k][-1], partsy[k][-1], tiemps[k][-1]
            dist = (ptrx - pacx)**2 + (ptry - pacy)**2

            if dist < delim**2 and (tac-ttr) < dtm:  #and (tac-ttr)>0:
                partsx[k].append(pacx)
                partsy[k].append(pacy)
                tiemps[k].append(tac)
                break
        else: 
            partsx.append([pacx])
            partsy.append([pacy])
            tiemps.append([tac])
    
    return partsx,partsy,tiemps

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def density_millero(t, s):
    """
    Computes density of seawater in kg/m^3.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    return rho_sw

def R_rho(t,s):
    """
    Computes R_rho for initial temperature t and salinity s.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010
    
    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    
    drho0_T = 6.793952e-2 - 2 * 9.095290e-3 * t68 + 3 * 1.001685e-4 * t68 ** 2 - 4 * 1.120083e-6 * t68 ** 3 + 5 * 6.536336e-9 * t68 ** 4
    dA_T = -4.0899e-3 + 2 * 7.6438e-5 * t68 - 3 * 8.2467e-7 * t68 ** 2 + 4 * 5.3875e-9 * t68 ** 3
    dB_T = 1.0227e-4 - 2 * 1.6546e-6 * t68
    
    rho = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    drho_s = (A + 3/2 * B * sp**(1/2) + 2* C * sp) / 1.00472
    drho_t = (drho0_T + dA_T * sp + dB_T * sp**(3/2)) / (1 - 2.5e-4)
    
    bett,bets = -drho_t / rho, drho_s / rho 
    rrho = (bets * s) / (bett * t)
    return rrho

def barviolin(data, ax, x = [0], height=1.0, width=1.0, bins=20, alpha=0.5, color=[], marker='o' ):
    """
    data: list of data for histogram
    x: list of len(data). horizontal position of the data 
    height: float 0-1. Percentage of the height of the bar (1 full bar, 0 no bar)
    width: float. Width of the bar 
    """
    nda = len(data)
    for i in range(nda):
        daa = data[i]
        wid,bed = np.histogram(daa, bins=bins, density=True)
        bec, hei = (bed[1:] + bed[:-1])/2, bed[1:] - bed[:-1]
        lefts = x[i] - 0.5 * wid * width
        if len(color) < 1:
            ax.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha )
            # ax.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
            ax.plot( x[i], np.mean(daa), marker, color='black', markersize=4 )
        else:
            col = color[i]
            ax.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha, color=col )
            # ax.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
            ax.plot( x[i], np.mean(daa), marker, color='black', markersize=4 )

def diff_dens(salis, temps, resol=100):
    drho = []
    for n in range(len(salis)):
        sa_ar, te_ar = np.meshgrid( np.linspace(0,salis[n],resol), np.linspace(0,temps[n],resol) )
        dens_ar = density_millero(te_ar, sa_ar)
        drho.append( ( np.max(dens_ar) - np.min(dens_ar) ) / density_millero(temps[n], salis[n]) )
    return np.array(drho)

#%%


sal_v = ['5','15','23','8(1)','0','2','4','6','8(3)','12','14','18','15(2)','22','27','10']

salis = [8,15.2,23.38,8,0,2.31,4.06,6.40,8.26,12.87,14.8,18.13,15.2,21.9,27.39,10.26]

ds_v = [ 0.37566904425667014, 0.39250081308793844, 0.3957774907299554, 0.39250081308793844, 0.3779326761033847, 0.37940313057152375,
       0.37917605048206754, 0.3780874065675317, 0.3773925197773744, 0.37804894658912125, 0.37799748108152204, 0.37903057740207474,
       0.3788519951979505, 0.3795787182927243, 0.3782083129763212, 0.3783574601007215 ]

Ls = [2499.999999932132, 2499.9998858815215, 1900.0000000000023, 923.8014551576581, 1900.000000000452,
     2499.999999773106, 2499.999999905409, 1900.000065431102, 2499.9999998625904, 1900.0019814700518,
     2499.999999995713, 2499.9999997968257, 2499.9999999787165, 1900.0000917628493, 1900.0000000000182,
     2499.999998376778]

temp = [ 20.69, 20.08, 21.40, 19.95, 21.00, 20.65, 19.6, 20.14, 19.13, 18.57, 20.04, 20.37, 19.46, 18.60, 17.76, 19.22 ]

angys = [-0.4989198571584637, 1.051061984149644, 2.079515379592808, 0.3985240654858331, -1.9507403600527167, 1.7299203972639903, 
         -0.5955648596154526, 0.04606949625386255, 1.070896888539351, 1.2734103782185764, -0.3066349762054198, 1.802580128658579, 
         0.5823129218796972, 0.569349668086476, 1.0872516904725684, 0.6295984837347782]

salis_v = np.array(salis)
Ls_v = np.array(Ls)
temp_v = np.array(temp)
angys_v = np.array(angys)

sal_t = ['0','0','0','0','6','6','6','6','12','12','12','12','20','20','20','20','27','27','27','27','0','0','6','12','20','35','35','20','12','6',
       '35','0', '6','12','12','20',  '1','2','3',]
inc_t = ['0(3)','30','15(3)','45(2)','45(ng)','30','15','0','0','15','45','30','30','45','15','0','0','15','30','45','15(4)','15(5)',
       '45(2)','0(2)','30(2)','30','n15','n15','n15','n15','n15(2)','n15', '0(s)','0(s)','30(s)','0(s)',  '30','30','30',]
salis = [0, 0, 0, 0, 6.8, 7.0, 6.8, 6.8, 13.2, 13.3, 13.2, 13.3, 20.1, 20.1, 19.8, 20.0, 26.1, 26.1, 26.2, 26.2, 0, 0, 6.9, 12.9, 20.1,34.5,34.5,19.7, 
         13.0, 7.0, 34.3, 0, 5.8, 12.3, 12.4, 18.5, 1.2,1.8,2.9]
ang = [0,30,15,45,45,30,15,0,0,15,45,30,30,45,15,0,0,15,30,45,15,15, 45, 0, 30, 30, -15, -15, -15, -15, -15,  0, 0, 30, 0, -15, 30,30,30 ]
ds_t = [ 0.4019629986999602, 0.4079970860258477, 0.400784094531674, 0.4019866341895751, 0.4140232465732999, 0.4108146995511450, 0.405185916205820,
       0.3985171779558734, 0.4082777044681367, 0.399457458948112, 0.429624677624675, 0.4002974355025642, 0.3962951595980395, 0.4158467162824917,
       0.405560070548485, 0.406690755428839, 0.3986160751319692, 0.406029247234490, 0.403935349493398, 0.4274366684657002, 0.39842940043484637,
       0.3944444940666371, 0.42941247988993125, 0.3986508813811391, 0.41300121024756764, 0.39724780723266606, 0.3991597643255994, 0.4114831458151559,
       0.4034206887886922, 0.3963784515420281, 0.406148619144839, 0.40500559156677696,  
       0.4038900423099877, 0.4088590866221042, 0.40465748645946387, 0.4070953514695665, 0.4129039076897664, 0.4030493020902331, 0.39538334537770237]

Ls = [ 2099.9999966941955, 2097.2250082592454, 2100.003821154219, 2098.7729533565816, 2092.3116009701507, 2100.7921001119767, 2102.061867627691,
       2104.0217430207585, 2097.0613260850837, 2108.070882626512, 2101.360857870178, 2106.0608525686807, 2103.8609505143863, 2158.7930487367967,
       2112.093781969593, 2113.181455334151, 2104.4329627701713, 2092.891945129341, 2103.955261899692, 2107.0490635783035, 2100.163205198002,
       2102.2972076507995, 2103.9171613871736, 2099.967887686282, 2110.996676678041, 2099.255549750563, 2098.0517570404286, 2102.9647844012497,
       2101.945064364561, 2100.341294595434, 2103.0141188891594, 2101.2510931269435,
       2100.01045350225, 2103.588545842329, 2100.408699264378, 2103.0207273047863,  2111.085100071991, 2107.6124628616667, 2100.365882979232]

angys = [0.2752835802019292, 34.47170331852835, 13.06499750709190, 43.6538844493011, 42.63317480202235, 29.31556395720869, 15.64436382383116, 
         0.379506977103330, 2.122818303993659, 16.82489775638404, 47.66715755358717, 28.76640507341181, 31.15486854660724, 50.65480071850850, 
         19.29185570378762, -0.876278847660974, -1.445891941223953, 15.68633111320458, 28.61178195105051, 46.98820258183241, 14.85643305810015,
         18.694159382293858, 44.69885882480372, 0.4091221586816418, 29.57628972829382, 29.370955852274992, -18.732894940242627, -16.522627138169074,
         -16.27828076769843, -17.33545209415416, -16.58142778295442, -17.450360846901734,
         -1.6926163253504223, -2.671232787825012, 27.052620463266, 5.031218324124902,  29.316888864559527, 29.121527035263778, 28.25041391205312]

temp = [19.0, 19.0, 19.3, 19.7, 20.1, 20.3, 19.5, 20.0, 19.0, 19.0, 19.7, 19.4, 18.8, 19.2, 19.8, 19.5, 19.8, 19.4, 19.1, 20.0, 20.1, 19.4,
        19.4, 19.3, 19.2, 19.2, 18.7, 19.3, 18.9, 19.0, 19.1, 19.3, 19.2, 19.1, 18.8, 19.3, 19.3,19.0,19.0]

salis_t = np.array(salis)
ang_t = np.array(ang)
angys_t = np.array(angys)
Ls_t = np.array(Ls)
temp_t = np.array(temp)

#%%
# =============================================================================
# Reading the files
# =============================================================================
with h5py.File('/Users/tomasferreyrahauchar/Documents/Height profiles/npys/heights(s2).hdf5', 'r') as f:

    hints_v, xns_v, yns_v, ts_v = [], [], [], []
    for n in tqdm(range(len(salis_v))):
        
        hints_v.append( f['h'][n].reshape( f['n'][n] ) )
        xns_v.append( f['x'][n].reshape( (f['n'][n])[1:] ) )
        yns_v.append( f['y'][n].reshape( (f['n'][n])[1:] ) )
        ts_v.append( np.arange(0, (f['n'][n])[0] ) * 30 )


with h5py.File('/Users/tomasferreyrahauchar/Documents/Height profiles/npys/sloped_heights(s0)_final.hdf5', 'r') as f:

    hints_t, xns_t, yns_t, ts_t = [], [], [], []
    for n in tqdm(range(len(salis_t))):
        
        hints_t.append( f['h'][n].reshape( f['n'][n] ) )
        xns_t.append( f['x'][n].reshape( (f['n'][n])[1:] ) )
        yns_t.append( f['y'][n].reshape( (f['n'][n])[1:] ) )
        ts_t.append( np.arange(0, (f['n'][n])[0] ) * 30 )
        
#%%
orderpol = 4 #order 4 seems bettter for wtershed. Order 2 gives slightly better results for tracking maxima

difs_v, coeffs_v = [], []
for n in tqdm(range(len(salis_v))):
    difes, coeffs = [], []
    for i in (range(len(ts_v[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_v,xns_v,yns_v, order=orderpol)
        cuapla = poln(coeff,xns_v[n],yns_v[n], order=orderpol) 
        
        difes.append( (hints_v[n][i]-cuapla) )
        coeffs.append(coeff)

    difes = np.array(difes)
    difs_v.append(difes)
    coeffs_v.append(np.array(coeffs))

difs_t, coeffs_t = [], []
for n in tqdm(range(len(salis_t))):
    difes, coeffs = [], []
    for i in (range(len(ts_t[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_t,xns_t,yns_t, order=orderpol)
        cuapla = poln(coeff,xns_t[n],yns_t[n], order=orderpol) 
        
        difes.append( (hints_t[n][i]-cuapla) )
        coeffs.append(coeff)

    difes = np.array(difes)
    difs_t.append(difes)
    coeffs_t.append(np.array(coeffs))

#%%
# =============================================================================
# Melting rates
# =============================================================================
mes_v, sed_v = [], []
for n in tqdm(range(len(ds_v))):
    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    gt,gy,gx = np.gradient(hints_v[n], t,y,x)
    xs,ys = xns_v[n], yns_v[n]
    
    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    
    gt[np.isnan(gt)] = 0.0
    meltr = np.trapz( np.trapz( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    
    tmelr = np.trapz( meltr / area, x=t ) / t[-1]
    mes_v.append( tmelr )
    sed_v.append( 0 )

mes_t, sed_t = [], []
for n in tqdm(range(len(ds_t))):
    t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
    gt,gy,gx = np.gradient(hints_t[n], t,y,x)
    xs,ys = xns_t[n], yns_t[n]
    
    area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    
    gt[np.isnan(gt)] = 0.0
    meltr = np.trapz( np.trapz( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    
    tmelr = np.trapz( meltr / area, x=t ) / t[-1]
    mes_t.append( tmelr )
    sed_t.append( 0 )

# drho_rho_v = np.abs( density_millero(0, salis_v) - density_millero(temp_v, salis_v) ) / density_millero(temp_v, salis_v)
# drho_rho_t = np.abs( density_millero(0, salis_t) - density_millero(temp_t, salis_t) ) / density_millero(temp_t, salis_t)
# drho_rho_v = np.abs( density_millero(temp_v, 0) - density_millero(temp_v, salis_v) ) / density_millero(temp_v, salis_v)
# drho_rho_t = np.abs( density_millero(temp_t, 0) - density_millero(temp_t, salis_t) ) / density_millero(temp_t, salis_t)
drho_rho_v = np.abs( density_millero(0, 0) - density_millero(temp_v, salis_v) ) / density_millero(temp_v, salis_v)
drho_rho_t = np.abs( density_millero(0, 0) - density_millero(temp_t, salis_t) ) / density_millero(temp_t, salis_t)
# drho_rho_v = diff_dens(salis_v, temp_v, resol=100)
# drho_rho_t = diff_dens(salis_t, temp_t, resol=100)


# not sure if I'm calculating R_rho correctly. This values are taken from simen paper
beta_t, beta_s = 6e-5, 7.8e-4 # 1/K, 1/(g/kg)
orrho_v = (beta_s * salis_v) / (beta_t * (temp_v + 273.15)) 
orrho_t = (beta_s * salis_t) / (beta_t * (temp_t + 273.15))

rrho_v = np.abs(density_millero(temp_v, 0) - density_millero(temp_v, salis_v)) / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v)) #R_rho(temp_v, salis_v)
rrho_t = np.abs(density_millero(temp_t, 0) - density_millero(temp_t, salis_t)) / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t)) #R_rho(temp_t, salis_t)
# rrho_v = np.abs(density_millero(0, 0) - density_millero(0, salis_v)) / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v)) #R_rho(temp_v, salis_v)
# rrho_t = np.abs(density_millero(0, 0) - density_millero(0, salis_t)) / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t)) #R_rho(temp_t, salis_t)

# Nusselt number (with initial length scale )
rho_ice = 916.8 # kg / m^3
latent = 334e3 # m^2 / s^2
length0 = 32 / 100 # m
length0_short = 23 / 100 #m

thcon = 0.6 # m kg / s^3 °C
mes_v, mes_t = np.array(mes_v), np.array(mes_t)
Nu_v = -mes_v/1000 * rho_ice * latent * length0 / (thcon * temp_v )
Nu_t = -mes_t/1000 * rho_ice * latent * length0 / (thcon * temp_t )
Nu_t[-7:-3] = -mes_t[-7:-3]/1000 * rho_ice * latent * length0_short / (thcon * temp_t[-7:-3] )

#Rayleigh number
g = 9.81 #m / s^2
mu = 0.00103 #kg / m s
kt = 1.4e-7 #m^2 / s

Ra_v = g * np.cos(angys_v * np.pi/180)**1 * drho_rho_v * length0**3 / kt / mu 
Ra_t = g * np.cos(angys_t * np.pi/180)**1 * drho_rho_t * length0**3 / kt / mu 
Ra_t[-7:-3] = g * np.cos(angys_t[-7:-3] * np.pi/180)**1 * drho_rho_t[-7:-3] * length0_short**3 / kt / mu 

# beta_s = 7.8e-4 # (g/kg)^-1
# nu = 1.03e-6 # m^2 / s
# ks = kt/100 # m^2 / s

Ra_t[-7:-3] 


#%%
# Print 

# order = np.argsort(salis_v)
# for j in order:
#     print( "& {:.1f} \t& {:.1f} \t& {:.1f} \t& {:.3f} \t& {:.2f} \t& {:.0f} \t{}".format( salis_v[j], angys_v[j], temp_v[j], Ra_v[j]/1e7, rrho_v[j], Nu_v[j],r'\\') )


order = np.lexsort((angys_t,salis_t))
# for j in order:
#     if j in [32,33,34,35]:
#         continue
#     print("& {:.1f} \t& {:.1f} \t& {:.1f} \t& {:.3f} \t& {:.2f} \t& {:.0f} \t{}".format( salis_t[j], angys_t[j], temp_t[j], Ra_t[j]/1e7, rrho_t[j], Nu_t[j],r'\\') )
# for j in order:
#     if j in [32,33,34,35]:
#         print("& {:.1f} \t& {:.1f} \t& {:.1f} \t& {:.3f} \t& {:.2f} \t& {:.0f} \t{}".format( salis_t[j], angys_t[j], temp_t[j], Ra_t[j]/1e7, rrho_t[j], Nu_t[j],r'\\') )


for j in [4,10,14]:
    print( "& {:.1f} \t& {:.1f} \t& {:.1f} \t& {:.3f} \t& {:.2f} \t& {:.0f} \t{}".format( salis_v[j], angys_v[j], temp_v[j], Ra_v[j]/1e7, rrho_v[j], Nu_v[j],r'\\') )
for j in [1,5,11]:
    print("& {:.1f} \t& {:.1f} \t& {:.1f} \t& {:.3f} \t& {:.2f} \t& {:.0f} \t{}".format( salis_t[j], angys_t[j], temp_t[j], Ra_t[j]/1e7, rrho_t[j], Nu_t[j],r'\\') )


#%%

save_name = '' # 'melt_rates', 'nus', 'nus_rrho', 'rrho_nus(2)'
reference = False
shadowgraphy = False
small_ice = True
axis_x = 'rrho'
axis_y = 'nu'
mfc = None

plt.rcParams.update({'font.size':16})

cols = np.linspace(-20,51,256)
comap = np.array( [(cols+20)/71 , 0.5 *np.ones_like(cols) , 1-(cols+20)/71 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,6), sharey=True)

if axis_x == 'salinity':
    xvariable_v, xvariable_t = salis_v, salis_t
    # ax[r'$a)$'].set_xlim(-1.65,36.17)
    ax[r'$a)$'].set_xlim(-0.7,35.1)
    ax[r'$a)$'].set_xticks( list(range(0,36,5)) )
    ax[r'$a)$'].set_xlabel(r'$S$ (g/kg)')
elif axis_x == 'density':
    xvariable_v, xvariable_t = drho_rho_v, drho_rho_t
    ax[r'$a)$'].set_xticks([0.0015, 0.0018, 0.0021, 0.0024, 0.0027, 0.0030])        
    ax[r'$a)$'].set_xlabel(r'$\Delta \rho / \rho$ (g/kg)')    
elif axis_x == 'rrho':
    xvariable_v, xvariable_t = rrho_v, rrho_t
    ax[r'$a)$'].set_xlabel(r'$R_\rho$')    
elif axis_x == 'ra':
    xvariable_v, xvariable_t = Ra_v, Ra_t
    ax[r'$a)$'].set_xlabel(r'Ra$_{S}$')
    # ax[r'$a)$'].set_xscale('log')
    # ax[r'$a)$'].set_yscale('log')
if axis_y == 'melt rate':
    yvariable_v, yvariable_t = -mes_v, -mes_t
    yvarerr_v, yvarerr_t = [0.0018] * len(mes_v), [0.0011] * len(mes_t)
    ax[r'$a)$'].set_ylabel(r'$\dot{m}$ (mm/s)')
if axis_y == 'nu':
    yvariable_v, yvariable_t = Nu_v, Nu_t
    yvarerr_v, yvarerr_t = 0.0018/1000 * rho_ice * latent * length0 / (thcon * temp_v ), 0.0011/1000 * rho_ice * latent * length0 / (thcon * temp_t )
    ax[r'$a)$'].set_ylabel(r'Nu')


for n in range(len(ds_v)):
    ax[r'$a)$'].errorbar(xvariable_v[n], yvariable_v[n] * 1 , yerr=yvarerr_v[n], fmt='^', label=str(n)+'°', markersize=6, \
                 color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), capsize=3, mfc=mfc)
for n in [j for j in range(len(ds_t)) if j not in range(32,36)]:        
    ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='o', label=str(n)+'°', markersize=5, \
                  color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71), capsize=3, mfc=mfc)
if small_ice:
    for n in range(-7,-3):        
        ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='d', label=str(n)+'°', markersize=6, \
                      color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71), capsize=3, mfc=mfc)
        
ax[r'$a)$'].set_ylim(70.,190.)
ax[r'$a)$'].set_yticks(range(70,191,20))

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-20, 51), cmap=newcmp), ax=ax[r'$a)$'], location='top')
# cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticks( list(range(-20,51,10)) )
cbar.set_label( label=r"$\theta$ (°)") #, size=12)

leg1 = [ax[r'$a)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
lgd = ax[r'$a)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.07,1.3], frameon=False, ncols=3)

co2 = [(i/35,0,1-i/35) for i in salis]

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

# if axis_x == 'ra':
#     ax[r'$a)$'].set_xscale('log')
#     ax[r'$a)$'].set_yscale('log')

for n in [j for j in range(len(ds_t)) if j not in range(32,36)]: 
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='o', markersize=5,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3, mfc=mfc) #label=str(i)+'g/kg', \
for n in range(-7,-3):
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='d', markersize=6,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3, mfc=mfc) #label=str(i)+'g/kg', \

if shadowgraphy:
    # shadowgraphy experiments, they dont say much 
    ax[r'$b)$'].errorbar( [0,30], [0.017391, 0.017927], yerr=[0.000014, 0.000036], fmt='s', color='black' ) # "clear" (not really that clear)
    ax[r'$b)$'].errorbar( [0,30], [0.014225, 0.018498], yerr=[0.000014, 0.000021], fmt='d', color='black' ) # opaque

# ax[r'$b)$'].set_xlim(-20.,52)
if axis_x == 'salinity':
    cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$b)$'], location='top')
    # cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_xticks( list(range(0,36,5)) )
    cbar.set_label( label=r"$S$ (g/kg)") #, size=12)
elif axis_x == 'rrho':
    cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 9), cmap=newcmp), ax=ax[r'$b)$'], location='top')
    # cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_xticks( list(range(0,10,1)) )
    cbar.set_label( label=r"$R_\rho$") #, size=12)

if reference:
    th = np.linspace(0,50*np.pi/180,50)
    plt.plot( th*180/np.pi, 0.016 * np.cos(th)**(2/3), 'k--' , label=r'McConnochie & Kerr 2018 ($\propto \cos^{2/3}(\theta)$)')
    ax[r'$b)$'].legend()


ax[r'$b)$'].set_xlabel(r'$\theta$ (°)')
ax[r'$b)$'].set_xticks( list(range(-20,51,10)) )
# ax[r'$b)$'].set_ylabel(r'Melting rate (mm/s)')


# for labels,axs in ax.items():
#     axs.annotate(labels, (-0.15,1), xycoords = 'axes fraction')

if len(save_name) > 0: plt.savefig('./Documents/Figs morpho draft/'+save_name+'.pdf',dpi=400, bbox_inches='tight')
plt.show()


#%%
# =============================================================================
# Local Nu vs Ra
# =============================================================================
rho_ice = 916.8 # kg / m^3
latent = 334e3 # m^2 / s^2
length0 = 32 / 100 # m
length0_short = 23 / 100 #m

thcon = 0.6 # m kg / s^3 °C

#Rayleigh number
g = 9.81 #m / s^2
mu = 0.00103 #kg / m s
kt = 1.4e-7 #m^2 / s

# drho_rho_v = diff_dens(salis_v, temp_v, resol=100)
# drho_rho_t = diff_dens(salis_t, temp_t, resol=100)
drho_rho_v = np.abs( density_millero(0, salis_v) - density_millero(temp_v, salis_v) ) / density_millero(temp_v, salis_v)
drho_rho_t = np.abs( density_millero(0, salis_t) - density_millero(temp_t, salis_t) ) / density_millero(temp_t, salis_t)

compensate = 1/3

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,6), sharey=False)

for n, set_n in zip([4,5,6,0,2,29,31],['v','v','v','t','t','t','t' ]):     
# for n, set_n in zip([4,14],['v','v']):     
    if set_n == 'v':
        temp_b, angys_b = temp_v, angys_v
        t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
        hints_b = hints_v
        gt,gy,gx = np.gradient(hints_v[n], t,y,x)
        xs,ys = xns_v[n], yns_v[n]
        drho = drho_rho_v
    elif set_n == 't':
        temp_b, angys_b = temp_t, angys_t
        t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
        hints_b = hints_t
        gt,gy,gx = np.gradient( nangauss(hints_t[n], 2), t,y,x)
        xs,ys = xns_t[n], yns_t[n]
        drho = drho_rho_t
    
    fil = ~np.isnan(gt)  #(gt!=0)
    # gtn = np.copy(gt)
    # gtn[~fil] = np.nan
    
    ymax = np.array( [ np.where(fil[i])[0][0] for i in range(len(hints_b[n]))  ])
    
    gtx = np.nanmean(gt, axis=2)
    rgtx = np.empty_like(gtx)
    for i in range(len(gtx)): rgtx[i] = np.roll(gtx[i], -ymax[i])

    Nu = -rgtx/1000 * rho_ice * latent * (y[0]-y)/1000 / (thcon * temp_b[n] )
    Ra = g * np.cos(angys_b[n] * np.pi/180)**3 * drho[n] * ((y[0]-y)/1000)**3 / kt / mu
    
    Num = np.nanmedian( Nu[10:], axis=0 )
    # for i in [0,10,20,30,40,50,60]:
    #     plt.plot(Ra, Nu[i] / Ra**(1/3), '.-' , label=i)
    if set_n == 'v':
        ax[r'$a)$'].plot(Ra[1:], Num[1:] / Ra[1:]**(compensate), '-' , label= r'$S={:.1f}$, $\theta={:.1f}°$'.format(salis_v[n], angys_v[n]),
                 color = np.array([0.5 , salis_v[n]/7 , 1-salis_v[n]/7 ]) )
    elif set_n == 't':
        ax[r'$a)$'].plot(Ra[1:], Num[1:] / Ra[1:]**(compensate), '--' , label= r'$S={:.1f}$, $\theta={:.1f}°$'.format(salis_t[n], angys_t[n]),
                 color = np.array([0.5 , salis_t[n]/7 , 1-salis_t[n]/7 ]) )

    # rgm = np.nanmedian( -rgtx[10:], axis=0 )
    # plt.plot(y[0]-y, rgm, label=n)
# plt.plot(y[0]-y, (y[0]-y)**(-1/4) * 5e-2, 'k--')

# ax[r'$a)$'].plot( Ra[1:], Ra[1:]**(1/4 - compensate) * 2.0 , 'k--' )
# ax[r'$a)$'].plot( Ra[1:], Ra[1:]**(1/3 - compensate) * 0.5 , 'k--' )
ax[r'$a)$'].set_xscale('log')
ax[r'$a)$'].set_yscale('log')

ax[r'$a)$'].set_yticks([0.4,1,4])
ax[r'$a)$'].get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax[r'$a)$'].set_yticklabels([], minor=True)


ax[r'$a)$'].set_xlabel(r'$\textrm{Ra(y)}$')
ax[r'$a)$'].set_ylabel(r'$\textrm{Nu(y)} / \textrm{Ra(y)}^{1/4}$')

ax[r'$a)$'].legend(loc='lower right')
# plt.show()

compensate = 1/3

for n, set_n in zip([14,14,16,26,30],['v','t','t','t','t', 'v','v']):     
# for n, set_n in zip([14,4],['v','v']):     
    if set_n == 'v':
        temp_b, angys_b = temp_v, angys_v
        t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
        hints_b = hints_v
        gt,gy,gx = np.gradient(hints_v[n], t,y,x)
        xs,ys = xns_v[n], yns_v[n]
        drho = drho_rho_v
    elif set_n == 't':
        temp_b, angys_b = temp_t, angys_t
        t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
        hints_b = hints_t
        gt,gy,gx = np.gradient( nangauss(hints_t[n], 2), t,y,x)
        xs,ys = xns_t[n], yns_t[n]
        drho = drho_rho_t
    
    fil = ~np.isnan(gt)  #(gt!=0)
    # gtn = np.copy(gt)
    # gtn[~fil] = np.nan
    
    ymin = np.array( [ np.where(fil[i])[0][-1] for i in range(len(hints_b[n]))  ])
    
    gtx = np.nanmean(gt, axis=2)
    rgtx = np.empty_like(gtx)
    for i in range(len(gtx)): rgtx[i] = np.roll(gtx[i], len(y)-ymin[i]-1)

    Nu = -rgtx/1000 * rho_ice * latent * (y-y[-1])/1000 / (thcon * temp_b[n] )
    Ra = g * np.cos(angys_b[n] * np.pi/180)**3 * drho[n] * ((y-y[-1])/1000)**3 / kt / mu
    
    Num = np.nanmedian( Nu[10:], axis=0 )

    # for i in [0,10,20,30,40,50,60]:
    #     plt.plot(Ra, Nu[i] / Ra**(compensate), '.-' , label=i)
    # plt.plot(Ra[1:], Num[1:] / Ra[1:]**(compensate), '-' , label=n)
    if set_n == 'v':
        ax[r'$b)$'].plot(Ra[:-1], Num[:-1] / Ra[:-1]**(compensate), '-' , label= r'$S={:.1f}$, $\theta={:.1f}°$'.format(salis_v[n], angys_v[n]),
                 color = np.array([0.5 , (salis_v[n]-19)/16 , 1-(salis_v[n]-19)/16 ]) )
    elif set_n == 't':
        ax[r'$b)$'].plot(Ra[:-1], Num[:-1] / Ra[:-1]**(compensate), '--' , label= r'$S={:.1f}$, $\theta={:.1f}°$'.format(salis_t[n], angys_t[n]),
                 color = np.array([0.5 , (salis_t[n]-19)/16 , 1-(salis_t[n]-19)/16 ]) )
    
    # rgm = np.nanmedian( -rgtx[10:], axis=0 )
    # plt.plot(y-y[-1], rgm, label=n)
 
# plt.plot( Ra[:-1], Ra[:-1]**(1/4 - compensate) * 2.0 , 'k--' )
# plt.plot( Ra[:-1], Ra[:-1]**(1/3 - compensate) * 0.4 , 'k--' )

ax[r'$b)$'].set_xscale('log')
ax[r'$b)$'].set_yscale('log')

ax[r'$b)$'].set_yticks([0.2,0.4,0.6])
ax[r'$b)$'].get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax[r'$b)$'].set_yticklabels([], minor=True)

ax[r'$b)$'].set_xlabel(r'$\textrm{Ra(y)}$')
ax[r'$b)$'].set_ylabel(r'$\textrm{Nu(y)} / \textrm{Ra(y)}^{1/3}$')

ax[r'$b)$'].legend()

for j,(labels,axs) in enumerate(ax.items()):
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman','fontsize':12})

plt.savefig('./Documents/Figs morpho draft/scaling_nuy_ray.pdf',dpi=400, bbox_inches='tight')

plt.show()

#%%

np.random.seed(10)
b = np.sort( np.random.random(100) * 5e7 )
a = np.sort( np.random.rand(100) * 0.6 + 0.12 )


fig,ax = plt.subplots()
ax.plot( b, a, '--' )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks( np.array([0.2,0.4,0.6]) ) #, labels=[0.2,0.4,0.6], minor=False )
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_yticklabels([], minor=True)
# ax.ticklabel_format(axis='y', style='plain')  # Prevent scientific notation

plt.show()

#%%
# with time
compensate = 0

plt.figure()
for n, set_n in zip([4,5,6,0,2,29,31],['v','v','v','t','t','t','t']):     
# for n, set_n in zip([0],['v']):     
    if set_n == 'v':
        temp_b, angys_b = temp_v, angys_v
        t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
        hints_b = hints_v
        gt,gy,gx = np.gradient(hints_v[n], t,y,x)
        xs,ys = xns_v[n], yns_v[n]
        drho = drho_rho_v
    elif set_n == 't':
        temp_b, angys_b = temp_t, angys_t
        t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
        hints_b = hints_t
        gt,gy,gx = np.gradient( nangauss(hints_t[n], 2), t,y,x)
        xs,ys = xns_t[n], yns_t[n]
        drho = drho_rho_t

    fil = ~np.isnan(gt)  #(gt!=0)

    ymax = np.array( [ np.where(fil[i])[0][0] for i in range(len(hints_b[n]))  ])
    ymin = np.array( [ np.where(fil[i])[0][-1] for i in range(len(hints_b[n]))  ])
    # H = 
    
    area = np.trapezoid( np.trapezoid(~np.isnan(hints_b[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    gt[~fil] = 0.0
    meltr = np.trapezoid( np.trapezoid( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1) / area

    Nu = -meltr/1000 * rho_ice * latent * (y[ymax]-y[ymin])/1000 / (thcon * temp_b[n] )
    Ra = g * np.cos(angys_b[n] * np.pi/180) * drho[n] * ((y[ymax]-y[ymin])/1000)**3 / kt / mu
    
    plt.plot(Ra, Nu / Ra**(compensate), '.-' , label=n)
 
plt.plot( Ra[1:], Ra[1:]**(1/4 - compensate) * 1.0 , 'k--' )
plt.plot( Ra[:], Ra[:]**(1/3 - compensate) * 0.6 , 'k--' )
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$\textrm{Ra}$')
plt.ylabel(r'$\textrm{Nu} / \textrm{Ra}^{1/4}$')

plt.legend()
plt.show()

compensate = 0

plt.figure()
for n, set_n in zip([14,14,16,26,30],['v','t','t','t','t', 'v','v']):     
# for n, set_n in zip([0],['v']):     
    if set_n == 'v':
        temp_b, angys_b = temp_v, angys_v
        t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
        hints_b = hints_v
        gt,gy,gx = np.gradient(hints_v[n], t,y,x)
        xs,ys = xns_v[n], yns_v[n]
        drho = drho_rho_v
    elif set_n == 't':
        temp_b, angys_b = temp_t, angys_t
        t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
        hints_b = hints_t
        gt,gy,gx = np.gradient( nangauss(hints_t[n], 2), t,y,x)
        xs,ys = xns_t[n], yns_t[n]
        drho = drho_rho_t

    fil = ~np.isnan(gt)  #(gt!=0)

    ymax = np.array( [ np.where(fil[i])[0][0] for i in range(len(hints_b[n]))  ])
    ymin = np.array( [ np.where(fil[i])[0][-1] for i in range(len(hints_b[n]))  ])
    # H = 
    
    area = np.trapezoid( np.trapezoid(~np.isnan(hints_b[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    gt[~fil] = 0.0
    meltr = np.trapezoid( np.trapezoid( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1) / area

    Nu = -meltr/1000 * rho_ice * latent * (y[ymax]-y[ymin])/1000 / (thcon * temp_b[n] )
    Ra = g * np.cos(angys_b[n] * np.pi/180) * drho[n] * ((y[ymax]-y[ymin])/1000)**3 / kt / mu
    
    plt.plot(Ra, Nu / Ra**(compensate), '.-' , label=n)
 
plt.plot( Ra[:], Ra[:]**(1/4 - compensate) * 1.0 , 'k--' )
plt.plot( Ra[:], Ra[:]**(1/3 - compensate) * 0.6 , 'k--' )
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$\textrm{Ra}$')
plt.ylabel(r'$\textrm{Nu} / \textrm{Ra}^{1/4}$')

plt.legend()
plt.show()

#%%
n = 14
temp_b, angys_b = temp_v, angys_v
t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
hints_b = hints_v
gt,gy,gx = np.gradient(hints_v[n], t,y,x)
xs,ys = xns_v[n], yns_v[n]
drho = drho_rho_v

fil = ~np.isnan(gt)  #(gt!=0)
ymin = np.array( [ np.where(fil[i])[0][-1] for i in range(len(hints_b[n]))  ])
    
gtx = np.nanmean(gt, axis=2)
rgtx = np.empty_like(gtx)
for i in range(len(gtx)): rgtx[i] = np.roll(gtx[i], len(y)-ymin[i]-1)

plt.figure()
# plt.plot(gtx[10])
# plt.plot(gtx[20])
# plt.plot(rgtx[10],'--')
# plt.plot(rgtx[20],'--')
# plt.plot(rgtx[30],'--')
# plt.plot(rgtx[40],'--')
plt.show()


#%%
# =============================================================================
# Tracking maxima (to see scallops velocity)
# =============================================================================
rrho_v = np.abs(density_millero(temp_v, 0) - density_millero(temp_v, salis_v)) / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v))
rrho_t = np.abs(density_millero(temp_t, 0) - density_millero(temp_t, salis_t)) / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t))

myss_v, mxss_v, ttss_v = [],[],[]
for n,ss in tqdm(enumerate(salis_v)):
# for n in [0]:

    dife = np.copy( difs_v[n] )
    # dife = np.copy( hints_v[n][i] )
    dife[np.isnan(dife)] = -1000

    mys, mxs, tts = [],[],[]
    for i in range(len(dife)):
        
        my,mx = peak_local_max( dife[i], labels=~np.isnan(hints_v[n][i]), min_distance=18 ).T
        # my,mx = np.where( maximum_filter(-iceh[i], footprint=disk(30)) == -iceh[i] )
        mys = mys + list((yns_v[n][:,0])[my])
        mxs = mxs + list((xns_v[n][0,:])[mx])
        tts = tts + [ts_v[n][i]/60] * len(my)

    myss_v.append(mys)
    mxss_v.append(mxs)
    ttss_v.append(tts)
    

myss_t, mxss_t, ttss_t = [],[],[]
for n,ss in tqdm(enumerate(salis_t)):
# for n in [0]:

    dife = np.copy( difs_t[n] )
    # dife = np.copy( hints_t[n][i] )
    dife[np.isnan(dife)] = -1000

    mys, mxs, tts = [],[],[]
    for i in range(len(dife)):
        
        my,mx = peak_local_max( dife[i], labels=~np.isnan(hints_t[n][i]), min_distance=18 ).T
        # my,mx = np.where( maximum_filter(-iceh[i], footprint=disk(30)) == -iceh[i] )
        mys = mys + list((yns_t[n][:,0])[my])
        mxs = mxs + list((xns_t[n][0,:])[mx])
        tts = tts + [ts_t[n][i]/60] * len(my)

    myss_t.append(mys)
    mxss_t.append(mxs)
    ttss_t.append(tts)    
    
t1 = time()
mve_v, msd_v = [],[]
mxe_v, mxd_v = [],[]


for i,ss in enumerate(salis_v):
# for i in [0]:
    ss = salis[i]
    
    mxs,mys,tts = mxss_v[i], myss_v[i], ttss_v[i]
    
    colos = [ [i,1-i ,0]  for i in np.linspace(0.1,1.0,25) ]
    
    mxs,mys,tts = np.array(mxs), np.array(mys), np.array(tts)
    
    # plt.figure()
    # plt.imshow( hints_v[i][30], extent=(np.min(xns_v[i]), np.max(xns_v[i]), np.min(yns_v[i]), np.max(yns_v[i]) )  )
    # plt.scatter(  mxs[tts==30], mys[tts==30], s=2, c='k')
    # plt.show()
    
    
    mint = 12
    # fig,ax = plt.subplots()
    # ppl = ax.scatter(mxs[tts>mint],mys[tts>mint], c=tts[tts>mint], cmap='viridis')
    
    # ax.set_aspect('equal')
    # ax.invert_yaxis()
    # ax.set_xlim([-53,53])
    # ax.set_ylim([170,-69])
    
    # fig.colorbar(ppl,ax=ax,label='time (min)')
    
    # plt.gca().invert_yaxis()
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    # plt.grid()
    # plt.show()
    
    
    trax,tray,tiemps = track2d(mxs[tts>mint],mys[tts>mint],tts[tts>mint], delim=4, dtm=1)
    
    trxl, tryl, tiel = [],[],[]
    for i in range(len(trax)):
        if len(trax[i]) > 25: 
            trxl.append(trax[i])
            tryl.append(tray[i])
            tiel.append(tiemps[i])
    del trxl[1], tryl[1], tiel[1]
    
    # fig,ax = plt.subplots()
    # for i in range(len(trxl)):
    #     ppil = ax.plot(trxl[i],tryl[i],'.-', label=i)
    
    # ax.set_aspect('equal')
    # ax.invert_yaxis()
    # # ax.set_xlim([-53,53])
    # # ax.set_ylim([170,-69])
    # ax.legend( loc='upper center', bbox_to_anchor=(1.25, 0.8), ncol=1 )#, frame=False)
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    
    # plt.gca().invert_yaxis()
    # plt.grid()
    # plt.show()
    
    slops, eslo = [],[]
    slpx,eslx = [],[]
    # plt.figure()
    for i in range(len(trxl)):
        lnry = linregress(tiel[i], tryl[i])
        slopy, ely = lnry[0], lnry[4]
        
        lnrx = linregress(tiel[i], trxl[i])
        slopx, elx = lnrx[0], lnrx[4]
    
        # plt.plot(tiel[i] , tryl[i] , '.-', label=i)
        # plt.plot(tiel[i] , trxl[i] , '.-', label=i)
        
        slops.append(slopy)
        eslo.append(ely)
        slpx.append(slopx)
        eslx.append(elx)
        
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    # fig,ax = plt.subplots(2,1, sharex=True)
    # ax[0].errorbar(np.arange(len(slops)),slops, yerr=eslo, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    # ax[1].errorbar(np.arange(len(slpx)),slpx, yerr=eslx, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    # ax[0].grid()
    # ax[1].grid()
    # ax[0].set_ylabel('vel y (mm/min)', fontsize=12)
    # ax[1].set_ylabel('vel x (mm/min)', fontsize=12)
    # ax[1].set_xlabel('Scallop', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.savefig('./Documents/vels_'+ str(ss) +'.png',dpi=400, bbox_inches='tight')
    # plt.show()
    
    # plt.figure()
    # plt.errorbar(slpx, slops, xerr=eslx, yerr=eslo, fmt='.')
    # plt.grid()
    # plt.show()
    
    # print(ss, np.mean(slops), np.median(slops), np.std(slops))
    mve_v.append( np.mean(slops) )
    msd_v.append( np.std(slops) )
    
    mxe_v.append( np.mean(slpx) )
    mxd_v.append( np.std(slpx) )


mve_t, msd_t = [],[]
mxe_t, mxd_t = [],[]
# for n,ss in enumerate(salis_t):
# for n,j in enumerate(range(len(salis_t))):
# for n,j in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
for n,j in enumerate([4,5,6,7,8,9,15,23,27,28,32,33,34,35,38]):
# for n,ss in enumerate([7,8,15]):
# for n,ss in enumerate([7,6,5]):
# for n in [0]:
    ss = salis_t[j]
    
    mxs,mys,tts = mxss_t[j], myss_t[j], ttss_t[j]
    
    colos = [ [i,1-i ,0]  for i in np.linspace(0.1,1.0,25) ]
    
    mxs,mys,tts = np.array(mxs), np.array(mys), np.array(tts)
    
    # plt.figure()
    # plt.imshow( hints_t[i][30], extent=(np.min(xns_t[i]), np.max(xns_t[i]), np.min(yns_t[i]), np.max(yns_t[i]) )  )
    # plt.scatter(  mxs[tts==30], mys[tts==30], s=2, c='k')
    # plt.show()
    
    
    mint = 12
    # fig,ax = plt.subplots()
    # ppl = ax.scatter(mxs[tts>mint],mys[tts>mint], c=tts[tts>mint], cmap='viridis')
    
    # ax.set_aspect('equal')
    # ax.invert_yaxis()
    # ax.set_xlim([-53,53])
    # ax.set_ylim([170,-69])
    
    # fig.colorbar(ppl,ax=ax,label='time (min)')
    
    # plt.gca().invert_yaxis()
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    # plt.grid()
    # plt.show()
    
    
    trax,tray,tiemps = track2d(mxs[tts>mint],mys[tts>mint],tts[tts>mint], delim=4, dtm=1)
    
    trxl, tryl, tiel = [],[],[]
    for i in range(len(trax)):
        if len(trax[i]) > 25: 
            trxl.append(trax[i])
            tryl.append(tray[i])
            tiel.append(tiemps[i])
    del trxl[1], tryl[1], tiel[1]
    
    # fig,ax = plt.subplots()
    # for i in range(len(trxl)):
    #     ppil = ax.plot(trxl[i],tryl[i],'.-', label=i)
    
    # ax.set_aspect('equal')
    # ax.invert_yaxis()
    # # ax.set_xlim([-53,53])
    # # ax.set_ylim([170,-69])
    # ax.legend( loc='upper center', bbox_to_anchor=(1.25, 0.8), ncol=1 )#, frame=False)
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    
    # plt.gca().invert_yaxis()
    # plt.grid()
    # plt.show()
    
    slops, eslo = [],[]
    slpx,eslx = [],[]
    # plt.figure()
    for i in range(len(trxl)):
        lnry = linregress(tiel[i], tryl[i])
        slopy, ely = lnry[0], lnry[4]
        
        lnrx = linregress(tiel[i], trxl[i])
        slopx, elx = lnrx[0], lnrx[4]
    
        # plt.plot(tiel[i] , tryl[i] , '.-', label=i)
        # plt.plot(tiel[i] , trxl[i] , '.-', label=i)
        
        slops.append(slopy)
        eslo.append(ely)
        slpx.append(slopx)
        eslx.append(elx)
        
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    
    # fig,ax = plt.subplots(2,1, sharex=True)
    # ax[0].errorbar(np.arange(len(slops)),slops, yerr=eslo, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    # ax[1].errorbar(np.arange(len(slpx)),slpx, yerr=eslx, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    # ax[0].grid()
    # ax[1].grid()
    # ax[0].set_ylabel('vel y (mm/min)', fontsize=12)
    # ax[1].set_ylabel('vel x (mm/min)', fontsize=12)
    # ax[1].set_xlabel('Scallop', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.savefig('./Documents/vels_'+ str(ss) +'.png',dpi=400, bbox_inches='tight')
    # plt.show()
    
    # plt.figure()
    # plt.errorbar(slpx, slops, xerr=eslx, yerr=eslo, fmt='.')
    # plt.grid()
    # plt.show()
    
    # print(ss, np.mean(slops), np.median(slops), np.std(slops))
    mve_t.append( np.mean(slops) )
    msd_t.append( np.std(slops) )
    
    mxe_t.append( np.mean(slpx) )
    mxd_t.append( np.std(slpx) )

t2 = time()
print(t2-t1)
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size':25
})

graph_velx = False
graph_peaks = [False, 7, 41]
    
mve_v, msd_v = np.array(mve_v), np.array(msd_v)
mxe_v, mxd_v = np.array(mxe_v), np.array(mxd_v)

mve_t, msd_t = np.array(mve_t), np.array(msd_t)
mxe_t, mxd_t = np.array(mxe_t), np.array(mxd_t)

filv = (salis_v > 6) * (salis_v < 25)
# fil = salis_v<50
filt = [5,6,7,8,9,15,23,27,28,32,33,35,38] #[7,6,5,28,23,8,9,15]

cols = np.linspace(-20,51,256)
comap = np.array( [(cols+20)/71 , 0.5 *np.ones_like(cols) , 1-(cols+20)/71 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplots(1,1,layout='constrained')

li1ys = []
for i,j in enumerate(range(len(ds_v))):
    # if salis_v[j] > 7 and salis_v[j] < 25:
    if filv[j]:
        # li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=msd_v[j], fmt='o', capsize=2, \
        #               color=((angys_v[j]+20)/71,0.5,1-(angys_v[j]+20)/71), markersize=5, mfc='w')
        li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=0.097, fmt='^', capsize=2, \
                  color=((angys_v[j]+20)/71,0.5,1-(angys_v[j]+20)/71)) #, markersize=5 )#, mfc='w')
        li1ys.append(li1y)

li2ys = []
for i,j in enumerate(filt):
    # li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=msd_t[i], fmt='o', capsize=2., \
    #               color=((angys_t[j]+17)/47,0.5,1-(angys_t[j]+17)/47), markersize=5)
    if j in [32,33,35]:
        li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='d', capsize=2., \
                      color=((angys_t[j]+20)/71,0.5,1-(angys_t[j]+20)/71)) #, markersize=5)
    else:
        li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='o', capsize=2., \
                      color=((angys_t[j]+20)/71,0.5,1-(angys_t[j]+20)/71)) #, markersize=5)
    li2ys.append(li2y)

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

# ax.legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                    bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5 )


# ax.grid()
ax.set_xlabel(r'$S$ (g/kg)')
ax.set_ylabel(r'$v_y$ (mm/s)')
ax.set_xticks([5,10,15,20,25])
ax.set_ylim(top=0)
# plt.savefig('./Documents/Figs morpho draft/poster_all_vely.png',dpi=500, bbox_inches='tight', transparent=True)
plt.show()

if graph_velx:
    plt.figure()
    
    plt.errorbar(salis_v[filv], mxe_v[filv], yerr=mxd_v[filv], fmt='o', capsize=2, \
                  color=((angys_v[i]+20)/71,0.5,1-(angys_v[i]+20)/71), markersize=5, mfc='w')
    
    for i,j in enumerate(filt):
        plt.errorbar(salis_t[j], mxe_t[i], yerr=mxd_t[i], fmt='o', capsize=2., \
                      color=((angys_t[j]+20)/71,0.5,1-(angys_t[j]+20)/71), markersize=5)
    
    plt.grid()
    plt.xlabel('S (g/kg)')
    plt.ylabel(r'$v_x$ (mm/s)')
    # plt.savefig('./Documents/velsvsal_min.png',dpi=400, bbox_inches='tight')
    plt.show()

if graph_peaks[0]:
    n = graph_peaks[1]
    i = graph_peaks[2]
    dife = np.copy( difs_t[n] )
    # dife = np.copy( hints_t[n][i] )
    dife[np.isnan(dife)] = -1000
    
    my,mx = peak_local_max( dife[i], labels=~np.isnan(hints_t[n][i]), min_distance=18 ).T
    
    
    plt.figure()
    plt.imshow( difs_t[n][i] )
    plt.plot(mx,my,'k.')
    plt.show()


#%%
plt.rcParams.update({'font.size':16})
mfc = None

xparam = 'rrho' # 'sal' or 'rrho'

mve_v, msd_v = np.array(mve_v), np.array(msd_v)
mxe_v, mxd_v = np.array(mxe_v), np.array(mxd_v)

mve_t, msd_t = np.array(mve_t), np.array(msd_t)
mxe_t, mxd_t = np.array(mxe_t), np.array(mxd_t)

filv = (salis_v > 6) * (salis_v < 25)
# filt = [5,6,7,8,9,15,23,27,28,32,33,35,38, 4,34] #[7,6,5,28,23,8,9,15]
filt = [4,5,6,7,8,9,15,23,27,28,32,33,34,35,38] #[7,6,5,28,23,8,9,15]

cols = np.linspace(-20,51,256)
comap = np.array( [(cols+20)/71 , 0.5 *np.ones_like(cols) , 1-(cols+20)/71 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

# fig, ax = plt.subplots()
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$b)$']], layout='constrained', figsize=(12,5) , sharex=False)

li1ys = []
for i,j in enumerate(range(len(ds_v))):
    if filv[j]:
        
        if   xparam == 'sal' : xvalues = salis_v[j]
        elif xparam == 'rrho': xvalues = rrho_v[j]
    
        li1y = ax[r'$a)$'].errorbar(xvalues, mve_v[j], yerr=0.097, fmt='^', capsize=2, \
                  color=((angys_v[j]+20)/71,0.5,1-(angys_v[j]+20)/71), markersize=6, mfc=mfc)
        li1ys.append(li1y)

li2ys = []
for i,j in enumerate(filt):
    
    if   xparam == 'sal' : xvalues = salis_t[j]
    elif xparam == 'rrho': xvalues = rrho_t[j]
    
    if j in [32,33,35]:
        li2y = ax[r'$a)$'].errorbar(xvalues, mve_t[i], yerr=0.097, fmt='d', capsize=2., \
                      color=((angys_t[j]+17)/62,0.5,1-(angys_t[j]+17)/62), markersize=6, mfc=mfc)
    else:
        li2y = ax[r'$a)$'].errorbar(xvalues, mve_t[i], yerr=0.097, fmt='o', capsize=2., \
                      color=((angys_t[j]+17)/71,0.5,1-(angys_t[j]+17)/62), markersize=5, mfc=mfc)
    li2ys.append(li2y)

if xparam == 'sal' :
    ax[r'$a)$'].set_xlim(0,25)
    ax[r'$a)$'].set_xlabel(r'$S$ (g/kg)')
if xparam == 'rrho' :
    ax[r'$a)$'].set_xlim(1,8)
    ax[r'$a)$'].set_xlabel(r'$R_\rho$')

ax[r'$a)$'].set_ylabel(r'$v_y$ (mm/s)')
ax[r'$a)$'].set_ylim(-1,0)

# ax[r'$a)$'].legend([li1ys[7],li2ys[2],li2ys[1],li2ys[0]],[r'-$15°$',r'$0°$',r'$15°$',r'$30°$'],\
#                    bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5 )
# colores = [ ((angulo+20)/71,0.5,1-(angulo+20)/71) for angulo in [-15,0,15,30] ]
colores = [ ((angulo+17)/62,0.5,1-(angulo+17)/62) for angulo in [-15,0,15,30,45] ]
leg1 = [ax[r'$a)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$a)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]
lgd1 = ax[r'$a)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.17,1.03], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
lgd2 = ax[r'$a)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.34,1.03], frameon=False, ncols=5, columnspacing=0.35, handletextpad=0.05)
ax[r'$a)$'].add_artist(lgd1)

# ax.grid()



n = 10 #8 #15 #23
i = 60

exp = 'v'
if exp == 'v':
    hints_b = hints_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
elif exp == 't': 
    hints_b = hints_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t

halg = nangauss(hints_b[n],5)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])


ax1 = ax[r'$b)$']
ax1.set_ylim(-160,120)

color = 'tab:red'
ax1.set_xlabel(r'$h$ (mm)', color=color)
ax1.set_ylabel(r'$y$ (mm)')
ax1.set_xlim(-2.5,10)
ax1.set_xticks(np.arange(-2.5,11,2.5))
ax1.plot( hints_b[n][i][:,200], yns_b[n][:,0], '-', color=color )
ax1.tick_params(axis='x', labelcolor=color)

ax2 = ax1.twiny()
color = 'tab:blue'
ax2.set_xlabel(r'$-\partial h/\partial t$ (mm/min)', color=color)
ax2.set_xlim(0.4,0.9)
ax2.set_xticks(np.arange(0.4,0.95,.1))
ax2.plot( -gt[i][:,200], yns_b[n][:,0], '-', color=color )
ax2.tick_params(axis='x', labelcolor=color)

for labels,axs in ax.items():
    if labels == r'$a)$':
        axs.annotate(labels, (-0.17,0.98), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
    else:
        axs.annotate(labels, (-0.3,0.98), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# ax.set_ylim(top=0)
# plt.savefig('./Documents/Figs morpho draft/rrho_vely(2).pdf',dpi=400, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Curvature graphs
# =============================================================================
i = 80
exp = 't'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t


fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(11.5,5) , sharex=False)

ns = [1,7,8]

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    
    gghi = nangauss(hints_b[n],5)
    gt,gy,gx = np.gradient( gghi , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    
    # barme, barlen = np.nanmean(-gt[i]), np.nanstd(-gt[i]) * 4
    # barme, barlen = np.nanmean(gy[i]), np.nanstd(-gy[i]) * 4
    barme, barlen = np.nanmean(gx[i]), np.nanstd(gx[i]) * 4
    print(barme, barlen)

    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    # imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), aspect= dy/dx )
    # imhe = axs.imshow( gghi[i], extent=(minx,maxx,miny,maxy), aspect= dy/dx )
    # imhe = axs.imshow( -gt[i], extent=(minx,maxx,miny,maxy), aspect= dy/dx, vmin=barme-barlen, vmax=barme+barlen )
    # imhe = axs.imshow( gy[i], extent=(minx,maxx,miny,maxy), aspect= dy/dx, vmin=barme-barlen, vmax=barme+barlen )
    # imhe = axs.imshow( gx[i], extent=(minx,maxx,miny,maxy), aspect= dy/dx, vmin=barme-barlen, vmax=barme+barlen )
    imhe = axs.imshow( kurv[i], extent=(minx,maxx,miny,maxy), aspect= dy/dx) #, vmin=barme-barlen, vmax=barme+barlen )
    
    topy,boty = np.max( (yns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(hints_b[n][i])] )
    topx,botx = np.max( (xns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(hints_b[n][i])] )
    midx = np.mean( (xns_b[n])[~np.isnan(hints_b[n][i])] )

    fig.colorbar(imhe, label=r'$\dot{m}$ (mm/min)', location='right', shrink=0.9) 

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/profiles_s1.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


#%%
# =============================================================================
# Showing of height profiles
# =============================================================================
plt.rcParams.update({'font.size':14})

i = 60
exp = 'v'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
    ns = [4,10,14]
    titles = ['Temperature-driven','Competing','Salinity-driven']
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t
    ns = [1,5,11]
    titles = ['Temperature-driven','Competing','Competing']

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(12,5) , sharex=False)

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    imhe = axs.imshow( hints_b[n][i] - np.nanmean(hints_b[n][i]) , extent=(minx,maxx,miny,maxy), aspect= dy/dx, vmin=-10, vmax=8 )
    
    topy,boty = np.max( (yns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(hints_b[n][i])] )
    topx,botx = np.max( (xns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(hints_b[n][i])] )
    midx = np.mean( (xns_b[n])[~np.isnan(hints_b[n][i])] )

    # if j == 0: fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9, ticks=list(range(-20,-42,-2)))
    # else: fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9) 
    cbar = fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9 )
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(x)}"))

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    axs.set_title(titles[j]) #, fontsize=12)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman','fontsize':15})

# plt.savefig('./Documents/Figs morpho draft/profiles_vert.pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
i = 60
exp = 't'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(12,5) , sharex=False)

ns = [17,18,24] #[36,37,38]
# ns = [0,4,5]

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    imhe = axs.imshow( hints_b[n][i] - np.nanmean(hints_b[n][0]), extent=(minx,maxx,miny,maxy), aspect= dy/dx )
    
    topy,boty = np.max( (yns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(hints_b[n][i])] )
    topx,botx = np.max( (xns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(hints_b[n][i])] )
    midx = np.mean( (xns_b[n])[~np.isnan(hints_b[n][i])] )

    fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9) 

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/profiles_s1.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%

ff =  (20<salis_t) & (salis_t< 30)

salis_t[ff], rrho_t[ff], angys_t[ff], np.where(ff)[0]

#%%
sa_ar, te_ar = np.meshgrid( np.linspace(0,salis_v[n],1000), np.linspace(0,temp_v[n],1000) )
dens_ar = density_millero(te_ar, sa_ar)

np.max(dens_ar), np.min(dens_ar), density_millero(temp_v[n], salis_v[n])

#%%
exp = 't'
gau = True

# ns = [1,1,1] #,10,18]
# ies = [20,40,60]

ns = [1,5,11] #,25] # 7,8]
ies = [60,60,60]

if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    if gau: 
        hints_b = [ np.zeros_like( hints_t[j] ) for j in range(len(hints_t)) ]
        for n in np.unique(ns):            
            hints_b[n] = nangauss(hints_t[n], 2) 
    else: hints_b = hints_t


fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(10,4.15) , sharex=False)

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    i = ies[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    if j ==0: imhe = axs.imshow( hints_b[n][i] - np.nanmean(hints_b[n][0]), extent=(minx,maxx,miny,maxy), vmax=-30, vmin=-45, aspect= dy/dx )
    # elif j == 1: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx )
    # elif j == 2: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx  )
    # elif j == 2: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=-85, vmin=-105  )
    else: imhe = axs.imshow( hints_b[n][i] - np.nanmean(hints_b[n][0]), extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx )
    # imhe = axs.imshow( hints_b[n][i] - np.nanmean(hints_b[n][0]), extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx )
    
    topy,boty = np.max( (yns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(hints_b[n][i])] )
    topx,botx = np.max( (xns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(hints_b[n][i])] )
    midx = np.mean( (xns_b[n])[~np.isnan(hints_b[n][i])] )

    # cbar_ax = fig.add_axes([0.18, 0.18, 0.015, 0.65])
    # if j == 0: fig.colorbar(imhe, label='height (mm)', location='right', shrink=0.9, ticks=list(range(-41,-61,-3))) #, cax=cbar_ax)
    # else: fig.colorbar(imhe, label='height (mm)', location='right', shrink=0.9) #, cax=cbar_ax)
    fig.colorbar(imhe, label=r'$h$ (mm)', location='right', shrink=0.7) #, cax=cbar_ax)

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.tight_layout()
# plt.savefig('./Documents/Figs morpho draft/profiles_tt.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
# =============================================================================
# Attempt dividing things numerically
# =============================================================================

def kurvature(halg, hints_b, yns_b, xns_b):
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2)     
    
    return kurv

def countour_mean(wtssal_b, difs_b,i,labe, all_wat=True):
    if all_wat:
        sccaa = wtssal_b[i] == labe
    else:
        sccaa = wtssal_b == labe
    ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )
    conval = difs_b[i][ccoo]

    # if ve:    
    #     if dif: conval = difs_v[n][i][ccoo]
    #     else: conval = hints_v[n][i][ccoo]
    # else:    
    #     if dif: conval = difs_t[n][i][ccoo]
    #     else: conval = hints_t[n][i][ccoo]
    return np.nanmean(conval)

def normalize(array):
    return ( array - np.nanmin(array) ) / (np.nanmax(array) - np.nanmin(array))
def image_nanstdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanstd(intensities[region]) #, ddof=0)
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region]) #, ddof=0)
def image_minmax(region, intensities):
    return np.nanmax(intensities[region]) - np.nanmin(intensities[region])  
def image_min(region, intensities):
    return np.nanmin(intensities[region])
#%%
exp = 't'
gau = True
eros = True

sw_t,sw_v = [],[]
v1_t,v2_t,v3_t,v4_t,v5_t = [],[],[],[],[]
v1_v,v2_v,v3_v,v4_v,v5_v = [],[],[],[],[]

# s1,s2,s3 = 7,10,15
s1,s2,s3 = 6,8,17
# for n in [8,9,10]:
# for nn in range(39):
for nn in range(55):
    
    t1 = time()
    
    if nn < 16: 
        n = nn
        exp='v'
    else: 
        n = nn - 16
        exp='t'
    
    if exp == 'v': 
        xns_b, yns_b = xns_v[n], yns_v[n]
        # hints_b = hints_v[n]
        # halg = nangauss(hints_v[n],5)
        hints_b = difs_v[n]
        halg = nangauss(difs_v[n],s1)
            
    elif exp == 't':
        xns_b, yns_b = xns_t[n], yns_t[n]
        # hints_b = hints_t[n]
        # halg = nangauss(hints_t[n],5)
        hints_b = difs_t[n]
        halg = nangauss(difs_t[n],s1)
        
    
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_b))*0.5, yns_b[:,0], xns_b[0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2)     
    
    flp, fln = kurv>0, kurv<0 
    if nn < 16: im_wat = nangauss( kurv * flp, [0,s2,s3] ) + (kurv * fln)
    else: im_wat = nangauss( kurv * flp, [0,s2,s3] ) + (kurv * fln)
    
    # kurv_smo = kurvature( nangauss(hints_b, 10) , hints_b, yns_b, xns_b) 
    kurv_smo = kurv
        
    sws, scaprop, scapropk = [],[],[]
    # for i in tqdm(range(len(hints_b))):
    for i in tqdm(range(len(hints_b)-35, len(hints_b))):
        # wts = watershed( -im_wat[i], mask = binary_erosion( ~np.isnan(kurv[i]), disk(10) ) )
        wts = watershed( -im_wat[i], mask = binary_dilation( ~np.isnan(kurv[i]), disk(1) ) )
    
        sws.append(wts)
        scaprop.append( regionprops(wts, intensity_image= gaussian(hints_b[i],sigma=1) , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        scapropk.append( regionprops(wts, intensity_image= kurv_smo[i] , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
    val1,val2,val3,val4,val5 = [],[],[],[],[]
    # for i in range(len(hints_b)):
    for i in range(len(sws)):
        val1.append( np.max(sws[i]) )    
        
        par = [ scapropk[i][l].area for l in range(len(scapropk[i]))]
        val2.append( np.nanmedian(par) )
        
        par = [ scapropk[i][l].image_nanstdev for l in range(len(scapropk[i]))]
        val3.append( np.nanmedian(par) )

        # par = [ countour_mean(sws,hints_b,i,j+1) - scaprop[i][j].image_min for j in range(len(scaprop[i]))]
        par = [ scaprop[i][l].image_nanstdev  for l in range(len(scaprop[i]))]
        val4.append( np.nanmedian(par) )

        par = [ (scapropk[i][l].image_stdev  / scapropk[i][l].area) for l in range(len(scapropk[i]))]
        val5.append( np.nanmedian(par) )
                
    if nn < 16:
        v1_v.append(val1)
        v2_v.append(val2)
        v3_v.append(val3)
        v4_v.append(val4)
        v5_v.append(val5)
        sw_v.append(sws)
    else:
        v1_t.append(val1)
        v2_t.append(val2)
        v3_t.append(val3)
        v4_t.append(val4)
        v5_t.append(val5)
        sw_t.append(sws)

    t2 = time()
    
    # ss = np.nanstd(kurv, axis=(1,2))
    print(f'{nn}/54', t2-t1) #, salis_t[n], rrho_t[n], angys_t[n] )

    # sca.append(scaprop)
    # ska.append(scapropk)
#%%
n = 7
i = -20
# print( salis_t[n], rrho_t[n], angys_t[n] )
# print( salis_v[n], rrho_v[n], angys_v[n] )

showplot = 0
calc = 0

if showplot:     
    s1 = 7
    s2 = 10
    s3 = 15
    # kurv = kurvature( nangauss(difs_t[n], s1) , difs_t[n], yns_t[n], xns_t[n])
    kurv = kurvature( nangauss(difs_v[n], s1) , difs_v[n], yns_v[n], xns_v[n])
    # kurv = nangauss( kurv, [0,5,5] )
        
    if calc:
        flp, fln = kurv>0, kurv<0 
        im_wat = nangauss( kurv * flp, [0,s2,s3] ) + (kurv * fln)
        wts = watershed( -im_wat[i], mask = binary_dilation( ~np.isnan(kurv[i]), disk(1) ) )
    
        # kurvs = kurvature( nangauss(difs_t[n], 10) , difs_t[n], yns_t[n], xns_t[n])
        kurvs = kurvature( nangauss(difs_v[n], 10) , difs_v[n], yns_v[n], xns_v[n])
    
        reg = regionprops(wts, intensity_image= kurvs[i] , extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min])
        # reg = regionprops(wts, intensity_image= difs_t[n][i] , extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min])
        # reg = regionprops(wts, intensity_image= difs_v[n][i] , extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min])
    
        nstd = [ reg[l].image_nanstdev for l in range(len(reg))]
        ncont = [ countour_mean(wts,kurv,i,j+1,False) - reg[j].image_min for j in range(len(reg)) ]
        # ncontd = [ countour_mean(wts,difs_t[n],i,j+1,False) - reg[j].image_min for j in range(len(reg)) ]
        
        nar = [ reg[l].area for l in range(len(reg))]
        
        print( salis_t[n], rrho_t[n], angys_t[n] )
        # print( salis_v[n], rrho_v[n], angys_v[n] )
        print( f'Std = {np.median(nstd):.4f}, Area = {np.median(nar):.0f}, N = {np.max(wts)} ')
        # print( f'Count_k = {np.median(ncont):.4f}, Count_h = {np.median(ncontd):.4f}')
        # print( f'Means: Std = {np.mean(nstd):.4f}, Area = {np.mean(nar):.0f}, Count_k = {np.mean(ncont):.4f}, Count_h = {np.mean(ncontd):.4f}')

    fig,ax = plt.subplots(1,1, layout='constrained', figsize=(8,6))
    # ax.imshow( difs_t[n][i] )
    # ax.imshow( difs_v[n][i] )
    ax.imshow( kurv[i] )
    # ax.imshow( difs_t[n][i], vmin=-5, vmax=3 )
    # ax.imshow( mark_boundaries(normalize(difs_t[n][i]), wts) )
    # ax.imshow( mark_boundaries(normalize(difs_v[n][i]), wts) )
    # ax.imshow( mark_boundaries(normalize(kurv[i]) , wts)  )
    # ax.imshow( mark_boundaries(normalize(kurvs[i]) , wts)  )

    # plt.tight_layout()
    plt.title(rf'n = {n}, i={i}, $(\sigma_1,\sigma_2,\sigma_3)$ = {s1,s2,s3}' )
    plt.show()

# array([ 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 35, 38]) should be scallops
# [ 1,  3, 17, 20, 21, 36, 37] should be channels
esto =1
ct = 0
mo = 10
last = -30

if esto:
    fig,ax = plt.subplots(1,3,layout='constrained', figsize=(14,5))
    for m in range(0,39):    
        fff = ['o','s','^','*']
        if m in [5,6,7,8,9,15,23,27,28,32,33,35,38]: c = ['b',None][ct]
        elif m in [4,34]: c = ['b',None][ct]
        elif m in [1,3,17,20,21,36,37]: c = ['g',None][ct]
        else: c = ['r',None][ct]
            
        ax[0].scatter( np.median(v3_t[m][last:]), np.median(v4_t[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )
        ax[1].scatter( np.median(v3_t[m][last:]), np.median(v2_t[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )
        ax[2].scatter( np.median(v2_t[m][last:]), np.median(v4_t[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )

    for m in range(0,16):    
        fff = ['v','d']
        if m in [4,5,6,14]: c = ['r',None][ct]
        else: c = ['b',None][ct]
            
        ax[0].scatter( np.median(v3_v[m][last:]), np.median(v4_v[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )
        ax[1].scatter( np.median(v3_v[m][last:]), np.median(v2_v[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )
        ax[2].scatter( np.median(v2_v[m][last:]), np.median(v4_v[m][last:]), label=m, marker=fff[m//mo], s=15, c=c )

    ax[0].set_xlabel(r'$H_s$')
    ax[0].set_ylabel(r'$N$')
    ax[1].set_xlabel(r'$H_s$')
    ax[1].set_ylabel(r'$A_s$')
    ax[2].set_xlabel(r'$A_s$')
    ax[2].set_ylabel(r'$N$')
    
    # ax[0].legend()
    # ax[1].legend()
    ax[2].legend(ncols=3, fontsize=10, loc=[1,0.2])
    plt.show()
    
#%%
# orderpol = 2
# co2_t = []
# for n in tqdm(range(len(salis_t))):
#     coeffs = []
#     for i in (range(len(ts_t[n]))):
#         coeff, r, rank, s = polyfitn(n,i,hints_t,xns_t,yns_t, order=orderpol)
#         # cuapla = poln(coeff,xns_t[n],yns_t[n], order=orderpol) 
#         coeffs.append(coeff)

#     co2_t.append(np.array(coeffs))

orderpol = 1
co1_t,co1_v = [],[]
for n in tqdm(range(len(salis_t))):
    coeffs = []
    for i in (range(len(ts_t[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_t,xns_t,yns_t, order=orderpol)
        coeffs.append(coeff)
    co1_t.append(np.array(coeffs))
for n in tqdm(range(len(salis_v))):
    coeffs = []
    for i in (range(len(ts_v[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_v,xns_v,yns_v, order=orderpol)
        coeffs.append(coeff)
    co1_v.append(np.array(coeffs))

#%%
order = 1
expos = list(dict.fromkeys( [expo for expo in itertools.permutations(list(range(order+1))*2,2) if sum(expo) < order+1] ))

i = -20


# for tm in range(1):
#     plt.figure()
#     for n in [0,2,10,11,12,13,14,16,18,19,22,24,25,26,29,30,31,34]:
#         if   order==4: coeffs_b = coeffs_t[n]
#         elif order==2: coeffs_b = co2_t[n]
#         elif order==1: coeffs_b = co1_t[n]
#         xns_b, yns_b = xns_t[n], yns_t[n]
#         terms = np.array( [xns_b**e1 * yns_b**e2 for (e1,e2) in expos ] )
        
#         # plt.scatter(n, np.median(coeffs_b[i:,0]) , label=n )
#         # plt.scatter(salis_t[n], np.median(coeffs_b[i:,tm]) , label=n )
#         plt.scatter(rrho_t[n], np.median(coeffs_b[i:,tm]) , label=n )
#     plt.title(tm)
#     plt.grid()
#     plt.show()

# n = 14
# if   order==4: coeffs_b = coeffs_t[n]
# elif order==2: coeffs_b = co2_t[n]
# elif order==1: coeffs_b = co1_t[n]

# xns_b, yns_b = xns_t[n], yns_t[n]
# terms = np.array( [xns_b**e1 * yns_b**e2 for (e1,e2) in expos ] )

# i = -20
# coso = np.sum((coeffs_b[i].T * terms.T).T, axis=0)
# fli = np.isnan( hints_t[n][i] )
# coso[fli] = np.nan

# plt.figure()
# if order == 0: plt.imshow( hints_t[n][i] )
# else: plt.imshow( coso )
# # plt.imshow( hints_t[n][i] - coso )
# plt.show()

esto = 1
ct = 0
mas,men,neu = [],[],[]    

i = -10
thres = 0.03

if esto:
    plt.figure()
    for n in [0,2,10,11,12,13,14,16,18,19,22,24,25,26,29,30,31]:
    # for n in [0]:
        if   order==4: coeffs_b = coeffs_t[n]
        elif order==2: coeffs_b = co2_t[n]
        elif order==1: coeffs_b = co1_t[n]
        
        if n in [0,29,31]: c=['m',None][ct]
        elif n in [14,16,26,30]: c=['r',None][ct] 
        elif n in [25,19, 13,18]: c=['r',None][ct] 
        else: c=['gray',None][ct]
        
        mmm, eee = np.median(coeffs_b[i:,0]), np.std(coeffs_b[i:,0])
        # plt.plot( coeffs_b[:,0], label=n )
        plt.errorbar(rrho_t[n], mmm, yerr=eee , label=n, c=c, fmt='o' )
    
        if mmm>thres: mas.append(n)
        elif mmm<-thres: men.append(n)
        else: neu.append(n)
    
    for n in [4,5,6,14]:
        if   order==4: coeffs_b = coeffs_v[n]
        # elif order==2: coeffs_b = co2_v[n]
        elif order==1: coeffs_b = co1_v[n]
        
        if n in [4,5,6]: c=['m',None][ct]
        elif n in [14]: c=['r',None][ct] 
        else: c=['gray',None][ct]
        
        mmm, eee = np.median(coeffs_b[i:,0]), np.std(coeffs_b[i:,0])
        # plt.plot( coeffs_b[:,0], label=n )
        plt.errorbar(rrho_v[n], mmm, yerr=eee , label=n, c=c, fmt='s' )
    
        if mmm>thres: mas.append(n)
        elif mmm<-thres: men.append(n)
        else: neu.append(n)
    
        
    plt.axhline( thres, color='k', ls='--', lw=1)
    plt.axhline(-thres, color='k', ls='--', lw=1)
    
    plt.legend(loc=[1,0.1], fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    print(mas)
    print(men)
    print(neu)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # for l in [0,1,2,3,4,5]:
# for l in range(15):
#     ax.plot_wireframe(xns_b, yns_b, coeffs_b[i][l] * terms[l], rstride=20, cstride=20, color='C'+str(l), alpha=0.5 )
# ax.plot_wireframe(xns_b, yns_b, np.sum((coeffs_b[i].T * terms.T).T, axis=0), rstride=20, cstride=20, color='k' )
# plt.show()
 
#%%

dictco = {0:'magenta', 1:'blue', 2:'green', 3:'orange', 4:'red'}
shape_t = [0,2,3,2,1,1,1,1,1,1,3,3,3,4,4,1,4,2,4,4,2,2,3,1,3,4,4,1,1,0,4,0,1,1,1,1,2,2,1]
# shape_t[9] = 3
# shape_t = [2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2]
# shape_t =   [0,2,3,2,1,1,1,1,1,1,3,3,3,4,4,1,4,2,3,4,2,2,3,1,3,4,4,1,1,0,4,0,1,1,1,1,2,2,1]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,4,1]


plt.rcParams.update({'font.size':14})
# fig,ax = plt.subplots(1,2,layout='constrained', figsize=(12,5))
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='tight', figsize=(12/1.,4/1.))

last = -20
ct = 0
for m in range(0,39):    
    fff = 'o'
    if m in [32,33,34,35]: fff = 'd'
    # ax.scatter( np.median(v3_t[m][last:]), np.median(v2_t[m][last:]), label=m, marker=fff, s=20, c=dictco[shape_t[m]]) 
    ax[r'$a)$'].scatter( np.median(v3_t[m][last:]), np.median(v2_t[m][last:]), label=m, marker=fff, s=20, edgecolors=dictco[shape_t[m]], facecolors='none' )

for m in range(0,16):
    fff = '^'
    # ax.scatter( np.median(v3_v[m][last:]), np.median(v2_v[m][last:]), label=m, marker=fff, s=20, c=dictco[shape_v[m]]) 
    ax[r'$a)$'].scatter( np.median(v3_v[m][last:]), np.median(v2_v[m][last:]), label=m, marker=fff, s=20, edgecolors=dictco[shape_v[m]], facecolors='none' )

ax[r'$a)$'].set_ylim(0,250)
ax[r'$a)$'].set_xlim(0,0.045)
ax[r'$a)$'].set_xlabel(r'$\sigma_{kr}$')
ax[r'$a)$'].set_ylabel(r'$A_r$')

th_el = np.linspace(0,2*np.pi,1000)
rxe, rye = 0.0065, 16
cxe, cye = 0.009, 57
x_el, y_el = rxe*np.cos(th_el)+cxe, rye*np.sin(th_el)+cye
ax[r'$a)$'].plot( x_el, y_el, 'k--', alpha=0.5 )



order = 1
expos = list(dict.fromkeys( [expo for expo in itertools.permutations(list(range(order+1))*2,2) if sum(expo) < order+1] ))
lasts = -20
thres = 0.03
for m in [0,2,10,11,12,13,14,16,18,19,22,24,25,26,29,30,31]:
    fff = 'o'
    if m in [32,33,34,35]: fff = 'd'
    coeffs_b = co1_t[m]
    mmm, eee = np.median(coeffs_b[lasts:,0]), np.std(coeffs_b[lasts:,0])
    # ax[r'$b)$'].scatter(rrho_t[m], mmm, label=m, marker=fff, s=20, c=dictco[shape_t[m]]) 
    ax[r'$b)$'].scatter(rrho_t[m], mmm, label=m, marker=fff, s=20, edgecolors=dictco[shape_t[m]], facecolors='none' ) 
for m in [4,5,6,14]:
    coeffs_b = co1_v[m]
    mmm, eee = np.median(coeffs_b[lasts:,0]), np.std(coeffs_b[lasts:,0])
    # ax[r'$b)$'].scatter(rrho_v[m], mmm, label=m, marker='o', s=20, c=dictco[shape_v[m]]) 
    ax[r'$b)$'].scatter(rrho_v[m], mmm, label=m, marker='^', s=20, edgecolors=dictco[shape_v[m]], facecolors='none' ) 
ax[r'$b)$'].axhline( thres, color='k', ls='--', lw=1)
ax[r'$b)$'].axhline(-thres, color='k', ls='--', lw=1)

ax[r'$b)$'].set_ylim(-0.1,0.15)
ax[r'$b)$'].set_xlim(-0.3,9)
ax[r'$b)$'].set_xlabel(r'$R_\rho$')
ax[r'$b)$'].set_ylabel(r'$a$')

leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', facecolors='none', s=30) for i in ['^','o','d']]
lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1,0.1], frameon=False)
ax[r'$b)$'].add_artist(lgd1)
leg2 = [ax[r'$b)$'].scatter([], [], marker='o', edgecolors=i, facecolors='none', s=30) for i in ['magenta','red','blue','green','orange']]
lgd2 = ax[r'$b)$'].legend(leg2, ['Top melting','Bottom melting', 'Scalloped', 'Channelized','Incurved'], loc=[1,0.5], frameon=False)

for labels,axs in ax.items():
    if labels == r'$a)$':
        axs.annotate(labels, (-0.16,0.98), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
    else: axs.annotate(labels, (-0.19,0.98), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})


# plt.savefig('./Documents/Figs morpho draft/separation_morph.pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


#%%
# =============================================================================
# Graph of phase space of experiments
# =============================================================================
Kerr_angle = [0,20,20,30,30,30,34.2,34.2,34.2,39.5,39.5,39.5] 
Kerr_sal = [35.9,35.7,35.7,35.9,35.5,35.3,35.1,35.6,35.7,36.4,35.4,35.3] 
# poner de forma distinta los experimentos que muestran scallops? Y canales?
 

# fig, ax = plt.subplot_mosaic([[r'$a)$']], layout='tight', figsize=(12,5) , sharex=False)
plt.figure()
plt.plot( salis_v, angys_v, 'o', mfc='none', color='#3B719F')
plt.plot( salis_t[:-4], angys_t[:-4], 'o', color='#3B719F' )
plt.plot( salis_t[-4:], angys_t[-4:], 'd', color='#3B719F' )
plt.plot( Kerr_sal, Kerr_angle, 'o' , color='#ba3c3c' )
plt.xlabel(r'$S$ (g/kg)')
plt.ylabel(r'$\theta$ (°)')
# plt.savefig('./Documents/Figs morpho draft/experiments.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
mfc=None
xaxis = 'rrho' # 'sal' or 'rrho'

plt.rcParams.update({'font.size':12})

# shape_t = [0,2,0,2,3,1,1,1,1,1,3,3,3,3,0,1,0,2,3,3,2,2,3,1,3,3,0,1,1,0,0,0,1,1,3,1,2,2,1]
# shape_t = [0,2,0,2,3,1,1,1,1,1,3,3,3,3,4,1,4,2,3,3,2,2,3,1,3,3,4,1,1,0,4,0,1,1,3,1,2,2,1]

# 14,16,26,30
# 13,19,18,25
#          0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38
shape_t = [0 ,2 ,3 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,3 ,3 ,3 ,4 ,4 ,1 ,4 ,2 ,4 ,4 ,2 ,2 ,3 ,1 ,3 ,4 ,4 ,1 ,1 ,0 ,4 ,0 ,1 ,1 ,1 ,1 ,2 ,2 ,1]
# shape_t = [0 ,2 ,3 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,3 ,3 ,3 ,3 ,4 ,1 ,4 ,2 ,3 ,4 ,2 ,2 ,3 ,1 ,3 ,4 ,4 ,1 ,1 ,0 ,4 ,0 ,1 ,1 ,1 ,1 ,2 ,2 ,1]
# shape_t[9] = 3

marker_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,4,1] 

rrho_v = np.abs(density_millero(temp_v, 0) - density_millero(temp_v, salis_v)) \
                / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v))
rrho_t = np.abs(density_millero(temp_t, 0) - density_millero(temp_t, salis_t)) \
                / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t))
# rrho_v = np.abs(density_millero(0, 0) - density_millero(0, salis_v)) \
#                 / np.abs(density_millero(temp_v, 0) - density_millero(0, 0))
# rrho_t = np.abs(density_millero(0, 0) - density_millero(0, salis_t)) \
#                 / np.abs(density_millero(temp_t, 0) - density_millero(0, 0))


# dictco = {0:(0,0,1), 1:(1,0,0), 2:(0,1,0), 3:(0.5,0.5,0.5)}
# dictco = {0:(0.5,0.5,0.5), 1:'#3B719F', 2:'#6aa84f', 3:'#ba3c3c'}
dictco = {0:'magenta', 1:'blue', 2:'green', 3:'orange', 4:'red'}
# dictco = {0:'magenta', 1:'cyan', 2:'green', 3:'orange', 4:'red'}
dictma = {0:'o',1:'d'}

cols_v = np.array( [dictco[j] for j in shape_v] )
mars_t = np.array( [dictma[j] for j in marker_t] )
cols_t = np.array( [dictco[j] for j in shape_t] )

if xaxis == 'sal':
    hollow_sv, hollow_st = [salis_v[4],salis_v[14],salis_v[10]],[salis_t[1],salis_t[5],salis_t[11]]
if xaxis == 'rrho':
    hollow_sv, hollow_st = [rrho_v[4],rrho_v[14],rrho_v[10]],[rrho_t[1],rrho_t[5],rrho_t[11]]
hollow_av, hollow_at = [angys_v[4],angys_v[14],angys_v[10]],[angys_t[1],angys_t[5],angys_t[11]]


# plt.figure()
fig,ax = plt.subplots(1,1,layout='constrained', figsize=(7,4))

if xaxis == 'sal':
    ax.scatter( np.roll(salis_t,3)[:-4], np.roll(angys_t,3)[:-4], edgecolors=np.roll(cols_t,3,axis=0)[:-4], marker='o', facecolors='none') #, facecolors=mfc)
    # ax.scatter( np.roll(salis_t,0)[:], np.roll(angys_t,0)[:], c=np.roll(cols_t,0)[:], marker='o' )
    ax.scatter(salis_t[-7:-3], angys_t[-7:-3], edgecolors=cols_t[-7:-3], marker='d', facecolors='none') #, facecolors=mfc )
    ax.scatter(salis_v, angys_v, marker='^',  edgecolors=cols_v, facecolors='none') # , facecolors=mfc )

elif xaxis == 'rrho':
    ax.scatter( np.roll(rrho_t,3)[:-4], np.roll(angys_t,3)[:-4], edgecolors=np.roll(cols_t,3,axis=0)[:-4], marker='o', facecolors='none') #, facecolors=mfc)
    ax.scatter(rrho_t[-7:-3], angys_t[-7:-3], edgecolors=cols_t[-7:-3], marker='d', facecolors='none') #, facecolors=mfc )
    ax.scatter(rrho_v, angys_v, marker='^',  edgecolors=cols_v, facecolors='none') # , facecolors=mfc )


# ttt = np.linspace(-15,50) * np.pi/180
# cte = 5.5
# ds = cte * (np.cos(ttt))**1
# ax.plot( ds, ttt * 180/np.pi, 'k--' )
# cte = 24
# ds = cte * (np.cos(ttt))**1
# ax.plot( ds, ttt * 180/np.pi, 'k--' )

if xaxis == 'sal': ax.set_xlabel(r'$S$ (g/kg)')
if xaxis == 'rrho': ax.set_xlabel(r'$R_\rho$ ')
ax.set_ylabel(r'$\theta$ (°)')

plt.scatter( hollow_sv, hollow_av, marker='^', c=cols_v[[4,14,10]] )
plt.scatter( hollow_st, hollow_at, marker='o', c=cols_t[[1,5,11]] )

leg1 = [ax.scatter([], [], marker=i, edgecolors='gray', facecolors='none', s=30) for i in ['^','o','d']]
lgd1 = ax.legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1,0.1], frameon=False)
ax.add_artist(lgd1)
leg2 = [ax.scatter([], [], marker='o', edgecolors=i, facecolors='none', s=30) for i in ['magenta','red','blue','green','orange']]
lgd2 = ax.legend(leg2, ['Top melting','Bottom melting', 'Scalloped', 'Channelized','Incurved'], loc=[1,0.5], frameon=False)

# ax.add_artist(lgd)

# coutl = 'k'
# ax.xaxis.label.set_color(coutl)        #setting up X-axis label color to yellow
# ax.yaxis.label.set_color(coutl)          #setting up Y-axis label color to blue

# ax.tick_params(axis='x', colors=coutl)    #setting up X-axis tick color to red
# ax.tick_params(axis='y', colors=coutl)  #setting up Y-axis tick color to black

# ax.spines['left'].set_color(coutl) 
# ax.spines['top'].set_color(coutl) 
# ax.spines['right'].set_color(coutl)
# ax.spines['bottom'].set_color(coutl)

# plt.legend(leg1, ['Set 1','Set 2','Set 3'], bbox_to_anchor=[1.5,0.5], frameon=False)
# plt.savefig('./Documents/Figs morpho draft/rrho_morphologs.pdf',dpi=400, bbox_inches='tight', transparent=True) #, bbox_extra_artists=(lgd))
fig.show()

#%%
# =============================================================================
# Phase space graph with sketches
# =============================================================================
import matplotlib.patches as patches

def bounding_box( data, pad = 20, ratio=3/2, size=(230,170) ):
    datab = data[:,:,1]<255
    reg = regionprops( label(datab) )
    bb = reg[0].bbox

    bp = np.zeros(4, dtype=int)
    hei, wid = bb[2]-bb[0], bb[3]-bb[1]
    ratio_bb = hei / wid

    if ratio_bb > ratio:
        bp[0],bp[2] = bb[0]-pad, bb[2]+pad
        
        pad_w = (int( (hei+2*pad) / ratio) - wid)//2
        bp[1],bp[3] = bb[1]-pad_w, bb[3]+pad_w

    elif ratio_bb < ratio:
        bp[1],bp[3] = bb[1]-pad, bb[3]+pad
        
        pad_h = (int( (wid+2*pad) * ratio) - hei)//2
        bp[0],bp[2] = bb[0]-pad_h, bb[2]+pad_h

    else:
        bp[1],bp[3] = bb[1]-pad, bb[3]+pad
        bp[0],bp[2] = bb[0]-pad, bb[2]+pad
    
    return data[bp[0]:bp[2],bp[1]:bp[3]]

def get_image(i, halg, xrp, yrp, add=20 ):
    hice = halg[i]

    fog = plt.figure(figsize=(4,6))
    ex = plt.axes(projection='3d')

    ex.plot_surface(xrp[i], hice , yrp[i], ccount=300, rcount=300, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
                    antialiased=True,
                    facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

    ex.set_xlabel('x (mm)')
    ex.set_ylabel('z (mm)')
    ex.set_zlabel('y (mm)')
    # ax[key].invert_zaxis()
    ex.invert_xaxis()

    lxd, lxi = np.nanmax(xrp), np.nanmin(xrp)
    lzd, lzi = np.nanmin(yrp), np.nanmax(yrp)
    lyi, lyd = np.nanmin(halg), np.nanmax(halg)

    add = 20
    ex.set_zlim(lzd-add,lzi+add)
    ex.set_xlim(lxd+add,lxi-add)
    ex.set_ylim(lyi-add,lyd+add)
    ex.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1)

    ex.view_init(5,120)    
    ex.axis('off')

    fog.canvas.draw()
    buf = np.asarray(fog.canvas.buffer_rgba())
    data = buf.copy()
    fog.show()
    plt.close(fog)
    
    return data

plt.rcParams.update({'font.size':16})
# fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$b)$',r'$b)$',r'$c)$',r'$c)$',r'$d)$',r'$d)$'],
#                               [r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$', '.'   ,r'$e)$',r'$e)$',r'$f)$',r'$f)$', '.'   ]], 
#                               figsize=(12/1.,5/1.), layout='constrained')
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$b)$',r'$b)$',r'$c)$',r'$c)$',r'$d)$',r'$d)$'],
                              [r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$a)$',r'$e)$',r'$e)$',r'$f)$',r'$f)$',r'$g)$',r'$g)$']], 
                              figsize=(12/1.,5/1.), layout='constrained')

i = 60
for n,key,expt,col,mark in zip([1,5,11,4,10,14], [r'$b)$',r'$c)$',r'$d)$',r'$e)$',r'$f)$',r'$g)$'], ['t','t','t','v','v','v'], 
                               ['green','blue','orange','magenta','blue','red'],['o','o','o','^','^','^']):
# for n,key,expt,l in zip([1,8,11,0,16], [r'$b)$',r'$c)$',r'$d)$',r'$e)$',r'$f)$'], ['t','t','t','t','t'], [0,1,2,3,4]):

    if expt == 't':
        halt = np.load('./Documents/Height profiles/profile_s'+sal_t[n]+'_t'+inc_t[n]+'.npy')
        halg = nangauss(halt, 2)
        ds_b, Ls_b = ds_t, Ls_t
        angys_b = angys_t
    elif expt == 'v':
        halt = np.load('./Documents/Height profiles/ice_block_0_'+sal_v[n]+'.npy')
        halg = nangauss(halt, 2)
        ds_b, Ls_b = ds_v, Ls_v
        angys_b = angys_v    
    
    nt,ny,nx = np.shape(halt)
    x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_b[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_b[n]
    t = np.arange(nt) * 30
    tr,yr,xr = np.meshgrid(t,y,x, indexing='ij')
    xrp, yrp = xr - halg/Ls_b[n] * xr, yr - halg/Ls_b[n] * yr
    
    data = get_image(i, halg, xrp, yrp)
    bdat = bounding_box(data, pad=10, ratio=4/3)

    ax[key].imshow( bdat )
    if expt == 't':
        ax[key].scatter( 10,10, marker=mark, s=40, c=col )
    if expt == 'v':
        ax[key].scatter( 30,10, marker=mark, s=40, c=col )
    ax[key].axis('off')


mfc=None
xaxis = 'rrho' # 'sal' or 'rrho'

shape_t = [0 ,2 ,3 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,3 ,3 ,3 ,4 ,4 ,1 ,4 ,2 ,4 ,4 ,2 ,2 ,3 ,1 ,3 ,4 ,4 ,1 ,1 ,0 ,4 ,0 ,1 ,1 ,1 ,1 ,2 ,2 ,1]
marker_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,4,1] 

dictco = {0:'magenta', 1:'blue', 2:'green', 3:'orange', 4:'red'}
dictma = {0:'o',1:'d'}

rrho_v = np.abs(density_millero(temp_v, 0) - density_millero(temp_v, salis_v)) \
                / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v))
rrho_t = np.abs(density_millero(temp_t, 0) - density_millero(temp_t, salis_t)) \
                / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t))

cols_v = np.array( [dictco[j] for j in shape_v] )
mars_t = np.array( [dictma[j] for j in marker_t] )
cols_t = np.array( [dictco[j] for j in shape_t] )

filled_v, filled_t = [4,10,14], [1,5,11]
if xaxis == 'sal':
    # hollow_sv, hollow_st = [salis_v[4],salis_v[14],salis_v[10]],[salis_t[1],salis_t[5],salis_t[11]]
    hollow_sv, hollow_st = [salis_v[k] for k in filled_v], [salis_t[k] for k in filled_t]
if xaxis == 'rrho':
    # hollow_sv, hollow_st = [rrho_v[4],rrho_v[14],rrho_v[10]],[rrho_t[1],rrho_t[5],rrho_t[11]]
    hollow_sv, hollow_st = [rrho_v[k] for k in filled_v],[rrho_t[k] for k in filled_t]
# hollow_av, hollow_at = [angys_v[4],angys_v[14],angys_v[10]],[angys_t[1],angys_t[5],angys_t[11]]
hollow_av, hollow_at = [angys_v[k] for k in filled_v],[angys_t[k] for k in filled_t]

if xaxis == 'sal':
    ax[r'$a)$'].scatter( np.roll(salis_t,3)[:-4], np.roll(angys_t,3)[:-4], edgecolors=np.roll(cols_t,3,axis=0)[:-4], marker='o', facecolors='none') #, facecolors=mfc)
    # ax[r'$a)$'].scatter( np.roll(salis_t,0)[:], np.roll(angys_t,0)[:], c=np.roll(cols_t,0)[:], marker='o' )
    ax[r'$a)$'].scatter(salis_t[-7:-3], angys_t[-7:-3], edgecolors=cols_t[-7:-3], marker='d', facecolors='none') #, facecolors=mfc )
    ax[r'$a)$'].scatter(salis_v, angys_v, marker='^',  edgecolors=cols_v, facecolors='none') # , facecolors=mfc )

elif xaxis == 'rrho':
    ax[r'$a)$'].scatter( np.roll(rrho_t,3)[:-4], np.roll(angys_t,3)[:-4], edgecolors=np.roll(cols_t,3,axis=0)[:-4], marker='o', facecolors='none') #, facecolors=mfc)
    ax[r'$a)$'].scatter(rrho_t[-7:-3], angys_t[-7:-3], edgecolors=cols_t[-7:-3], marker='d', facecolors='none') #, facecolors=mfc )
    ax[r'$a)$'].scatter(rrho_v, angys_v, marker='^',  edgecolors=cols_v, facecolors='none') # , facecolors=mfc )

if xaxis == 'sal': ax[r'$a)$'].set_xlabel(r'$S$ (g/kg)')
if xaxis == 'rrho': ax[r'$a)$'].set_xlabel(r'$R_\rho$ ')
ax[r'$a)$'].set_ylabel(r'$\theta$ (°)')

# ax[r'$a)$'].scatter( hollow_sv, hollow_av, marker='^', c=cols_v[[4,14]] ) #,10]] )
# ax[r'$a)$'].scatter( hollow_st, hollow_at, marker='o', c=cols_t[[1,8,11]] )
ax[r'$a)$'].scatter( hollow_sv, hollow_av, marker='^', c=cols_v[filled_v] ) #,10]] )
ax[r'$a)$'].scatter( hollow_st, hollow_at, marker='o', c=cols_t[filled_t] )

leg1 = [ax[r'$a)$'].scatter([], [], marker=i, edgecolors='gray', facecolors='none', s=30) for i in ['^','o','d']]
lgd1 = ax[r'$a)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1,0.1], frameon=False, labelspacing=0.7)
ax[r'$a)$'].add_artist(lgd1)
leg2 = [ax[r'$a)$'].scatter([], [], marker='o', edgecolors=i, facecolors='none', s=30) for i in ['magenta','red','blue','green','orange']]
lgd2 = ax[r'$a)$'].legend(leg2, ['Top melting','Bottom melting', 'Scalloped', 'Channelized','Incurved'], loc=[1,0.5], frameon=False, labelspacing=0.7)


# fig.canvas.draw()
# # for a in ax.values():
# for key,col in zip([r'$b)$',r'$c)$',r'$d)$',r'$e)$',r'$f)$'], ['green','blue','orange','magenta','red']):
#     pos = ax[key].get_position()  # now position is correct
#     rect = patches.Rectangle( (pos.x0, pos.y0), pos.width, pos.height, transform=fig.transFigure,
#                                 fill=False, linewidth=1, edgecolor=col )
#     fig.add_artist(rect)

# plt.savefig('./Documents/Figs morpho draft/rrho_morphologs_sketch(2).pdf',dpi=400, bbox_inches='tight', transparent=True)

fig.show()



#%%
# =============================================================================
# Try for wavelength channels
# =============================================================================
def channels(kurv, n, i, delimx=10, delimy=25, min_tsize=70):
    kurg = nangauss(-kurv[i], 5 )

    lines, piks = [], []
    for line in range(350):
        peks = find_peaks( kurg[line,:], prominence=0.018, distance=30 ) #,wlen=50)
        lines += list( [line]*len(peks[0]) )
        piks += list( peks[0] )

    lines, piks = np.array(lines), np.array(piks)

    tracs = [0]
    count = 1
    for j in range(1,len(lines)):
        
        distsx = (piks[:j] - piks[j])**2 
        distsy = (lines[:j] - lines[j])**2 
        
        dists = (distsx / delimx**2) + (distsy / delimy**2)
        
        if np.min(dists) < 2 : tracs.append( tracs[np.argmin(dists)] )
        else:
            tracs.append( count )
            count+=1
    tracs = np.array(tracs)

    xmean = []
    for j in range(np.max(tracs)+1):
        if np.sum(tracs == j) < min_tsize: tracs[tracs == j] = -1
        else: xmean.append( np.mean(piks[tracs==j]) )
    filtr = tracs >= 0
    piks, lines, tracs, xmean = piks[filtr], lines[filtr], tracs[filtr], np.array(xmean)

    nlines = len(np.unique(tracs))
    lams = []
    for j in np.unique(lines):
        if len( piks[lines==j] ) == nlines:
            sortp = np.sort(piks[lines==j])
            lams += list( sortp[1:]-sortp[:-1] )
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    lams = np.array(lams) * dx
    return lams, piks, lines, tracs

def channels_gxx(gxx,n,i, delimx=10, delimy=25, min_tsize=70, filtering=False):
    lines, piks = [], []
    for line in range(350):
        peks = find_peaks( -gxx[i][line,:], prominence=0.01, distance=30, height=0 ) #,wlen=50)
        lines += list( [line]*len(peks[0]) )
        piks += list( peks[0] )
    lines, piks = np.array(lines), np.array(piks)

    tracs = [0]
    count = 1
    for j in range(1,len(lines)):
    # for j in range(1,40):
        
        distsx = (piks[:j] - piks[j])**2 
        distsy = (lines[:j] - lines[j])**2 
        
        dists = (distsx / delimx**2) + (distsy / delimy**2)
        
        if np.min(dists) < 2 : tracs.append( tracs[np.argmin(dists)] )
        else:
            tracs.append( count )
            count+=1
    tracs = np.array(tracs)

    xmean = []
    for j in range(np.max(tracs)+1):
        if np.sum(tracs == j) < min_tsize: tracs[tracs == j] = -1
        else: xmean.append( np.mean(piks[tracs==j]) )
    filtr = tracs >= 0
    piks, lines, tracs, xmean = piks[filtr], lines[filtr], tracs[filtr], np.array(xmean)

    nlines = len(np.unique(tracs))
    lams = []
    if filtering:
        for j in np.unique(lines):
            if len( piks[lines==j] ) == nlines:
                sortp = np.sort(piks[lines==j])
                lams += list( sortp[1:]-sortp[:-1] )
    else:
        for j in np.unique(lines):
            sortp = np.sort(piks[lines==j])
            lams += list( sortp[1:]-sortp[:-1] )

    dx = xns_t[n][0,1] - xns_t[n][0,0]
    lams = np.array(lams) * dx

    return lams, piks, lines, tracs

def peak_mask(ceros, pek):
    left, right = ceros[:-1], ceros[1:]

    lefr, rigr = left.reshape(-1,1), right.reshape(-1,1)
    ppes = pek.reshape(1,-1)
    mask = (lefr<ppes) & (rigr>ppes)

    count = np.where( np.sum( mask, axis=1 )!=1 )[0]
    mask[count] = False

    proper_sides, proper_peaks = np.where(mask)
    wid = right[proper_sides] - left[proper_sides]
    
    return pek[proper_peaks], wid


def channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.01, dx=True):
    lines, piks, widths = [], [], []
    for line in range(350):
        gline = gxx[i][line,:]
        peks = find_peaks( gline , prominence=0.01, distance=30, height=height ) #,wlen=50)
        ceros = np.where( (gline[:-1]*gline[1:])<0 )[0]
    
        peaks, width = peak_mask( ceros, peks[0] )    
        
        if len(peaks) != len(width): 
            print(line, len(peaks) > len(width) )
    
        lines += list( [line]*len(peaks) )    
        piks += list( peaks )
        widths += list(width)
    lines, piks, widths = np.array(lines), np.array(piks), np.array(widths)

    tracs = [0]
    count = 1
    for j in range(1,len(lines)):
    # for j in range(1,40):
        
        distsx = (piks[:j] - piks[j])**2 
        distsy = (lines[:j] - lines[j])**2 
        
        dists = (distsx / delimx**2) + (distsy / delimy**2)
        
        if np.min(dists) < 2 : tracs.append( tracs[np.argmin(dists)] )
        else:
            tracs.append( count )
            count+=1
    tracs = np.array(tracs)

    # xmean = []
    for j in range(np.max(tracs)+1):
        if np.sum(tracs == j) < min_tsize: tracs[tracs == j] = -1
        # else: xmean.append( np.mean(piks[tracs==j]) )
    filtr = tracs >= 0
    # piks, lines, tracs, xmean = piks[filtr], lines[filtr], tracs[filtr], np.array(xmean)
    piks, lines, tracs, widths = piks[filtr], lines[filtr], tracs[filtr], widths[filtr]

    # nlines = len(np.unique(tracs))
    # lams = []
    # if filtering:
    #     for j in np.unique(lines):
    #         if len( piks[lines==j] ) == nlines:
    #             sortp = np.sort(piks[lines==j])
    #             lams += list( sortp[1:]-sortp[:-1] )
    # else:
    #     for j in np.unique(lines):
    #         sortp = np.sort(piks[lines==j])
    #         lams += list( sortp[1:]-sortp[:-1] )

    if dx:
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        lams = widths * dx
    else:
        lams = widths

    return lams, piks, lines, tracs

#%%
n = 36
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam36, slam36 = [], []
for i in tqdm(range(93)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=100, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)
    
    mlam36.append( np.median(lams) )
    slam36.append(  np.std(lams) )

n = 37
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam37, slam37 = [], []
for i in tqdm(range(122)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=150, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)
    
    mlam37.append( np.median(lams) )
    slam37.append(  np.std(lams) )

n = 1
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam1, slam1 = [], []
for i in tqdm(range(91)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=150, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)

    mlam1.append( np.median(lams) )
    slam1.append(  np.std(lams) )

n = 3
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam3, slam3 = [], []
for i in tqdm(range(76)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=150, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)
    
    mlam3.append( np.median(lams) )
    slam3.append(  np.std(lams) )

n = 20
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam20, slam20 = [], []
for i in tqdm(range(80)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=150, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)

    mlam20.append( np.median(lams) )
    slam20.append(  np.std(lams) )

n = 21
halg = nangauss(hints_t[n],8)
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

mlam21, slam21 = [], []
for i in tqdm(range(65)):
    # lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=150, filtering=False)
    lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005)

    mlam21.append( np.median(lams) )
    slam21.append(  np.std(lams) )

#%%
tt36 = np.linspace(0,92,93) / 2
tt37 = np.linspace(0,121,122) / 2
tt1  = np.linspace(0,90,91) / 2
tt3  = np.linspace(0,75,76) / 2
tt20 = np.linspace(0,79,80) / 2
tt21 = np.linspace(0,64,65) / 2

plt.figure()
plt.errorbar(tt36*2, mlam36, yerr=slam36, fmt='.', capsize=2, label='36')
plt.errorbar(tt37*2, mlam37, yerr=slam37, fmt='.', capsize=2, label='37')
plt.errorbar(tt1*2, mlam1, yerr=slam1, fmt='.', capsize=2, label='1')
plt.errorbar(tt3*2, mlam3, yerr=slam3, fmt='.', capsize=2, label='3')
plt.errorbar(tt20*2, mlam20, yerr=slam20, fmt='.', capsize=2, label='20')
plt.errorbar(tt21*2, mlam21, yerr=slam21, fmt='.', capsize=2, label='21')
plt.grid()
plt.legend()
plt.show()

#%%

def theory_delta(tb,ti,tf, mu,ks,lat,cp, rb,rf,ri,rs, g,the ):
    beta = rs/ri
    topc = 3 * ks * mu * rs
    top1, top2 = (tb-ti), (lat*rb - cp*tb*ri)
    botc = 4 * g * np.cos(the) * (rb - rf)
    bot1, bot2 = (tf*rb-tb*rf), (-beta*cp*ti*ri + lat*rs)
    top = (topc*top1*top2)
    bot = botc*bot1*bot2
    # print(top, bot)
    return (top/bot)**(1/3)

lams = []
for n in [36,37,1,3,20,21]:
    
    tb = temp_t[n]  #°C
    ti = 0 #°C
    tf = (tb-ti)/2
    
    lat = 334000 # J/kg
    cp = 4184  # J/(kg°C)
    ks = 1.43e-7 # m^2/s
    mu = 0.001 # N s/m^2
    
    rb = density_millero(tb, salis_t[n]) #kg/m^3
    ri = density_millero(ti, 0) #kg/m^3
    # ri = density_millero(ti, salis_t[n]) #kg/m^3
    rf = density_millero(tf, salis_t[n]) #kg/m^3
    rs = 917 #kg/m^3
    
    g = 9.81 # m/s^2
    the = angys_t[n] * np.pi / 180
    
    lams.append(theory_delta(tb, ti, tf, mu, ks, lat, cp, rb, rf, ri, rs, g, the) * 1000)

dels = [ mlam36[60], mlam37[60], mlam1[60], mlam3[60], mlam20[60], mlam21[60] ]
dels_s = [ slam36[60], slam37[60], slam1[60], slam3[60], slam20[60], slam21[60] ]

plt.figure()
plt.ylabel(r'$\lambda_{exp}$ (mm)')

order = np.argsort(lams)
# plt.errorbar( np.array(lams) * 63.9, dels, yerr=dels_s, fmt='.', capsize=2 )
plt.xlabel(r'$\lambda_s$ (mm)')
plt.errorbar( np.array(lams), dels, yerr=dels_s, fmt='.', capsize=2 )
plt.plot( np.array(lams)[order], np.array(lams)[order]*50.558, 'k--' )
plt.plot( np.array(lams)[order], np.array(lams)[order]*63.9 - 3.1957 , 'r--' )

# plt.errorbar( angys_t[[36,37,1,3,20,21]], dels, yerr=dels_s, fmt='.', capsize=2 )
# plt.xlabel(r'$\theta$ (°)')

# plt.xscale('log')
# plt.yscale('log')

plt.show()

def lin(v):
    return np.sum( ( np.array(dels) - np.array(lams)*v[0] )**2 )
    
linregress(np.array(lams), np.array(dels))[0], least_squares(lin, [50]).x

#%%
n = 1
halg = nangauss(hints_t[n],(5,15,5))
gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

# mlam36, slam36 = [], []
# for i in tqdm(range(93)):

t1 = time()
i = 80
# llin, ppik = [],[]
# for i in tqdm(range(65)):
#     lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=100, filtering=False)
#     llin.append(lines)
#     ppik.append(piks)

medlam = []
lams, piks, lines, tracs = channels_gxx_width(gxx,n,i, delimx=10, delimy=25, min_tsize=70, height=0.005, dx=False)
# lams, piks, lines, tracs = channels_gxx(gxx, n, i, delimx=5, delimy=15, min_tsize=100, filtering=False)
for j in np.unique(tracs):
    medlam.append( np.median(lams[tracs==j]) )



t2 = time()
print(t2-t1)

# print( np.mean(medlam),np.std(medlam) )
print( np.mean(lams), np.median(lams), np.std(lams) )

plt.figure()
plt.imshow( halg[i] )# , vmax=-79)
# for j in np.unique(tracs):
#     plt.plot(piks[tracs==j], lines[tracs==j], '.-', label=j)
# plt.legend()
plt.colorbar()
plt.title(i)
plt.show()

plt.figure()
plt.imshow(gxx[i])
plt.plot( piks, lines, 'k.' )
# plt.scatter( piks, lines, c=tracs, cmap='gray' )
# plt.errorbar( piks, lines, xerr=widths/2, capsize=5, fmt='.', color='red')
plt.colorbar()
plt.title(i)
plt.show()

# plt.figure()
# for j in np.unique(tracs):
#     plt.plot(lines[tracs==j], lams[tracs==j], '.-', label=j)
# plt.grid()
# plt.legend()
# plt.show()

ll = 250
plt.figure()
plt.plot( gxx[i][ll,:], '.-' )
plt.plot( piks[lines==ll], gxx[i][ll,piks[lines==ll]], 'k.' )
plt.grid()
plt.show()


# ll = 250
# plt.figure()
# plt.plot( halg[i][250,:] )
# plt.plot( halg[i][200,:] )
# plt.plot( halg[i][150,:] )
# # plt.plot( np.gradient(halg[i][250,:]) )
# # plt.plot( np.gradient(np.gradient(halg[i][250,:])) )
# # plt.plot( gxx[i][ll,:], '.-' )
# # plt.plot( piks[lines==ll], gxx[i][ll,piks[lines==ll]], 'k.' )
# plt.grid()
# plt.show()

#%%
# ll = 100
# for ll in [100,150,200,250]:
#     plt.figure()
#     for i in range(0,65,2):
#         plt.plot( halg[i][ll,:], c=(i/65,0,0) )
#         plt.plot( ppik[i][llin[i]==ll], halg[i][ll, ppik[i][llin[i]==ll]], 'k.'  )
#         # plt.plot( np.gradient( halg[i][ll,:] ), c=((i)/80,0,0) )
#         # plt.plot( np.gradient(np.gradient( halg[i][ll,:])), c=((i)/80,0,0) )
#     plt.show()

i = 40
plt.figure()
for ll in range(100,300,5):
    plt.plot( halg[i][ll,:] - np.nanmean(halg[i][ll,:]) , c=((ll-100)/200,0,0), label=ll )
    # plt.plot( ppik[i][llin[i]==ll], halg[i][ll, ppik[i][llin[i]==ll]], 'k.'  )
plt.legend()
plt.show()

# nt,ny,nx = np.shape(halg)
# ll = 200
# line_t = np.zeros((nt,nx))
# for i in range(0,nt,1):
#     line_t[i] = halg[i][ll,:] - np.nanmean(halg[i][ll,:])
    
    
# plt.figure()
# plt.imshow(line_t.T)
# # plt.axis('equal')
# plt.show()

# halg = nangauss(hints_t[n],(5,15,5))

# i = 80
# wts = watershed( halg[i][:350],  mask= binary_erosion( ~np.isnan(halg[i][:350]), disk(5) ) )
# # wts = watershed( -gxx[i],  mask= dilation( ~np.isnan(halg[i]) ))

# def normalize(array):
#     return ( array - np.nanmin(array) ) / (np.nanmax(array) - np.nanmin(array))

# plt.figure()
# plt.imshow( mark_boundaries( normalize(halg[i][:350]), wts) )
# # plt.imshow( mark_boundaries( normalize(gxx[i]), wts) )
# plt.show()


#%%
# =============================================================================
# Watershed gaussian
# =============================================================================

def countour_mean(wtssal_b,n,i,labe, ve=True, dif=True):

    sccaa = wtssal_b[n][i] == labe
    ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )

    if ve:    
        if dif: conval = difs_v[n][i][ccoo]
        else: conval = hints_v[n][i][ccoo]
    else:    
        if dif: conval = difs_t[n][i][ccoo]
        else: conval = hints_t[n][i][ccoo]
    
    return np.nanmean(conval)

def image_nanstdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanstd(intensities[region]) #, ddof=0)
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region]) #, ddof=0)
def image_minmax(region, intensities):
    return np.nanmax(intensities[region]) - np.nanmin(intensities[region])  
def image_min(region, intensities):
    return np.nanmin(intensities[region])  

# with finding minima + watershed
wtssal_v, propscal_v, totars_v = [],[], []
for n in tqdm(range(len(ds_v))):
    wats, scaprop = [], []
    
    # halg = nangauss(difs_v[n] , 7)
    for i in range(len(difs_v[n])):
        dife = np.copy( difs_v[n][i] )
        dife[np.isnan(dife)] = -1
        # wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_v[n][i]) )
        wts = watershed( nangauss(dife,[5,10]), mask= dilation( ~np.isnan(difs_v[n][i]) ) )
        
        # dife = np.copy( halg[i] )
        # dife[np.isnan(dife)] = 1000
        # wts = watershed( dife , mask= ~np.isnan(difs_v[n][i]) )
        
        wats.append(wts)
        # scaprop.append( regionprops(wts, intensity_image= gaussian(difs_v[n][i],sigma=1) , \
        #                             extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        scaprop.append( regionprops(wts, intensity_image= gaussian(hints_v[n][i],sigma=1) , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
    xs,ys = xns_v[n], yns_v[n]
    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_v.append(area)
    wtssal_v.append(wats)
    propscal_v.append( scaprop )

wtssal_t, propscal_t, totars_t = [],[], []
for n in tqdm(range(len(ds_t))):
    wats, scaprop = [], []
    
    # halg = nangauss(difs_t[n] , 7)
    for i in range(len(difs_t[n])):
        dife = np.copy( difs_t[n][i] )
        dife[np.isnan(dife)] = -1
        # wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_t[n][i]) )
        wts = watershed( nangauss(dife,[5,10]), mask= dilation( ~np.isnan(difs_t[n][i]) ) )
        
        # dife = np.copy( halg[i] )
        # dife[np.isnan(dife)] = 1000
        # wts = watershed( dife , mask= ~np.isnan(difs_t[n][i]) )
        
        wats.append(wts)
        # scaprop.append( regionprops(wts, intensity_image= gaussian(difs_t[n][i],sigma=1) , \
        #                             extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )    
        scaprop.append( regionprops(wts, intensity_image= gaussian(hints_t[n][i],sigma=1) , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
    xs,ys = xns_t[n], yns_t[n]
    area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_t.append(area)
    wtssal_t.append(wats)
    propscal_t.append( scaprop )

#%%

nss_v = list(range(len(ds_v)))
lxs_v,lys_v = [],[]
sscas_v, centss_v, centws_v, nscas_v, nscafs_v, labs_v = [], [], [], [], [], []
ssds_v, smms_v, smes_v = [], [], []

for n in tqdm(nss_v):
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme = [], [], []

    scaprop = propscal_v[n]
    for i in range(len(scaprop)):
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        cenw = np.array( [ scaprop[i][j].centroid_weighted for j in range(len(scaprop[i])) ] )
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        slab = np.array( [ scaprop[i][j].label for j in range(len(scaprop[i])) ] )
        
        nsd = np.array( [ scaprop[i][j].image_nanstdev for j in range(len(scaprop[i])) ] )
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )

        me = np.array( [ countour_mean(wtssal_v,n,i,j+1,ve=True,dif=False) - scaprop[i][j].image_min for j in range(len(scaprop[i])) ] )

        nijs = np.array( [ (scaprop[i][j].moments_normalized).T for j in range(len(scaprop[i])) ] )

        bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
        hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        
        fil = (scas>1000) * (scas<18000) * ( sd>0.4 ) * ((scas/sd)<18000) #(scas>2000) * ( sd>0.3 )
        
        nsca.append( len(scaprop[i]) )
        nscaf.append( np.sum( fil ) )

        ssca.append(scas[fil])
        cents.append(cen[fil])
        centws.append(cenw[fil])
        labe.append(slab[fil])
        lx.append(bn[fil])
        ly.append(hn[fil])
        ssd.append(sd[fil])
        smm.append(mm[fil])
        sme.append(me[fil])
        
    sscas_v.append(ssca)
    centss_v.append(cents)
    centws_v.append(centws)
    labs_v.append(labe)
    nscas_v.append(nsca)
    nscafs_v.append(nscaf)
    ssds_v.append(ssd)
    smms_v.append(smm)
    smes_v.append(sme)
    lxs_v.append(lx)
    lys_v.append(ly)


nss_t = list(range(len(ds_t)))
lxs_t,lys_t = [],[]
sscas_t, centss_t, centws_t, nscas_t, nscafs_t, labs_t = [], [], [], [], [], []
ssds_t, smms_t, smes_t = [], [], []

for n in tqdm(nss_t):
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme = [], [], []

    scaprop = propscal_t[n]
    for i in range(len(scaprop)):
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        cenw = np.array( [ scaprop[i][j].centroid_weighted for j in range(len(scaprop[i])) ] )
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        slab = np.array( [ scaprop[i][j].label for j in range(len(scaprop[i])) ] )
        
        nsd = np.array( [ scaprop[i][j].image_nanstdev for j in range(len(scaprop[i])) ] )
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )
        
        me = np.array( [ countour_mean(wtssal_t,n,i,j+1,ve=False,dif=False) - scaprop[i][j].image_min for j in range(len(scaprop[i])) ] )

        nijs = np.array( [ (scaprop[i][j].moments_normalized).T for j in range(len(scaprop[i])) ] )

        bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
        hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        
        fil =(scas>1400) * ( sd>0.4 ) * ((scas/sd)<18000) # (scas>2000) * (sd>0.3)
        
        nsca.append( len(scaprop[i]) )
        nscaf.append( np.sum( fil ) )

        ssca.append(scas[fil])
        cents.append(cen[fil])
        centws.append(cenw[fil])
        labe.append(slab[fil])
        lx.append(bn[fil])
        ly.append(hn[fil])
        ssd.append(sd[fil])
        smm.append(mm[fil])
        sme.append(me[fil])
        
    sscas_t.append(ssca)
    centss_t.append(cents)
    centws_t.append(centws)
    labs_t.append(labe)
    nscas_t.append(nsca)
    nscafs_t.append(nscaf)
    ssds_t.append(ssd)
    smms_t.append(smm)
    smes_t.append(sme)
    lxs_t.append(lx)
    lys_t.append(ly)

#%%

n = 10 #8 #15 #23
i = 60

exp = 'v'
if exp == 'v': 
    wtssal_b = wtssal_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
    labs_b, sscas_b = labs_v, sscas_v
elif exp == 't': 
    wtssal_b = wtssal_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t
    labs_b, sscas_b = labs_t, sscas_t

sobb = thin(sobel(wtssal_b[n][i]) > 0)
soy,sox = np.where(sobb)

minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
dx = xns_b[n][0,1] - xns_b[n][0,0]
dy = yns_b[n][0,0] - yns_b[n][1,0]

fig,axs = plt.subplots(1,3, figsize=(12,5), sharey=True)

ims1 = axs[1].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
axs[1].plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
for j in range(len(sscas_b[n][i])):
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(lys_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(sscas_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(ssds_b[n][i][j]/1. ,2)) )
    pass

axs[2].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
axs[2].plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
mask = np.zeros_like(wtssal_b[n][i])
for j in range(len(labs_b[n][i])):
    mask += wtssal_b[n][i] == labs_b[n][i][j]
mask += np.isnan(difs_b[n][i])
amask = np.ma.masked_where(mask, mask)
axs[2].imshow( amask, extent=(minx,maxx,miny,maxy), alpha = 0.5 )


ims2 = axs[0].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy) )#, vmax=5) #, vmin=-5)

topy,boty = np.max( (yns_b[n])[~np.isnan(difs_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(difs_b[n][i])] )
topx,botx = np.max( (xns_b[n])[~np.isnan(difs_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(difs_b[n][i])] )
midx = np.mean( (xns_b[n])[~np.isnan(difs_b[n][i])] )
for j in range(3):
    axs[j].plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs[j].text(midx, boty-27, '5 cm')
    axs[j].axis('off')
    axs[j].set_ylim( top = topy + 5 )
    axs[j].set_xlim(botx-2,topx+2)
    pass


fig.subplots_adjust(left=0.2)
cbar_ax = fig.add_axes([0.18, 0.18, 0.015, 0.65])
fig.colorbar(ims2, cax=cbar_ax, label=r'$h$ (mm)', location='left')

# plt.tight_layout()
# plt.savefig('./Documents/Figs morpho draft/watershed_s15_t0.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()
#%%

n = 0

plt.figure()
for i in [30,40,50,60]:
# for i in [60,61,62,63]:
    plt.scatter( centss_v[n][i][:,1], centss_v[n][i][:,0], c=(i/70,0.5,1-i/70) )
    # plt.scatter( centws_v[n][i][:,1], centws_v[n][i][:,0], c=(i/70,0.5,1-i/70) )
plt.axis('equal')
plt.show()

#%%
# =============================================================================
# Wavelengths
# =============================================================================
mtim = -20
spread = True




cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$b)$',r'$b)$'],
                              [r'$a)$',r'$a)$',r'$c)$',r'$c)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharex=True)

# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([7,8,15]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx = [], []
    sly,slx = [], []
    for i in range(len(lxs_t[n])):
        mlx.append(np.nanmean(lxs_t[n][i]) * dx)
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        slx.append(np.nanstd(lxs_t[n][i]) * dx)
        sly.append(np.nanstd(lys_t[n][i]) * dy)
    mly,mlx = np.array(mly), np.array(mlx)
    sly,slx = np.array(sly), np.array(slx)
    
    ax[r'$a)$'].plot(ts_t[n]/60, mly , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
                     label=r'$S$ = '+str(salis_t[n])+' g/kg')
    ax[r'$a)$'].fill_between(ts_t[n]/60, mly-sly/2, mly+sly/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

    # ax[r'$a)$'].plot(ts_t[n]/60, mlx , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
    #                  label=str(salis_t[n])+'; '+str(ang_t[n]))
    # ax[r'$a)$'].fill_between(ts_t[n]/60, mlx-slx/2, mlx+slx/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$a)$'])
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].set_xlabel(r'$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$\lambda_y$ (mm)')
# ax[r'$a)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$a)$'].legend(loc='lower right')


li1ys,li1xs = [],[]
for l,n in enumerate(nss_v):
    dx = xns_v[l][0,1] - xns_v[l][0,0]
    dy = yns_v[l][0,0] - yns_v[l][1,0]
    if salis_v[l] > 7 and salis_v[l] < 25: 
        mly, mlx, mld = [], [], []
        sly, slx, sld = [], [], []
        for i in range(mtim,0):
            mlx += list(lxs_v[l][i] * dx )
            mly += list(lys_v[l][i] * dy )

        mey, eey = np.nanmean(mly), np.nanstd(mly) 
        li1y = ax[r'$b)$'].errorbar(salis_v[l], mey, yerr=eey, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+20)/71,0.5,1-(angys_v[l]+20)/71), mfc='w'  )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx), np.nanstd(mlx) 
        li1x = ax[r'$c)$'].errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+20)/71,0.5,1-(angys_v[l]+20)/71), mfc='w'  )        
        li1xs.append(li1x)

li2ys,li2xs = [],[]
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx, mld = [], [], []
    sly, slx, sld = [], [], []
    for i in range(mtim,0):
        mlx += list(lxs_t[n][i] * dx )
        mly += list(lys_t[n][i] * dy )

    if n in [32,33,35]:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='d', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='d', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
        li2xs.append(li2x)
    else:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
        li2xs.append(li2x)


ax[r'$b)$'].set_ylim(10.2,53)
ax[r'$c)$'].set_ylim(10.2,53)
ax[r'$b)$'].sharex(ax[r'$c)$'])
ax[r'$b)$'].tick_params(axis='x',length=3,labelsize=0)

ax[r'$b)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$c)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$c)$'].set_xlabel(r'$S$ (g/kg)')


ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
                    bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
# ax[r'$c)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(1.1,0.5), loc='upper center' , ncols=1 )

for labels,axs in ax.items():
    if labels == r'$a)$':
        axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
    else:
        axs.annotate(labels, (-0.16,0.91), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/wavelengths.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
def barviolin(data, ax, x = [0], height=1.0, width=1.0, bins=20, alpha=0.5, color=[] ):
    """
    data: list of data for histogram
    x: list of len(data). horizontal position of the data 
    height: float 0-1. Percentage of the height of the bar (1 full bar, 0 no bar)
    width: float. Width of the bar 
    """
    nda = len(data)
    for i in range(nda):
        daa = data[i]
        wid,bed = np.histogram(daa, bins=bins, density=True)
        bec, hei = (bed[1:] + bed[:-1])/2, bed[1:] - bed[:-1]
        lefts = x[i] - 0.5 * wid * width
        if len(color) < 1:
            ax.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha )
            # ax.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
            ax.plot( x[i], np.mean(daa), 'o', color='black', markersize=5 )
        else:
            ax.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha, color=color )
            # ax.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
            ax.plot( x[i], np.mean(daa), 'o', color='black', markersize=5 )
            
#%%
# =============================================================================
# Graph Area
# =============================================================================

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharex=True)

for l,n in enumerate([7,8,15]):
    if salis_t[n] > 6.0 and salis_t[n] < 25: 
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        dy = yns_t[n][0,0] - yns_t[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_t[n])):
            sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )

        # ax[r'$a)$'].errorbar(ts_t[n]/60, np.array(sarm)/100, yerr=np.array(sars)/100,  capsize=2, fmt='.-', errorevery=(l*2,20), \
        #               label=r'$S = $'+str(salis_t[n])+' g/kg' )#, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )
        ax[r'$a)$'].plot(ts_t[n]/60, np.array(sarm)/100, '.-', label=r'$S = $'+str(salis_t[n])+' g/kg', \
                       color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ) )
            
        ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(sarm)-np.array(sars)/2)/100, (np.array(sarm)+np.array(sars)/2)/100, \
                                 color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

ax[r'$a)$'].set_xlabel('$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$A_{scallop}$ (cm$^2$)')
# ax[r'$a)$'].legend(bbox_to_anchor=(0.5,1.15), loc='upper center' , ncol=3, columnspacing=0.5)
ax[r'$a)$'].legend(loc='upper left')

# li1ys = []
# for n in nss_v:
#     if salis_v[n] > 7 and salis_v[n] < 25: 
#         dx = xns_v[n][0,1] - xns_v[n][0,0]
#         dy = yns_v[n][0,0] - yns_v[n][1,0]
        
#         sarm,sars = [], []
#         # for i in range(len(sscas_v[n])):
#         #     sarm.append( np.nanmean(sscas_v[n][i] * dx*dy) )
#         #     sars.append( np.nanstd(sscas_v[n][i] * dx*dy) )
#         for i in range(-30,0):
#             sarm += list(sscas_v[n][i] * dx*dy) 
        
#         # li1y = ax[r'$b)$'].errorbar(salis_v[n], np.nanmean(sarm[-30:])/100, yerr=np.nanmean(sars[-30:])/100, capsize=2, fmt='o', \
#         #               markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
#         li2y = ax[r'$b)$'].errorbar(salis_v[n], np.nanmean(sarm)/100, yerr=np.nanstd(sarm)/100, capsize=2, fmt='o', \
#                       markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
#         li1ys.append(li1y)

# li2ys = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):

#     dx = xns_t[n][0,1] - xns_t[n][0,0]
#     dy = yns_t[n][0,0] - yns_t[n][1,0]
    
#     sarm,sars = [], []
#     # for i in range(len(sscas_t[n])):
#     #     sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
#     #     sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )
#     for i in range(-30,0):
#         sarm += list(sscas_t[n][i] * dx*dy) 
#         # sars.append( np.array(sscas_t[n][i] * dx*dy) )
    
#     # li2y = ax[r'$b)$'].errorbar(salis_t[n], np.nanmean(sarm[-30:])/100, yerr=np.nanmean(sars[-30:])/100, capsize=2, fmt='o', \
#     #               markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
#     li2y = ax[r'$b)$'].errorbar(salis_t[n], np.nanmean(sarm)/100, yerr=np.nanstd(sarm)/100, capsize=2, fmt='o', \
#                   markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
#     li2ys.append(li2y)

# ax[r'$b)$'].set_xlabel('Salinity (g/kg)')
# ax[r'$b)$'].set_ylabel(r'Area (cm$^2$)')
# ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.5,1.15), loc='upper center' , ncol=5, columnspacing=0.5 )

ns_v = np.array([3,9,11,12,13,15,2])
ns_t = np.array([7,8,15])

sarms_t = [] 
for n in ns_t:
    sarm = []
    for i in range(-20,0):
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        dy = yns_t[n][0,0] - yns_t[n][1,0]
        sarm +=  list( sscas_t[n][i] * dx*dy / 100 ) 
    sarms_t.append(sarm)

sarms_v = [] 
for n in ns_v:
    sarm = []
    for i in range(-20,0):
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        sarm +=  list( sscas_v[n][i] * dx*dy / 100 ) 
    sarms_v.append(sarm)

# plt.figure()
barviolin( sarms_v, ax[r'$b)$'], x=salis_v[ns_v], bins=30, width=10, color='blue') #, labe='Set 1' )
barviolin( sarms_t, ax[r'$b)$'], x=salis_t[ns_t], bins=30, width=10, color='red' ) #, labe='Set 2')
# plt.xticks(salis_v[ns_v])
# plt.xticks(salis_t[ns_t])
ax[r'$b)$'].set_ylim(0,25)
ax[r'$b)$'].set_xlim(6,24.5)
ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
ax[r'$b)$'].set_ylabel(r'$A_{scallop}$ (cm$^2$)')
    
for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/areas.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
# =============================================================================
# Graph amplitude
# =============================================================================
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharex=True)

for n in [7,8,15]:
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        # mimam.append( np.nanmean(smms_t[n][i]) )  #min-max
        # mimas.append( np.nanstd(smms_t[n][i]) )   #min-max
        mimam.append( np.nanmean(smes_t[n][i]) )  #mean to edge
        mimas.append( np.nanstd(smes_t[n][i]) )   #mean to edge
        # mimam.append( np.nanmean(ssds_t[n][i]) )  #std
        # mimas.append( np.nanstd(ssds_t[n][i]) )   #std
    
    # ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg', \
    #                      color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
    ax[r'$a)$'].plot(ts_t[n]/60, mimam,'.-', label=r'$S = $'+str(salis_t[n])+' g/kg', \
                         color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
        
    ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(mimam)-np.array(mimas)/2), (np.array(mimam)+np.array(mimas)/2), \
                             color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )


ax[r'$a)$'].set_xlabel(r'$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$H_{scallop}$ (mm)')
ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].legend(loc='upper left')

ave = 20
li1s = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(len(smms_v[n])):
            # mimam.append( np.nanmean(smms_v[n][i]) )
            # mimas.append( np.nanstd(smms_v[n][i]) )
            mimam.append( np.nanmean(smes_v[n][i]) )
            mimas.append( np.nanstd(smes_v[n][i]) )
            # mimam.append( np.nanmean(ssds_v[n][i]) )
            # mimas.append( np.nanstd(ssds_v[n][i]) )
        
        ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
        li1 = ax[r'$b)$'].errorbar(salis_v[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
        li1s.append(li1)
    
li2s = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        # mimam.append( np.nanmean(smms_t[n][i]) )
        # mimas.append( np.nanstd(smms_t[n][i]) )
        mimam.append( np.nanmean(smes_t[n][i]) )
        mimas.append( np.nanstd(smes_t[n][i]) )
        # mimam.append( np.nanmean(ssds_t[n][i]) )
        # mimas.append( np.nanstd(ssds_t[n][i]) )

    if n in [32,33,35]:
        ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='d', \
                     markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
    else:
        ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
    li2s.append(li2)
        
ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
ax[r'$b)$'].set_ylabel(r'$H_{scallop}$ (mm)')
# ax.set_ylim(bottom=0)

ax[r'$b)$'].legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing=0.5 )


for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/amplitudes.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()
#%%
def linear(x,a,b):
    return a*x + b

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharex=True)

for n in [7,8,15]:
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        # mimam.append( np.nanmean(smms_t[n][i]) )
        # mimas.append( np.nanstd(smms_t[n][i]) )
        mimam.append( np.nanmean(smes_t[n][i]) )
        mimas.append( np.nanstd(smes_t[n][i]) )
        # mimam.append( np.nanmean(ssds_t[n][i]) )
        # mimas.append( np.nanstd(ssds_t[n][i]) )
    
    ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg')

ax[r'$a)$'].set_xlabel('time (min)')
ax[r'$a)$'].set_ylabel('Amplitud (mm)')
ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].legend(loc='upper left')

ave = 40
li1s = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(len(smms_v[n])):
            mimam.append( np.nanmean(smms_v[n][i]) )
            mimas.append( np.nanstd(smms_v[n][i]) )
            # mimam.append( np.nanmean(smes_v[n][i]) )
            # mimas.append( np.nanstd(smes_v[n][i]) )
            # mimam.append( np.nanmean(ssds_v[n][i]) )
            # mimas.append( np.nanstd(ssds_v[n][i]) )
        
        lof = curve_fit(linear,ts_v[n][-ave:]/60, mimam[-ave:], sigma=mimas[-ave:], absolute_sigma=True)
        li1 = ax[r'$b)$'].errorbar(salis_v[n], lof[0][0], yerr = np.sqrt(lof[1][0,0]), capsize=2, fmt='o', \
                     markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
        
        # lres = linregress(ts_v[n][-ave:]/60, mimam[-ave:])        
        # li1 = ax[r'$b)$'].errorbar(salis_v[n], lres[0], yerr = lres[4], capsize=2, fmt='o', \
        #              markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
        li1s.append(li1)
    
li2s = []
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        mimam.append( np.nanmean(smms_t[n][i]) )
        mimas.append( np.nanstd(smms_t[n][i]) )
        # mimam.append( np.nanmean(smes_t[n][i]) )
        # mimas.append( np.nanstd(smes_t[n][i]) )
        # mimam.append( np.nanmean(ssds_t[n][i]) )
        # mimas.append( np.nanstd(ssds_t[n][i]) )

    lof = curve_fit(linear,ts_t[n][-ave:]/60, mimam[-ave:], sigma=mimas[-ave:], absolute_sigma=True)
    li2 = ax[r'$b)$'].errorbar(salis_t[n], lof[0][0], yerr = np.sqrt(lof[1][0,0]), capsize=2, fmt='o', \
                 markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
        
    # lres = linregress(ts_t[n][-ave:]/60, mimam[-ave:])        
    # li2 = ax[r'$b)$'].errorbar(salis_t[n], lres[0], yerr = lres[4], capsize=2, fmt='o', \
    #              markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
    li2s.append(li2)
        
ax[r'$b)$'].set_xlabel('Salinity (g/kg)')
ax[r'$b)$'].set_ylabel('Growth rate (mm/min)')
# ax.set_ylim(bottom=0)

ax[r'$b)$'].legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing=0.5 )


for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/amplitudes.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%
# =============================================================================
# Density figure
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmcrameri.cm as cmc


def density_fig(safi=False):
    """ Show density as function of T and S """
    CMAP = ListedColormap([cmc.devon_r(val) for val in np.linspace(0, .8, 250)])
    contour_levels = [1000,1005,1010,1015,1020,1025,1030]
    MANUAL_CLABELS = [(x, -0.5 * x + 25.5) for x in np.linspace(5, 40.1, len(contour_levels))]

    min_t, min_sal = -5, 0
    max_t, max_sal = 30.1, 40.1
    t, sal = np.meshgrid(np.arange(min_t, max_t, .05), np.arange(min_sal, max_sal, .05))
    rho = density_millero(t, sal)
    t_fr = freezing_temperature(sal)

    plt.figure()
    plt.imshow(np.flipud(rho.T), extent=(np.min(sal), np.max(sal), np.min(t), np.max(t)), cmap=CMAP, aspect='auto', vmin=990, vmax=1030)
    plt.xlabel('Salinity (g/kg)', fontsize=12)
    plt.ylabel('Temperature ($\degree$C)', fontsize=12)
    plt.tick_params(labelsize=12)
    cb = plt.colorbar() #extend='both')
    cb.ax.tick_params(labelsize=12)
    # cb.ax.set_title(r"$\rho$ (kg/m$^3$)", fontsize=12)
    plt.title(r'Density (kg/m$^3$)', fontsize=12)

    t1 = plt.Polygon(np.array([[np.min(sal), np.min(t)], [np.min(sal), 0], [np.max(sal), np.min(t_fr)], [np.max(sal), np.min(t)]]), color=[.6, .6, .6])
    plt.gca().add_patch(t1)

    crho = rho + 0.0
    crho[t < t_fr] = np.nan

    plt.plot([np.min(sal), np.max(sal)], [0, 0], '--k', lw=1)
    cont = plt.contour(np.flipud(sal.T), np.flipud(t.T), np.flipud(crho.T), levels=contour_levels, colors='black',\
                       linewidths=[2, 1] + [1 for _ in range(len(contour_levels)-2)])
    plt.clabel(cont, cont.levels, manual=MANUAL_CLABELS, fmt=lambda x: "{:.0f}".format(x), fontsize=10)


    # maximum density
    s_col = sal[:, 0]
    s_col = s_col[s_col < 26.25]
    plt.plot(s_col[s_col < 7.6], 4 - 0.216 * s_col[s_col < 7.6], color=(.3, .3, .3), lw=1.5)
    plt.plot(s_col[s_col > 10], 4 - 0.216 * s_col[s_col > 10], color=(.3, .3, .3), lw=1.5)
    plt.text(8.0, 1.5, 'T$*$', color=(.3, .3, .3), fontsize=11)

    # freezing temperature
    plt.plot(sal[:, 0], t_fr[:, 0], '-', color=(.2, .2, .2), lw=2)
    plt.text(15, -3, 'T$_{fp}$', color=(.2, .2, .2), fontsize=11)
    plt.text(2, -4, 'Ice', color='k', fontsize=12)

    # if safi: plt.savefig('./Documents/densityst.png', dpi=400, bbox_inches='tight', transparent=True)
    plt.show()


def density_millero(t, s):
    """
    Computes density of seawater in kg/m^3.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    return rho_sw


def freezing_temperature(sal):
    Kf = 1.86  # [K.kg/mol] cryoscopic constant of water
    M = 58.44  # [g/mol] molar mass of NaCl
    i = 2      # [-] Van 't Hoff constant of NaCl
    return 0.0 - Kf*i*sal/M  # Bagden's law for freezing depression


# density_fig()
density_millero(0, 0), density_millero(19, 0), density_millero(0, 1)

#%%
# =============================================================================
# 3D graphs
# =============================================================================
n = 9 # 3,16,6,8,12
i = 60

# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan

# halt = np.load('./Documents/Height profiles/ice_block_0_'+sal_v[n]+'.npy')
# halg = nangauss(halt, 2)

halt = np.load('./Documents/Height profiles/profile_s'+sal_t[n]+'_t'+inc_t[n]+'.npy')
halg = nangauss(halt, 2)

nt,ny,nx = np.shape(halt)

x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_t[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_t[n]
# x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_v[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_v[n]
t = np.arange(nt) * 30
tr,yr,xr = np.meshgrid(t,y,x, indexing='ij')

# xrp, yrp = xr - halg/Ls_v[n] * xr, yr - halg/Ls_v[n] * yr
xrp, yrp = xr - halg/Ls_t[n] * xr, yr - halg/Ls_t[n] * yr
# xrp, yrp = xr - halt/Ls_c[n] * xr, yr - halt_v/Ls[n] * yr



#%%

# with h5py.File('./Documents/profile_s20_t0.hdf5', 'w') as f:
#     # dt = h5py.special_dtype(vlen=np.dtype('float64'))
#     ghj = halt[60].flatten()
#     hh = f.create_dataset('h', len(ghj)) #array with all untilted slopes in regurlar grid
#     hh = ghj
#     # f.create_dataset?


# with h5py.File('./Documents/profile_s20_t0.hdf5', 'r') as f:
#     fff = f['h']
#     print(fff)



# np.shape(halt)

plt.figure()
# plt.imshow(halt[60])
plt.imshow(hints_t[n][60])
plt.colorbar()
plt.show()

# vvv = np.where(halt[60] > -10000)

# print( np.min(vvv[0]),np.max(vvv[0])) 
# print( np.min(vvv[1]),np.max(vvv[1])) 

#%%
i = 60
hice = halg[i]

az,al = 0 , 90
ver,fra = 0.1 , 100.

blue = np.array([1., 1., 1.])
rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

fig = plt.figure(figsize=(500/192 * 2,400/192 * 2))
ax = plt.axes(projection='3d')

ls = LightSource()
illuminated_surface = ls.shade_rgb(rgb, hice)

ax.plot_surface(xrp[i], hice , yrp[i], ccount=300, rcount=300, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
                antialiased=True,
                facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')
# ax.plot_surface(xns[n], hice , yns[n], ccount=300, rcount=300, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
#                 antialiased=True,
#                 facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

yli, zli = 100 * np.cos(angys_t[n]*np.pi/180), 100 * np.sin(angys_t[n]*np.pi/180)
# ax.plot( [-180,-180], [0-140,zli-140], [-yli/2,yli/2], 'k-' )

ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_zaxis()
ax.invert_xaxis()

lxd, lxi = np.nanmax(xrp), np.nanmin(xrp)
lzd, lzi = np.nanmin(yrp), np.nanmax(yrp)
lyi, lyd = np.nanmin(halg), np.nanmax(halg)

print( lzd,lzi )
print( lxd,lxi )
print( lyd,lyi )

# ax.set_box_aspect([2,1.6,4])
# ax.set_zlim(-180,140)
# ax.set_xlim(110,-70)
# ax.set_ylim(-60,0),.

# ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.25)
ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.05)
ax.set_zlim(lzd-5,lzi+5)
ax.set_xlim(lxd+5,lxi-5)
ax.set_ylim(lyi-5,lyd+5)

plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=3)
# ax.text(30,40,180, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title( 't = '+str(0.5*n)+'s', fontsize=12)

# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
# ax.view_init(17,120)
ax.view_init(5,120)

# ax.text(-21,60,240, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title('t = '+str(0.5*n)+'min', fontsize=12)

plt.axis('off')
# plt.savefig('./Documents/blok27.png',dpi=192 * 5, transparent=True , bbox_inches='tight')

plt.show()
#%%
i = 60
hice = halg[i]

az,al = 0 , 90
ver,fra = 0.1 , 100.

blue = np.array([1., 1., 1.])
rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

# plt.close('all')
# plt.ioff()

# fig = plt.figure(figsize=(500/192 * 2,400/192 * 2))
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

ls = LightSource()
illuminated_surface = ls.shade_rgb(rgb, hice)

ax.plot_surface(xrp[i], hice , yrp[i], ccount=300, rcount=300, # * mkf[i], yrts[n][i], ccount=300, rcount=300,
                antialiased=True,
                facecolors=illuminated_surface, label='t = '+str(0.5*i)+'seg')

ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_zaxis()
ax.invert_xaxis()

lxd, lxi = np.nanmax(xrp), np.nanmin(xrp)
lzd, lzi = np.nanmin(yrp), np.nanmax(yrp)
lyi, lyd = np.nanmin(halg), np.nanmax(halg)

ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=0.95)
ax.set_zlim(lzd-5,lzi+5)
ax.set_xlim(lxd+5,lxi-5)
ax.set_ylim(lyi-5,lyd+5)

# plt.locator_params(axis='y', nbins=1)
# plt.locator_params(axis='x', nbins=6, integer=True, steps=[1, 2, 10])
ax.xaxis.set_ticks([-50,0,50])
ax.yaxis.set_ticks([-20,-70])

ax.view_init(17,120)

ax.text(0,60,180, 't = '+str(0.5*i)+'min', fontsize=15)
# ax.set_title('t = '+str(0.5*i)+'min', fontsize=13)

# plt.savefig('./Documents/imgifi.png',dpi=400) #, bbox_inches='tight')


plt.show()
# lista_im.append(imageio.imread('imgifi.png')[:,:,0])
    
#%%
lista_im = []
# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan
for i in tqdm(range(len(halg))):
# for i in tqdm(range(5)):
# for n in [10]:
    hice = halg[i]

    az,al = 0 , 90
    ver,fra = 0.1 , 100.

    blue = np.array([1., 1., 1.])
    rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

    plt.close('all')
    plt.ioff()

    # fig = plt.figure(figsize=(500/192 * 2,400/192 * 2))
    fig = plt.figure(figsize=(10,7))
    fig.canvas.draw()
    ax = plt.axes(projection='3d')

    ls = LightSource()
    illuminated_surface = ls.shade_rgb(rgb, hice)

    ax.plot_surface(xrp[i], hice , yrp[i], ccount=300, rcount=300, 
                    antialiased=True,
                    facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    # ax.invert_zaxis()
    ax.invert_xaxis()

    lxd, lxi = np.nanmax(xrp), np.nanmin(xrp)
    lzd, lzi = np.nanmin(yrp), np.nanmax(yrp)
    lyi, lyd = np.nanmin(halg), np.nanmax(halg)

    ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=0.95)
    ax.set_zlim(lzd-5,lzi+5)
    ax.set_xlim(lxd+5,lxi-5)
    ax.set_ylim(lyi-5,lyd+5)

    # plt.locator_params(axis='y', nbins=2)
    # plt.locator_params(axis='x', nbins=3)
    ax.xaxis.set_ticks([-50,0,50])
    ax.yaxis.set_ticks([-20,-70])
    
    ax.view_init(17,120)
    plt.axis('off')
    
    # ax.text(0,60,180, 't = '+str(0.5*i)+'min', fontsize=15)
    # # ax.text(-101,60,200, 't = '+str(0.5*n)+'min', fontsize=15)
    
    # # plt.savefig('imgifi'+str(n)+'.jpg',dpi=400, transparent=True) #, bbox_inches='tight')
    # # lista_im.append(imageio.imread('imgifi.png') )#[:,:,0])
    # # lista_im.append(  Image.open('imgifi'+str(n)+'.jpg') )#[:,:,0])
    # # lista_im.append(  Image.frombytes('RGB', fig.canvas.get_width_height()[::-1],fig.canvas.tostring_rgb()) ) 
    # # lista_im.append(  Image.frombytes('RGBa', fig.canvas.get_width_height(),fig.canvas.buffer_rgba()) ) 
    
    # # lst = list(fig.canvas.get_width_height())[::-1]
    # # lst.append(3)
    # # imi = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(lst) 
    # # mim = Image.fromarray( imi )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, transparent=True)    
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    imi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # mim = Image.fromarray( imi[70:1350,640:1500] )
    mim = Image.fromarray( imi[350:1050,670:1350] )
    lista_im.append( mim )
    
# imageio.mimsave('./Documents/ice_s0_t0r.gif', lista_im, fps=10, format='gif')
# plt.close('all')

# plt.figure()
# plt.imshow(lista_im[0])
# plt.show()

frame_one = lista_im[0] #lista_im[0]
frame_one.save("./Documents/ice_s0_t45.gif", format="GIF", append_images=lista_im[1:], save_all=True, duration=5, loop=0, transparent=True)
#%%

plt.figure()
plt.imshow( frame_one )
plt.show()

#%%
# =============================================================================
# Temperature and salinity
# =============================================================================
sal = ['0','0','0','0','6','6','6','6','12','12','12','12','20','20','20','20','27','27','27','27','0','0','6','12','20','35','35','20','12','6',
       '35','0']
inc = ['0(3)','30','15(3)','45','45','30','15','0','0','15','45','30','30','45','15','0','0','15','30','45','15(4)','15(5)',
       '45(2)','0(2)','30(2)','30','n15','n15','n15','n15','n15(2)','n15']
pos = ['dd','md','mu','uu']

startimes = [23,34,10,33,22,23,14,15,10,12,16,16,14,26,18,15,21,32,36,274, 37,29,27,27,3,27,18,20,13,43,27,13]
times,temps,ssali= [],[],[]
for n in tqdm(range(len(sal))):    
    file = '/Volumes/Ice blocks/s'+sal[n]+'_t'+inc[n]+'/ice_s'+sal[n]+'_t'+inc[n]+'_dd.csv'
    dats1 = np.genfromtxt(file, delimiter=';', skip_header=1)
    file = '/Volumes/Ice blocks/s'+sal[n]+'_t'+inc[n]+'/ice_s'+sal[n]+'_t'+inc[n]+'_md.csv'
    dats2 = np.genfromtxt(file, delimiter=';', skip_header=1)
    file = '/Volumes/Ice blocks/s'+sal[n]+'_t'+inc[n]+'/ice_s'+sal[n]+'_t'+inc[n]+'_mu.csv'
    dats3 = np.genfromtxt(file, delimiter=';', skip_header=1)
    try:
        file = '/Volumes/Ice blocks/s'+sal[n]+'_t'+inc[n]+'/ice_s'+sal[n]+'_t'+inc[n]+'_uu.csv'
        dats4 = np.genfromtxt(file, delimiter=';', skip_header=1)
    except FileNotFoundError:
        dats4 = dats3
    # file = '/Volumes/Ice blocks/s'+sal[n]+'_t'+inc[n]+'/ice_s'+sal[n]+'_t'+inc[n]+'_uu.csv'
    # dats4 = np.genfromtxt(file, delimiter=';', skip_header=1)
    
    lt = min(len(dats1),len(dats2),len(dats3),len(dats4))
    time = ( np.vstack( (dats1[:lt,0],dats2[:lt,0],dats3[:lt,0],dats4[:lt,0]) ) - dats1[startimes[n],0] ) * 100000/60
    # time = ( np.vstack( (dats1[:lt,0],dats2[:lt,0],dats3[:lt,0],dats4[:lt,0]) ) - dats1[0,0] ) * 100000/60
    temp = np.vstack( (dats1[:lt,1],dats2[:lt,1],dats3[:lt,1],dats4[:lt,1]) )
    sali = np.vstack( (dats1[:lt,2],dats2[:lt,2],dats3[:lt,2],dats4[:lt,2]) )
    
    times.append(time)
    temps.append(temp)
    ssali.append(sali)

#%%
n = 0
plt.figure()
for i in range(4):
    plt.plot(times[n][i], temps[n][i],'.-',label=pos[i])
    # plt.plot(np.arange(len(times[n][i])), temps[n][i],'.-',label=pos[i])
plt.grid()
plt.legend()
plt.show()


plt.figure()
for i in range(4):
    plt.plot(times[n][i], ssali[n][i],'.-',label=pos[i])
plt.grid()
plt.legend()
plt.show()

#%%
i = 2
ns = [12,24]
starttime = [23,34,10,33,22,23,14,15,10,12,16,16,14,26,18,15,21,32,36,274, 37,29,27,27,3,27,18,20,13,43,27,13]

plt.figure()
for n in ns:
    # plt.plot(times[n][i] - (times[n][i])[starttime[n]], temps[n][i] - temps[n][i][starttime[n]-2]*1,'.-',label=ang[n])
    plt.plot(times[n][i] - (times[n][i])[starttime[n]], temps[n][i],'.-',label=ang[n])
    # plt.plot( temps[n][i],'.-',label=ang[n])
plt.grid()
plt.legend()
plt.show()
#%%
#%%
# =============================================================================
# Pruebas con curvature
# =============================================================================

i = 60
n = 8

exp = 't'
if exp == 'v': 
    wtssal_b = wtssal_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
    hints_b = hints_v
    labs_b, sscas_b = labs_v, sscas_v
elif exp == 't': 
    wtssal_b = wtssal_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t
    hints_b = hints_t
    labs_b, sscas_b = labs_t, sscas_t


t1 = time()

gghi = nangauss(hints_b[n],5)
gt,gy,gx = np.gradient( gghi , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
_,gyy,gyx = np.gradient( gy , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
_,gxy,gxx = np.gradient( gx , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
flp, fln = kurv>0, kurv<0 

# im_wat = gaussian( kurv * flp, 10 ) + (kurv * fln)
im_wat = nangauss( kurv * flp, [0,7,20] ) + (kurv * fln)
# im_wat = nangauss( kurv * flp, [5,7,10] ) + (kurv * fln)

t2 = time()

# dife = np.copy( difs_v[n][i] )
# dife[np.isnan(dife)] = -1
# wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_v[n][i]) )
# wtsn = watershed( -gaussian(kurv[i] * fln[i],0), mask = dilation( ~np.isnan(kurv[i]) ) )
wtsn = watershed( -im_wat[i], mask = binary_erosion( ~np.isnan(kurv[i]), disk(10) ) )
# wtsn = watershed( -im_wat[i], mask = dilation( ~np.isnan(kurv[i]) ) )

t3 = time()
print(t2-t1, t3-t2)
#%%
minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
dx = xns_b[n][0,1] - xns_b[n][0,0]
dy = yns_b[n][0,0] - yns_b[n][1,0]

# plt.figure()
# plt.imshow(gghi[i], aspect= dy/dx )
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow(difs_b[n][i], aspect= dy/dx )
# plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(kurv[i], aspect= dy/dx )
plt.colorbar()
plt.show()


sobb = thin(sobel(wtsn) > 0)
soy,sox = np.where(sobb)

plt.figure()
# plt.imshow( wtsn )
plt.imshow( kurv[i], extent=(minx,maxx,miny,maxy))
# plt.imshow( difs_b[n][i], extent=(minx,maxx,miny,maxy))
plt.plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.show()


# sobb = thin(sobel(wtssal_b[n][i]) > 0)
# soy,sox = np.where(sobb)

# minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
# miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
# dx = xns_b[n][0,1] - xns_b[n][0,0]
# dy = yns_b[n][0,0] - yns_b[n][1,0]

plt.figure()
plt.imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
# plt.plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.show()

#%%
flp, fln = kurv>0, kurv<0 

plt.figure()
plt.imshow( im_wat[i] )
# plt.imshow( gaussian( kurv[i] * flp[i], 0 ) )
# plt.imshow( gaussian( kurv[i] * flp[i], 10 ) + (kurv[i] * fln[i]) )
plt.colorbar()
plt.show( )

plt.figure()
plt.imshow( kurv[i] )
plt.colorbar()
plt.show( )

#%%
# n = 0 #14,13, 12
i = 50
for n in range(len(ds_v)):
    plt.figure()
    plt.imshow( difs_v[n][i] )
    plt.show()


#%%
# =============================================================================
# Watershed gaussian curvature
# =============================================================================

def countour_mean(wtssal_b,n,i,labe, ve=True, dif=True):

    sccaa = wtssal_b[n][i] == labe
    ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )

    if ve:    
        if dif: conval = difs_v[n][i][ccoo]
        else: conval = hints_v[n][i][ccoo]
    else:    
        if dif: conval = difs_t[n][i][ccoo]
        else: conval = hints_t[n][i][ccoo]
    
    return np.nanmean(conval)

def image_nanstdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanstd(intensities[region]) #, ddof=0)
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region]) #, ddof=0)
def image_minmax(region, intensities):
    return np.nanmax(intensities[region]) - np.nanmin(intensities[region])  
def image_min(region, intensities):
    return np.nanmin(intensities[region])  

# s1,s2,s3 = [7,10,15] #5,7,20
# s1,s2,s3 = [5,7,20] #5,7,20
s1,s2,s3 = [6,8,17] #5,7,20

wtssal_v, propscal_v, totars_v, kpropscal_v = [],[], [], []
for n in tqdm(range(len(ds_v))):
    if n not in [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]:
        totars_v.append(np.nan)
        wtssal_v.append(np.nan)
        propscal_v.append( np.nan )
        kpropscal_v.append( np.nan )
        continue
        
    
    wats, scaprop, scapropk = [], [], []
    
    # halg = nangauss(hints_v[n],s1)
    halg = nangauss(difs_v[n],s1)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    flp, fln = kurv>0, kurv<0 

    im_wat = nangauss( kurv * flp, [0,s2,s3] ) + (kurv * fln)
    
    for i in range(len(difs_v[n])):
        # wts = watershed( -im_wat[i], mask = binary_erosion( ~np.isnan(kurv[i]), disk(10) ) )
        wts = watershed( -im_wat[i], mask = binary_dilation( ~np.isnan(kurv[i]), disk(1) ) )
        
        wats.append(wts)
        # scaprop.append( regionprops(wts, intensity_image= gaussian(difs_v[n][i],sigma=1) , \
        #                             extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        scaprop.append( regionprops(wts, intensity_image= gaussian(hints_v[n][i],sigma=1) , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
        scapropk.append( regionprops(wts, intensity_image= kurv[i] , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
    xs,ys = xns_v[n], yns_v[n]
    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_v.append(area)
    wtssal_v.append(wats)
    propscal_v.append( scaprop )
    kpropscal_v.append( scapropk )
    
# s1,s2,s3 = [7,10,15] #5,7,20
# s1,s2,s3 = [7,10,15] #5,7,20

wtssal_t, propscal_t, totars_t, kpropscal_t = [],[], [], []
for n in tqdm(range(len(ds_t))):
    # if n not in [ 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 35, 38]:
    if n not in [4, 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 34, 35, 38]:
        totars_t.append(np.nan)
        wtssal_t.append(np.nan)
        propscal_t.append( np.nan )
        kpropscal_t.append( np.nan )
        continue

    wats, scaprop, scapropk = [], [], []
    
    # halg = nangauss(hints_t[n],s1)
    halg = nangauss(difs_t[n],s1)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    flp, fln = kurv>0, kurv<0 

    im_wat = nangauss( kurv * flp, [0,s2,s3] ) + (kurv * fln)
    
    for i in range(len(difs_t[n])):
        # wts = watershed( -im_wat[i], mask = binary_erosion( ~np.isnan(kurv[i]), disk(10) ) )
        wts = watershed( -im_wat[i], mask = binary_dilation( ~np.isnan(kurv[i]), disk(1) ) )
        
        wats.append(wts)
        # scaprop.append( regionprops(wts, intensity_image= gaussian(difs_t[n][i],sigma=1) , \
        #                             extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )    
        scaprop.append( regionprops(wts, intensity_image= gaussian(hints_t[n][i],sigma=1) , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
        scapropk.append( regionprops(wts, intensity_image= kurv[i] , \
                                    extra_properties=[image_stdev, image_nanstdev, image_minmax, image_min]) )
    
    xs,ys = xns_t[n], yns_t[n]
    area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_t.append(area)
    wtssal_t.append(wats)
    propscal_t.append( scaprop )
    kpropscal_t.append( scapropk )

#%%

for n in tqdm(nss_v):
    if n not in [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]:
        continue
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]
    print(f"{dx:.3f},\t{dy:.3f},\t{dx*dy:.3f},\t{1000*dx*dy:.1f},\t{12000*dx*dy:.1f}")
print()
for n in tqdm(nss_t):
    if n not in [4, 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 34, 35, 38]:
        continue
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    print(f"{dx:.3f},\t{dy:.3f},\t{dx*dy:.3f},\t{1000*dx*dy:.1f},\t{12000*dx*dy:.1f}")

#%%
rrho_v = np.abs(density_millero(temp_v, 0) - density_millero(temp_v, salis_v)) / np.abs(density_millero(temp_v, salis_v) - density_millero(0, salis_v)) 
rrho_t = np.abs(density_millero(temp_t, 0) - density_millero(temp_t, salis_t)) / np.abs(density_millero(temp_t, salis_t) - density_millero(0, salis_t))

nss_v = list(range(len(ds_v)))
lxs_v,lys_v = [],[]
sscas_v, centss_v, centws_v, nscas_v, nscafs_v, labs_v = [], [], [], [], [], []
ssds_v, smms_v, smes_v = [], [], []

for n in tqdm(nss_v):
    if n not in [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]:
        sscas_v.append( np.nan )
        centss_v.append( np.nan )
        centws_v.append( np.nan )
        labs_v.append( np.nan )
        nscas_v.append( np.nan )
        nscafs_v.append( np.nan )
        ssds_v.append( np.nan )
        smms_v.append( np.nan )
        smes_v.append( np.nan )
        lxs_v.append( np.nan )
        lys_v.append( np.nan )
        continue
    
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme = [], [], []
    
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]

    scaprop = propscal_v[n]
    scapropk = kpropscal_v[n]
    for i in range(len(scaprop)):
        cen, cenw, scas, slab = [], [], [], []
        nsd, sd, mm = [], [], []
        me, nijs = [], []
        
        for j in range(len(scaprop[i])):
            ksd = scapropk[i][j].image_stdev
            # sarea = scaprop[i][j].area
            # if sarea > 1000 and sarea < 12000 and ksd > 0.025: #0.03 o 0.036
            sarea = scaprop[i][j].area * dx*dy
            # if sarea > 100 and sarea < 20000 and ksd > 0.025: #0.03 o 0.036
            if sarea > 100 and ksd > 0.025: #0.03 o 0.036
                cen.append( scaprop[i][j].centroid )
                cenw.append( scaprop[i][j].centroid_weighted )
                scas.append( sarea /dx/dy )
                slab.append( scaprop[i][j].label )
                
                nsd.append( scaprop[i][j].image_nanstdev )
                sd.append( scapropk[i][j].image_stdev )  #scaprop[i][j].image_stdev )
                mm.append( scaprop[i][j].image_minmax )
                
                me.append( countour_mean(wtssal_v,n,i,j+1,ve=True,dif=False) - scaprop[i][j].image_min )
                nijs.append( (scaprop[i][j].moments_normalized).T )

        cen, cenw, scas, slab = np.array(cen), np.array(cenw), np.array(scas), np.array(slab)
        nsd, sd, mm = np.array(nsd), np.array(sd), np.array(mm)
        me, nijs = np.array(me), np.array(nijs)
        
        if len(nijs) > 0:
            bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
            hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        else:
            be,hn = np.nan, np.nan
        
        # fil = scas>1 #(scas>1000) * (scas<18000) * ( sd>0.4 ) * ((scas/sd)<18000) #(scas>2000) * ( sd>0.3 )
        
        nsca.append( len(cen) )
        # nscaf.append( np.sum( fil ) )

        ssca.append(scas )
        cents.append(cen )
        centws.append(cenw )
        labe.append(slab )
        lx.append(bn )
        ly.append(hn )
        ssd.append(sd )
        smm.append(mm )
        sme.append(me)
        
    sscas_v.append(ssca)
    centss_v.append(cents)
    centws_v.append(centws)
    labs_v.append(labe)
    nscas_v.append(nsca)
    nscafs_v.append(nscaf)
    ssds_v.append(ssd)
    smms_v.append(smm)
    smes_v.append(sme)
    lxs_v.append(lx)
    lys_v.append(ly)


nss_t = list(range(len(ds_t)))
lxs_t,lys_t = [],[]
sscas_t, centss_t, centws_t, nscas_t, nscafs_t, labs_t = [], [], [], [], [], []
ssds_t, smms_t, smes_t = [], [], []

for n in tqdm(nss_t):
    if n not in [4, 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 34, 35, 38]:
        sscas_t.append( np.nan )
        centss_t.append( np.nan )
        centws_t.append( np.nan )
        labs_t.append( np.nan )
        nscas_t.append( np.nan )
        nscafs_t.append( np.nan )
        ssds_t.append( np.nan )
        smms_t.append( np.nan )
        smes_t.append( np.nan )
        lxs_t.append( np.nan )
        lys_t.append( np.nan )
        continue
    
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme = [], [], []

    scaprop = propscal_t[n]
    scapropk = kpropscal_t[n]
    for i in range(len(scaprop)):
        cen, cenw, scas, slab = [], [], [], []
        nsd, sd, mm = [], [], []
        me, nijs = [], []
        
        for j in range(len(scaprop[i])):
            ksd = scapropk[i][j].image_stdev
            # sarea = scaprop[i][j].area
            # if sarea > 1000 and sarea < 12000 and ksd > 0.025: #0.036
            sarea = scaprop[i][j].area * dx*dy
            # if sarea > 100 and sarea < 20000 and ksd > 0.025: #0.03 o 0.036
            if sarea > 100 and ksd > 0.025: #0.03 o 0.036
                cen.append( scaprop[i][j].centroid )
                cenw.append( scaprop[i][j].centroid_weighted )
                scas.append( sarea /dx/dy )
                slab.append( scaprop[i][j].label )
                
                nsd.append( scaprop[i][j].image_nanstdev )
                sd.append( scapropk[i][j].image_stdev )  #scaprop[i][j].image_stdev )
                mm.append( scaprop[i][j].image_minmax )
                
                me.append( countour_mean(wtssal_t,n,i,j+1,ve=False,dif=False) - scaprop[i][j].image_min )
                nijs.append( (scaprop[i][j].moments_normalized).T )
                
        cen, cenw, scas, slab = np.array(cen), np.array(cenw), np.array(scas), np.array(slab)
        nsd, sd, mm = np.array(nsd), np.array(sd), np.array(mm)
        me, nijs = np.array(me), np.array(nijs)
                
        if len(nijs) > 0:
            bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
            hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        else:
            be,hn = np.nan, np.nan

        nsca.append( len(scaprop[i]) )
        # nscaf.append( np.sum( fil ) )

        ssca.append(scas )
        cents.append(cen )
        centws.append(cenw )
        labe.append(slab )
        lx.append(bn )
        ly.append(hn )
        ssd.append(sd )
        smm.append(mm )
        sme.append(me )
        
    sscas_t.append(ssca)
    centss_t.append(cents)
    centws_t.append(centws)
    labs_t.append(labe)
    nscas_t.append(nsca)
    nscafs_t.append(nscaf)
    ssds_t.append(ssd)
    smms_t.append(smm)
    smes_t.append(sme)
    lxs_t.append(lx)
    lys_t.append(ly)

#%%
# [ 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 35, 38] for t
n = 8 #8 #15 #23
i = 60

# print(salis_t[n], angys_t[n], len(hints_t[n]))

plt.rcParams.update({'font.size':14})

cuv = True
exp = 'v'
if exp == 'v':
    hints_b = hints_v
    wtssal_b = wtssal_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
    labs_b, sscas_b = labs_v, sscas_v
elif exp == 't': 
    hints_b = hints_t
    wtssal_b = wtssal_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t
    labs_b, sscas_b = labs_t, sscas_t

numim = 3
if cuv:
    halg = nangauss(hints_b[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    numim = 4


sobb = thin( sobel(wtssal_b[n][i]) > 0)
soy,sox = np.where(sobb)

minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
dx = xns_b[n][0,1] - xns_b[n][0,0]
dy = yns_b[n][0,0] - yns_b[n][1,0]

# fig,axs = plt.subplots(1,numim, figsize=(12,5), sharey=True)
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$',r'$d)$'],
                              [r'$a)$',r'$b)$',r'$c)$',r'$d)$']], figsize=(12/1.,5/1.), sharey=True, #layout='constrained',
                             gridspec_kw={
                            "bottom": 0.02,
                            "top": 1.0,
                            "left": 0.05,
                            "right": 0.945,
                            "wspace": -0.08,
                            "hspace": 0.0})

ims1 = ax[r'$b)$'].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
ax[r'$b)$'].plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
for j in range(len(sscas_b[n][i])):
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(lys_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(sscas_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(ssds_b[n][i][j]/1. ,2)) )
    pass

ax[r'$c)$'].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
ax[r'$c)$'].plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
mask = np.zeros_like(wtssal_b[n][i])
for j in range(len(labs_b[n][i])):
    mask += wtssal_b[n][i] == labs_b[n][i][j]
mask += np.isnan(difs_b[n][i])
amask = np.ma.masked_where(mask, mask)
ax[r'$c)$'].imshow( amask, extent=(minx,maxx,miny,maxy), alpha = 0.6 )


ims2 = ax[r'$a)$'].imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy) )#, vmax=5) #, vmin=-5)

topy,boty = np.max( (yns_b[n])[~np.isnan(difs_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(difs_b[n][i])] )
topx,botx = np.max( (xns_b[n])[~np.isnan(difs_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(difs_b[n][i])] )
midx = np.mean( (xns_b[n])[~np.isnan(difs_b[n][i])] )
# for j in range(3):
for labels,axs in ax.items():
    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-23, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    pass


fig.subplots_adjust(left=0.2)
cbar_ax = fig.add_axes([0.05, 0.1, 0.015, 0.85])
fig.colorbar(ims2, cax=cbar_ax, label=r'$h$ (mm)', location='left')

if cuv:
    imsk = ax[r'$d)$'].imshow(kurv[i], extent=(minx,maxx,miny,maxy), cmap='gray')
    ax[r'$d)$'].plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    # ax[r'$d)$'].text(midx, boty-27, '5 cm')
    ax[r'$d)$'].axis('off')
    ax[r'$d)$'].set_ylim( top = topy + 5 )
    ax[r'$d)$'].set_xlim(botx-2,topx+2)
    cbar_axk = fig.add_axes([0.925, 0.1, 0.015, 0.85])
    fig.colorbar(imsk, cax=cbar_axk, label=r'$k$ (mm$^{-1}$)', location='right')

for labels,axs in ax.items():
    axs.annotate(labels, (-0.03,0.97), xycoords = 'axes fraction') #, **{'fontname':'Times New Roman'})

# plt.tight_layout()
# plt.savefig('./Documents/Figs morpho draft/watershed_s15_t0(2).pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()



#%%
n = 5
i = 60

exp = 'v'
if exp == 'v': 
    wtssal_b = wtssal_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
    labs_b, sscas_b, centss_b, ssds_b, smms_b = labs_v, sscas_v, centss_v, ssds_v, smms_v
elif exp == 't': 
    wtssal_b = wtssal_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t
    labs_b, sscas_b, centss_b, ssds_b, smms_b = labs_t, sscas_t, centss_t, ssds_t, smms_t

sobb = sobel(wtssal_b[n][i]) > 0
soy,sox = np.where(sobb)

plt.figure()
plt.imshow(difs_b[n][i])
plt.show()
plt.figure()
plt.imshow(difs_b[n][i])
plt.plot(sox,soy, 'r.', markersize=1)
plt.show()

plt.figure()
plt.imshow(difs_b[n][i])
plt.plot(sox,soy, 'r.', markersize=3)
for j in range(len(sscas_b[n][i])):
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(labs_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(lys_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(sscas_b[n][i][j]/1. ,2)) )
    plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(ssds_b[n][i][j]/1. ,3)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(smms_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( countour_mean(wtssal_b, n, i, labs_b[n][i][j],dif=False) - \
    #                                                                 propscal_t[n][i][labs_b[n][i][j]].intensity_min,2 ) ) ) 
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( countour_mean(wtssal_b, n, i, labs_b[n][i][j],dif=False),2 ) )) 
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( propscal_t[n][i][labs_b[n][i][j]].intensity_min,2 )  ) ) 
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( (ssds_b[n][i][j] / sscas_b[n][i][j])**(-1)  ,2)) )
plt.colorbar()
plt.show()
#%%
# =============================================================================
# Wavelengths
# =============================================================================
mtim = -20
spread = True
mfc = None #'none'

xparam = 'rrho' # 'sal' or 'rrho'
 
plt.rcParams.update({'font.size':16})

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$b)$',r'$b)$'],
                              [r'$a)$',r'$a)$',r'$c)$',r'$c)$']], layout='constrained', figsize=(12/1.,5/1.) ) #, sharex=True)

# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([7,8,15]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx = [], []
    sly,slx = [], []
    for i in range(len(lxs_t[n])):
        mlx.append(np.nanmean(lxs_t[n][i]) * dx)
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        slx.append(np.nanstd(lxs_t[n][i]) * dx)
        sly.append(np.nanstd(lys_t[n][i]) * dy)
    mly,mlx = np.array(mly), np.array(mlx)
    sly,slx = np.array(sly), np.array(slx)
    
    if   xparam == 'sal' : labplot = r'$S$ = '+str(salis_t[n])+' g/kg'
    elif xparam == 'rrho': labplot = rf'$R_\rho$ = {rrho_t[n]:.2f}'
    
    ax[r'$a)$'].plot(ts_t[n]/60, mly , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
                     label=labplot, mfc=mfc)
    ax[r'$a)$'].fill_between(ts_t[n]/60, mly-sly, mly+sly, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

    # ax[r'$a)$'].plot(ts_t[n]/60, mlx , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
    #                  label=str(salis_t[n])+'; '+str(ang_t[n]))
    # ax[r'$a)$'].fill_between(ts_t[n]/60, mlx-slx/2, mlx+slx/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$a)$'])
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

ax[r'$a)$'].set_ylim(bottom=0,top=45)
ax[r'$a)$'].set_xlim(left=0)
ax[r'$a)$'].set_xlabel(r'$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$\lambda_y$ (mm)')
# ax[r'$a)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$a)$'].legend(loc='lower right', ncols=1 ) #loc='lower right')


li1ys,li1xs = [],[]
for l,n in enumerate(nss_v):
    dx = xns_v[l][0,1] - xns_v[l][0,0]
    dy = yns_v[l][0,0] - yns_v[l][1,0]
    # if salis_v[l] > 7 and salis_v[l] < 25: 
    if l in [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]: 
        mly, mlx, mld = [], [], []
        sly, slx, sld = [], [], []
        for i in range(mtim,0):
            mlx += list(lxs_v[l][i] * dx )
            mly += list(lys_v[l][i] * dy )

        mey, eey = np.nanmean(mly), np.nanstd(mly) 
        
        if   xparam == 'sal' : xvalues = salis_v[l]
        elif xparam == 'rrho': xvalues = rrho_v[l]
        
        li1y = ax[r'$b)$'].errorbar(xvalues, mey, yerr=eey, capsize=2, fmt='^', markersize=5, \
                             color=((angys_v[l]+20)/71,0.5,1-(angys_v[l]+20)/71), mfc=mfc )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx), np.nanstd(mlx) 
        li1x = ax[r'$c)$'].errorbar(xvalues, mex, yerr=eex, capsize=2, fmt='^', markersize=5, \
                             color=((angys_v[l]+20)/71,0.5,1-(angys_v[l]+20)/71), mfc=mfc  )        
        li1xs.append(li1x)
        

li2ys,li2xs = [],[]
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([4,5,6,7,8,9,15,23,27,28,32,33,35,34,38]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx, mld = [], [], []
    sly, slx, sld = [], [], []
    for i in range(mtim,0):
        mlx += list(lxs_t[n][i] * dx )
        mly += list(lys_t[n][i] * dy )

    if   xparam == 'sal' : xvalues = salis_t[n]
    elif xparam == 'rrho': xvalues = rrho_t[n]

    if n in [32,33,35]:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(xvalues, mey, yerr=eey, capsize=2, fmt='d', markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(xvalues, mex, yerr=eex, capsize=2, fmt='d', markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc )
        li2xs.append(li2x)
    else:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(xvalues, mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(xvalues, mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc )
        li2xs.append(li2x)
                
if xparam == 'sal' :
    ax[r'$b)$'].set_xlim(0,25)
    ax[r'$c)$'].set_xlim(0,25)
elif xparam == 'rrho' :
    pass
    ax[r'$b)$'].set_xlim(1,8)
    ax[r'$c)$'].set_xlim(1,8)

ax[r'$b)$'].set_ylim(5,50)
ax[r'$c)$'].set_ylim(5,50)
ax[r'$b)$'].set_yticks([10,30,50])
ax[r'$c)$'].set_yticks([10,30,50])
ax[r'$b)$'].sharex(ax[r'$c)$'])
ax[r'$b)$'].tick_params(axis='x',length=3,labelsize=0)

ax[r'$b)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$c)$'].set_ylabel(r'$\lambda_x$ (mm)')

if   xparam == 'sal' : ax[r'$c)$'].set_xlabel(r'$S$ (g/kg)')
elif xparam == 'rrho': ax[r'$c)$'].set_xlabel(r'$R_\rho$')

colores = [ ((angulo+17)/62,0.5,1-(angulo+17)/62) for angulo in [-15,0,15,30,45] ]
leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.55,1.03], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.16,1.23], frameon=False, ncols=5, columnspacing=0.3, handletextpad=0.1)

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[0.37,1.03], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.16,1.23], frameon=False, ncols=5, columnspacing=0.3, handletextpad=0.1)

lgd1 = ax[r'$c)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1.0,0.4], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1, labelspacing=0.7)
lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[1.0,0.1], frameon=False, ncols=1, columnspacing=0.3, handletextpad=0.1, labelspacing=0.7)

# ax[r'$b)$'].add_artist(lgd1)
ax[r'$b)$'].add_artist(lgd2)


# ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
# ax[r'$b)$'].legend([li2ys[7],li2ys[2],li2ys[1],li2ys[0]],[r'$-15°$', r'$0°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
# ax[r'$c)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(1.1,0.5), loc='upper center' , ncols=1 )

for labels,axs in ax.items():
    if labels == r'$a)$':
        axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
    else:
        axs.annotate(labels, (-0.16,0.91), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})        
        

# coutl = 'white'
# ax.xaxis.label.set_color(coutl)        #setting up X-axis label color to yellow
# ax.yaxis.label.set_color(coutl)          #setting up Y-axis label color to blue

# ax.tick_params(axis='x', colors=coutl)    #setting up X-axis tick color to red
# ax.tick_params(axis='y', colors=coutl)  #setting up Y-axis tick color to black

# ax.spines['left'].set_color(coutl) 
# ax.spines['top'].set_color(coutl) 
# ax.spines['right'].set_color(coutl)
# ax.spines['bottom'].set_color(coutl)

# plt.savefig('./Documents/Figs morpho draft/rrho_wavelengths(2).pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
# =============================================================================
# Graph Area
# =============================================================================
mfc = None
xparam = 'rrho' # 'sal' or 'rrho'

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$b)$',r'$b)$',r'$b)$']], layout='constrained', figsize=(12/1.,5/1.), sharey=True)

for l,n in enumerate([7,8,15]):
    if salis_t[n] > 6.0 and salis_t[n] < 25: 
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        dy = yns_t[n][0,0] - yns_t[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_t[n])):
            sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )
            
        if   xparam == 'sal' : labplot = r'$S$ = '+str(salis_t[n])+' g/kg'
        elif xparam == 'rrho': labplot = rf'$R_\rho$ = {rrho_t[n]:.2f}'

        # print(n, f"{np.mean(sarm[40:]):.1f}, {np.min(sarm[40:]):.1f}, {np.max(sarm[40:]):.1f}, {np.std(sarm[40:]):.1f}" )

        # ax[r'$a)$'].errorbar(ts_t[n]/60, np.array(sarm)/100, yerr=np.array(sars)/100,  capsize=2, fmt='.-', errorevery=(l*2,20), \
        #               label=r'$S = $'+str(salis_t[n])+' g/kg' )#, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )
        ax[r'$a)$'].plot(ts_t[n]/60, np.array(sarm)/100, '.-', label=labplot, \
                       color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), mfc=mfc )
            
        ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(sarm)-np.array(sars))/100, (np.array(sarm)+np.array(sars))/100, \
                                 color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

ax[r'$a)$'].set_xlim(left=0)
ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].set_xlabel('$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$A_{s}$ (cm$^2$)')
# ax[r'$a)$'].legend(bbox_to_anchor=(0.5,1.15), loc='upper center' , ncol=3, columnspacing=0.5)
ax[r'$a)$'].legend(loc='upper left')

# li1ys = []
# for n in nss_v:
#     if salis_v[n] > 7 and salis_v[n] < 25: 
#         dx = xns_v[n][0,1] - xns_v[n][0,0]
#         dy = yns_v[n][0,0] - yns_v[n][1,0]
        
#         sarm,sars = [], []
#         # for i in range(len(sscas_v[n])):
#         #     sarm.append( np.nanmean(sscas_v[n][i] * dx*dy) )
#         #     sars.append( np.nanstd(sscas_v[n][i] * dx*dy) )
#         for i in range(-30,0):
#             sarm += list(sscas_v[n][i] * dx*dy) 
        
#         # li1y = ax[r'$b)$'].errorbar(salis_v[n], np.nanmean(sarm[-30:])/100, yerr=np.nanmean(sars[-30:])/100, capsize=2, fmt='o', \
#         #               markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
#         li2y = ax[r'$b)$'].errorbar(salis_v[n], np.nanmean(sarm)/100, yerr=np.nanstd(sarm)/100, capsize=2, fmt='o', \
#                       markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), mfc='w' )
#         li1ys.append(li1y)

# li2ys = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):

#     dx = xns_t[n][0,1] - xns_t[n][0,0]
#     dy = yns_t[n][0,0] - yns_t[n][1,0]
    
#     sarm,sars = [], []
#     # for i in range(len(sscas_t[n])):
#     #     sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
#     #     sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )
#     for i in range(-30,0):
#         sarm += list(sscas_t[n][i] * dx*dy) 
#         # sars.append( np.array(sscas_t[n][i] * dx*dy) )
    
#     # li2y = ax[r'$b)$'].errorbar(salis_t[n], np.nanmean(sarm[-30:])/100, yerr=np.nanmean(sars[-30:])/100, capsize=2, fmt='o', \
#     #               markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
#     li2y = ax[r'$b)$'].errorbar(salis_t[n], np.nanmean(sarm)/100, yerr=np.nanstd(sarm)/100, capsize=2, fmt='o', \
#                   markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
#     li2ys.append(li2y)

# ax[r'$b)$'].set_xlabel('Salinity (g/kg)')
# ax[r'$b)$'].set_ylabel(r'Area (cm$^2$)')
# ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.5,1.15), loc='upper center' , ncol=5, columnspacing=0.5 )

# ns_v = np.array([3,9,11,12,13,15,2])
# ns_t = np.array([7,8,15])
# ns_v = np.array([3,9,13,15,7,2]) # [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]
# ns_t = np.array([7,8,15,38]) # [4,5,6,7,8,9,15,23,27,28,32,33,35,34,38]

ns_v = np.array([7,3,15,2,13])
ns_t = np.array([38,9,15,4,28])

last = -40

sarms_t = [] 
for n in ns_t:
    sarm = []
    for i in range(last,0):
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        dy = yns_t[n][0,0] - yns_t[n][1,0]
        sarm +=  list( sscas_t[n][i] * dx*dy / 100 ) 
    sarms_t.append(sarm)

sarms_v = [] 
for n in ns_v:
    sarm = []
    for i in range(last,0):
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        sarm +=  list( sscas_v[n][i] * dx*dy / 100 ) 
    sarms_v.append(sarm)

# plt.figure()

if xparam == 'sal' : 
    for n in ns_v:
        colv = ((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71) 
        barviolin( sarms_v, ax[r'$b)$'], x=salis_v[ns_v], bins=30, width=7, color=colv) # 'blue') #, labe='Set 1' )
    for n in ns_t:
        colt = ((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62)
        barviolin( sarms_t, ax[r'$b)$'], x=salis_t[ns_t], bins=30, width=7, color=colt) # 'red' ) #, labe='Set 2')
    ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
    ax[r'$b)$'].set_xlim(5,25)
    
elif xparam == 'rrho': #needs some work to show with Rp (makes everything to cluster together)
    colsv = [ ((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71) for n in ns_v]
    colst = [ ((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62) for n in ns_t]
    # colsv = [ 'blue' for n in ns_v]
    # colst = [ 'red' for n in ns_t]

    barviolin( sarms_v, ax[r'$b)$'], x=rrho_v[ns_v], bins=30, width=2.5, marker='^', color=colsv) # 'blue') #, labe='Set 1' )
    barviolin( sarms_t, ax[r'$b)$'], x=rrho_t[ns_t], bins=30, width=2.5, marker='o', color=colst) # 'red' ) #, labe='Set 2')
    ax[r'$b)$'].set_xlabel(r'$R_\rho$')
    ax[r'$b)$'].set_xlim(1,8)

ax[r'$b)$'].axhline(1, linestyle='dashed', alpha=0.5, color='black')

# plt.xticks(salis_v[ns_v])
# plt.xticks(salis_t[ns_t])
# ax[r'$b)$'].set_ylim(0,25)
ax[r'$b)$'].set_ylim(0,28)
ax[r'$b)$'].yaxis.set_tick_params(labelleft=True)
ax[r'$b)$'].set_ylabel(r'$A_{s}$ (cm$^2$)')
    
colores = [ ((angulo+17)/62,0.5,1-(angulo+17)/62) for angulo in [-15,0,15,30,45] ]
leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$b)$'].barh([100],[1],1, color=i, alpha=0.5) for i in colores]

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.1,1.03], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, [' $-15$°',' $0$°',' $15$°',' $30$°',' $45$°'], loc=[0.37,1.03], frameon=False, ncols=5, columnspacing=0.8, handletextpad=0.5)

lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1.,.2], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1, labelspacing=0.7)
lgd2 = ax[r'$b)$'].legend(leg2, [' $-15$°',' $0$°',' $15$°',' $30$°',' $45$°'], loc=[1.,.55], frameon=False, ncols=1, columnspacing=0.8, handletextpad=0.5, labelspacing=0.7)

ax[r'$b)$'].add_artist(lgd1)

for labels,axs in ax.items():
    if labels == r'$a)$':
        axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
    else: axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/rrho_areas(2).pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()
#%%
# =============================================================================
# Graph amplitude
# =============================================================================
mfc = None
xparam = 'rrho' # 'sal' or 'rrho'

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='constrained', figsize=(12/1.,5/1.), sharey=True, sharex=False)

for n in [7,8,15]:
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        # mimam.append( np.nanmean(smms_t[n][i]) )  #min-max
        # mimas.append( np.nanstd(smms_t[n][i]) )   #min-max
        mimam.append( np.nanmean(smes_t[n][i]) )  #mean to edge
        mimas.append( np.nanstd(smes_t[n][i]) )   #mean to edge
        # mimam.append( np.nanmean(ssds_t[n][i]) )  #std
        # mimas.append( np.nanstd(ssds_t[n][i]) )   #std
    
    if   xparam == 'sal' : labplot = r'$S$ = '+str(salis_t[n])+' g/kg'
    elif xparam == 'rrho': labplot = rf'$R_\rho$ = {rrho_t[n]:.2f}'
    
    # ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg', \
    #                      color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
    ax[r'$a)$'].plot(ts_t[n]/60, mimam,'.-', label=labplot, \
                         color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70), mfc=mfc )
        
    ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(mimam)-np.array(mimas)), (np.array(mimam)+np.array(mimas)), \
                             color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

ax[r'$a)$'].set_ylim(0,10)
ax[r'$a)$'].set_xlim(0,70)
ax[r'$a)$'].set_xlabel(r'$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$H_{s}$ (mm)')
ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].legend(loc='upper left')

ave = 20
li1s = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(-ave,0):
            mimam += list(smes_v[n][i])
            mimas += list(smes_v[n][i]) 
            
        if   xparam == 'sal' : xvalues = salis_v[n]
        elif xparam == 'rrho': xvalues = rrho_v[n]
        
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        li1 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='^', \
                     markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71) )#, mfc='w' )
        li1s.append(li1)
    
li2s = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([4,5,6,7,8,9,15,23,27,28,32,33,34,35,38]):
    mimam, mimas = [],[]
    for i in range(-ave,0):
        mimam += list(smes_t[n][i])
        mimas += list(smes_t[n][i]) 

    if   xparam == 'sal' : xvalues = salis_t[n]
    elif xparam == 'rrho': xvalues = rrho_t[n]

    if n in [32,33,35]:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='d', \
                     markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc ) 
    else:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc ) 
    li2s.append(li2)

if xparam == 'sal' :
    ax[r'$b)$'].set_xlim(0,25)
    ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
if xparam == 'rrho' :
    ax[r'$b)$'].set_xlim(1,8)
    ax[r'$b)$'].set_xlabel(r'$R_\rho$')

ax[r'$b)$'].set_ylabel(r'$H_{s}$ (mm)')
ax[r'$b)$'].yaxis.set_tick_params(labelleft=True)
# ax.set_ylim(bottom=0)


colores = [ ((angulo+17)/62,0.5,1-(angulo+17)/62) for angulo in [-15,0,15,30,45] ]
leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.15,1.02], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.34,1.02], frameon=False, ncols=5, columnspacing=0.4, handletextpad=0.1)

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.15,1.02], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.34,1.02], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1)
lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1.,.2], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1, labelspacing=0.7)
lgd2 = ax[r'$b)$'].legend(leg2, [' $-15$°',' $0$°',' $15$°',' $30$°',' $45$°'], loc=[1.,.55], frameon=False, ncols=1, columnspacing=0.8, handletextpad=0.5, labelspacing=0.7)

ax[r'$b)$'].add_artist(lgd1)




for labels,axs in ax.items():
    axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/rrho_amplitudes(2).pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%

# =============================================================================
# Graph melt rate dif wall minima
# =============================================================================
def countour_melt(wtssal_b, gt, n,i,labe):

    sccaa = wtssal_b[n][i] == labe
    ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )

    conval = gt[i][ccoo]
    
    return np.nanmean(conval)


nss_v = list(range(len(ds_v)))
mdif_v = []

for n in tqdm(nss_v):
    if n not in [ 0,  1,  2,  3,  7,  8,  9, 10, 11, 12, 13, 15]:
        mdif_v.append( np.nan )
        continue
    
    mdif = []
    
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]
    
    halg = nangauss(hints_v[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])

    scaprop = propscal_v[n]
    scapropk = kpropscal_v[n]
    for i in range(len(scaprop)):
        md = []
        
        for j in range(len(scaprop[i])):
            ksd = scapropk[i][j].image_stdev
            sarea = scaprop[i][j].area * dx*dy
            ceny,cenx = scaprop[i][j].centroid
            if sarea > 100 and ksd > 0.025: #0.03 o 0.036                
                
                md.append( countour_melt(wtssal_v, gt, n, i, j+1) - gt[i][int(ceny),int(cenx)] )

        md = np.array(md)
        mdif.append(md)
        
    mdif_v.append(mdif)

nss_t = list(range(len(ds_t)))
mdif_t = []

for n in tqdm(nss_t):
    if n not in [4, 5,  6,  7,  8,  9, 15, 23, 27, 28, 32, 33, 34, 35, 38]:
        mdif_t.append( np.nan )
        continue
    
    mdif = []
    
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    
    halg = nangauss(hints_t[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])

    scaprop = propscal_t[n]
    scapropk = kpropscal_t[n]
    for i in range(len(scaprop)):
        md = []
        
        for j in range(len(scaprop[i])):
            ksd = scapropk[i][j].image_stdev
            sarea = scaprop[i][j].area * dx*dy
            ceny,cenx = scaprop[i][j].centroid
            if sarea > 100 and ksd > 0.025: #0.03 o 0.036                
                
                md.append( countour_melt(wtssal_t, gt, n, i, j+1) - gt[i][int(ceny),int(cenx)] )

        md = np.array(md)
        mdif.append(md)
        
    mdif_t.append(mdif)

    
#%%
mfc = None
xparam = 'rrho' # 'sal' or 'rrho'

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='constrained', figsize=(12/1.,5/1.), sharey=True, sharex=False)

for n in [7,8,15]:
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        mimam.append( np.nanmean(mdif_t[n][i]) )  #mean to edge
        mimas.append( np.nanstd(mdif_t[n][i]) )   #mean to edge
    
    if   xparam == 'sal' : labplot = r'$S$ = '+str(salis_t[n])+' g/kg'
    elif xparam == 'rrho': labplot = rf'$R_\rho$ = {rrho_t[n]:.2f}'
    
    # ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg', \
    #                      color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
    ax[r'$a)$'].plot(ts_t[n]/60, mimam,'.-', label=labplot, \
                         color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70), mfc=mfc )
        
    ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(mimam)-np.array(mimas)), (np.array(mimam)+np.array(mimas)), \
                             color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

# ax[r'$a)$'].set_ylim(0,10)
ax[r'$a)$'].set_xlim(0,70)
ax[r'$a)$'].set_xlabel(r'$t$ (min)')
ax[r'$a)$'].set_ylabel(r'$H_{s}$ (mm)')
# ax[r'$a)$'].set_ylim(bottom=0)
ax[r'$a)$'].legend(loc='upper left')

ave = 20
li1s = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(-ave,0):
            mimam += list(mdif_v[n][i])
            mimas += list(mdif_v[n][i]) 
            
        if   xparam == 'sal' : xvalues = salis_v[n]
        elif xparam == 'rrho': xvalues = rrho_v[n]
        
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        li1 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='^', \
                     markersize=5, color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71) )#, mfc='w' )
        li1s.append(li1)
    
li2s = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([4,5,6,7,8,9,15,23,27,28,32,33,34,35,38]):
    mimam, mimas = [],[]
    for i in range(-ave,0):
        mimam += list(mdif_t[n][i])
        mimas += list(mdif_t[n][i]) 

    if   xparam == 'sal' : xvalues = salis_t[n]
    elif xparam == 'rrho': xvalues = rrho_t[n]

    if n in [32,33,35]:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='d', \
                     markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc ) 
    else:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(xvalues, ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color=((angys_t[n]+17)/62,0.5,1-(angys_t[n]+17)/62), mfc=mfc ) 
    li2s.append(li2)

if xparam == 'sal' :
    ax[r'$b)$'].set_xlim(0,25)
    ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
if xparam == 'rrho' :
    ax[r'$b)$'].set_xlim(1,8)
    ax[r'$b)$'].set_xlabel(r'$R_\rho$')

ax[r'$b)$'].set_ylabel(r'$H_{s}$ (mm)')
ax[r'$b)$'].yaxis.set_tick_params(labelleft=True)
# ax.set_ylim(bottom=0)


colores = [ ((angulo+17)/62,0.5,1-(angulo+17)/62) for angulo in [-15,0,15,30,45] ]
leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.15,1.02], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.34,1.02], frameon=False, ncols=5, columnspacing=0.4, handletextpad=0.1)

# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.15,1.02], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°','$45$°'], loc=[0.34,1.02], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1)
lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[1.,.2], frameon=False, ncols=1, columnspacing=0.4, handletextpad=0.1, labelspacing=0.7)
lgd2 = ax[r'$b)$'].legend(leg2, [' $-15$°',' $0$°',' $15$°',' $30$°',' $45$°'], loc=[1.,.55], frameon=False, ncols=1, columnspacing=0.8, handletextpad=0.5, labelspacing=0.7)

ax[r'$b)$'].add_artist(lgd1)




for labels,axs in ax.items():
    axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/rrho_amplitudes(2).pdf',dpi=400, bbox_inches='tight', transparent=False)
plt.show()




#%%




#%%
yvariable_v, yvariable_t = Nu_v, Nu_t
yvarerr_v, yvarerr_t = 0.0018/1000 * rho_ice * latent * length0 / (thcon * temp_v ), 0.0011/1000 * rho_ice * latent * length0 / (thcon * temp_t )

shape_t = [0,2,0,2,3,1,1,1,1,1,3,3,3,3,0,1,0,2,3,3,2,2,3,1,3,3,0,1,1,0,0,0,1,1,3,1,2,2,1]
marker_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1] 

# dictco = {0:(0,0,1), 1:(1,0,0), 2:(0,1,0), 3:(0.5,0.5,0.5)}
dictco = {0:(0.5,0.5,0.5), 1:'#3B719F', 2:'#6aa84f', 3:'#ba3c3c'}
dictma = {0:'o',1:'d'}

cols_v = [dictco[j] for j in shape_v]
mars_t = [dictma[j] for j in marker_t]
cols_t = [dictco[j] for j in shape_t]


plt.figure()

for n in [j for j in range(len(ds_t)) if j not in range(32,36)]: 
     plt.errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='.', markersize=10,  
                  color=cols_t[n], capsize=3) #label=str(i)+'g/kg', \
                 # color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \
for n in range(32,36):
     plt.errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='d', markersize=6,  
                 color=cols_t[n], capsize=3) #label=str(i)+'g/kg', \
                 # color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \


# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$b)$'], location='top')
# # cbar.ax.tick_params(labelsize=12)
# cbar.ax.set_xticks( list(range(0,36,5)) )
# cbar.set_label( label=r"$S$ (g/kg)") #, size=12)


plt.xlabel(r'$\theta$ (°)')
plt.xticks( list(range(-20,51,10)) )
# plt.set_ylabel(r'Melting rate (mm/s)')

plt.show()


plt.figure()
plt.scatter( np.roll(salis_t,3)[:-4], np.roll(angys_t,3)[:-4], c=np.roll(cols_t,3,axis=0)[:-4], marker='o' )
# plt.scatter( np.roll(salis_t,0)[:], np.roll(angys_t,0)[:], c=np.roll(cols_t,0)[:], marker='o' )
plt.scatter(salis_t[-7:-3], angys_t[-7:-3], c=cols_t[-7:-3], marker='d' )
plt.scatter(salis_v, angys_v, marker='o',  facecolors='none', edgecolors=cols_v )
plt.xlabel(r'$S$ (g/kg)')
plt.ylabel(r'$\theta$ (°)')
# plt.savefig('./Documents/Figs morpho draft/morphologs.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
save_name = '' # 'melt_rates', 'nus'
reference = False
shadowgraphy = False
small_ice = True
axis_x = 'salinity'
axis_y = 'nu'

shape_t = [0 ,2 ,3 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,3 ,3 ,3 ,4 ,4 ,1 ,4 ,2 ,4 ,4 ,2 ,2 ,3 ,1 ,3 ,4 ,4 ,1 ,1 ,0 ,4 ,0 ,1 ,1 ,1 ,1 ,2 ,2 ,1]
marker_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,4,1] 

dictco = {0:'magenta', 1:'blue', 2:'green', 3:'orange', 4:'red'}
dictma = {0:'o',1:'d'}

cols_v = [dictco[j] for j in shape_v]
mars_t = [dictma[j] for j in marker_t]
cols_t = [dictco[j] for j in shape_t]

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,5), sharey=True)

if axis_x == 'salinity':
    xvariable_v, xvariable_t = salis_v, salis_t
    ax[r'$a)$'].set_xlim(-1.65,36.17)
    ax[r'$a)$'].set_xticks( list(range(0,36,5)) )
    ax[r'$a)$'].set_xlabel(r'$S$ (g/kg)')
elif axis_x == 'density':
    xvariable_v, xvariable_t = drho_rho_v, drho_rho_t
    ax[r'$a)$'].set_xticks([0.0015, 0.0018, 0.0021, 0.0024, 0.0027, 0.0030])        
    ax[r'$a)$'].set_xlabel(r'$\Delta \rho / \rho$ (g/kg)')    
elif axis_x == 'rrho':
    xvariable_v, xvariable_t = rrho_v, rrho_t
    ax[r'$a)$'].set_xlabel(r'$R_\rho$ (g/kg)')    
elif axis_x == 'ra':
    xvariable_v, xvariable_t = Ra_v, Ra_t
    ax[r'$a)$'].set_xlabel(r'Ra')
if axis_y == 'melt rate':
    yvariable_v, yvariable_t = -mes_v, -mes_t
    yvarerr_v, yvarerr_t = [0.0018] * len(mes_v), [0.0011] * len(mes_t)
    ax[r'$a)$'].set_ylabel(r'$\dot{m}$ (mm/s)')
if axis_y == 'nu':
    yvariable_v, yvariable_t = Nu_v, Nu_t
    yvarerr_v, yvarerr_t = 0.0018/1000 * rho_ice * latent * length0 / (thcon * temp_v ), 0.0011/1000 * rho_ice * latent * length0 / (thcon * temp_t )
    ax[r'$a)$'].set_ylabel(r'Nu')


for n in range(len(ds_v)):
    ax[r'$a)$'].errorbar(xvariable_v[n], yvariable_v[n] * 1 , yerr=yvarerr_v[n], fmt='o', label=str(n)+'°', markersize=5, \
                 color=cols_v[n], capsize=3, mfc='w')
for n in [j for j in range(len(ds_t)) if j not in range(32,36)]:        
    ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='.', label=str(n)+'°', markersize=10, \
                  color=cols_t[n], capsize=3)
if small_ice:
    for n in range(-7,-3):        
        ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='d', label=str(n)+'°', markersize=5, \
                      color=cols_t[n], capsize=3)
        


for n in [j for j in range(len(ds_t)) if j not in range(32,36)]: 
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='.', markersize=10,  
                 color=cols_t[n], capsize=3) #label=str(i)+'g/kg', \
for n in range(-7,-3):
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='d', markersize=6,  
                 color=cols_t[n], capsize=3) #label=str(i)+'g/kg', \

if shadowgraphy:
    # shadowgraphy experiments, they dont say much 
    ax[r'$b)$'].errorbar( [0,30], [0.017391, 0.017927], yerr=[0.000014, 0.000036], fmt='s', color='black' ) # "clear" (not really that clear)
    ax[r'$b)$'].errorbar( [0,30], [0.014225, 0.018498], yerr=[0.000014, 0.000021], fmt='d', color='black' ) # opaque

if reference:
    th = np.linspace(0,50*np.pi/180,50)
    plt.plot( th*180/np.pi, 0.016 * np.cos(th)**(2/3), 'k--' , label=r'McConnochie & Kerr 2018 ($\propto \cos^{2/3}(\theta)$)')
    ax[r'$b)$'].legend()


ax[r'$b)$'].set_xlabel(r'$\theta$ (°)')
ax[r'$b)$'].set_xticks( list(range(-20,51,10)) )
# ax[r'$b)$'].set_ylabel(r'Melting rate (mm/s)')


# for labels,axs in ax.items():
#     axs.annotate(labels, (-0.15,1), xycoords = 'axes fraction')

if len(save_name) > 0: plt.savefig('./Documents/Figs morpho draft/'+save_name+'.png',dpi=400, bbox_inches='tight')
plt.show()
#%%


n = 27 #8 #15 #23
i = -10

cuv = True
exp = 't'
if exp == 'v':
    hints_b = hints_v
    wtssal_b = wtssal_v
    xns_b, yns_b = xns_v, yns_v
    difs_b = difs_v
    labs_b, sscas_b = labs_v, sscas_v
elif exp == 't': 
    hints_b = hints_t
    wtssal_b = wtssal_t
    xns_b, yns_b = xns_t, yns_t
    difs_b = difs_t
    labs_b, sscas_b = labs_t, sscas_t

numim = 3
if cuv:
    halg = nangauss(hints_b[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_b[n]))*0.5, yns_b[n][:,0], xns_b[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    numim = 4

sobb = thin(sobel(wtssal_b[n][i]) > 0)
soy,sox = np.where(sobb)

# minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
# miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
# dx = xns_b[n][0,1] - xns_b[n][0,0]
# dy = yns_b[n][0,0] - yns_b[n][1,0]

plt.figure()
plt.imshow(difs_b[n][i] ) #, extent=(minx,maxx,miny,maxy))
plt.plot( sox, soy, 'k.', markersize=1 ) # xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.show()

plt.figure()
plt.imshow(difs_b[n][i] ) #, extent=(minx,maxx,miny,maxy))
plt.plot( sox, soy, 'k.', markersize=1 ) # xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
mask = np.zeros_like(wtssal_b[n][i])
for j in range(len(labs_b[n][i])):
    mask += wtssal_b[n][i] == labs_b[n][i][j]
mask += np.isnan(difs_b[n][i])
amask = np.ma.masked_where(mask, mask)
plt.imshow( amask, alpha = 0.5 ) # extent=(minx,maxx,miny,maxy), 

plt.show()


plt.figure()
plt.imshow(kurv[i])
plt.show()

# # plt.figure()
# # plt.imshow(halg[i])
# # plt.show()
# plt.figure()
# plt.imshow(difs_b[n][i])
# plt.show()
#%%
n = -1 #8 #15 #23
i = -20

plt.figure()
plt.imshow(kurv_t[n][i])
plt.show()

filsat = frangi(kurv_t[n][i])
nfisa = filsat / np.nanmax(filsat)

plt.figure()
plt.imshow( nfisa  )
plt.colorbar()
plt.show()


plt.figure()
plt.hist( nfisa[nfisa>-1] , bins=50 )
plt.show()
plt.figure()
plt.hist( nfisa[nfisa>0] , bins=50 )
plt.show()

kurtosis( nfisa[nfisa>0] ), kurtosis( nfisa[nfisa>-1] )

#%%
i = -10

rat_ars_v = []
for n in range(len(ds_v)):
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]
    rat_ars_v.append( np.sum(sscas_v[n][i]) * dx*dy / totars_v[n][i] )

plt.figure()
plt.scatter(salis_v, rat_ars_v, c=cols_v)
plt.grid()
plt.show()
#%%

i = -10

rat_ars_t, amp_t = [], []
for n in range(len(ds_t)):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    rat_ars_t.append( np.sum(sscas_t[n][i]) * dx*dy / totars_t[n][i] )
    # amp_t.append(  np.nanstd(smes_t[n][i])  )
    # amp_t.append(  np.nanstd(ssds_t[n][i])  )
    
    filsat = frangi(kurv_t[n][i])
    nfisa = filsat / np.nanmax(filsat)
    
    amp_t.append(  kurtosis( nfisa[nfisa>-1] ) )


plt.figure()
# plt.scatter(salis_t, rat_ars_t, c=cols_t)
plt.scatter(salis_t, amp_t, c=cols_t)
# plt.scatter(rat_ars_t, amp_t, c=cols_t)
# plt.plot(rat_ars_t,'.')
plt.grid()
plt.show()

#%%

n = 0

t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
gt,gy,gx = np.gradient(hints_t[n], t,y,x)
xs,ys = xns_t[n], yns_t[n]

area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)

gt[np.isnan(gt)] = 0.0
# meltr = np.trapz( np.trapz( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
# tmelr = np.trapz( meltr / area, x=t ) / t[-1]
tmel = np.trapz( gt, x=t, axis=0 ) / t[-1]
tmel[np.isnan(gy[-1])] = np.nan

plt.figure()
plt.imshow( - tmel )
plt.colorbar()
plt.show()

plt.figure()
# plt.plot( -tmel[300,:] )
# plt.plot( np.gradient( -np.nanmean(tmel,axis=0) ) )
plt.plot( -np.nanmean(tmel,axis=0) ) 
plt.plot( -np.nanmean(tmel,axis=1) ) 
plt.show()

# fil = np.where(~np.isnan(np.nanmean(tmel,axis=0)))
# plt.figure()
# # plt.plot( -tmel[300,:] )
# # plt.plot( np.gradient( -np.nanmean(tmel,axis=0) ) )
# plt.plot( np.gradient(-np.nanmean(tmel,axis=0)[fil]), '.' )
# # plt.yscale('log')
# plt.show()
#%%

n = 2

t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
gt,gy,gx = np.gradient(hints_v[n], t,y,x)
xs,ys = xns_v[n], yns_v[n]

area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)

gt[np.isnan(gt)] = 0.0
# meltr = np.trapz( np.trapz( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
# tmelr = np.trapz( meltr / area, x=t ) / t[-1]
tmel = np.trapz( gt, x=t, axis=0 ) / t[-1]
tmel[np.isnan(gy[-3])] = np.nan

plt.figure()
plt.imshow( - tmel )
plt.colorbar()
plt.show()

#%%
sdml = []
for n in tqdm(range(len(ds_v))):
    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    gt,gy,gx = np.gradient(hints_v[n], t,y,x)
    xs,ys = xns_v[n], yns_v[n]

    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)

    gt[np.isnan(gt)] = 0.0
    # meltr = np.trapz( np.trapz( gt, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
    # tmelr = np.trapz( meltr / area, x=t ) / t[-1]
    tmel = np.trapz( gt, x=t, axis=0 ) / t[-1]
    tmel[np.isnan(gy[-3])] = np.nan
    sdml.append( np.nanstd(tmel) )


plt.figure()
plt.plot(salis_v, sdml,'.')
plt.show()

#%%

rat_ars_t = []
sadf = []
# for n in [0,1,2,3,4,10]:
for n in range(len(ds_t)):
    t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
    gt,gy,gx = np.gradient(hints_t[n], t,y,x)
    xs,ys = xns_t[n], yns_t[n]
    gt[np.isnan(gt)] = 0.0
    tmel = np.trapz( gt, x=t, axis=0 ) / t[-1]
    tmel[np.isnan(gy[-1])] = np.nan
    
    sadf.append( np.nanstd(-np.nanmean(tmel,axis=0)) )
    
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    rat_ars_t.append( np.sum(sscas_t[n][i]) * dx*dy / totars_t[n][i] )



plt.figure()
# plt.plot([0,1,2,3,4,10], sadf,'.')
plt.plot(rat_ars_t, sadf,'.')
plt.show()




#%%

n = 0
i = 60

exp = 'v'
if exp == 'v':
    hints_b, difs_b, coeffs_b = hints_v, difs_v, coeffs_v
    xns_b, yns_b = xns_v, yns_v
elif exp == 't': 
    hints_b, difs_b, coeffs_b = hints_t, difs_t, coeffs_t
    xns_b, yns_b = xns_t, yns_t

plt.figure()
plt.imshow( hints_b[n][i] ) 
plt.show()

dife = np.copy( hints_b[n][i] )
dife[np.isnan(dife)] = 0

ny,nx = np.shape(dife)
kx,ky = np.fft.fftfreq(nx), np.fft.fftfreq(ny)
kxg,kyg = np.meshgrid(kx,ky) 

ftx = np.abs( np.fft.fft(dife,axis=1) )**2
lx = np.trapz(ftx[:,1:int(nx/2)] / kxg[:,1:int(nx/2)], x=kx[1:int(nx/2)], axis=1) * np.pi/2 / ((nx-1)/2)**2

fty = np.abs( np.fft.fft(dife,axis=0) )**2
ly = np.trapz(fty[1:int(ny/2)] / kyg[1:int(ny/2)], x=ky[1:int(ny/2)], axis=0) * np.pi/2 / ((ny-1)/2)**2

plt.figure()
plt.imshow( np.log(fty) )
# plt.yscale('log') 
plt.show()

# lx,ly
print(np.mean([i for i in ly if i != 0.]), np.median([i for i in ly if i != 0.]))
print(np.mean([i for i in lx if i != 0.]), np.median([i for i in lx if i != 0.]))
# coeffs_b[n][i]

#%%


n = 8
i = 60

exp = 'v'
if exp == 'v':
    hints_b, difs_b, coeffs_b = hints_v, difs_v, coeffs_v
    xns_b, yns_b = xns_v, yns_v
elif exp == 't': 
    hints_b, difs_b, coeffs_b = hints_t, difs_t, coeffs_t
    xns_b, yns_b = xns_t, yns_t

plt.figure()
plt.imshow( hints_b[n][i] ) 
plt.show()

dife = np.copy( hints_b[n][i] )
dife -= np.nanmean( hints_b[n][i] )
dife[np.isnan(dife)] = 0

ny,nx = np.shape(dife)

# for i in range(-int(ny/2),int(ny/2)):
#     conv = dife * np.roll(dife,i,axis=0)
# conv = fftconvolve(dife, dife[::-1,:], mode='valid',axes=0)
# conv = fftconvolve(hints_b[n][i], hints_b[n][i][::-1,:], mode='valid',axes=0)

pdfl = np.pad( dife[:,200], int(ny/2) )
cfn = fftconvolve(pdfl, pdfl[::-1], mode='full')

plt.figure()
plt.plot(pdfl)
plt.show()
plt.figure()
plt.plot( cfn )
plt.show()

pdfl = np.pad( dife[400], int(nx/2) )
cfn = fftconvolve(pdfl, pdfl[::-1], mode='full')

plt.figure()
plt.plot(pdfl)
plt.show()
plt.figure()
plt.plot( cfn )
plt.show()
#%%
i = -10
fgh = []
for n in range(len(ds_v)):
    fgh.append( np.nanstd(difs_v[n][i]) )

plt.figure()
# plt.plot(salis_v, fgh, '.')
plt.plot(fgh, fgh, '.')
plt.show()

# n = 10
# plt.figure()
# plt.imshow( difs_v[n][i] )
# plt.show()
# np.nanstd(difs_v[n][i])

#%%
kurv_v = []
for n in tqdm(range(len(ds_v))):
    # halg = nangauss(hints_v[n],5)
    halg = nangauss(difs_v[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    kurv_v.append(kurv)

kurv_t = []
for n in tqdm(range(len(ds_t))):
    # halg = nangauss(hints_t[n],5)
    halg = nangauss(difs_t[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    kurv_t.append(kurv)
#%%
n = 27
i = -10

# fil = binary_erosion(~np.isnan(kurv_t[n][i]), disk(30)) * 1.
# fil[fil==0.] = np.nan
kusd, kume = np.nanstd( kurv_t[n][i] ), np.nanmean( kurv_t[n][i] )
fil = np.zeros_like(kurv_t[n][i]) * False
fil[ (kurv_t[n][i] < (kume+2*kusd)) * (kurv_t[n][i] > (kume-2*kusd)) ] = True
# fil = ~np.isnan(kurv_t[n][i])

# print(salis_t[n], angys_t[n], np.nanstd( kurv_t[n][i] )) #, np.nanstd( kurv_t[n][i] * fil ) )
# print( skew( kurv_t[n][i][fil] ), kurtosis( kurv_t[n][i][fil] ), skew( kurv_t[n][i][fil] ) / kurtosis( kurv_t[n][i][fil] )   )
print( skew( kurv_t[n][i][fil>0] ), kurtosis( kurv_t[n][i][fil>0] ), skew( kurv_t[n][i][fil>0] ) / kurtosis( kurv_t[n][i][fil>0] )   )
# print( skew( difs_t[n][i][fil] ), kurtosis( difs_t[n][i][fil] ), skew( difs_t[n][i][fil] ) / kurtosis( difs_t[n][i][fil] )   )

# plt.figure()
# plt.imshow( kurv_v[n][i] )
# plt.colorbar()
# plt.show()
plt.figure()
plt.imshow( kurv_t[n][i] )
plt.colorbar()
plt.show()
# plt.figure()
# plt.imshow( fil * kurv_t[n][i] )
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow( difs_t[n][i] )
# plt.colorbar()
# plt.show()

plt.figure()
# plt.hist( ((kurv_t[n][i] - np.nanmean(kurv_t[n][i])) / np.nanstd(kurv_t[n][i])  ).flatten(), bins=90 )
plt.hist( kurv_t[n][i][fil>0].flatten(), bins=90, density=True )
plt.hist( kurv_t[n][i].flatten(), bins=90, alpha=0.5, density=True )
# plt.yscale('log')
plt.show()
# # plt.figure()
# # plt.hist( difs_t[n][i].flatten(), bins=90 )
# # # plt.yscale('log')
# # plt.show()

#%%
# i = -20
kuku_v, skku_v = [],[]
for n in tqdm(range(len(ds_v))):
    kks, skk = [],[]
    for i in range(len(kurv_v[n])):
        kusd, kume = np.nanstd( kurv_v[n][i] ), np.nanmean( kurv_v[n][i] )
        fil = np.zeros_like(kurv_v[n][i]) * False
        fil[ (kurv_v[n][i] < (kume+2*kusd)) * (kurv_v[n][i] > (kume-2*kusd)) ] = True
        # fil = ~np.isnan(kurv_v[n][i])
        
        skk.append( skew( kurv_v[n][i][fil>0] )) # [fil>0] ) )
        kks.append( kurtosis( kurv_v[n][i][fil>0] )) #[fil>0] ) )
    skku_v.append( skk )
    kuku_v.append( kks )

kuku_t, skku_t = [],[]
for n in tqdm(range(len(ds_t))):
    kks, skk = [],[]
    for i in range(len(kurv_t[n])):
        kusd, kume = np.nanstd( kurv_t[n][i] ), np.nanmean( kurv_t[n][i] )
        fil = np.zeros_like(kurv_t[n][i]) * False
        fil[ (kurv_t[n][i] < (kume+2*kusd)) * (kurv_t[n][i] > (kume-2*kusd)) ] = True
        # fil = ~np.isnan(kurv_t[n][i])
        
        skk.append( skew( kurv_t[n][i][fil>0] ))# [fil>0] ) )
        kks.append( kurtosis( kurv_t[n][i][fil>0] ))# [fil>0] ) )
    skku_t.append( skk )
    kuku_t.append( kks )
#%%
# n = 0

plt.figure()
for n in [0,4,10,14,13]:
    # plt.plot( skku_v[n], label=n )
    plt.plot( kuku_v[n], label=n )
# plt.plot( kuku_v[n] )
plt.legend()
plt.show()

plt.figure()
for n in range(len(ds_v)):
    plt.plot(salis_v[n], np.nanmin(skku_v[n]),'.' )
    # plt.plot( kuku_v[n], label=n )
# plt.plot( kuku_v[n] )
plt.show()




#%%

shape_t = [0,2,0,2,3,1,1,1,1,1,3,3,3,3,0,1,0,2,3,3,2,2,3,1,3,3,0,1,1,0,0,0,1,1,3,1,2,2,1]
marker_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
shape_v = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1] 

# dictco = {0:(0,0,1), 1:(1,0,0), 2:(0,1,0), 3:(0.5,0.5,0.5)}
# dictco = {0:(0.5,0.5,0.5), 1:'#3B719F', 2:'#6aa84f', 3:'#ba3c3c'}
dictco = {0:'magenta', 1:'blue', 2:'green', 3:'orange'}
dictma = {0:'o',1:'d'}

cols_v = [dictco[j] for j in shape_v]
mars_t = [dictma[j] for j in marker_t]
cols_t = [dictco[j] for j in shape_t]


plt.figure()
# plt.scatter(salis_v, sdku_v,marker='.', c=cols_v)
# plt.scatter(salis_v, skku_v,marker='.', c=cols_v)
plt.scatter(kuku_v, skku_v,marker='.', c=cols_v)
# plt.scatter(sdku_v, sdku_v, marker='.')
plt.show()
plt.figure()
# plt.scatter(salis_t, sdku_t, marker='.', c=cols_t )
# plt.scatter(salis_t, skku_t, marker='.', c=cols_t )
plt.scatter(kuku_t, skku_t, marker='.', c=cols_t )
# plt.scatter(sdku_t, sdku_t, marker='.', c=cols_t )
plt.show()

#%%
plt.figure()
for n in [7,14,27]:
    i = -20
    
    # kuce = np.copy(kurv_t[n][i])
    # kuce = np.copy(difs_t[n][i])
    # kuce[ np.isnan(kuce) ] = 0.
    # kft = np.fft.fft2( kuce )
    # kx, ky = np.fft.fftfreq( np.shape(kuce)[1] ), np.fft.fftfreq( np.shape(kuce)[0] )
    # kx,ky = np.meshgrid( kx,ky )
    # k, thk = np.sqrt(kx**2+ky**2), np.arctan2(ky,kx)
    
    # kg,thg = np.arange(0,0.5,0.0015), np.linspace(-np.pi,np.pi,1000)
    # kg,thg = np.meshgrid(kg,thg)
    # kfg = griddata( np.array([kx.flatten(),ky.flatten()]).T, kft.flatten(), (kg*np.cos(thg), kg*np.sin(thg)), method='linear')
    
    # sng = np.trapz( np.abs(kfg)**2 * kg, np.linspace(-np.pi,np.pi,1000), axis=0 )
    

    # plt.figure()
    # plt.imshow( np.log(np.abs(kfg)) )
    # plt.imshow( thg )
    # plt.plot( kg[0,:], sng, '.-'  )
    # plt.yscale('log')
    # plt.xscale('log')
    
    # plt.hist( kurv_t[n][i].flatten(), bins=90, alpha=0.5, label=n, density=True )
    plt.hist( ((kurv_t[n][i] - np.nanmean(kurv_t[n][i])) / np.nanstd(kurv_t[n][i])).flatten(), bins=90, alpha=0.5, label=n, density=True )
plt.legend()    
plt.show()

# plt.figure()
# # plt.imshow( np.fft.fftshift(np.log(np.abs(kft)**2)) )
# # plt.imshow( np.fft.fftshift(k) )
# plt.imshow( kuce )
# plt.colorbar()
# plt.show()

#%%



#%%
# =============================================================================
# Local (inside scallop region) melt rate diferences
# =============================================================================
i = -20 #-10, -20, 40

medif_v, medis_v = [],[]
gtsd_v = []
for n in tqdm(range(len(xns_v))):
    halg = nangauss(hints_v[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    
    scc, mesd = np.zeros_like(hints_v[n][i]), []
    for i in range(-20,0):
        for j in range(len(labs_v[n][i])):
            # scc += (wtssal_v[n][i] == labs_v[n][i][j]) * labs_v[n][i][j]
            scpo = np.where( wtssal_v[n][i] == labs_v[n][i][j] )
            mesd.append( np.std( gt[i][scpo] ) )

    medif_v.append( np.nanmean(mesd) )
    medis_v.append( np.nanstd(mesd) )
    # gtsd_v.append( np.nanstd(gt[i]) )

medif_t, medis_t = [],[]
gtsd_t = []
for n in tqdm(range(len(xns_t))):
    halg = nangauss(hints_t[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    
    scc, mesd = np.zeros_like(hints_t[n][i]), []
    for i in range(-20,0):
        for j in range(len(labs_t[n][i])):
            # scc += (wtssal_t[n][i] == labs_t[n][i][j]) * labs_t[n][i][j]
            scpo = np.where( wtssal_t[n][i] == labs_t[n][i][j] )
            mesd.append( np.std( gt[i][scpo] ) )

    medif_t.append( np.nanmean(mesd) )
    medis_t.append( np.nanstd(mesd) )
    # gtsd_t.append( np.nanstd(gt[i]) )

#%%

plt.figure()


for n in range(len(ds_v)):
    if salis_v[n] > 5 and salis_v[n] < 25: 
        plt.errorbar(salis_v[n], medif_v[n], yerr=medis_v[n],fmt='^', color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71)) #(0.5,1-salis_t[n]/35,salis_t[n]/35)
for l,n in enumerate([5,6,7,8,9,15,23,27,28,38]):
    plt.errorbar(salis_t[n], medif_t[n], yerr=medis_t[n],fmt='o', color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
    # plt.plot(salis_t[n], gtsd_t[n],'o', color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
for l,n in enumerate([32,33,35]):
    plt.errorbar(salis_t[n], medif_t[n], yerr=medis_t[n],fmt='d', color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
    # plt.plot(salis_t[n], gtsd_t[n],'d', color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
# plt.title(r'$t$ = '+str(i/2)+' min')
plt.title(r'$t$ = last '+str(20/2)+' min')
plt.xlabel(r'$S$ (g/kg)')
plt.ylabel(r'$\langle \sigma_{\dot{m}} \rangle_{scallop}$ (mm/min)')
plt.show()

#%%

i = -20

# mesd, mssd = [], []
# for n in tqdm(range(len(ds_v))):
#     halg = nangauss(hints_v[n],2)
#     gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    
#     gtm = np.nanmean(gt[i:],axis=0)
    
#     # img = gt[i]  
#     img = gtm 
#     img2 = img**2
#     kernel = disk(30)
#     kernel = kernel / np.sum(kernel)

#     img_mean = convolve2d(img, kernel, mode="valid")
#     img2_mean = convolve2d(img2, kernel, mode="valid")
#     losd = np.sqrt( img2_mean - img_mean**2 )
    
#     mesd.append( np.nanmedian(losd) )
#     mssd.append( np.nanstd(losd) )

mesd, mssd = [], []
for n in tqdm(range(len(ds_t))):
    halg = nangauss(hints_t[n],2)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    
    gtm = np.nanmean(gt[i:],axis=0)
    
    # img = gt[i]  
    img = gtm 
    img2 = img**2
    kernel = disk(30)
    kernel = kernel / np.sum(kernel)

    img_mean = convolve2d(img, kernel, mode="valid")
    img2_mean = convolve2d(img2, kernel, mode="valid")
    losd = np.sqrt( img2_mean - img_mean**2 )
    
    mesd.append( np.nanmedian(losd) )
    mssd.append( np.nanstd(losd) )
#%%

# plt.figure()
# # plt.errorbar( salis_v, np.array(mesd)**2, yerr=np.array(mssd)**2, fmt='.' )
# plt.errorbar( salis_v, mesd, yerr=mssd, fmt='.' )
# # plt.plot( salis_v, mesd, '.' )
# plt.show()

plt.figure()
# plt.errorbar( salis_t, np.array(mesd)**2, yerr=np.array(mssd)**2, fmt='.' )
for n in tqdm(range(len(ds_t))):
    plt.errorbar( salis_t[n], mesd[n], yerr=mssd[n], fmt='.', color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
# plt.plot( salis_t, mesd, '.' )
plt.show()
#%%
# =============================================================================
# Poster
# =============================================================================
#wavelength
plt.rcParams.update({'font.size':20})

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplot_mosaic([[r'$b)$',r'$b)$'],
                              [r'$c)$',r'$c)$']], layout='constrained') #, sharex=True)

# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
# for l,n in enumerate([7,8,15]):
#     dx = xns_t[n][0,1] - xns_t[n][0,0]
#     dy = yns_t[n][0,0] - yns_t[n][1,0]

#     mly, mlx = [], []
#     sly,slx = [], []
#     for i in range(len(lxs_t[n])):
#         mlx.append(np.nanmean(lxs_t[n][i]) * dx)
#         mly.append(np.nanmean(lys_t[n][i]) * dy)
#         slx.append(np.nanstd(lxs_t[n][i]) * dx)
#         sly.append(np.nanstd(lys_t[n][i]) * dy)
#     mly,mlx = np.array(mly), np.array(mlx)
#     sly,slx = np.array(sly), np.array(slx)
    
#     ax[r'$a)$'].plot(ts_t[n]/60, mly , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
#                      label=r'$S$ = '+str(salis_t[n])+' g/kg', mfc=mfc)
#     ax[r'$a)$'].fill_between(ts_t[n]/60, mly-sly, mly+sly, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

    # ax[r'$a)$'].plot(ts_t[n]/60, mlx , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), 
    #                  label=str(salis_t[n])+'; '+str(ang_t[n]))
    # ax[r'$a)$'].fill_between(ts_t[n]/60, mlx-slx/2, mlx+slx/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$a)$'])
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# ax[r'$a)$'].set_ylim(bottom=0)
# ax[r'$a)$'].set_xlim(left=0)
# ax[r'$a)$'].set_xlabel(r'$t$ (min)')
# ax[r'$a)$'].set_ylabel(r'$\lambda_y$ (mm)')
# # ax[r'$a)$'].set_ylabel(r'$\lambda_x$ (mm)')
# ax[r'$a)$'].legend(loc='lower right', ncols=1 ) #loc='lower right')

# tupcol = ((angys_t[n]+17)/47,1-(angys_t[n]+17)/47,0.5)
# tupcol = ((angys_t[n]+17)/47,1-(angys_t[n]+17)/47,0.5)

blue = 0.1

li1ys,li1xs = [],[]
for l,n in enumerate(nss_v):
    dx = xns_v[l][0,1] - xns_v[l][0,0]
    dy = yns_v[l][0,0] - yns_v[l][1,0]
    
    # tupcol = ((angys_v[l]+20)/71,0.5,1-(angys_v[l]+20)/71)
    tupcol = ( 1-(angys_v[l]+17)/47, 1-(angys_v[l]+17)/47, blue)
    
    if salis_v[l] > 7 and salis_v[l] < 25: 
        mly, mlx, mld = [], [], []
        sly, slx, sld = [], [], []
        for i in range(mtim,0):
            mlx += list(lxs_v[l][i] * dx )
            mly += list(lys_v[l][i] * dy )

        mey, eey = np.nanmean(mly), np.nanstd(mly) 
        li1y = ax[r'$b)$'].errorbar(salis_v[l], mey, yerr=eey, capsize=2, fmt='^', markersize=5, \
                             color= tupcol , mfc=mfc )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx), np.nanstd(mlx) 
        li1x = ax[r'$c)$'].errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='^', markersize=5, \
                             color= tupcol, mfc=mfc  )        
        li1xs.append(li1x)

li2ys,li2xs = [],[]
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx, mld = [], [], []
    sly, slx, sld = [], [], []
    for i in range(mtim,0):
        mlx += list(lxs_t[n][i] * dx )
        mly += list(lys_t[n][i] * dy )
        
    # tupcol = ((angys_t[n]+17)/47, 0.5, 1-(angys_t[n]+17)/47)
    tupcol = ( 1-(angys_t[n]+17)/47, 1-(angys_t[n]+17)/47, blue)

    if n in [32,33,35]:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='d', markersize=5, color= tupcol, mfc=mfc )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='d', markersize=5, color= tupcol, mfc=mfc )
        li2xs.append(li2x)
    else:
        mey, eey = np.nanmean(mly), np.nanstd(mly)  
        li2y = ax[r'$b)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color= tupcol, mfc=mfc )
        li2ys.append(li2y)
    
        mex, eex = np.nanmean(mlx), np.nanstd(mlx)  
        li2x = ax[r'$c)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color= tupcol, mfc=mfc )
        li2xs.append(li2x)

ax[r'$b)$'].set_xlim(0,25)
ax[r'$c)$'].set_xlim(0,25)
ax[r'$b)$'].set_ylim(0,50)
ax[r'$c)$'].set_ylim(0,50)

# ax[r'$b)$'].set_yticks([0,10,20,30,40,50])
# ax[r'$c)$'].set_yticks([0,10,20,30,40,50])
ax[r'$b)$'].set_yticks([0,20,40])
ax[r'$c)$'].set_yticks([0,20,40])

ax[r'$b)$'].sharex(ax[r'$c)$'])
ax[r'$b)$'].tick_params(axis='x',length=3,labelsize=0)

ax[r'$b)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$c)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$c)$'].set_xlabel(r'$S$ (g/kg)')

# colores = [ ((angulo+17)/47,1-(angulo+17)/47,0.5)  for angulo in [-15,0,15,30] ]
# leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
# leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]
# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.05,1.03], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°'], loc=[0.47,1.03], frameon=False, ncols=4, columnspacing=0.4, handletextpad=0.1)
# ax[r'$b)$'].add_artist(lgd1)


# ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
# ax[r'$b)$'].legend([li2ys[7],li2ys[2],li2ys[1],li2ys[0]],[r'$-15°$', r'$0°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
# ax[r'$c)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(1.1,0.5), loc='upper center' , ncols=1 )

# for labels,axs in ax.items():
#     if labels == r'$a)$':
#         axs.annotate(labels, (-0.16,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})
#     else:
#         axs.annotate(labels, (-0.16,0.91), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})        


coutl = 'white'
for labels,axs in ax.items():
    axs.xaxis.label.set_color(coutl)        #setting up X-axis label color to yellow
    axs.yaxis.label.set_color(coutl)          #setting up Y-axis label color to blue
    
    axs.tick_params(axis='x', colors=coutl)    #setting up X-axis tick color to red
    axs.tick_params(axis='y', colors=coutl)  #setting up Y-axis tick color to black
    
    axs.spines['left'].set_color(coutl) 
    axs.spines['top'].set_color(coutl) 
    axs.spines['right'].set_color(coutl)
    axs.spines['bottom'].set_color(coutl)

# plt.savefig('./Documents/Figs morpho draft/poster_wavelengths_scallops.pdf',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%

mfc = None

fig, ax = plt.subplot_mosaic([[r'$b)$',r'$b)$'],
                              [r'$b)$',r'$b)$']], layout='constrained')

# for n in [7,8,15]:
#     mimam, mimas = [],[]
#     for i in range(len(smms_t[n])):
#         # mimam.append( np.nanmean(smms_t[n][i]) )  #min-max
#         # mimas.append( np.nanstd(smms_t[n][i]) )   #min-max
#         mimam.append( np.nanmean(smes_t[n][i]) )  #mean to edge
#         mimas.append( np.nanstd(smes_t[n][i]) )   #mean to edge
#         # mimam.append( np.nanmean(ssds_t[n][i]) )  #std
#         # mimas.append( np.nanstd(ssds_t[n][i]) )   #std
    
#     # ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg', \
#     #                      color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
#     ax[r'$a)$'].plot(ts_t[n]/60, mimam,'.-', label=r'$S = $ '+str(salis_t[n])+' g/kg', \
#                          color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70), mfc=mfc )
        
#     ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(mimam)-np.array(mimas)), (np.array(mimam)+np.array(mimas)), \
#                              color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

# ax[r'$a)$'].set_ylim(0,10)
# ax[r'$a)$'].set_xlim(0,70)
# ax[r'$a)$'].set_xlabel(r'$t$ (min)')
# ax[r'$a)$'].set_ylabel(r'$H_{\textrm{scallop}}$ (mm)')
# ax[r'$a)$'].set_ylim(bottom=0)
# ax[r'$a)$'].legend(loc='upper left')

ave = 20
li1s = []
for n in nss_v:
    
    # tupcol = ((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71
    tupcol = ( 1-(angys_v[n]+17)/47, 1-(angys_v[n]+17)/47, 0.1)
    
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(-ave,0):
            mimam += list(smes_v[n][i])
            mimas += list(smes_v[n][i]) 
        
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        li1 = ax[r'$b)$'].errorbar(salis_v[n], ascm, yerr = ascs, capsize=2, fmt='^', \
                     markersize=5, color= tupcol )#, mfc='w' )
        li1s.append(li1)
    
li2s = []
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
    
    # tupcol = ((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47)
    tupcol = ( 1-(angys_t[n]+17)/47, 1-(angys_t[n]+17)/47, 0.1)
    
    mimam, mimas = [],[]
    for i in range(-ave,0):
        mimam += list(smes_t[n][i])
        mimas += list(smes_t[n][i]) 

    if n in [32,33,35]:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='d', \
                     markersize=5, color= tupcol, mfc=mfc ) 
    else:
        ascm, ascs = np.nanmean(mimam), np.nanstd(mimas)
        # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
        #              markersize=5, color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71) )
        li2 = ax[r'$b)$'].errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color= tupcol, mfc=mfc ) 
    li2s.append(li2)
        
ax[r'$b)$'].set_xlim(0,25)
ax[r'$b)$'].set_ylim(0,8.9)
ax[r'$b)$'].set_xlabel(r'$S$ (g/kg)')
ax[r'$b)$'].set_ylabel(r'$H_{\textrm{scallop}}$ (mm)')
ax[r'$b)$'].yaxis.set_tick_params(labelleft=True)
# ax.set_ylim(bottom=0)


# colores = [ ((angulo+17)/47,0.5,1-(angulo+17)/47) for angulo in [-15,0,15,30] ]
# leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
# leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]
# lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.05,1.02], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1)
# lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°'], loc=[0.47,1.02], frameon=False, ncols=4, columnspacing=0.4, handletextpad=0.1)
# ax[r'$b)$'].add_artist(lgd1)

# ax[r'$b)$'].legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
#                    bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing=0.5 )
# ax[r'$b)$'].legend([li2s[7],li2s[2],li2s[1],li2s[0]],[r'$-15°$', r'$0°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing = 0.5 )

# for labels,axs in ax.items():
#     axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

coutl = 'white'
for labels,axs in ax.items():
    axs.xaxis.label.set_color(coutl)        #setting up X-axis label color to yellow
    axs.yaxis.label.set_color(coutl)          #setting up Y-axis label color to blue
    
    axs.tick_params(axis='x', colors=coutl)    #setting up X-axis tick color to red
    axs.tick_params(axis='y', colors=coutl)  #setting up Y-axis tick color to black
    
    axs.spines['left'].set_color(coutl) 
    axs.spines['top'].set_color(coutl) 
    axs.spines['right'].set_color(coutl)
    axs.spines['bottom'].set_color(coutl)
    
# plt.savefig('./Documents/Figs morpho draft/poster_amplitudes_scallops.pdf',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%
from matplotlib.transforms import Bbox

coutl = 'white'

fig, ax = plt.subplot_mosaic([[r'$b)$',r'$b)$'],
                              [r'$b)$',r'$b)$']], layout='constrained', figsize=(12/1.,5/1.))

colores = [ (1-(angulo+17)/47,1-(angulo+17)/47, 0.1) for angulo in [-15,0,15,30] ]
leg1 = [ax[r'$b)$'].scatter([], [], marker=i, edgecolors='gray', s=30, facecolors={None:'gray','none':'none'}[mfc]) for i in ['^','o','d']]
leg2 = [ax[r'$b)$'].scatter([],[],marker='o',edgecolors=i,s=30,facecolors={None:i,'none':'none'}[mfc]) for i in colores]
lgd1 = ax[r'$b)$'].legend(leg1, ['Set 1','Set 2','Set 3'], loc=[-0.05,1.02], frameon=False, ncols=3, columnspacing=0.4, handletextpad=0.1, labelcolor=coutl)
lgd2 = ax[r'$b)$'].legend(leg2, ['$-15$°','$0$°','$15$°','$30$°'], loc=[0.47,1.02], frameon=False, ncols=4, columnspacing=0.4, handletextpad=0.1, labelcolor=coutl)
ax[r'$b)$'].add_artist(lgd1)

bbox1 = np.array( lgd1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) )
bbox2 = np.array( lgd2.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) )
bbox = bbox1 
bbox[0,0] = bbox[0,0] - 1
bbox[1,0] = bbox2[1,0] + 1
bbox[1,0] = bbox[1,0] 
bbox[1,1] = bbox[1,1]

for labels,axs in ax.items():
    axs.xaxis.label.set_color(coutl)        #setting up X-axis label color to yellow
    axs.yaxis.label.set_color(coutl)          #setting up Y-axis label color to blue
    
    axs.tick_params(axis='x', colors=coutl)    #setting up X-axis tick color to red
    axs.tick_params(axis='y', colors=coutl)  #setting up Y-axis tick color to black
    
    axs.spines['left'].set_color(coutl) 
    axs.spines['top'].set_color(coutl) 
    axs.spines['right'].set_color(coutl)
    axs.spines['bottom'].set_color(coutl)    

plt.savefig('./Documents/Figs morpho draft/poster_legend.pdf',dpi=600, bbox_inches=Bbox(bbox), transparent=True)
plt.show()


#%%