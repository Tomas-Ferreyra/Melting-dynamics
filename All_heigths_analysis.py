#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:17:46 2024

@author: tomasferreyrahauchar
"""

import numpy as np
import numpy.ma as ma

# from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, correlate, fftconvolve, peak_prominences, correlate2d, peak_prominences, savgol_filter
import scipy.ndimage as snd # from scipy.ndimage import rotate from scipy.ndimage import convolve
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline, Rbf, griddata, splrep, splev
from scipy.ndimage import maximum_filter

# import rawpy
import imageio
from tqdm import tqdm
from time import time
import h5py

from skimage.filters import gaussian, frangi, sato, hessian, meijering, roberts, sobel #, gabor
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion, thin, skeletonize, binary_dilation
from skimage.morphology import remove_small_objects, binary_opening, dilation
from skimage.segmentation import felzenszwalb, mark_boundaries, watershed
from skimage.restoration import unwrap_phase as unwrap
from skimage.feature import peak_local_max

import uncertainties as un
from uncertainties import unumpy

from PIL import Image, ImageDraw
import io
import cv2

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

plt.rcParams.update({'font.size':12})

# os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
# mpl.use('Agg') 


#%%

def nangauss(altus, sigma):
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
        
orderpol = 4 #order 4 seems bettter for wtershed. Order 2 gives slightly better results for tracking maxima

difs_v = []
for n in tqdm(range(len(salis_v))):
    difes = []
    for i in (range(len(ts_v[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_v,xns_v,yns_v, order=orderpol)
        cuapla = poln(coeff,xns_v[n],yns_v[n], order=orderpol) 
        
        difes.append( (hints_v[n][i]-cuapla) )

    difes = np.array(difes)
    difs_v.append(difes)

difs_t = []
for n in tqdm(range(len(salis_t))):
    difes = []
    for i in (range(len(ts_t[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_t,xns_t,yns_t, order=orderpol)
        cuapla = poln(coeff,xns_t[n],yns_t[n], order=orderpol) 
        
        difes.append( (hints_t[n][i]-cuapla) )

    difes = np.array(difes)
    difs_t.append(difes)

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

drho_rho_v = np.abs( density_millero(0, salis_v) - density_millero(temp_v, salis_v) ) / density_millero(temp_v, salis_v)
drho_rho_t = np.abs( density_millero(0, salis_t) - density_millero(temp_t, salis_t) ) / density_millero(temp_t, salis_t)

# not sure if I'm calculating R_rho correctly. This values are taken from simen paper
beta_t, beta_s = 6e-5, 7.8e-4 # 1/K, 1/(g/kg)
rrho_v = (beta_s * salis_v) / (beta_t * (temp_v + 273.15)) 
rrho_t = (beta_s * salis_t) / (beta_t * (temp_t + 273.15))

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

Ra_v = g * np.cos(angys_v * np.pi/180) * drho_rho_v * length0**3 / kt / mu 
Ra_t = g * np.cos(angys_t * np.pi/180) * drho_rho_t * length0**3 / kt / mu 
Ra_t[-7:-3] = g * np.cos(angys_t[-7:-3] * np.pi/180) * drho_rho_t[-7:-3] * length0**3 / kt / mu 

# beta_s = 7.8e-4 # (g/kg)^-1
# nu = 1.03e-6 # m^2 / s
# ks = kt/100 # m^2 / s

#%%
save_name = '' # 'melt_rates', 'nus'
reference = False
shadowgraphy = False
small_ice = True
axis_x = 'salinity'
axis_y = 'nu'

cols = np.linspace(-20,51,256)
comap = np.array( [(cols+20)/71 , 0.5 *np.ones_like(cols) , 1-(cols+20)/71 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,6), sharey=True)

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
                 color=((angys_v[n]+20)/71,0.5,1-(angys_v[n]+20)/71), capsize=3, mfc='w')
for n in [j for j in range(len(ds_t)) if j not in range(32,36)]:        
    ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='.', label=str(n)+'°', markersize=10, \
                  color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71), capsize=3)
if small_ice:
    for n in range(-7,-3):        
        ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='d', label=str(n)+'°', markersize=5, \
                      color=((angys_t[n]+20)/71,0.5,1-(angys_t[n]+20)/71), capsize=3)
        
cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-20, 51), cmap=newcmp), ax=ax[r'$a)$'], location='top')
# cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticks( list(range(-20,51,10)) )
cbar.set_label( label=r"$\theta$ (°)") #, size=12)

co2 = [(i/35,0,1-i/35) for i in salis]

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)


for n in [j for j in range(len(ds_t)) if j not in range(32,36)]: 
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='.', markersize=10,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \
for n in range(-7,-3):
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='d', markersize=6,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \

if shadowgraphy:
    # shadowgraphy experiments, they dont say much 
    ax[r'$b)$'].errorbar( [0,30], [0.017391, 0.017927], yerr=[0.000014, 0.000036], fmt='s', color='black' ) # "clear" (not really that clear)
    ax[r'$b)$'].errorbar( [0,30], [0.014225, 0.018498], yerr=[0.000014, 0.000021], fmt='d', color='black' ) # opaque

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$b)$'], location='top')
# cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticks( list(range(0,36,5)) )
cbar.set_label( label=r"$S$ (g/kg)") #, size=12)

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
# =============================================================================
# Tracking maxima (to see scallops velocity)
# =============================================================================

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
# for n,j in enumerate([7,6,5,28,23,8,9,15 ]):
for n,j in enumerate([5,6,7,8,9,15,23,27,28,32,33,35,38]):
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

fig, ax = plt.subplots()

li1ys = []
for i,j in enumerate(range(len(ds_v))):
    # if salis_v[j] > 7 and salis_v[j] < 25:
    if filv[j]:
        # li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=msd_v[j], fmt='o', capsize=2, \
        #               color=((angys_v[j]+20)/71,0.5,1-(angys_v[j]+20)/71), markersize=5, mfc='w')
        li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=0.097, fmt='o', capsize=2, \
                  color=((angys_v[j]+20)/71,0.5,1-(angys_v[j]+20)/71), markersize=5, mfc='w')
        li1ys.append(li1y)

li2ys = []
for i,j in enumerate(filt):
    # li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=msd_t[i], fmt='o', capsize=2., \
    #               color=((angys_t[j]+17)/47,0.5,1-(angys_t[j]+17)/47), markersize=5)
    if j in [32,33,35]:
        li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='d', capsize=2., \
                      color=((angys_t[j]+20)/71,0.5,1-(angys_t[j]+20)/71), markersize=5)
    else:
        li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='o', capsize=2., \
                      color=((angys_t[j]+20)/71,0.5,1-(angys_t[j]+20)/71), markersize=5)
    li2ys.append(li2y)

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

ax.legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5 )


# ax.grid()
ax.set_xlabel(r'$S$ (g/kg)')
ax.set_ylabel(r'$v_y$ (mm/s)')
# ax.set_ylim(top=0)
# plt.savefig('./Documents/Figs morpho draft/all_vely.png',dpi=400, bbox_inches='tight')
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
i = 60
exp = 'v'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(11.5,5) , sharex=False)

ns = [4,10,14]

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), aspect= dy/dx )
    
    topy,boty = np.max( (yns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (yns_b[n])[~np.isnan(hints_b[n][i])] )
    topx,botx = np.max( (xns_b[n])[~np.isnan(hints_b[n][i])] ), np.min( (xns_b[n])[~np.isnan(hints_b[n][i])] )
    midx = np.mean( (xns_b[n])[~np.isnan(hints_b[n][i])] )

    if j == 0: fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9, ticks=list(range(-41,-61,-3)))
    else: fig.colorbar(imhe, label='$h$ (mm)', location='right', shrink=0.9) 

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/profiles_s1.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%
i = 80
exp = 't'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(11.5,5) , sharex=False)

ns = [36,37,38]

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), aspect= dy/dx )
    
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
exp = 't'
gau = True

# ns = [1,1,1] #,10,18]
# ies = [20,40,60]

ns = [1,7,8]
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


fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(12,5) , sharex=False)

for j,(labels,axs) in enumerate(ax.items()):
    
    n = ns[j]
    i = ies[j]
    minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
    miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )
    dx = xns_b[n][0,1] - xns_b[n][0,0]
    dy = yns_b[n][0,0] - yns_b[n][1,0]
    if j ==0: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx )
    elif j == 1: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx )
    elif j == 2: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=None, vmin=None, aspect= dy/dx  )
    # elif j == 2: imhe = axs.imshow( hints_b[n][i], extent=(minx,maxx,miny,maxy), vmax=-85, vmin=-105  )
    
    if j == 0:
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
plt.scatter( np.roll(salis_t,3)[:-4], np.roll(angys_t,3)[:-4], c=np.roll(cols_t,3,axis=0)[:-4], marker='o' )
# plt.scatter( np.roll(salis_t,0)[:], np.roll(angys_t,0)[:], c=np.roll(cols_t,0)[:], marker='o' )
plt.scatter(salis_t[-7:-3], angys_t[-7:-3], c=cols_t[-7:-3], marker='d' )
plt.scatter(salis_v, angys_v, marker='o',  facecolors='none', edgecolors=cols_v )
plt.xlabel(r'$S$ (g/kg)')
plt.ylabel(r'$\theta$ (°)')
# plt.savefig('./Documents/Figs morpho draft/morphologs.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


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
n = 1 #1,5 ,11
i = 60

# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan

# halt = np.load('./Documents/Height profiles/ice_block_0_'+sal_v[n]+'.npy')
# halg = nangauss(halt, 2)

halt = np.load('./Documents/Height profiles/profile_s'+sal_t[n]+'_t'+inc_t[n]+'.npy')
halg = nangauss(halt, 2)

nt,ny,nx = np.shape(halt)

x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_v[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_v[n]
t = np.arange(nt) * 30
tr,yr,xr = np.meshgrid(t,y,x, indexing='ij')

xrp, yrp = xr - halg/Ls_v[n] * xr, yr - halg/Ls_v[n] * yr
# xrp, yrp = xr - halt/Ls_c[n] * xr, yr - halt_v/Ls[n] * yr
#%%
# i = 80
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
ax.plot( [-180,-180], [0-140,zli-140], [-yli/2,yli/2], 'k-' )

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
# plt.savefig('./Documents/blok0_30(2).png',dpi=192 * 5, transparent=True) # , bbox_inches='tight')

plt.show()
#%%

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
    
    ax.text(0,60,180, 't = '+str(0.5*i)+'min', fontsize=15)
    # ax.text(-101,60,200, 't = '+str(0.5*n)+'min', fontsize=15)
    
    # plt.savefig('imgifi'+str(n)+'.jpg',dpi=400, transparent=True) #, bbox_inches='tight')
    # lista_im.append(imageio.imread('imgifi.png') )#[:,:,0])
    # lista_im.append(  Image.open('imgifi'+str(n)+'.jpg') )#[:,:,0])
    # lista_im.append(  Image.frombytes('RGB', fig.canvas.get_width_height()[::-1],fig.canvas.tostring_rgb()) ) 
    # lista_im.append(  Image.frombytes('RGBa', fig.canvas.get_width_height(),fig.canvas.buffer_rgba()) ) 
    
    # lst = list(fig.canvas.get_width_height())[::-1]
    # lst.append(3)
    # imi = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8).reshape(lst) 
    # mim = Image.fromarray( imi )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)    
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    imi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mim = Image.fromarray( imi[70:1350,640:1500] )
    lista_im.append( mim )
    
# imageio.mimsave('./Documents/ice_s0_t0r.gif', lista_im, fps=10, format='gif')
# plt.close('all')

# plt.figure()
# plt.imshow(lista_im[0])
# plt.show()

frame_one = lista_im[0] #lista_im[0]
frame_one.save("./Documents/ice_s27_t0.gif", format="GIF", append_images=lista_im[1:],save_all=True, duration=5, loop=0)
#%%

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


wtssal_v, propscal_v, totars_v, kpropscal_v = [],[], [], []
for n in tqdm(range(len(ds_v))):
    wats, scaprop, scapropk = [], [], []
    
    halg = nangauss(hints_v[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_v[n]))*0.5, yns_v[n][:,0], xns_v[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    flp, fln = kurv>0, kurv<0 

    im_wat = nangauss( kurv * flp, [0,7,20] ) + (kurv * fln)
    
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
    

wtssal_t, propscal_t, totars_t, kpropscal_t = [],[], [], []
for n in tqdm(range(len(ds_t))):
    wats, scaprop, scapropk = [], [], []
    
    halg = nangauss(hints_t[n],5)
    gt,gy,gx = np.gradient( halg , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gyy,gyx = np.gradient( gy , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    _,gxy,gxx = np.gradient( gx , np.arange(len(hints_t[n]))*0.5, yns_t[n][:,0], xns_t[n][0])
    kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2) 
    flp, fln = kurv>0, kurv<0 

    im_wat = nangauss( kurv * flp, [0,7,20] ) + (kurv * fln)
    
    for i in range(len(difs_t[n])):
        # wts = watershed( -im_wat[i], mask = binary_erosion( ~np.isnan(kurv[i]), disk(5) ) )
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

nss_v = list(range(len(ds_v)))
lxs_v,lys_v = [],[]
sscas_v, centss_v, centws_v, nscas_v, nscafs_v, labs_v = [], [], [], [], [], []
ssds_v, smms_v, smes_v = [], [], []

for n in tqdm(nss_v):
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm, sme = [], [], []

    scaprop = propscal_v[n]
    scapropk = kpropscal_v[n]
    for i in range(len(scaprop)):
        cen, cenw, scas, slab = [], [], [], []
        nsd, sd, mm = [], [], []
        me, nijs = [], []
        
        for j in range(len(scaprop[i])):
            sarea = scaprop[i][j].area
            ksd = scapropk[i][j].image_stdev
            if sarea > 1000 and sarea < 12000 and ksd > 0.036:
                cen.append( scaprop[i][j].centroid )
                cenw.append( scaprop[i][j].centroid_weighted )
                scas.append( sarea )
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
            sarea = scaprop[i][j].area
            ksd = scapropk[i][j].image_stdev
            if sarea > 1000 and sarea < 12000 and ksd > 0.03:
                cen.append( scaprop[i][j].centroid )
                cenw.append( scaprop[i][j].centroid_weighted )
                scas.append( sarea )
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
n = 7
i = 60

exp = 't'
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


ax[r'$b)$'].set_ylim(10.2,48)
ax[r'$c)$'].set_ylim(10.2,48)
ax[r'$b)$'].sharex(ax[r'$c)$'])
ax[r'$b)$'].tick_params(axis='x',length=3,labelsize=0)

ax[r'$b)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$c)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$c)$'].set_xlabel(r'$S$ (g/kg)')


# ax[r'$b)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
#                     bbox_to_anchor=(0.48,1.3), loc='upper center' , ncol=5, columnspacing = 0.5 )
ax[r'$b)$'].legend([li2ys[7],li2ys[2],li2ys[1],li2ys[0]],[r'$-15°$', r'$0°$',r'$15°$',r'$30°$'],\
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
# =============================================================================
# Graph Area
# =============================================================================

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$a)$',r'$b)$',r'$b)$',r'$b)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharex=True)

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
barviolin( sarms_v, ax[r'$b)$'], x=salis_v[ns_v], bins=30, width=7, color='blue') #, labe='Set 1' )
barviolin( sarms_t, ax[r'$b)$'], x=salis_t[ns_t], bins=30, width=7, color='red' ) #, labe='Set 2')
# plt.xticks(salis_v[ns_v])
# plt.xticks(salis_t[ns_t])
# ax[r'$b)$'].set_ylim(0,25)
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
fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$']], layout='tight', figsize=(12/1.,5/1.) ) #, sharey=True ) #, sharex=True)

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

ave = 10
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

# ax[r'$b)$'].legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
#                    bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing=0.5 )
ax[r'$b)$'].legend([li2s[7],li2s[2],li2s[1],li2s[0]],[r'$-15°$', r'$0°$',r'$15°$',r'$30°$'],\
                    bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing = 0.5 )


for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/amplitudes.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()
#%%












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


