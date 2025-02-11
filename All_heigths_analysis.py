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
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion, thin, skeletonize
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
       '35','0','6','12','12','20']
inc_t = ['0(3)','30','15(3)','45(2)','45(ng)','30','15','0','0','15','45','30','30','45','15','0','0','15','30','45','15(4)','15(5)',
       '45(2)','0(2)','30(2)','30','n15','n15','n15','n15','n15(2)','n15','0(s)','0(s)','30(s)','0(s)']
salis = [0, 0, 0, 0, 6.8, 7.0, 6.8, 6.8, 13.2, 13.3, 13.2, 13.3, 20.1, 20.1, 19.8, 20.0, 26.1, 26.1, 26.2, 26.2, 0, 0, 6.9, 12.9, 20.1,34.5,34.5,19.7, 
         13.0, 7.0, 34.3, 0, 5.8, 12.3, 12.4, 18.5]
ang = [0,30,15,45,45,30,15,0,0,15,45,30,30,45,15,0,0,15,30,45,15,15, 45, 0, 30, 30, -15, -15, -15, -15, -15, -15, 0, 0, 30, 0 ]
ds_t = [ 0.4019629986999602, 0.4079970860258477, 0.400784094531674, 0.4019866341895751, 0.4140232465732999, 0.4108146995511450, 0.405185916205820,
       0.3985171779558734, 0.4082777044681367, 0.399457458948112, 0.429624677624675, 0.4002974355025642, 0.3962951595980395, 0.4158467162824917,
       0.405560070548485, 0.406690755428839, 0.3986160751319692, 0.406029247234490, 0.403935349493398, 0.4274366684657002, 0.39842940043484637,
       0.3944444940666371, 0.42941247988993125, 0.3986508813811391, 0.41300121024756764, 0.39724780723266606, 0.3991597643255994, 0.4114831458151559,
       0.4034206887886922, 0.3963784515420281, 0.406148619144839, 0.40500559156677696, 
       0.4038900423099877, 0.4088590866221042, 0.40465748645946387, 0.4070953514695665]

Ls = [ 2099.9999966941955, 2097.2250082592454, 2100.003821154219, 2098.7729533565816, 2092.3116009701507, 2100.7921001119767, 2102.061867627691,
       2104.0217430207585, 2097.0613260850837, 2108.070882626512, 2101.360857870178, 2106.0608525686807, 2103.8609505143863, 2158.7930487367967,
       2112.093781969593, 2113.181455334151, 2104.4329627701713, 2092.891945129341, 2103.955261899692, 2107.0490635783035, 2100.163205198002,
       2102.2972076507995, 2103.9171613871736, 2099.967887686282, 2110.996676678041, 2099.255549750563, 2098.0517570404286, 2102.9647844012497,
       2101.945064364561, 2100.341294595434, 2103.0141188891594, 2101.2510931269435,
       2100.01045350225, 2103.588545842329, 2100.408699264378, 2103.0207273047863]

angys = [0.2752835802019292, 34.47170331852835, 13.06499750709190, 43.6538844493011, 42.63317480202235, 29.31556395720869, 15.64436382383116, 
         0.379506977103330, 2.122818303993659, 16.82489775638404, 47.66715755358717, 28.76640507341181, 31.15486854660724, 50.65480071850850, 
         19.29185570378762, -0.876278847660974, -1.445891941223953, 15.68633111320458, 28.61178195105051, 46.98820258183241, 14.85643305810015,
         18.694159382293858, 44.69885882480372, 0.4091221586816418, 29.57628972829382, 29.370955852274992, -18.732894940242627, -16.522627138169074,
         -16.27828076769843, -17.33545209415416, -16.58142778295442, -17.450360846901734,
         -1.6926163253504223, -2.671232787825012, 27.052620463266, 5.031218324124902]

temp = [19.0, 19.0, 19.3, 19.7, 20.1, 20.3, 19.5, 20.0, 19.0, 19.0, 19.7, 19.4, 18.8, 19.2, 19.8, 19.5, 19.8, 19.4, 19.1, 20.0, 20.1, 19.4,
        19.4, 19.3, 19.2, 19.2, 18.7, 19.3, 18.9, 19.0, 19.1, 19.3, 19.2, 19.1, 18.8, 19.3]

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


with h5py.File('/Users/tomasferreyrahauchar/Documents/Height profiles/npys/sloped_heights(s0)_all.hdf5', 'r') as f:

    hints_t, xns_t, yns_t, ts_t = [], [], [], []
    for n in tqdm(range(len(salis_t))):
        
        hints_t.append( f['h'][n].reshape( f['n'][n] ) )
        xns_t.append( f['x'][n].reshape( (f['n'][n])[1:] ) )
        yns_t.append( f['y'][n].reshape( (f['n'][n])[1:] ) )
        ts_t.append( np.arange(0, (f['n'][n])[0] ) * 30 )
        
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
thcon = 0.6 # m kg / s^3 °C
mes_v, mes_t = np.array(mes_v), np.array(mes_t)
Nu_v = -mes_v/1000 * rho_ice * latent * length0 / (thcon * temp_v )
Nu_t = -mes_t/1000 * rho_ice * latent * length0 / (thcon * temp_t )

#%%
save_name = 'melt_rates' # 'melt_rates'
reference = False
shadowgraphy = False
small_ice = True
axis_x = 'salinity'
axis_y = 'nu'

cols = np.linspace(-19,51,256)
comap = np.array( [(cols+19)/70 , 0.5 *np.ones_like(cols) , 1-(cols+19)/70 , 1*np.ones_like(cols) ] )
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
                 color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), capsize=3, mfc='w')
for n in range(len(ds_t)-4):        
    ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='.', label=str(n)+'°', markersize=10, \
                  color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70), capsize=3)
if small_ice:
    for n in range(-4,0):        
        ax[r'$a)$'].errorbar(xvariable_t[n], yvariable_t[n] * 1 , yerr=yvarerr_t[n], fmt='d', label=str(n)+'°', markersize=5, \
                      color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70), capsize=3)
        
cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax[r'$a)$'], location='top')
# cbar.ax.tick_params(labelsize=12)
cbar.ax.set_xticks( list(range(-10,51,10)) )
cbar.set_label( label=r"$\theta$ (°)") #, size=12)

co2 = [(i/35,0,1-i/35) for i in salis]

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)


for n in range(len(ds_t)-4):
    ax[r'$b)$'].errorbar( angys_t[n],  yvariable_t[n], yerr=yvarerr_t[n] ,fmt='.', markersize=10,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \
for n in range(-4,0):
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
# Wavelength with distance between max
# =============================================================================
difs_v = []
for n in tqdm(range(len(salis_v))):
    difes = []
    for i in (range(len(ts_v[n]))):
        coeff, r, rank, s = polyfit(n,i,hints_v,xns_v,yns_v)
        cuapla = pol2(coeff,xns_v[n],yns_v[n]) 
        
        difes.append( (hints_v[n][i]-cuapla) )

    difes = np.array(difes)
    difs_v.append(difes)

difs_t = []
for n in tqdm(range(len(salis_t))):
    difes = []
    for i in (range(len(ts_t[n]))):
        coeff, r, rank, s = polyfit(n,i,hints_t,xns_t,yns_t)
        cuapla = pol2(coeff,xns_t[n],yns_t[n]) 
        
        difes.append( (hints_t[n][i]-cuapla) )

    difes = np.array(difes)
    difs_t.append(difes)

#%%
lmeas_v, lmeds_v, lstds_v = [],[],[]
for n in tqdm(range(len(ds_v))):
    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    lmea,lmed,lsd = [],[],[]
    for i in range(len(t)):
        lons =[]
        for l in range(len(x)):
            line = difs_v[n][i,:,l]
            # pek = find_peaks(line, prominence=0.5)[0]
            pek = find_peaks(line, prominence=1.0)[0]
            long = y[pek][:-1] - y[pek][1:]
            lons += list(long) 
        lmea.append( np.mean(lons) )
        lmed.append( np.median(lons) )
        lsd.append( np.std(lons) )
    lmeas_v.append(lmea)
    lmeds_v.append(lmed)
    lstds_v.append(lsd)

lmeas_t, lmeds_t, lstds_t = [],[],[]
for n in tqdm(range(len(ds_t))):
    t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
    lmea,lmed,lsd = [],[],[]
    for i in range(len(t)):
        lons =[]
        for l in range(len(x)):
            line = difs_t[n][i,:,l]
            # pek = find_peaks(line, prominence=0.5)[0]
            pek = find_peaks(line, prominence=1.0)[0]
            long = y[pek][:-1] - y[pek][1:]
            lons += list(long) 
        lmea.append( np.mean(lons) )
        lmed.append( np.median(lons) )
        lsd.append( np.std(lons) )
    lmeas_t.append(lmea)
    lmeds_t.append(lmed)
    lstds_t.append(lsd)
    
#%%

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,5))

ini = 0
# for i,j in enumerate(range(len(ds))):
#     # if salis[j] > 5 and salis[j] < 25:
#     # if salis[j] > 5 and salis[j] < 10:
#     # if salis[j] < 5 or salis[j] > 25:
        
for i,j in enumerate([0,3,8,10,2,13]):
    ax[r'$a)$'].errorbar(ts_v[j][ini:]/60, lmeas_v[j][ini:], yerr=lstds_v[j][ini:], capsize=2, fmt='.-', label=str(salis_v[j])+' g/kg', \
                          errorevery=(5*i,30), color=np.array([0.5,salis_v[j]/27.4,1-salis_v[j]/27.4]) ) #, markersize=5, mfc='w' )
        
# # # for i,j in enumerate([7,6,5,28,23,8,9,15 ]):
# for i,j in enumerate([7,8,15]):
#     ax[r'$a)$'].errorbar(ts_t[j][ini:]/60, lmeas_t[j][ini:], yerr=lstds_t[j][ini:], capsize=2, fmt='.-', \
#     label=str(salis_t[j])+' g/kg; '+str(ang_t[j])+'°', \
#                   errorevery=(5+9*i,55), color=np.array([0.5,salis_t[j]/26.2,1-salis_t[j]/26.2]) * (1 - (ang_t[j])/70 ) )


# plt.grid()
ax[r'$a)$'].legend(loc='upper right')#fontsize=12)
ax[r'$a)$'].set_ylabel(r'$\lambda$ (mm)')#, fontsize=12)
ax[r'$a)$'].set_xlabel('t (min)')#, fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlim(10,70)
# plt.ylim(0,90)
# plt.ylim(0,70)
ax[r'$a)$'].set_ylim(bottom=0)
# plt.gca().set_ylim(bottom=0)
# plt.savefig('./Documents/wals.png',dpi=400, bbox_inches='tight')
# plt.show()

cols = np.linspace(-19,51,256)
comap = np.array( [(cols+19)/70 , 0.5 *np.ones_like(cols) , 1-(cols+19)/79 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

for i,j in enumerate(range(len(ds_v))):
    if salis_v[j] > 7 and salis_v[j] < 25:
        argu =  np.argmin( lmeas_v[j][:])
        
        ax[r'$b)$'].errorbar(salis_v[j], lmeas_v[j][argu], yerr=lstds_v[j][argu], capsize=2, fmt='o', markersize=5, \
                  color=((angys_v[j]+19)/70,0.5,1-(angys_v[j]+19)/70), mfc='w' )

# for i,j in enumerate(range(len(ds_t))): 
for i,j in enumerate([7,6,5,28,23,8,9,15 ]):
    # if salis_t[j] > 7 and salis_t[j] < 25:
    argu =  np.argmin( lmeas_t[j][:])
    
    ax[r'$b)$'].errorbar(salis_t[j], lmeas_t[j][argu], yerr=lstds_t[j][argu], capsize=2, fmt='o', \
              color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5)

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax[r'$b)$'])
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

# plt.grid()
ax[r'$b)$'].set_ylabel(r'$\lambda$ (mm)')#, fontsize=12)
ax[r'$b)$'].set_xlabel('S (g/kg)')#, fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlim(10,70)
# plt.ylim(0,90)
ax[r'$b)$'].set_ylim(0,72)

for labels,axs in ax.items():
    axs.annotate(labels, (-0.05,1), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/all_wale.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Amplitud scallops
# =============================================================================
prmm_v, prme_v = [], []
for n in tqdm(range(len(ds_v))):
    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    promm, prome = [],[]
    for i in range(len(t)):
    
        hice = hints_v[n][i]
        
        prom = []
        for j in range(len(x)):
            linea = hice[:,j]
            argp = find_peaks(linea,distance=10, prominence=1.3)[0]
            prom = prom + list( peak_prominences(linea, argp, wlen=50)[0] )
    
        promm.append(np.mean(prom)) 
        prome.append(np.std(prom))
    prmm_v.append(promm)
    prme_v.append(prome)

prmm_t, prme_t = [], []
for n in tqdm(range(len(ds_t))):
    t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
    promm, prome = [],[]
    for i in range(len(t)):
    
        hice = hints_t[n][i]
        
        prom = []
        for j in range(len(x)):
            linea = hice[:,j]
            argp = find_peaks(linea,distance=10, prominence=1.3)[0]
            prom = prom + list( peak_prominences(linea, argp, wlen=50)[0] )
    
        promm.append(np.mean(prom)) 
        prome.append(np.std(prom))
    prmm_t.append(promm)
    prme_t.append(prome)

#%%

fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,5))

ini = 0
# for i,j in enumerate(range(len(ds))):
#     # if salis[j] > 5 and salis[j] < 25:
#     # if salis[j] > 5 and salis[j] < 10:
#     # if salis[j] < 5 or salis[j] > 25:
        
for j in [8,10,2]:
    ax[r'$a)$'].errorbar(ts_v[j]/60, prmm_v[j], yerr=prme_v[j] , fmt='.-', capsize=2, label=str(salis_v[j])+' g/kg', \
                  color=np.array([0.5,salis_v[j]/27.4,1-salis_v[j]/27.4]) )

# for i,j in enumerate([7,6,5,28,23,8,9,15 ]):
for i,j in enumerate([7,8,15]):
# for i,j in enumerate([7,6,5]):
    ax[r'$a)$'].errorbar(ts_t[j]/60, prmm_t[j], yerr=prme_t[j] , fmt='.-', capsize=2,label=str(salis_t[j])+' g/kg; '+str(ang_t[j])+'°', \
                  color=np.array([0.5,salis_t[j]/27.4,1-salis_t[j]/27.4]) * (1 - (ang_t[j])/70 ) )

# plt.grid()
ax[r'$a)$'].legend(loc='upper left')#fontsize=12)
ax[r'$a)$'].set_ylabel(r'$A$ (mm)')#, fontsize=12)
ax[r'$a)$'].set_xlabel('t (min)')#, fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlim(10,70)
# plt.ylim(0,90)
# plt.ylim(0,3.5)
ax[r'$a)$'].set_ylim(bottom=0)
# plt.savefig('./Documents/wals.png',dpi=400, bbox_inches='tight')
# plt.show()


cols = np.linspace(-19,51,256)
comap = np.array( [(cols+19)/70 , 0.5 *np.ones_like(cols) , 1-(cols+19)/79 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

# plt.figure()
for i,j in enumerate(range(len(ds_v))):
    if salis_v[j] > 7 and salis_v[j] < 25:
        argu =  np.argmax( prmm_v[j] )
        
        print(salis_v[j], prmm_v[j][argu])
        
        # ax[r'$b)$'].errorbar(salis_v[j], prmm_v[j][argu], yerr=prme_v[j][argu], capsize=2, fmt='o', \
        #           color=((angys_v[i]+19)/70,0.5,1-(angys_v[i]+19)/70), markersize=5, mfc='w' )
        ax[r'$b)$'].errorbar(salis_v[j], prmm_v[j][argu], yerr=0.33, capsize=2, fmt='o', \
                  color=((angys_v[j]+19)/70,0.5,1-(angys_v[j]+19)/70), markersize=5, mfc='w' )

for i,j in enumerate([7,6,5,28,23,8,9,15 ]):
    argu =  np.argmax( prmm_t[j] )
    
    # ax[r'$b)$'].errorbar(salis_t[j], prmm_t[j][argu], yerr=prme_t[j][argu], capsize=2, fmt='o', \
    #           color=((angys_t[i]+19)/70,0.5,1-(angys_t[i]+19)/70), markersize=5 )
    ax[r'$b)$'].errorbar(salis_t[j], prmm_t[j][argu], yerr=0.33, capsize=2, fmt='o', \
              color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5 )

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax[r'$b)$'])
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

# plt.grid()
ax[r'$b)$'].set_ylabel(r'$A$ (mm)')#, fontsize=12)
ax[r'$b)$'].set_xlabel('t (min)')#, fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlim(10,70)
# plt.ylim(0,4.2)
# ax[r'$b)$'].set_ylim(bottom=0)

for labels,axs in ax.items():
    axs.annotate(labels, (-0.05,1.01), xycoords = 'axes fraction')

# plt.savefig('./Documents/all_amp.png',dpi=400, bbox_inches='tight')
plt.show()


#%%
# rms = []
# for i in range(difs_v[n]):

srms = []
    
for n in tqdm(range(len(salis_v))):
    rms = []
    for i in range(len(difs_v[n])):
        rms.append( np.nanstd( hints_v[n][i] ) )
    srms.append( rms )

plt.figure()
for n in range(len(salis_v)):
    plt.plot(srms[n], label=n, color=(salis_v[n]/28, 0, 1-salis_v[n]/28))
plt.legend()
plt.show()


srms = []
    
for n in tqdm(range(len(salis_t))):
    rms = []
    for i in range(len(difs_t[n])):
        rms.append( np.nanstd( hints_t[n][i] ) )
    srms.append( rms )

plt.figure()
for n in range(len(salis_t)):
    plt.plot(srms[n], label=n, color=(salis_t[n]/35, 0, 1-salis_t[n]/35))
plt.legend()
plt.show()
#%%
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
    
#%%

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
for n,j in enumerate([7,6,5,28,23,8,9,15 ]):
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

#%%
mve_v, msd_v = np.array(mve_v), np.array(msd_v)
mxe_v, mxd_v = np.array(mxe_v), np.array(mxd_v)

mve_t, msd_t = np.array(mve_t), np.array(msd_t)
mxe_t, mxd_t = np.array(mxe_t), np.array(mxd_t)

filv = (salis_v > 7) * (salis_v < 25)
# fil = salis_v<50
filt = [7,6,5,28,23,8,9,15]

cols = np.linspace(-19,51,256)
comap = np.array( [(cols+19)/70 , 0.5 *np.ones_like(cols) , 1-(cols+19)/79 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig, ax = plt.subplots()

li1ys = []
for i,j in enumerate(range(len(ds_v))):
    if salis_v[j] > 7 and salis_v[j] < 25:
        li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=msd_v[j], fmt='o', capsize=2, \
                      color=((angys_v[j]+19)/70,0.5,1-(angys_v[j]+19)/70), markersize=5, mfc='w')
        #li1y = ax.errorbar(salis_v[j], mve_v[j], yerr=0.097, fmt='o', capsize=2, \
        #           color=((angys_v[j]+19)/70,0.5,1-(angys_v[j]+19)/70), markersize=5, mfc='w')
        li1ys.append(li1y)

li2ys = []
for i,j in enumerate(filt):
    li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=msd_t[i], fmt='o', capsize=2., \
                  color=((angys_t[j]+17)/47,0.5,1-(angys_t[j]+17)/47), markersize=5)
    # li2y = ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='o', capsize=2., \
    #               color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5)
    li2ys.append(li2y)

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

ax.legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5 )


# ax.grid()
ax.set_xlabel('S (g/kg)')
ax.set_ylabel(r'$v_y$ (mm/s)')
# ax.set_ylim(top=0)
# plt.savefig('./Documents/all_vely.png',dpi=400, bbox_inches='tight')
plt.show()


# plt.figure()

# plt.errorbar(salis_v[filv], mxe_v[filv], yerr=mxd_v[filv], fmt='o', capsize=2, \
#              color=((angys_v[i]+19)/70,0.5,1-(angys_v[i]+19)/70), markersize=5, mfc='w')

# for i,j in enumerate(filt):
#     plt.errorbar(salis_t[j], mxe_t[i], yerr=mxd_t[i], fmt='o', capsize=2., \
#                  color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5)

# plt.grid()
# plt.xlabel('S (g/kg)')
# plt.ylabel(r'$v_x$ (mm/s)')
# # plt.savefig('./Documents/velsvsal_min.png',dpi=400, bbox_inches='tight')
# plt.show()


#%%
n = 7
i = 41

dife = np.copy( difs_t[n] )
# dife = np.copy( hints_t[n][i] )
dife[np.isnan(dife)] = -1000

my,mx = peak_local_max( dife[i], labels=~np.isnan(hints_t[n][i]), min_distance=18 ).T


plt.figure()
plt.imshow( difs_t[n][i] )
plt.plot(mx,my,'k.')
plt.show()
#%%
# =============================================================================
# Correlation gt gy
# =============================================================================
mmm_v = []
for n,ss in tqdm(enumerate(salis_v)):
    mm = []
    for i in range(len(hints_v[n])):
        mmm = binary_erosion( ~np.isnan(hints_v[n][i]), disk(100) ) *1.
        mmm[ mmm==0.0 ] = np.nan
        mm.append(mmm)
    mmm_v.append( np.array(mm) )
        


#%%
coexs_v,coeys_v = [],[]
for n,ss in tqdm(enumerate(salis_v)):
    
    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    gt,gy,gx = np.gradient( nangauss(hints_v[n],10) , t,y,x)
    
    gt = ma.masked_invalid(gt)    
    gy = ma.masked_invalid(gy)    
    gx = ma.masked_invalid(gx)    

    coey, coex = [], []
    for i in range(len(gt)):
        coey.append( ma.corrcoef( gy[i].flatten(), gt[i].flatten() )[0,1] )
        coex.append( ma.corrcoef( gx[i].flatten(), gt[i].flatten() )[0,1] )
    coexs_v.append(coex)
    coeys_v.append(coey)
    
#%%

plt.figure()
ccm = []
# for n,ss in enumerate(salis_v):
for n in [0,3,8]:
    ss = salis_v[n]
    plt.plot(ts_v[n]/60, coeys_v[n], '-', c=(salis_v[n]/35,0,0), label=str(ss)+' g/kg')  #label='dh/dy')
    # plt.plot(ts_v[n]/60 ,coexs_v[n], '--', color=(salis_v[n]/35,0,0))  #label='dh/dy')
    # plt.plot(t/60, coex, label='dh/dx')
    coei = coeys_v[n]
    ccm.append( coei[np.nanargmax(np.abs(coei))] )
    
plt.xlabel('time (min)',fontsize=12)
plt.ylabel('Correlation',fontsize=12)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
# plt.savefig('./Documents/corrs8.png',dpi=400, bbox_inches='tight')
plt.show()    

#%%
plt.figure()
plt.plot(salis, ccm, 'k.' )
plt.grid()
plt.xlabel('Salinity (g/kg)',fontsize=12)
plt.ylabel('Maximum correlation',fontsize=12)
# plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/corrmax.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
n = 0
i = 60

t1 = time()
mmm = binary_erosion( ~np.isnan(hints_v[n][i]), disk(5) ) *1.
mmm[ mmm==0.0 ] = np.nan

t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
gt,gy,gx = np.gradient( nangauss(hints_v[n],5) , t,y,x)
# t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
# gt,gy,gx = np.gradient( nangauss(hints_t[n],5) , t,y,x)

gt = ma.masked_invalid(gt * mmm)    
gy = ma.masked_invalid(gy * mmm)    
gx = ma.masked_invalid(gx * mmm)    
t2 = time()
print( ma.corrcoef( gy[i].flatten(), gt[i].flatten() ) )
t2-t1

#%%
ll = 220
print( ma.corrcoef( gy[i].flatten(), gt[i].flatten() ) )
print( ma.corrcoef( gy[i,100:-100,ll], gt[i,100:-100,ll] ) )

# plt.figure()
# # plt.scatter( gy[i].flatten(), gt[i].flatten() )
# plt.scatter( gy[i,:,ll], gt[i,:,ll] )
# plt.scatter( gy[i,100:-100,ll], gt[i,100:-100,ll] )
# plt.show()


# plt.figure()
# plt.imshow(gy[i])
# plt.show()
# plt.figure()
# plt.imshow(gt[i])
# plt.show()

# fig, ax1 = plt.subplots(figsize=(12,5))

# color = 'tab:red'
# ax1.set_xlabel('y (mm)')
# ax1.set_ylabel('h (mm)', color=color)
# # ax1.set_ylabel('dh/dy', color=color)
# ax1.plot(y, hints_v[n][i,:,ll], color=color)
# # ax1.plot(y, gy[i,:,ll], color=color)
# # ax1.plot( gy[i,:,ll], color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('-dh/dt (mm/min)', color=color)  # we already handled the x-label with ax1
# ax2.plot(y, -gt[i,:,ll] * 60, color=color)
# # ax2.plot( gt[i,:,ll] * 60, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# # ax1.invert_xaxis()
# ax1.grid()
# # plt.savefig('./Documents/scamelt.png',dpi=400, bbox_inches='tight')
# plt.show()

fig, ax1 = plt.subplots(figsize=(3.5,8))

color = 'tab:red'
ax1.set_ylabel('y (mm)')
# ax1.set_xlabel('h (mm)', color=color)
ax1.set_xlabel('dh/dy', color=color)
# ax1.plot(hints_v[n][i,:,ll], y, color=color)
ax1.plot(gy[i,:,ll], y, color=color)
# ax1.plot( gy[i,:,ll], color=color)
ax1.tick_params(axis='x', labelcolor=color)

ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_xlabel('-dh/dt (mm/min)', color=color)  # we already handled the x-label with ax1
ax2.plot(-gt[i,:,ll] * 60, y, color=color)
# ax2.plot( gt[i,:,ll] * 60, color=color)
ax2.tick_params(axis='x', labelcolor=color)

# ax1.invert_xaxis()
# ax1.grid()
plt.savefig('./Documents/gymelt.png',dpi=400, bbox_inches='tight')
plt.show()


# plt.figure()
# ylt,ylb = 343, 423
# cmap = plt.get_cmap('bwr')
# norm = plt.Normalize(np.nanmin(-gt[i,ylt:ylb,ll]*60), np.nanmax(-gt[i,ylt:ylb,ll]*60))
# line_colors = cmap( norm(-gt[i,ylt:ylb,ll]*60) )

# lines = plt.scatter(hints_v[n][i,ylt:ylb,ll], y[ylt:ylb], color=line_colors)

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=plt.Normalize(np.nanmin(-gt[i,ylt:ylb,ll]*60), np.nanmax(-gt[i,ylt:ylb,ll]*60)), cmap=cmap )) #
#                     # , ax=ax[r'$a)$'], location='top')
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label=r"-dh/dt (mm/min)") #, size=12)

# # plt.ylim(25,55)
# # plt.xlim(-6,-1)
# plt.axis('equal')

# # fig1, ax1 = plt.subplots()
# # lines = colored_line(hints_v[n][i,:,ll], y, -gt[i,:,ll]*60, ax1, linewidth=10, cmap="viridis")
# # fig1.colorbar(lines)  # add a color legend

plt.show()


fig, axs = plt.subplots()
minx,maxx = np.min( xns_v[n] ), np.max( xns_v[n] )
miny,maxy = np.min( yns_v[n] ), np.max( yns_v[n] )
ims2 = axs.imshow(hints_v[n][i], extent=(minx,maxx,miny,maxy) )#, vmax=5) #, vmin=-5)

topy,boty = np.max( (yns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (yns_v[n])[~np.isnan(hints_v[n][i])] )
topx,botx = np.max( (xns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (xns_v[n])[~np.isnan(hints_v[n][i])] )
midx = np.mean( (xns_v[n])[~np.isnan(hints_v[n][i])] )

axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
# axs.plot([x[ll],x[ll]],[y[ylt],y[ylb]],'r-', linewidth=3 )
axs.plot([x[ll],x[ll]],[topy,boty],'r-', linewidth=3 )
axs.text(midx-20, boty-27, '5 cm')
axs.axis('off')
axs.set_ylim( top = topy + 5 )
axs.set_xlim(botx-2,topx+2)


fig.subplots_adjust(left=0.2)
cbar_ax = fig.add_axes([0.38, 0.18, 0.015, 0.65])
fig.colorbar(ims2, cax=cbar_ax, label='height (mm)', location='left')
# plt.savefig('./Documents/h830.png', dpi=400, bbox_inches='tight', transparent=True)
plt.show()

# fig, axs = plt.subplots()
# minx,maxx = np.min( xns_v[n] ), np.max( xns_v[n] )
# miny,maxy = np.min( yns_v[n] ), np.max( yns_v[n] )
# ims2 = axs.imshow(-(gt[i] - np.nanmean(gt[i])) * 60, extent=(minx,maxx,miny,maxy), cmap='bwr' )#, vmax=5) #, vmin=-5)
# topy,boty = np.max( (yns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (yns_v[n])[~np.isnan(hints_v[n][i])] )
# topx,botx = np.max( (xns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (xns_v[n])[~np.isnan(hints_v[n][i])] )
# midx = np.mean( (xns_v[n])[~np.isnan(hints_v[n][i])] )
# axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
# axs.text(midx-20, boty-27, '5 cm')
# axs.axis('off')
# axs.set_ylim( top = topy + 5 )
# axs.set_xlim(botx-2,topx+2)
# fig.subplots_adjust(left=0.2)
# cbar_ax = fig.add_axes([0.38, 0.18, 0.015, 0.65])
# fig.colorbar(ims2, cax=cbar_ax, label='-dh/dt (mm/min)', location='left')
# # plt.savefig('./Documents/gt830.png', dpi=400, bbox_inches='tight', transparent=True)
# plt.show()

# fig, axs = plt.subplots()
# minx,maxx = np.min( xns_v[n] ), np.max( xns_v[n] )
# miny,maxy = np.min( yns_v[n] ), np.max( yns_v[n] )
# ims2 = axs.imshow(gy[i], extent=(minx,maxx,miny,maxy) )#, vmax=5) #, vmin=-5)
# topy,boty = np.max( (yns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (yns_v[n])[~np.isnan(hints_v[n][i])] )
# topx,botx = np.max( (xns_v[n])[~np.isnan(hints_v[n][i])] ), np.min( (xns_v[n])[~np.isnan(hints_v[n][i])] )
# midx = np.mean( (xns_v[n])[~np.isnan(hints_v[n][i])] )
# axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
# axs.text(midx-20, boty-27, '5 cm')
# axs.axis('off')
# axs.set_ylim( top = topy + 5 )
# axs.set_xlim(botx-2,topx+2)
# fig.subplots_adjust(left=0.2)
# cbar_ax = fig.add_axes([0.38, 0.18, 0.015, 0.65])
# fig.colorbar(ims2, cax=cbar_ax, label=r'dh/dy', location='left')
# # plt.savefig('./Documents/gy830.png', dpi=400, bbox_inches='tight', transparent=True)
# plt.show()


#%%
# =============================================================================
# Nu vs Ra with time
# =============================================================================
llss_v, lfit_v = [], []
mfis_v = []

for n in range(len(ds_v)):
    t1 = time()
    halt = np.load('./Documents/Height profiles/ice_block_0_'+sal_v[n]+'.npy')
    halg = nangauss(halt, 2)
    
    nt,ny,nx = np.shape(halt)
    
    x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_v[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_v[n]
    t = np.arange(nt) * 30
    
    # if ang[n] > 0: mmm = area_interest(halg, t, x, y, sob=10000)
    mmm = area_interest_0a(halg, t, x, y, sob=10000) 
    
    lls = []
    for i in range(len(ts_v[n])):
        icebox = ~np.isnan(halt[i] * mmm[i])
        lens = np.sum(icebox,axis=0)
        # filt = lens>380
        # lls.append(( np.mean(lens[filt]) * ds[n] ) / np.cos(angys[n]*np.pi/180) )
        if n == 2: lls.append(( np.mean(lens[300:450]) * ds_v[n] ) / np.cos(angys_v[n]*np.pi/180) )
        elif n == 6: lls.append(( np.mean(lens[350:500]) * ds_v[n] ) / np.cos(angys_v[n]*np.pi/180) )
        else: lls.append(( np.mean(lens[350:550]) * ds_v[n] ) / np.cos(angys_v[n]*np.pi/180) )
        
    A = np.array([t*0+1,t,t**2,t**3]).T
    co, r, rank, s = np.linalg.lstsq(A, lls, rcond=None)
    lfi = co[0] + co[1]*t + co[2]*t**2 + co[3]*t**3 
        
    lfit_v.append(lfi)
    llss_v.append(lls)
    
    
    mhe = np.nanmean(hints_v[n], axis=(1,2) )
    a = ts_v[n]
    A = np.array([a*0+1,a,a**2,a**3]).T
    co, r, rank, s = np.linalg.lstsq(A, mhe, rcond=None)
    # mfi = co[0] + co[1]*a + co[2]*a**2 + co[3]*a**3 
    mfi_d = co[1] + 2*co[2]*a + 3*co[3]*a**2 
    mfis_v.append(mfi_d)
    
    t2 = time()
    print(n,t2-t1)


# llss_t, lfit_t = [], []
# mfis_t = []

# for n in range(len(ds_t)):
#     t1 = time()
#     halt = np.load('./Documents/Height profiles/profile_s'+sal_t[n]+'_t'+inc_t[n]+'.npy')
#     halg = nangauss(halt, 2)
    
#     nt,ny,nx = np.shape(halt)
    
#     x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_t[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_t[n]
#     t = np.arange(nt) * 30
    
#     # if ang[n] > 0: mmm = area_interest(halg, t, x, y, sob=10000)
#     mmm = area_interest_0a(halg, t, x, y, sob=10000) 
    
#     lls = []
#     for i in range(len(ts_t[n])):
#         icebox = ~np.isnan(halt[i] * mmm[i])
#         lens = np.sum(icebox,axis=0)
#         # filt = lens>380
#         # lls.append(( np.mean(lens[filt]) * ds[n] ) / np.cos(angys[n]*np.pi/180) )
#         if n == 2: lls.append(( np.mean(lens[300:450]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
#         elif n == 6: lls.append(( np.mean(lens[350:500]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
#         else: lls.append(( np.mean(lens[350:550]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
        
#     A = np.array([t*0+1,t,t**2,t**3]).T
#     co, r, rank, s = np.linalg.lstsq(A, lls, rcond=None)
#     lfi = co[0] + co[1]*t + co[2]*t**2 + co[3]*t**3 
        
#     lfit_t.append(lfi)
#     llss_t.append(lls)
    
    
#     mhe = np.nanmean(hints_t[n], axis=(1,2) )
#     a = ts_t[n]
#     A = np.array([a*0+1,a,a**2,a**3]).T
#     co, r, rank, s = np.linalg.lstsq(A, mhe, rcond=None)
#     # mfi = co[0] + co[1]*a + co[2]*a**2 + co[3]*a**3 
#     mfi_d = co[1] + 2*co[2]*a + 3*co[3]*a**2 
#     mfis_t.append(mfi_d)
    
#     t2 = time()
#     print(n,t2-t1)

#%%
g = 9.81 #m / s^2
mu = 0.00103 #kg / m s
kt = 1.4e-7 #m^2 / s
deni = 916.8 # kg / m^3
latent = 334e3 # m^2 / s^2
thcon = 0.6 # m kg / s^3 °C

beta_s = 7.8e-4 # (g/kg)^-1
nu = 1.03e-6 # m^2 / s
ks = kt/100 # m^2 / s

Ras_v,Nus_v = [],[]
# for n in [0,7,8,15,16]:
ns = [16,17,18,19]
# for n in ns:
for n in tqdm(range(len(ds_v))):

    t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
    gt,gy,gx = np.gradient(hints_v[n], t,y,x)
    drho = np.abs( water_dens(0, salis_v[n]) - water_dens(temp_v[n], salis_v[n]) )
    dT = temp_v[n]
    gt[np.isnan(gt)] = 0.0
    
    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=x, axis=2), x=-y, axis=1)
    tmelr = np.trapz( np.trapz( gt, x=x, axis=2), x=-y, axis=1) / area


    Ra, Nu = [],[]
    for i in range(len(t)):
    
        # finan = ~np.isnan(hints[n][i])
        # # L = (np.max( (yns[n])[finan] ) - np.min( (yns[n])[finan] )) / 1000
        # L = co[0] + co[1]*i + co[2]*i**2 + co[3]*i**3 + co[4]*i**4 
        L = lfit_v[n][i] / 1000
        
        mh = tmelr[i]
        # mh = np.nanmean(gt[i]) #mfis[n][i]
        
        Ra.append( g * np.cos(angys_v[n]*np.pi/180) * drho * L**3 / kt / mu )
        # Ra.append( g * np.cos(angys[n]*np.pi/180) * beta_s * salis[n] * L**3 / (ks * nu) )

        Nu.append( -mh/1000 * deni * latent * L / thcon / dT )
        # Nu.append( -np.nanmean(gt[i])/1000 * deni * latent * L / thcon / dT )

    Ra,Nu = np.array(Ra), np.array(Nu)
    Ras_v.append(Ra)
    Nus_v.append(Nu)
    
# Ras_t,Nus_t = [],[]
# # for n in [0,7,8,15,16]:
# ns = [16,17,18,19]
# # for n in ns:
# for n in tqdm(range(len(ds_t))):

#     t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
#     gt,gy,gx = np.gradient(hints_t[n], t,y,x)
#     drho = np.abs( water_dens(0, salis_t[n]) - water_dens(temp_t[n], salis_t[n]) )
#     dT = temp_t[n]
#     gt[np.isnan(gt)] = 0.0
    
#     area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=x, axis=2), x=-y, axis=1)
#     tmelr = np.trapz( np.trapz( gt, x=x, axis=2), x=-y, axis=1) / area


#     Ra, Nu = [],[]
#     for i in range(len(t)):
    
#         # finan = ~np.isnan(hints[n][i])
#         # # L = (np.max( (yns[n])[finan] ) - np.min( (yns[n])[finan] )) / 1000
#         # L = co[0] + co[1]*i + co[2]*i**2 + co[3]*i**3 + co[4]*i**4 
#         L = lfit_t[n][i] / 1000
        
#         mh = tmelr[i]
#         # mh = np.nanmean(gt[i]) #mfis[n][i]
        
#         Ra.append( g * np.cos(angys_t[n]*np.pi/180) * drho * L**3 / kt / mu )
#         # Ra.append( g * np.cos(angys[n]*np.pi/180) * beta_s * salis[n] * L**3 / (ks * nu) )

#         Nu.append( -mh/1000 * deni * latent * L / thcon / dT )
#         # Nu.append( -np.nanmean(gt[i])/1000 * deni * latent * L / thcon / dT )

#     Ra,Nu = np.array(Ra), np.array(Nu)
#     Ras_t.append(Ra)
#     Nus_t.append(Nu)
    
#%%
rs = np.logspace(7.9, 9.8, 10)
# ns = [0,1,2,3] #[0,7,8,15,16]

col = ['b','g','r','purple', 'brown']
mark = ['.-','v-','*-','P-']
ss,aa = np.array([0,7,13,20,26]),np.array([0,15,30,45])

plt.figure(figsize=(10,7))
plt.tight_layout()
for i in range(len(ds_v)):
# for i in [0,1,2,3]:
    # plt.plot(Ras_v[i], Nus_v[i] / Ras_v[i]**(1/3) , '.-', color=(0.5,1-salis_v[i]/35,salis_v[i]/35))
    # plt.plot(Ras_v[i], Nus_v[i] / Ras_v[i]**(1/4) , '.-', color=(0.5,1-salis_v[i]/35,salis_v[i]/35))
    plt.plot(Ras_v[i][5:-5], Nus_v[i][5:-5] , '.-', color=(0.5,1-salis_v[i]/35,salis_v[i]/35))


# plt.plot(rs, rs**(1/2) * 0.003, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
plt.plot(rs, rs**(1/3) * 0.11, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
plt.plot(rs, rs**(1/4) * 0.5, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# # plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3-1/3) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4-1/3) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# # plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3-1/4) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4-1/4) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$b)$'], location='top')
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="Salinity (g/kg)") #, size=12)


# plt.ylim(0.041, 0.154)
plt.grid()
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ra')
# plt.ylabel(r'Nu/Ra$^{1/3}$')
plt.ylabel(r'Nu')

# plt.title('S = '+str(salis[i])+'g/kg')
# plt.savefig('./Documents/2nura0.png',dpi=400, bbox_inches='tight')

plt.show()

#%%

rs = np.logspace(8.7, 10.0, 10)

plt.figure(figsize=(10,7))
plt.tight_layout()
for i in range(len(ds_t)):
    if angys_t[i] > 22 and angys_t[i] < 36:
    # for i in [0,1,2,3]:
        plt.plot(Ras_t[i], Nus_t[i] / (Ras_t[i] * np.cos(angys_t[n]*np.pi/180) )**(1/3) , '.-', color=(0.5,1-salis_t[i]/35,salis_t[i]/35))
        # plt.plot(Ras_t[i], Nus_t[i] / Ras_t[i]**(1/4) , '.-', color=(0.5,1-salis_t[i]/35,salis_t[i]/35))
        # plt.plot(Ras_t[i], Nus_t[i] , '.-', color=(0.5,1-salis_t[i]/35,salis_t[i]/35))


# # plt.plot(rs, rs**(1/2) * 0.003, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3) * 0.13, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4) * 0.3, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
plt.plot(rs, rs**(1/3-1/3) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
plt.plot(rs, rs**(1/4-1/3) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# # plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3-1/4) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4-1/4) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# plt.ylim(0.041, 0.154)
plt.grid()
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ra')
plt.ylabel(r'Nu/Ra$^{1/3}$')
# plt.ylabel(r'Nu'`)

# plt.title('S = '+str(salis[i])+'g/kg')
# plt.savefig('./Documents/2nura0.png',dpi=400, bbox_inches='tight', )

plt.show()

#%%
# =============================================================================
# Nu, Ra and Ras (1 per experiemnt)
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
#%%

g = 9.81 #m / s^2
mu = 0.00103 #kg / m s
kt = 1.4e-7 #m^2 / s
deni = 916.8 # kg / m^3
latent = 334e3 # m^2 / s^2
thcon = 0.6 # m kg / s^3 °C

beta_s = 7.8e-4 # (g/kg)^-1
nu = 1.03e-6 # m^2 / s
ks = kt/100 # m^2 / s

Nu_v,Ra_v,RaS_v = [],[],[]
for n in range(len(ds_v)):

    drho = np.abs( water_dens(0, salis_v[n]) - water_dens(temp_v[n], salis_v[n]) )
    dT = temp_v[n]
    mh = mes_v[n]
    L = 310 / 1000 #lfit_v[n][0] / 1000

    Ra_v.append( g * np.cos(angys_v[n]*np.pi/180) * drho * L**3 / kt / mu )
    RaS_v.append( g * np.cos(angys_v[n]*np.pi/180) * beta_s * salis_v[n] * L**3 / (ks * nu) )

    Nu_v.append( -mh/1000 * deni * latent * L / thcon / dT )

Nu_t,Ra_t,RaS_t = [],[],[]
for n in range(len(ds_t)):

    drho = np.abs( water_dens(0, salis_t[n]) - water_dens(temp_t[n], salis_t[n]) )
    dT = temp_t[n]
    mh = mes_t[n]
    L = 320/1000 #lfit_t[n][0] / 1000

    Ra_t.append( g * np.cos(angys_t[n]*np.pi/180) * drho * L**3 / kt / mu )
    RaS_t.append( g * np.cos(angys_t[n]*np.pi/180) * beta_s * salis_t[n] * L**3 / (ks * nu) )

    Nu_t.append( -mh/1000 * deni * latent * L / thcon / dT )
    
Nu_v,Ra_v,RaS_v = np.array(Nu_v), np.array(Ra_v), np.array(RaS_v)
Nu_t,Ra_t,RaS_t = np.array(Nu_t), np.array(Ra_t), np.array(RaS_t)
#%%
rs = np.logspace(9.4, 10.1)
plt.figure()
plt.plot(Ra_v, Nu_v, '.' )
plt.plot(Ra_t, Nu_t, '.' )

# plt.plot(Ra_v, Nu_v / Ra_v**(1/3), '.' )
# plt.plot(Ra_t, Nu_t / Ra_t**(1/3), '.' )

# plt.plot(rs, rs**(1/3) * 0.05, '--')
# plt.plot(rs, rs**(1/4) * 0.3, '--')

# plt.xscale('log')
# plt.yscale('log')
plt.show()

rs = np.logspace(11.6, 13.0)
plt.figure()
plt.plot(RaS_v, Nu_v, '.' )
plt.plot(RaS_t, Nu_t, '.' )

# plt.plot(RaS_v, Nu_v / RaS_v**(1/3), '.' )
# plt.plot(RaS_t, Nu_t / RaS_t**(1/3), '.' )


# plt.plot(rs, rs**(1/3) * 0.01, '--')
# plt.plot(rs, rs**(1/4) * 0.1, '--')

# plt.xscale('log')
# plt.yscale('log')
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

    if j == 0: fig.colorbar(imhe, label='height (mm)', location='right', shrink=0.9, ticks=list(range(-41,-61,-3)))
    else: fig.colorbar(imhe, label='height (mm)', location='right', shrink=0.9) 

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.savefig('./Documents/Figs morpho draft/profiles_s1.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()
#%%
exp = 't'
if exp == 'v': 
    xns_b, yns_b = xns_v, yns_v
    hints_b = hints_v
elif exp == 't': 
    xns_b, yns_b = xns_t, yns_t
    hints_b = hints_t

fig, ax = plt.subplot_mosaic([[r'$a)$',r'$b)$',r'$c)$']], layout='tight', figsize=(12,5) , sharex=False)

ns = [1,1,1] #,10,18]
ies = [20,40,60]

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
    fig.colorbar(imhe, label='height (mm)', location='right', shrink=0.7) #, cax=cbar_ax)

    axs.plot([midx-25,midx+25],[boty-10,boty-10],'k-', linewidth=3 )
    axs.text(midx, boty-27, '5 cm')
    axs.axis('off')
    axs.set_ylim( top = topy + 5 )
    axs.set_xlim(botx-2,topx+2)
    
    axs.annotate(labels, (-0.11,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

# plt.tight_layout()
# plt.savefig('./Documents/Figs morpho draft/profiles_tt.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%

#%%
# =============================================================================
# Watershed gaussian
# =============================================================================
difs_v = []
for n in tqdm(range(len(salis_v))):
    difes = []
    for i in (range(len(ts_v[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_v,xns_v,yns_v, order=4)
        cuapla = poln(coeff,xns_v[n],yns_v[n], order=4) 
        
        difes.append( (hints_v[n][i]-cuapla) )

    difes = np.array(difes)
    difs_v.append(difes)

difs_t = []
for n in tqdm(range(len(salis_t))):
    difes = []
    for i in (range(len(ts_t[n]))):
        coeff, r, rank, s = polyfitn(n,i,hints_t,xns_t,yns_t, order=4)
        cuapla = poln(coeff,xns_t[n],yns_t[n], order=4) 
        
        difes.append( (hints_t[n][i]-cuapla) )

    difes = np.array(difes)
    difs_t.append(difes)

#%%
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
n = 0
i = -1


# # wtssal_b,n,i,label, 
labe = 31
ve=True
dif=False

sccaa = wtssal_v[n][i] == labe+1
ccoo = np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )

if ve:    
    if dif: conval = difs_v[n][i][ccoo]
    else: conval = hints_v[n][i][ccoo]
else:    
    if dif: conval = difs_t[n][i][ccoo]
    else: conval = hints_t[n][i][ccoo]

# plt.figure()
# plt.imshow( hints_v[n][i] )
# plt.colorbar()
# # plt.imshow( binary_erosion(sccaa,disk(1)) )
# # plt.imshow(wtssal_v[n][i] == labe, alpha=0.2)
# plt.plot( np.where( sccaa ^ binary_erosion(sccaa,disk(1)))[1], np.where( sccaa ^ binary_erosion(sccaa,disk(1)) )[0], 'r.' )
# plt.show()

scaprop = propscal_v[n]
np.array( [ countour_mean(wtssal_v,n,i,j+1,dif=False) - scaprop[i][j].image_min for j in range(len(scaprop[i])) ] )
# # np.array( [ [countour_mean(wtssal_v,n,i,j,ve=True,dif=False), scaprop[i][j].image_min] for j in range(len(scaprop[i])) ] ), np.nanmean(conval)

# countour_mean(wtssal_v,n,i,labe+1,ve=True,dif=False),  np.nanmean(conval)
# scaprop[i][labe].image_min , np.nanmin([hints_v[n][i][np.where(sccaa)]]) #, np.nanmin([hints_v[n][i][np.where( binary_erosion(sccaa,disk(1)) )]]) 

# # np.shape(scaprop[i][labe].image_intensity)

#%%

n = 7 #8 #15 #23
i = 60

exp = 't'
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
fig.colorbar(ims2, cax=cbar_ax, label='height (mm)', location='left')

# plt.tight_layout()
# plt.savefig('./Documents/Figs morpho draft/watershed_s20_t0.png',dpi=400, bbox_inches='tight', transparent=True)
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

dx = xns_b[n][0,1] - xns_b[n][0,0]
dy = yns_b[n][0,0] - yns_b[n][1,0]
print(dx,dy, dx*dy)

plt.figure()
plt.imshow(difs_b[n][i])
plt.plot(sox,soy, 'r.', markersize=3)
for j in range(len(sscas_b[n][i])):
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(labs_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(lys_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(sscas_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(ssds_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round(smms_b[n][i][j]/1. ,2)) )
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( countour_mean(wtssal_b, n, i, labs_b[n][i][j],dif=False) - \
    #                                                                 propscal_t[n][i][labs_b[n][i][j]].intensity_min,2 ) ) ) 
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( countour_mean(wtssal_b, n, i, labs_b[n][i][j],dif=False),2 ) )) 
    plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( propscal_t[n][i][labs_b[n][i][j]].intensity_min,2 )  ) ) 
    # plt.text(centss_b[n][i][j][1], centss_b[n][i][j][0] , str(round( (ssds_b[n][i][j] / sscas_b[n][i][j])**(-1)  ,2)) )
plt.colorbar()
plt.show()

#%%
countour_mean(wtssal_b, n, i, 5)

# plt.figure()
# # plt.imshow( difs_b[n][i] )#, vmin=-8,vmax=6)
# # plt.imshow( sccaa, alpha=0.5 )#, vmin=-8,vmax=6)
# # plt.plot(ccoo[1],ccoo[0],'r.')
# # plt.colorbar()
# plt.plot(conval)
# plt.show()

#%%
fig,axs = plt.subplots(1)

axs.imshow(difs_v[n][i], extent=(minx,maxx,miny,maxy))
axs.plot( xns_v[n][0,:][sox], yns_v[n][:,0][soy], 'k.', markersize=1)


mask = np.zeros_like(wtssal_v[n][i])
for j in range(len(labs_v[n][i])):
    mask += wtssal_v[n][i] == labs_v[n][i][j]
mask += np.isnan(difs_v[n][i])
amask = np.ma.masked_where(mask, mask)

axs.imshow( amask, extent=(minx,maxx,miny,maxy), alpha = 0.5 )


plt.show()


#%%

n = 0

plt.figure()
# for i in [30,40,50,60]:
for i in [60,61,62,63]:
    plt.scatter( centss_v[n][i][:,1], centss_v[n][i][:,0], c=(i/70,0.5,1-i/70) )
    # plt.scatter( centws_v[n][i][:,1], centws_v[n][i][:,0], c=(i/70,0.5,1-i/70) )
plt.axis('equal')
plt.show()

#%%

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

fig,ax = plt.subplots()
for l,n in enumerate(nss_v):
# for l,n in enumerate([3,8,10,12,2]):
# for l,n in enumerate([8,13]):
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]
    if salis_v[n] > 7 and salis_v[n] < 25: 
        mly, mlx = [], []
        sly,slx = [], []
        for i in range(len(lxs_v[n])):
            mlx.append(np.nanmean(lxs_v[n][i]) * dx)
            mly.append(np.nanmean(lys_v[n][i]) * dy)
            slx.append(np.nanstd(lxs_v[n][i]) * dx)
            sly.append(np.nanstd(lys_v[n][i]) * dy)
            
        # ax.plot( mly , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg')
        ax.errorbar(ts_v[n]/60, mly, yerr=sly, capsize=2, fmt='.-', errorevery=(l*2,20), \
                      color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg' ) #, markersize=5, mfc='w' )

        # ax.plot( mlx , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg')
        # ax.errorbar(ts_v[n]/60, mlx, yerr=slx, capsize=2, fmt='.-', errorevery=(l*2,20), \
        #              color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg' ) #, markersize=5, mfc='w' )



# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# ax.set_ylim(bottom=0)
# plt.show()
# plt.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.set_xlabel('time (min)')
ax.set_ylabel(r'$\lambda_y$ (mm)')
# ax.set_ylabel(r'$\lambda_x$ (mm)')

ax.legend(loc='lower left')
# plt.savefig('./Documents/lamx_t.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

fig,ax = plt.subplots()

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
        
    # ax.plot(ts_t[n]/60, mly , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), label=str(salis_t[n])+'; '+str(ang_t[n]))
    # ax.fill_between(ts_t[n]/60, mly-sly/2, mly+sly/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )

    # ax.errorbar(ts_t[n]/60, mly, yerr=sly, capsize=2, fmt='.-', label=str(salis_t[n])+'; '+str(ang_t[n]), \
    #             errorevery=(l*2,20), color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ) )
    #             #color=np.array([0.5,salis_t[n]/35,1-salis_t[n]/35]) ) #, markersize=5, mfc='w' ) #  label=str(salis_t[n])+' g/kg', \

    # ax.plot( mlx , '.-', color=(0.5,1-salis_t[n]/35,salis_t[n]/35))
    # ax.plot( np.array(mlx) / np.array(mly) , '.-', color=(0.5,1-salis_t[n]/35,salis_t[n]/35))
 
    ax.plot(ts_t[n]/60, mlx , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), label=str(salis_t[n])+'; '+str(ang_t[n]))
    ax.fill_between(ts_t[n]/60, mlx-slx/2, mlx+slx/2, color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )


cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label="S (g/kg)") #, size=12)

# plt.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.set_xlabel('time (min)')
# ax.set_ylabel(r'$\lambda_y$ (mm)')
ax.set_ylabel(r'$\lambda_x$ (mm)')

ax.legend(loc='lower left')
# plt.savefig('./Documents/Figs morpho draft/lamx_t.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%
# Wavelengths with watershed

# fig,ax = plt.subplots()
fig, ax = plt.subplot_mosaic([[r'$a)$'],[r'$b)$']], layout='tight', figsize=(7,6) , sharex=True)


li1ys,li1xs = [],[]
for l,n in enumerate(nss_v):
    dx = xns_v[l][0,1] - xns_v[l][0,0]
    dy = yns_v[l][0,0] - yns_v[l][1,0]
    if salis_v[l] > 7 and salis_v[l] < 25: 
        mly, mlx, mld = [], [], []
        sly, slx, sld = [], [], []
        for i in range(len(lxs_v[l])):
            mlx.append(np.nanmean(lxs_v[l][i]) * dx)
            mly.append(np.nanmean(lys_v[l][i]) * dy)
            # mld.append(np.nanmean( (lys_v[l][i] * dy) / (lxs_v[l][i] * dx) ))
            # mlx.append(np.nanmedian(lxs_v[l][i]) * dx)
            # mly.append(np.nanmedian(lys_v[l][i]) * dy)
            slx.append(np.nanstd(lxs_v[l][i]) * dx)
            sly.append(np.nanstd(lys_v[l][i]) * dy)
            # sld.append(np.nanstd( (lys_v[l][i] * dy) / (lxs_v[l][i] * dx) ))
            
        indy, indx = np.argmin(mly[-30:]), np.argmin(mlx[-30:])
        # ax.errorbar(salis_v[l], mly[indy+30], yerr=sly[indy+30], capsize=2, fmt='.-'  )

        mey, eey = np.nanmean(mly[-30:]), np.nanstd(mly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        # mey, eey = np.nanmean(mly[-30:]), np.nanmean(sly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        # mey, eey = np.nanmean(mly[-20:]), np.nanmean(sly[-20:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        li1y = ax[r'$a)$'].errorbar(salis_v[l], mey, yerr=eey, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx[-30:]), np.nanstd(mlx[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        # mex, eex = np.nanmean(mlx[30:]), np.nanmean(slx[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        # mex, eex = np.nanmean(mlx[-20:]), np.nanmean(slx[-20:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        li1x = ax[r'$b)$'].errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1xs.append(li1x)
        # ax.errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='b.', markersize=15, mfc='w', alpha=0.5, label=r'$\lambda_x$'  )
        
        # med, eed = np.nanmean(mld[30:]), np.nanmean(sld[30:])  #np.sqrt(np.sum(np.array(sld[30:])**2)) #/ len(sly)
        # # mex, eex = np.nanmean(mld[-20:]), np.nanmean(sld[-20:])  #np.sqrt(np.sum(np.array(sld[30:])**2)) #/ len(sly)
        # li1x = ax[r'$c)$'].errorbar(salis_v[l], med, yerr=eed, capsize=2, fmt='o', markersize=5, \
        #                      color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        

        # ax.errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='b.', markersize=15, mfc='w', alpha=0.5, label=r'$\lambda_x$'  )

li2ys,li2xs = [],[]
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx, mld = [], [], []
    sly, slx, sld = [], [], []
    for i in range(len(lxs_t[n])):
        mlx.append(np.nanmean(lxs_t[n][i]) * dx)
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        # mld.append(np.nanmean( (lys_t[n][i] * dy) / (lxs_t[n][i] * dx) ))
    
        slx.append(np.nanstd(lxs_t[n][i]) * dx)
        sly.append(np.nanstd(lys_t[n][i]) * dy)
        # sld.append(np.nanstd( (lys_t[n][i] * dy) / (lxs_t[n][i] * dx) ))
        
    indy, indx = np.argmin(mly[-30:]), np.argmin(mlx[-30:])
    # ax.errorbar(salis_t[l], mly[indy+30], yerr=sly[indy+30], capsize=2, fmt='.-'  )

    mey, eey = np.nanmean(mly[-30:]), np.nanstd(mly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # mey, eey = np.nanmean(mly[-30:]), np.nanmean(sly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # mey, eey = np.nanmean(mly[-20:]), np.nanmean(sly[-20:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # li2y = ax[r'$a)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2y = ax[r'$a)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2ys.append(li2y)

    mex, eex = np.nanmean(mlx[-30:]), np.nanstd(mlx[-30:]) #np.nanmean(slx[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # mex, eex = np.nanmean(mlx[-30:]), np.nanmean(slx[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # mex, eex = np.nanmean(mlx[-20:]), np.nanmean(slx[-20:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # ax.errorbar(salis_t[l], mex, yerr=eex, capsize=2, fmt='b.', markersize=15, mfc='w', alpha=0.5, label=r'$\lambda_x$'  )
    # li2x = ax[r'$b)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2x = ax[r'$b)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2xs.append(li2x)
    
    # med, eed = np.nanmean(mld[30:]), np.nanmean(sld[30:])  #np.sqrt(np.sum(np.array(sld[30:])**2)) #/ len(sly)
    # # mex, eex = np.nanmean(mld[-20:]), np.nanmean(sld[-20:])  #np.sqrt(np.sum(np.array(sld[30:])**2)) #/ len(sly)
    # li1x = ax[r'$c)$'].errorbar(salis_v[l], med, yerr=eed, capsize=2, fmt='o', markersize=5, \
    #                      color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        



ax[r'$a)$'].set_ylim(18,42)
ax[r'$b)$'].set_ylim(18,42)

ax[r'$a)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$b)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$b)$'].set_xlabel('Salinity (g/kg)')


ax[r'$a)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],[r'Set 1, $0°$', r'Set 2, $0°$',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.2), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/lamxy_s.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()
#%%
# =============================================================================
# Wavelengths
# =============================================================================
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
ax[r'$a)$'].set_xlabel('Time (min)')
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
        for i in range(len(lxs_v[l])):
            mlx.append(np.nanmean(lxs_v[l][i]) * dx)
            mly.append(np.nanmean(lys_v[l][i]) * dy)
            # mld.append(np.nanmean( (lys_v[l][i] * dy) / (lxs_v[l][i] * dx) ))
            # mlx.append(np.nanmedian(lxs_v[l][i]) * dx)
            # mly.append(np.nanmedian(lys_v[l][i]) * dy)
            slx.append(np.nanstd(lxs_v[l][i]) * dx)
            sly.append(np.nanstd(lys_v[l][i]) * dy)
            # sld.append(np.nanstd( (lys_v[l][i] * dy) / (lxs_v[l][i] * dx) ))
            
        indy, indx = np.argmin(mly[-30:]), np.argmin(mlx[-30:])

        mey, eey = np.nanmean(mly[-30:]), np.nanstd(mly[-30:])  
        # mey, eey = np.nanmean(mly[-30:]), np.nanmean(sly[-30:]) 
        li1y = ax[r'$b)$'].errorbar(salis_v[l], mey, yerr=eey, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx[-30:]), np.nanstd(mlx[-30:])  
        # mex, eex = np.nanmean(mlx[-30:]), np.nanmean(slx[-30:]) 
        li1x = ax[r'$c)$'].errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1xs.append(li1x)

li2ys,li2xs = [],[]
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx, mld = [], [], []
    sly, slx, sld = [], [], []
    for i in range(len(lxs_t[n])):
        mlx.append(np.nanmean(lxs_t[n][i]) * dx)
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        # mld.append(np.nanmean( (lys_t[n][i] * dy) / (lxs_t[n][i] * dx) ))
    
        slx.append(np.nanstd(lxs_t[n][i]) * dx)
        sly.append(np.nanstd(lys_t[n][i]) * dy)
        # sld.append(np.nanstd( (lys_t[n][i] * dy) / (lxs_t[n][i] * dx) ))
        
    indy, indx = np.argmin(mly[-30:]), np.argmin(mlx[-30:])

    mey, eey = np.nanmean(mly[-30:]), np.nanstd(mly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # mey, eey = np.nanmean(mly[-30:]), np.nanmean(sly[-30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    li2y = ax[r'$b)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2ys.append(li2y)

    mex, eex = np.nanmean(mlx[-30:]), np.nanstd(mlx[-30:]) 
    # mex, eex = np.nanmean(mlx[-30:]), np.nanmean(slx[30:])
    li2x = ax[r'$c)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2xs.append(li2x)


ax[r'$b)$'].set_ylim(19,43)
ax[r'$c)$'].set_ylim(19,43)
ax[r'$b)$'].sharex(ax[r'$c)$'])
ax[r'$b)$'].tick_params(axis='x',length=3,labelsize=0)

ax[r'$b)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$c)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$c)$'].set_xlabel('Salinity (g/kg)')


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
n = 8
i = 70



plt.figure()

dx = xns_t[n][0,1] - xns_t[n][0,0]
dy = yns_t[n][0,0] - yns_t[n][1,0]
daa = [lys_t[7][i] * dy, lys_t[8][i] * dy, lys_t[15][i] * dy]
# col = (n/12,(n-10)/3,0)
barviolin(daa, bins=10, x=[1,2,3], width=10) # , x=0, width = 1500, bins=10, alpha=0.6, color=col)

plt.violinplot( daa, widths=.5) 
#, positions=ts_v[ns[1]][ies]/60, widths=3.5, showextrema=False, showmeans=True, points=100, bw_method='scott' )#, showmeans=True,

# plt.xlim(left=12)
plt.show()

#%%
# Area scallops (y quiza numero)

fig,ax = plt.subplots()
# for n in nss_v:
# for l,n in enumerate([8,10,13]):
for l,n in enumerate([8,9,10,11]):
# for l,n in enumerate([1,10,12]):
    if salis_v[n] > 7 and salis_v[n] < 25: 
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_v[n])):
            sarm.append( np.nanmean(sscas_v[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_v[n][i] * dx*dy) )

        plt.errorbar(ts_v[n]/60, np.array(sarm)/100, yerr=np.array(sars)/100,  capsize=2, fmt='.-', errorevery=(l*2,20), \
                      label=r'$S = $'+str(salis_v[n])+' g/kg' )#, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )

ax.set_xlabel('time (min)')
ax.set_ylabel(r'Area (cm$^2$)')
ax.legend(bbox_to_anchor=(0.5,1.18), loc='upper center' , ncol=3)

# plt.savefig('./Documents/aret.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

fig,ax = plt.subplots()
# for n in nss_v:
# for l,n in enumerate([8,10,13]):
for l,n in enumerate([7,8,15]):
# for l,n in enumerate([1,10,12]):
    if salis_t[n] > 6.0 and salis_v[n] < 25: 
        dx = xns_t[n][0,1] - xns_t[n][0,0]
        dy = yns_t[n][0,0] - yns_t[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_t[n])):
            sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )

        plt.errorbar(ts_t[n]/60, np.array(sarm)/100, yerr=np.array(sars)/100,  capsize=2, fmt='.-', errorevery=(l*2,20), \
                      label=r'$S = $'+str(salis_t[n])+' g/kg' )#, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )

ax.set_xlabel('time (min)')
ax.set_ylabel(r'Area (cm$^2$)')
ax.legend(bbox_to_anchor=(0.5,1.18), loc='upper center' , ncol=3)

# plt.savefig('./Documents/aret.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

fig,ax = plt.subplots()
li1ys = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25: 
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_v[n])):
            sarm.append( np.nanmean(sscas_v[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_v[n][i] * dx*dy) )
        
        # plt.error(ts_v[n]/60, sscas_v[n][i] * dx*dy, '.' )
        # plt.errorbar(ts_v[n]/60, sarm, yerr=sars, capsize=2, fmt='o-', markersize=5, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )
    
        # li1y = ax.errorbar(salis_v[n], np.nanmean(sarm[30:])/100, yerr=np.nanmean(sars[30:])/100, capsize=2, fmt='o', \
        #               markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        # li1y = ax.errorbar(salis_v[n], np.nanmean(sarm[-30:])/100, yerr=np.nanmean(sars[3-0:])/100, capsize=2, fmt='o', \
        #               markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        li1y = ax.errorbar(salis_v[n], np.nanmean(sarm[-30:])/100, yerr=np.nanstd(sars[-30:])/100, capsize=2, fmt='o', \
                      markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        # li1y = ax.errorbar(salis_v[n], np.nanmean(sarm[40:60])/100, fmt='o', markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        # li1y = ax.errorbar(salis_v[n], np.nanstd(sarm[40:60])/100, fmt='o', markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        li1ys.append(li1y)

li2ys = []
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):

    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    
    sarm,sars = [], []
    for i in range(len(sscas_t[n])):
        sarm.append( np.nanmean(sscas_t[n][i] * dx*dy) )
        sars.append( np.nanstd(sscas_t[n][i] * dx*dy) )
    
    # plt.error(ts_v[n]/60, sscas_v[n][i] * dx*dy, '.' )
    # plt.errorbar(ts_v[n]/60, sarm, yerr=sars, capsize=2, fmt='o-', markersize=5, color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )

    # li2y = ax.errorbar(salis_t[n], np.nanmean(sarm[30:])/100, yerr=np.nanmean(sars[30:])/100, capsize=2, fmt='o', \
    #              markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    # li2y = ax.errorbar(salis_t[n], np.nanmean(sarm[30:])/100, yerr=np.nanmean(sars[30:])/100, capsize=2, fmt='o', \
    #               markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2y = ax.errorbar(salis_t[n], np.nanmean(sarm[-30:])/100, yerr=np.nanstd(sars[-30:])/100, capsize=2, fmt='o', \
                  markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    # li2y = ax.errorbar(salis_t[n], np.nanmean(sarm[40:60])/100, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    # li2y = ax.errorbar(salis_t[n], np.nanstd(sarm[40:60])/100, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2ys.append(li2y)

ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel(r'Area (cm$^2$)')
ax.legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.12), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/ares.png',dpi=400, bbox_inches='tight', transparent=True)
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
            ax.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
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

ax[r'$a)$'].set_xlabel('Time (min)')
ax[r'$a)$'].set_ylabel(r'Area (cm$^2$)')
# ax[r'$a)$'].legend(bbox_to_anchor=(0.5,1.15), loc='upper center' , ncol=3, columnspacing=0.5)
ax[r'$a)$'].legend(loc='upper right')

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
#         #               markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
#         li2y = ax[r'$b)$'].errorbar(salis_v[n], np.nanmean(sarm)/100, yerr=np.nanstd(sarm)/100, capsize=2, fmt='o', \
#                       markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
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
ax[r'$b)$'].set_xlabel('Salinity (g/kg)')
ax[r'$b)$'].set_ylabel(r'Area (cm$^2$)')
    
for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

plt.savefig('./Documents/Figs morpho draft/areas.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


#%%
# n° de scallops

fig,ax = plt.subplots()
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
    
        ax.plot(ts_v[n]/60, nscafs_v[n] / totars_v[n] , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))
        # ax.plot(ts_v[n]/60, nscafs_v[n] , '.-')
plt.show()

fig,ax = plt.subplots()
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    ax.plot(ts_t[n]/60, nscafs_t[n] / totars_t[n] , '.-', color=(0.5,1-salis_t[n]/35,salis_t[n]/35))
    # ax.plot(ts_v[n]/60, nscafs_v[n] , '.-')
plt.show()


fig,ax = plt.subplots()
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        ascm, ascs = np.mean((nscafs_v[n] / totars_v[n])[-20:]), np.std((nscafs_v[n] / totars_v[n])[-20:])
        ax.errorbar(salis_v[n], ascm *100 , yerr = ascs*100, capsize=2, fmt='o-', \
                     markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):

    ascm, ascs = np.mean((nscafs_t[n] / totars_t[n])[-20:]), np.std((nscafs_t[n] / totars_t[n])[-20:])
    ax.errorbar(salis_t[n], ascm *100, yerr = ascs*100, capsize=2, fmt='o-', \
                 markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
        
ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel('n° scallops / area (1/cm$^2$)')

plt.show()
#%%
# amplitud scallops (use minmax probably)        


fig,ax = plt.subplots()

# for n in [1,10,12]:
# for n in [0,3,8]:
# for n in [8,10,2]:
for n in [8,9,10,11,12,13,15]:
    mimam, mimas = [],[]
    for i in range(len(smms_v[n])):
        mimam.append( np.nanmean(smms_v[n][i]) )
        mimas.append( np.nanstd(smms_v[n][i]) )
        # mimam.append( np.nanmean(smes_v[n][i]) )
        # mimas.append( np.nanstd(smes_v[n][i]) )
        # mimam.append( np.nanmean(ssds_v[n][i]) )
        # mimas.append( np.nanstd(ssds_v[n][i]) )
    
    ax.errorbar(ts_v[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2)

# for n in [8,23]:
#     mimam, mimas = [],[]
#     for i in range(len(smms_t[n])):
#         # mimam.append( np.nanmean(smms_t[n][i]) )
#         mimam.append( np.nanmean(ssds_t[n][i]) )
#         # mimas.append( np.nanstd(smms_t[n][i]) )
#         mimas.append( np.nanstd(ssds_t[n][i]) )
    
#     ax.errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2)

ax.set_xlabel('time (min)')
ax.set_ylabel('Amplitud (mm)')
ax.set_ylim(bottom=0)

# plt.savefig('./Documents/amp8.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()        
#%%
ave = 20
fig,ax = plt.subplots()
li1s = []
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        mimam, mimas = [],[]
        for i in range(len(smms_v[n])):
            mimam.append( np.nanmean(smms_v[n][i]) )
            mimas.append( np.nanstd(smms_v[n][i]) )
        
        ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
        li1 = ax.errorbar(salis_v[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                     markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        li1s.append(li1)
    
li2s = []
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        mimam.append( np.nanmean(smms_t[n][i]) )
        mimas.append( np.nanstd(smms_t[n][i]) )

    ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
    # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
    #              markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                 markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
    li2s.append(li2)
        
ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel('Amplitude (mm)')
# ax.set_ylim(bottom=0)

ax.legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/amps.png', dpi=400, bbox_inches='tight', transparent=True)
plt.show()
#%%
# =============================================================================
# Graph amplitude
# =============================================================================
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
    
    # ax[r'$a)$'].errorbar(ts_t[n]/60, mimam, yerr=mimas, fmt='o-', capsize=2, label=r'$S = $'+str(salis_t[n])+' g/kg', \
    #                      color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
    ax[r'$a)$'].plot(ts_t[n]/60, mimam,'.-', label=r'$S = $'+str(salis_t[n])+' g/kg', \
                         color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70) )
        
    ax[r'$a)$'].fill_between(ts_t[n]/60, (np.array(mimam)-np.array(mimas)/2), (np.array(mimam)+np.array(mimas)/2), \
                             color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), alpha=0.2 )


ax[r'$a)$'].set_xlabel('time (min)')
ax[r'$a)$'].set_ylabel('Amplitud (mm)')
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
                     markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        li1s.append(li1)
    
li2s = []
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    mimam, mimas = [],[]
    for i in range(len(smms_t[n])):
        # mimam.append( np.nanmean(smms_t[n][i]) )
        # mimas.append( np.nanstd(smms_t[n][i]) )
        mimam.append( np.nanmean(smes_t[n][i]) )
        mimas.append( np.nanstd(smes_t[n][i]) )
        # mimam.append( np.nanmean(ssds_t[n][i]) )
        # mimas.append( np.nanstd(ssds_t[n][i]) )

    ascm, ascs = np.mean((mimam)[-ave:]), np.mean(mimas[-ave:])
    # li2 = ax.errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o-', \
    #              markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2 = ax[r'$b)$'].errorbar(salis_t[n], ascm, yerr = ascs, capsize=2, fmt='o', \
                 markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) ) 
    li2s.append(li2)
        
ax[r'$b)$'].set_xlabel('Salinity (g/kg)')
ax[r'$b)$'].set_ylabel('Amplitude (mm)')
# ax.set_ylim(bottom=0)

ax[r'$b)$'].legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.14), loc='upper center' , ncol=5, columnspacing=0.5 )


for labels,axs in ax.items():
    axs.annotate(labels, (-0.13,0.96), xycoords = 'axes fraction', **{'fontname':'Times New Roman'})

    
# plt.savefig('./Documents/Figs morpho draft/amplitudes.png',dpi=400, bbox_inches='tight', transparent=True)
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
                     markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
        
        # lres = linregress(ts_v[n][-ave:]/60, mimam[-ave:])        
        # li1 = ax[r'$b)$'].errorbar(salis_v[n], lres[0], yerr = lres[4], capsize=2, fmt='o', \
        #              markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
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
# Things against height
n = 8
# i = 50

dx = xns_v[n][0,1] - xns_v[n][0,0]
dy = yns_v[n][0,0] - yns_v[n][1,0]


plt.figure()
for i in range(len(centss_v[n])):
#     # plt.plot( centss_v[n][i][:,0],  sscas_v[n][i]*dx*dy, '.', color=(i/100,0,0.5))
    # plt.plot( centss_v[n][i][:,0],  lys_v[n][i] * dy, '.', color=(i/100,0,0.5) )
    # plt.plot( [ts_v[n][i]/60] * len(lys_v[n][i]),  lys_v[n][i] * dy, '.', color=(i/120,0,0.5) )
    plt.plot( [ts_v[n][i]/60] * len(lys_v[n][i]),  sscas_v[n][i] *dy*dx, '.', color=(i/120,0,0.5) )
# plt.ylim(0,70)
plt.show()

# len(centss_v[n][i][:,0]), len( lys_v[n][i] * dy )
#%%
# Amplitud vs vertical wavelength

andre = np.loadtxt('./Downloads/plot-data-2.csv', delimiter=',', skiprows=1 )


#%%
ust = 1e-2 /1. # m/s
nu = 1e-6 * 2.25 # m^2/s
un = ust / nu

plt.figure()
myt, mmt = [],[]
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
# for l,n in enumerate([7,8,15]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]
    
    mly, mimam = [], []
    les = len(lys_t[n]) 
    # for i in range(len(lys_t[n])):
    for i in range(les-10,les):
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        mimam.append( np.nanmean(smms_t[n][i]) )
        
        plt.scatter( (lys_t[n][i] * dy * 1e-3 * un)**-1, smms_t[n][i] * 1e-3 * un, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    myt.append( np.nanmean(mly) )
    mmt.append( np.nanmean(mimam) )

plt.plot(andre[:,0], andre[:,1], 'k-')
plt.xlabel(r'$\lambda_y$ (mm)')
plt.ylabel(r'Amplitud (mm)')
# plt.title( 'Salinity = '+str(salis_t[n])+' g/kg' )
plt.xscale('log')
plt.show()

plt.figure()
myv, mmv = [],[]
for n in nss_v:
    if salis_v[n] > 7 and salis_v[n] < 25:
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        
        mly, mimam = [], []
        les = len(lys_v[n]) 
        # for i in range(len(lys_v[n])):
        for i in range(les-10,les):
            mly.append(np.nanmean(lys_v[n][i]) * dy)
            mimam.append( np.nanmean(smms_v[n][i]) )
            
            plt.scatter( (lys_v[n][i] * dy *1e-3 * un)**-1 , smms_v[n][i] * 1e-3 * un , color=(0.5,1-salis_v[n]/35,salis_v[n]/35) )
        myv.append( np.nanmean(mly) )
        mmv.append( np.nanmean(mimam) )
            
plt.plot(andre[:,0], andre[:,1], 'k-')
plt.xlabel(r'$\lambda_y$ (mm)')
plt.ylabel(r'Amplitud (mm)')
# plt.title( 'Salinity = '+str(salis_v[n])+' g/kg' )
plt.xscale('log')
plt.show()


plt.figure()
plt.scatter( (np.array(myt) *1e-3 * un)**(-1), np.array(mmt) * 1e-3 * un )
plt.scatter( (np.array(myv) *1e-3 * un)**(-1), np.array(mmv) * 1e-3 * un )
plt.plot(andre[:,0], andre[:,1], 'k-')
plt.xscale('log')
plt.show()


#%%
#Violin area
def barviolin(data, x = [0], height=1.0, width=1.0, bins=20, alpha=0.5, color=[], ):
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
            plt.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha )
            plt.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )
        else:
            plt.barh( bec, wid * width, hei * height, left = lefts, alpha=alpha, color=color )
            plt.errorbar( x[i], np.mean(daa), yerr=np.std(daa), color='black', capsize=2, fmt='o', markersize=5 )



plt.figure()

n = 12 #[1,10,12] #[8,10,13]
ies = [30,40,50,60,70,80,90]

for n in [11,12]:
    dx = xns_v[n][0,1] - xns_v[n][0,0]
    dy = yns_v[n][0,0] - yns_v[n][1,0]
    daa = [ np.concatenate(sscas_v[n][i-5:i+5]) * dx*dy for i in ies ] 
    # col = (0.5,1-salis_v[n]/35,salis_v[n]/35)
    col = (n/12,(n-10)/3,0)
    barviolin(daa, x=ts_v[n][ies]/60, width = 1500, bins=10, alpha=0.6, color=col)

# plt.violinplot( daa, positions=ts_v[ns[1]][ies]/60, widths=3.5, showextrema=False, showmeans=True, points=100, bw_method='scott' )#, showmeans=True,
plt.xlim(left=12)
plt.show()

#%%

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

plt.figure()
barviolin( sarms_v, x=salis_v[ns_v], bins=20, width=10, color='blue' )
barviolin( sarms_t, x=salis_t[ns_t], bins=20, width=10, color='red' )
# plt.xticks(salis_v[ns_v])
# plt.xticks(salis_t[ns_t])
plt.ylim(0,25)
plt.xlim(6,24.5)
plt.show()

#%%
dot = np.concatenate(sscas_v[10][-20:]) 
dot2= np.concatenate(sscas_v[12][-20:]) 

plt.figure()
# plt.hist(dot,bins=20, alpha=.5, density=True, range=(1000,18000))
# plt.hist(dot2,bins=20,alpha=.5, density=True, range=(1000,18000))

plt.hist(dot,bins='fd', alpha=.5, density=True, range=(1000,18000))
plt.hist(dot2,bins='fd',alpha=.5, density=True, range=(1000,18000))

plt.show()


#%%
n = 8
i = 40

dfe = np.copy(difs_v[n][i])
dfeg = nangauss(dfe, 5)
dfeg2 = nangauss(dfe, [5,8])
# dfeg = snd.gaussian_filter(dfe, 5)
# dfeg2 = snd.gaussian_filter(dfe, [5,10])


plt.figure()
plt.imshow(dfe)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(dfeg)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(dfeg2)
plt.colorbar()
plt.show()

#%% 

plt.figure()
plt.scatter( salis_v, angys_v )
plt.scatter( salis_t, angys_t )
plt.xlabel('Salinity (g/kg)')
plt.ylabel('Angle (°)')
plt.show()

#%%
n = 13
i = 40

exp = 'v'
if exp == 'v': 
    wtssal_b = wtssal_v
    xns_b, yns_b, ts_b = xns_v, yns_v, ts_v
    difs_b = difs_v
elif exp == 't': 
    wtssal_b = wtssal_t
    xns_b, yns_b, ts_b = xns_t, yns_t, ts_t
    difs_b = difs_t


t1 = time()
dife = np.copy(difs_b[n][i])
dife[np.isnan(dife)] = 10
sdi = sato(- gaussian(dife,0), range(10,12,1) )
wtse = watershed( sdi, mask= dilation( ~np.isnan(difs_b[n][i]),disk(3) ), connectivity=2 ) #~np.isnan(dife) )#difs_b[n][i]) )
t2 = time()
print(t2-t1)

plt.figure()
plt.imshow(difs_b[n][i])
plt.show()

# plt.figure()
# plt.imshow( np.gradient(np.gradient( gaussian(difs_b[n][i],10) ,axis=0),axis=0) )
# plt.show()

# plt.figure()
# plt.imshow( np.gradient(np.gradient( gaussian(difs_b[n][i],10) ,axis=1),axis=1) )
# plt.show()

# def curvature(  )
t1 = time()
t,x,y = ts_b[n], xns_b[n][0], yns_b[n][:,0]

gy,gx = np.gradient( nangauss(difs_b[n][i], 10) , y,x)
gyy,gyx = np.gradient(gy, y,x)
gxy,gxx = np.gradient(gx, y,x)

kurv = ( (1 + gyy**2)*gxx + (1+gxx**2)*gyy - 2*gx*gy*gxy) / (1+gx**2+gy**2)**(3/2)
wtse2 = watershed( -kurv, mask= dilation( ~np.isnan(difs_b[n][i]),disk(3) ), connectivity=1 ) #~np.isnan(dife) )#difs_b[n][i]) )
t2 = time()
print(t2-t1)

plt.figure()
# plt.imshow(  -gaussian(kurv,10) )
plt.imshow(  -kurv )
plt.show()
# plt.figure()
# plt.imshow(  gyy * gxx )
# plt.show()

# plt.figure()
# plt.imshow( sdi )
# plt.show()



minx,maxx = np.min( xns_b[n] ), np.max( xns_b[n] )
miny,maxy = np.min( yns_b[n] ), np.max( yns_b[n] )

sobb = thin(sobel(wtse) > 0)
soy,sox = np.where(sobb)

plt.figure()
plt.imshow( difs_b[n][i], extent=(minx,maxx,miny,maxy) )
plt.plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.show()


sobb = thin(sobel(wtse2) > 0)
soy,sox = np.where(sobb)

plt.figure()
plt.imshow( difs_b[n][i], extent=(minx,maxx,miny,maxy) )
plt.plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.show()

sobb = thin(sobel(wtssal_b[n][i]) > 0)
soy,sox = np.where(sobb)

plt.figure()
plt.plot( xns_b[n][0,:][sox], yns_b[n][:,0][soy], 'k.', markersize=1)
plt.imshow(difs_b[n][i], extent=(minx,maxx,miny,maxy))
plt.show()


#%%


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


