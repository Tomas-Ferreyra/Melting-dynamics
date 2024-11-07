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

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

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
       '35','0']
inc_t = ['0(3)','30','15(3)','45(2)','45(ng)','30','15','0','0','15','45','30','30','45','15','0','0','15','30','45','15(4)','15(5)',
       '45(2)','0(2)','30(2)','30','n15','n15','n15','n15','n15(2)','n15']
salis = [0, 0, 0, 0, 6.8, 7.0, 6.8, 6.8, 13.2, 13.3, 13.2, 13.3, 20.1, 20.1, 19.8, 20.0, 26.1, 26.1, 26.2, 26.2, 0, 0, 6.9, 12.9, 20.1,34.5,34.5,19.7, 
         13.0, 7.0, 34.3, 0]
ang = [0,30,15,45,45,30,15,0,0,15,45,30,30,45,15,0,0,15,30,45,15,15, 45, 0, 30, 30, -15, -15, -15, -15, -15, -15 ]
ds_t = [ 0.4019629986999602, 0.4079970860258477, 0.400784094531674, 0.4019866341895751, 0.4140232465732999, 0.4108146995511450, 0.405185916205820,
       0.3985171779558734, 0.4082777044681367, 0.399457458948112, 0.429624677624675, 0.4002974355025642, 0.3962951595980395, 0.4158467162824917,
       0.405560070548485, 0.406690755428839, 0.3986160751319692, 0.406029247234490, 0.403935349493398, 0.4274366684657002, 0.39842940043484637,
       0.3944444940666371, 0.42941247988993125, 0.3986508813811391, 0.41300121024756764, 0.39724780723266606, 0.3991597643255994, 0.4114831458151559,
       0.4034206887886922, 0.3963784515420281, 0.406148619144839, 0.40500559156677696]

Ls = [ 2099.9999966941955, 2097.2250082592454, 2100.003821154219, 2098.7729533565816, 2092.3116009701507, 2100.7921001119767, 2102.061867627691,
       2104.0217430207585, 2097.0613260850837, 2108.070882626512, 2101.360857870178, 2106.0608525686807, 2103.8609505143863, 2158.7930487367967,
       2112.093781969593, 2113.181455334151, 2104.4329627701713, 2092.891945129341, 2103.955261899692, 2107.0490635783035, 2100.163205198002,
       2102.2972076507995, 2103.9171613871736, 2099.967887686282, 2110.996676678041, 2099.255549750563, 2098.0517570404286, 2102.9647844012497,
       2101.945064364561, 2100.341294595434, 2103.0141188891594, 2101.2510931269435]

angys = [0.2752835802019292, 34.47170331852835, 13.06499750709190, 43.6538844493011, 42.63317480202235, 29.31556395720869, 15.64436382383116, 
         0.379506977103330, 2.122818303993659, 16.82489775638404, 47.66715755358717, 28.76640507341181, 31.15486854660724, 50.65480071850850, 
         19.29185570378762, -0.876278847660974, -1.445891941223953, 15.68633111320458, 28.61178195105051, 46.98820258183241, 14.85643305810015,
         18.694159382293858, 44.69885882480372, 0.4091221586816418, 29.57628972829382, 29.370955852274992, -18.732894940242627, -16.522627138169074,
         -16.27828076769843, -17.33545209415416, -16.58142778295442, -17.450360846901734]

temp = [19.0, 19.0, 19.3, 19.7, 20.1, 20.3, 19.5, 20.0, 19.0, 19.0, 19.7, 19.4, 18.8, 19.2, 19.8, 19.5, 19.8, 19.4, 19.1, 20.0, 20.1, 19.4,
        19.4, 19.3, 19.2, 19.2, 18.7, 19.3, 18.9, 19.0, 19.1, 19.3]

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


with h5py.File('/Users/tomasferreyrahauchar/Documents/Height profiles/npys/sloped_heights(s0)_r.hdf5', 'r') as f:

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

#%%
pt = 1./72.27
golden = (1+5**0.5)/2
width_l = 426. * pt * 3

cols = np.linspace(-19,51,256)
comap = np.array( [(cols+19)/70 , 0.5 *np.ones_like(cols) , 1-(cols+19)/79 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)


fig, ax = plt.subplot_mosaic([[r'$a)$', r'$b)$']], layout='tight', figsize=(12,4)) #, sharey=True)

for n in range(len(ds_v)):
    ax[r'$a)$'].errorbar(salis_v[n], -mes_v[n] * 1 , yerr=0.0018, fmt='o', label=str(n)+'°', markersize=5, \
                 color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), capsize=3, mfc='w')
        
        
for n in range(len(ds_t)):
        
    ax[r'$a)$'].errorbar(salis_t[n], -mes_t[n] * 1 , yerr=0.0011, fmt='.', label=str(n)+'°', markersize=10, \
                 color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70), capsize=3)
        
        
cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax[r'$a)$'])
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

ax[r'$a)$'].set_xlabel(r'$S$ (g/kg)')
ax[r'$a)$'].set_ylabel(r'$\dot{m}$ (mm/s)')


co2 = [(i/35,0,1-i/35) for i in salis]

cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)


for n in range(len(ds_t)):
    ax[r'$b)$'].errorbar( angys_t[n],  -mes_t[n], yerr=0.0011 ,fmt='.', markersize=10,  
                 color=(0.5,1-salis_t[n]/35,salis_t[n]/35), capsize=3) #label=str(i)+'g/kg', \
    
cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax[r'$b)$'])
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label="S (g/kg)") #, size=12)


ax[r'$b)$'].set_xlabel(r'$\mathrm{\theta}$ (°)')
ax[r'$b)$'].set_ylabel(r'$\dot{\mathrm{m}}$ (mm/s)')
# plt.legend()


for labels,axs in ax.items():
    axs.annotate(labels, (-0.15,1), xycoords = 'axes fraction')

plt.savefig('./Documents/all_mr.png',dpi=400, bbox_inches='tight')
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

plt.savefig('./Documents/all_amp.png',dpi=400, bbox_inches='tight')
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

# plt.errorbar(salis_v[filv], mve_v[filv], yerr=msd_v[filv], fmt='o', capsize=2, \
#              color=((angys_v[i]+19)/70,0.5,1-(angys_v[i]+19)/70), markersize=5, mfc='w')
ax.errorbar(salis_v[filv], mve_v[filv], yerr=0.097, fmt='o', capsize=2, \
             color=((angys_v[i]+19)/70,0.5,1-(angys_v[i]+19)/70), markersize=5, mfc='w')

for i,j in enumerate(filt):
    # plt.errorbar(salis_t[j], mve_t[i], yerr=msd_t[i], fmt='o', capsize=2., \
    #              color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5)
    ax.errorbar(salis_t[j], mve_t[i], yerr=0.097, fmt='o', capsize=2., \
                 color=((angys_t[j]+19)/70,0.5,1-(angys_t[j]+19)/70), markersize=5)

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(-19, 51), cmap=newcmp), ax=ax)
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label=r"$\mathrm{\theta}$ (°)") #, size=12)

# ax.grid()
ax.set_xlabel('S (g/kg)')
ax.set_ylabel(r'$v_y$ (mm/s)')
ax.set_ylim(top=0)
plt.savefig('./Documents/all_vely.png',dpi=400, bbox_inches='tight')
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


llss_t, lfit_t = [], []
mfis_t = []

for n in range(len(ds_t)):
    t1 = time()
    halt = np.load('./Documents/Height profiles/profile_s'+sal_t[n]+'_t'+inc_t[n]+'.npy')
    halg = nangauss(halt, 2)
    
    nt,ny,nx = np.shape(halt)
    
    x,y = (np.arange(0.5,nx+0.5) - nx/2) * ds_t[n], (-np.arange(0.5,ny+0.5) + ny/2) * ds_t[n]
    t = np.arange(nt) * 30
    
    # if ang[n] > 0: mmm = area_interest(halg, t, x, y, sob=10000)
    mmm = area_interest_0a(halg, t, x, y, sob=10000) 
    
    lls = []
    for i in range(len(ts_t[n])):
        icebox = ~np.isnan(halt[i] * mmm[i])
        lens = np.sum(icebox,axis=0)
        # filt = lens>380
        # lls.append(( np.mean(lens[filt]) * ds[n] ) / np.cos(angys[n]*np.pi/180) )
        if n == 2: lls.append(( np.mean(lens[300:450]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
        elif n == 6: lls.append(( np.mean(lens[350:500]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
        else: lls.append(( np.mean(lens[350:550]) * ds_t[n] ) / np.cos(angys_t[n]*np.pi/180) )
        
    A = np.array([t*0+1,t,t**2,t**3]).T
    co, r, rank, s = np.linalg.lstsq(A, lls, rcond=None)
    lfi = co[0] + co[1]*t + co[2]*t**2 + co[3]*t**3 
        
    lfit_t.append(lfi)
    llss_t.append(lls)
    
    
    mhe = np.nanmean(hints_t[n], axis=(1,2) )
    a = ts_t[n]
    A = np.array([a*0+1,a,a**2,a**3]).T
    co, r, rank, s = np.linalg.lstsq(A, mhe, rcond=None)
    # mfi = co[0] + co[1]*a + co[2]*a**2 + co[3]*a**3 
    mfi_d = co[1] + 2*co[2]*a + 3*co[3]*a**2 
    mfis_t.append(mfi_d)
    
    t2 = time()
    print(n,t2-t1)

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
    
Ras_t,Nus_t = [],[]
# for n in [0,7,8,15,16]:
ns = [16,17,18,19]
# for n in ns:
for n in tqdm(range(len(ds_t))):

    t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
    gt,gy,gx = np.gradient(hints_t[n], t,y,x)
    drho = np.abs( water_dens(0, salis_t[n]) - water_dens(temp_t[n], salis_t[n]) )
    dT = temp_t[n]
    gt[np.isnan(gt)] = 0.0
    
    area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=x, axis=2), x=-y, axis=1)
    tmelr = np.trapz( np.trapz( gt, x=x, axis=2), x=-y, axis=1) / area


    Ra, Nu = [],[]
    for i in range(len(t)):
    
        # finan = ~np.isnan(hints[n][i])
        # # L = (np.max( (yns[n])[finan] ) - np.min( (yns[n])[finan] )) / 1000
        # L = co[0] + co[1]*i + co[2]*i**2 + co[3]*i**3 + co[4]*i**4 
        L = lfit_t[n][i] / 1000
        
        mh = tmelr[i]
        # mh = np.nanmean(gt[i]) #mfis[n][i]
        
        Ra.append( g * np.cos(angys_t[n]*np.pi/180) * drho * L**3 / kt / mu )
        # Ra.append( g * np.cos(angys[n]*np.pi/180) * beta_s * salis[n] * L**3 / (ks * nu) )

        Nu.append( -mh/1000 * deni * latent * L / thcon / dT )
        # Nu.append( -np.nanmean(gt[i])/1000 * deni * latent * L / thcon / dT )

    Ra,Nu = np.array(Ra), np.array(Nu)
    Ras_t.append(Ra)
    Nus_t.append(Nu)
    
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
    plt.plot(Ras_v[i], Nus_v[i] , '.-', color=(0.5,1-salis_v[i]/35,salis_v[i]/35))


# plt.plot(rs, rs**(1/2) * 0.003, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
plt.plot(rs, rs**(1/3) * 0.11, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
plt.plot(rs, rs**(1/4) * 0.5, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# # plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3-1/3) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4-1/3) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

# # plt.plot(rs, rs**(1/2-1/3) * 0.0027, 'k--', label=r'Nu $\propto$ Ra$^{1/2}$')
# plt.plot(rs, rs**(1/3-1/4) * 0.09, 'c--', label=r'Nu $\propto$ Ra$^{1/3}$')
# plt.plot(rs, rs**(1/4-1/4) * 0.44, 'm--', label=r'Nu $\propto$ Ra$^{1/4}$')

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
# Separate scallops
# =============================================================================
# with tubeness filter sato
n = 1
skels, scaprop = [],[]

for i in tqdm(range(len(difs_v[n]))):
    saa = sato( -difs_v[n][i] )

    # skel = skeletonize( (saa / np.nanmax(saa)) > 0.15 )
    skel = thin( (saa / np.nanmax(saa)) > 0.15 )

    scals = label(~dilation(skel, np.ones((3,3))) )
    scals += -1
    scals[scals==-1] = 0
    propsca = regionprops(scals)
    
    skels.append(skel)
    scaprop.append(propsca)
    
#%%
nske, nsca, nscaf, cents = [], [], [], []
ssca = []
for i in range(len(skels)):
    scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
    cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )

    nske.append( np.sum(skels[i]) )
    nsca.append( len(scaprop[i]) )
    nscaf.append( np.sum( scas > 1000) )
    cents.append( cen )
    ssca.append(  np.mean(scas[scas>1000]) )


plt.figure()
plt.plot(nske,'.-')
plt.show()
plt.figure()
plt.plot(nsca,'.-')
plt.plot(nscaf,'.-')
plt.show()
plt.figure()
plt.plot(ssca,'.-')
plt.show()
#%%
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=0)
def image_minmax(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.max(intensities[region]) - np.min(intensities[region])  

# with finding minima + watershed
wtssal_v, propscal_v = [],[]
for n in tqdm(range(len(ds_v))):
    wats, scaprop = [], []
    for i in range(len(difs_v[n])):
        dife = np.copy( difs_v[n][i] )
        dife[np.isnan(dife)] = -1000
        mm3 = peak_local_max( -dife, labels=~np.isnan(difs_v[n][i]), min_distance=25 ) 
        
        markr = np.zeros_like(difs_v[n][i])
        markr[mm3[:,0],mm3[:,1]] = 1
        
        dife = np.copy( difs_v[n][i] )
        dife[np.isnan(dife)] = 1000
        
        wts = watershed(dife, markers=label(markr), mask= ~np.isnan(difs_v[n][i]) )
        
        wats.append(wts)
        scaprop.append( regionprops(wts, intensity_image=difs_v[n][i], extra_properties=[image_stdev,image_minmax]) )
        
    wtssal_v.append(wats)
    propscal_v.append( scaprop )
    

# wtssal_t, propscal_t = [],[]
# for n in tqdm(range(len(ds_t))):
#     wats, scaprop = [], []
#     for i in range(len(difs_t[n])):
#         dife = np.copy( difs_t[n][i] )
#         dife[np.isnan(dife)] = -1000
#         mm3 = peak_local_max( -dife, labels=~np.isnan(difs_t[n][i]), min_distance=25 ) 
        
#         markr = np.zeros_like(difs_t[n][i])
#         markr[mm3[:,0],mm3[:,1]] = 1
        
#         dife = np.copy( difs_t[n][i] )
#         dife[np.isnan(dife)] = 1000
        
#         wts = watershed(dife, markers=label(markr), mask= ~np.isnan(difs_t[n][i]) )
        
#         wats.append(wts)
#         scaprop.append( regionprops(wts, intensity_image=difs_t[n][i], extra_properties=[image_stdev,image_minmax]) )
        
#     wtssal_t.append(wats)
#     propscal_t.append( scaprop )
    
#%%
nss = list(range(len(ds_v)))
sscas, nscas, nscafs, centss = [], [], [], []
eccens, oriens, stdevs, mimas = [],[], [], []
laxmas, laxmis, exts = [], [], []
for n in tqdm(nss):
    ssca, nsca, nscaf, cents = [], [], [], []
    ecen,orien,stdev,mima = [],[], [], []
    laxma, laxmi, ext = [],[], []
    scaprop = propscal_v[n]
    for i in range(len(scaprop)):
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        ec = np.array( [ scaprop[i][j].eccentricity for j in range(len(scaprop[i])) ] )
        ori = np.array( [ scaprop[i][j].orientation for j in range(len(scaprop[i])) ] )
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )
        ama = np.array( [ scaprop[i][j].axis_major_length for j in range(len(scaprop[i])) ] )
        ami = np.array( [ scaprop[i][j].axis_minor_length for j in range(len(scaprop[i])) ] )   
        ex = np.array( [ scaprop[i][j].extent for j in range(len(scaprop[i])) ] )   
        
        # if n==15 and i==50: print(len(ec), len(ama))
    
        nsca.append( len(scaprop[i]) )
        nscaf.append( np.sum( scas > 1000) )
        cents.append( cen )
        ssca.append(  np.mean(scas[scas>1000]) )
        ecen.append( ec ) #[scas>1000] )
        orien.append( ori) #[scas>1000] )
        stdev.append( sd[scas>1000] )
        mima.append( mm[scas>1000] )
        laxma.append( ama )
        laxmi.append( ami )        
        ext.append( ex )        

    sscas.append(ssca)
    nscas.append(np.array(nsca))
    nscafs.append(nscaf)
    centss.append(cents)
    eccens.append(ecen)
    oriens.append(orien)
    stdevs.append(stdev)
    mimas.append(mima)
    laxmas.append( laxma )
    laxmis.append( laxmi )
    exts.append( ext )

# nss = list(range(len(ds_t)))
# sscas, nscas, nscafs, centss = [], [], [], []
# eccens, oriens, stdevs, mimas = [],[], [], []
# for n in nss:
#     ssca, nsca, nscaf, cents = [], [], [], []
#     ecen,orien,stdev,mima = [],[], [], []
#     scaprop = propscal_t[n]
#     for i in range(len(scaprop)):
#         scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
#         cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
#         ec = np.array( [ scaprop[i][j].eccentricity for j in range(len(scaprop[i])) ] )
#         ori = np.array( [ scaprop[i][j].orientation for j in range(len(scaprop[i])) ] )
#         ori = np.array( [ scaprop[i][j].orientation for j in range(len(scaprop[i])) ] )
#         sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
#         mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )
    
#         nsca.append( len(scaprop[i]) )
#         nscaf.append( np.sum( scas > 1000) )
#         cents.append( cen )
#         ssca.append(  np.mean(scas[scas>1000]) )
#         ecen.append( ec[scas>1000] )
#         orien.append( ori[scas>1000] )
#         stdev.append( sd[scas>1000] )
#         mima.append( mm[scas>1000] )

#     sscas.append(ssca)
#     nscas.append(np.array(nsca))
#     nscafs.append(nscaf)
#     centss.append(centss)
#     eccens.append(ecen)
#     oriens.append(orien)
#     stdevs.append(stdev)
#     mimas.append(mima)
#%%
n = 1
i = 50

sobb = sobel(wtssal_v[n][i]) > 0
soy,sox = np.where(sobb)

plt.figure()
plt.imshow(difs_v[n][i])
plt.plot(sox,soy, 'r.', markersize=3)
plt.show()

# plt.figure()
# plt.imshow(difs_v[n][i])
# plt.plot(sox,soy, 'r.', markersize=3)
# for j in range(len(laxmas[n][i])):
#     plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(oriens[n][i][j]/1. ,2)) )
# plt.show()

# # plt.figure()
# # plt.imshow(difs_v[n][i])
# # plt.plot(sox,soy, 'r.', markersize=3)
# # for j in range(len(laxmas[n][i])):
# #     plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(eccens[n][i][j],2)) )
# # plt.show()

# # plt.figure()
# # plt.imshow(difs_v[n][i])
# # plt.plot(sox,soy, 'r.', markersize=3)
# # for j in range(len(laxmas[n][i])):
# #     plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(laxmas[n][i][j],2)) )
# # plt.show()

# plt.figure()
# plt.imshow(difs_v[n][i])
# plt.plot(sox,soy, 'r.', markersize=3)
# for j in range(len(laxmas[n][i])):
#     plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(laxmis[n][i][j] / laxmas[n][i][j],2)) )
# plt.show()

# plt.figure()
# plt.imshow(difs_v[n][i])
# plt.plot(sox,soy, 'r.', markersize=3)
# for j in range(len(laxmas[n][i])):
#     plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(laxmis[n][i][j] / laxmas[n][i][j] * oriens[n][i][j],3)) )
# plt.show()

#%%

plt.figure()
for n in [1,4]:
    for i in range(len(difs_v[n])):
        fil = np.abs(oriens[n][i]) > np.pi/4
        # plt.errorbar(ts_v[n][i]/60, np.mean(laxmas[n][i][fil]), yerr=np.std(laxmas[n][i][fil]), fmt='r.')
        # plt.errorbar(ts_v[n][i]/60, np.mean(laxmis[n][i][fil]), yerr=np.std(laxmis[n][i][fil]), fmt='.', color=(n/15,(n%2)/2,1-n/15), label=salis_v[n])
        plt.errorbar(ts_v[n][i]/60, np.mean(laxmas[n][i][fil]), yerr=np.std(laxmas[n][i][fil]), fmt='.', color=(n/15,(n%2)/2,1-n/15), label=salis_v[n])
plt.show()

n = 1
plt.figure()
for i in range(len(difs_v[n])):
    fil = np.abs(oriens[n][i]) > np.pi/4
    plt.errorbar(ts_v[n][i]/60, np.mean(laxmas[n][i][fil] / laxmis[n][i][fil]), yerr=np.std(laxmas[n][i][fil] / laxmis[n][i][fil]), fmt='r.')
plt.show()

#%%
plt.figure()
for l,n in enumerate(nss):
    # plt.plot(nscas[l],'.--')
    plt.plot(nscafs[l] / ,'.-')
    # plt.plot(nscas[l] - nscafs[l],'.-')
plt.show()

# plt.figure()
# for l,n in enumerate(nss):
#     plt.plot(sscas[l],'.-')
# plt.show()


# plt.figure()
# l = 3
# for i in range(len(eccens[l])):
#     mec,sec = np.mean( eccens[l][i]), np.std( eccens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='k.' )
# l = 1
# for i in range(len(eccens[l])):
#     mec,sec = np.mean( eccens[l][i]), np.std( eccens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='r.' )
# plt.show()

# plt.figure()
# # for l,n in enumerate(nss):
# for i in range(len(oriens[l])):
#     # plt.plot( [i]*len(oriens[l][i]) , oriens[l][i],'.')
#     mec,sec = np.mean( oriens[l][i]), np.std( oriens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='k.' )
# plt.show()

# l = 1
# plt.figure()
# # for l,n in enumerate(nss):
# for l,n in enumerate([4,5,6,7,8,9]):
#     for i in range(len(stdevs[l])):
#         # plt.plot( [i]*len(stdevs[l][i]) , stdevs[l][i],'.')
#         mec,sec = np.mean( stdevs[l][i]), np.std( stdevs[l][i])
#         plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_v[n]/28,0,1-salis_v[n]/28) )
#     plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_v[n]/28,0,1-salis_v[n]/28), label=salis_v[n] )
# plt.legend(loc='upper left')
# plt.gca().set_ylim(bottom=0)
# plt.show()

# plt.figure()
# # for l,n in enumerate(nss):
# for l,n in enumerate(nss):
#     for i in range(len(mimas[l])):
#         # plt.plot( [i]*len(mimas[l][i]) , mimas[l][i],'.')
#         mec,sec = np.mean( mimas[l][i]), np.std( mimas[l][i])
#         plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_v[n]/28,0,1-salis_v[n]/28) )
#     plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_v[n]/28,0,1-salis_v[n]/28), label=salis_v[n] )
# plt.legend(loc='upper left')
# plt.gca().set_ylim(bottom=0)
# plt.show()

#%%

plt.figure()
# for l,n in enumerate(nss):
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([7,8,15]):
# for l,n in enumerate([7,6,5]):
    # plt.plot(nscas[l],'.--')
    # plt.plot(nscafs[l],'.-')
    plt.plot(nscas[l] - nscafs[l],'.-', label=salis_t[n])
plt.legend()
plt.show()

plt.figure()
# for l,n in enumerate(nss):
# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
for l,n in enumerate([7,8,15]):
# for l,n in enumerate([7,6,5]):
    plt.plot(sscas[l],'.-')
plt.show()

# plt.figure()
# # for l,n in enumerate(nss):
# # for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
# for l,n in enumerate([7,8,15]):
# # for l,n in enumerate([7,6,5]):
#     for i in range(len(stdevs[l])):
#         # plt.plot( [i]*len(stdevs[l][i]) , stdevs[l][i],'.')
#         mec,sec = np.mean( stdevs[l][i]), np.std( stdevs[l][i])
#         plt.errorbar(i, mec, yerr=sec, fmt='.', color=np.array([0.5,salis_t[n]/27.4,1-salis_t[n]/27.4]) * (1 - (ang_t[n])/70 ))
#     plt.errorbar(i, mec, yerr=sec, fmt='.', color=np.array([0.5,salis_t[n]/27.4,1-salis_t[n]/27.4]) * (1 - (ang_t[n])/70 ), label=salis_t[n] )
# plt.legend(loc='upper left')
# plt.gca().set_ylim(bottom=0)
# plt.show()

# plt.figure()
# # for l,n in enumerate(nss):
# for l,n in enumerate(nss):
#     for i in range(len(mimas[l])):
#         # plt.plot( [i]*len(mimas[l][i]) , mimas[l][i],'.')
#         mec,sec = np.mean( mimas[l][i]), np.std( mimas[l][i])
#         plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_t[n]/35,0,1-salis_t[n]/35) )
#     plt.errorbar(i, mec, yerr=sec, fmt='.', color=(salis_t[n]/35,0,1-salis_t[n]/35), label=salis_t[n] )
# plt.legend(loc='upper left')
# plt.gca().set_ylim(bottom=0)
# plt.show()

#%%
n = 7
i = 80
# normm = (difs_v[n][i] - np.nanmin(difs_v[n][i]) ) / (np.nanmax(difs_v[n][i]) - np.nanmin(difs_v[n][i]))

# plt.figure()
# plt.imshow( mark_boundaries(normm, wtssal_v[n][i], color=(1,1,1)) )
# plt.show()

normm = (difs_t[n][i] - np.nanmin(difs_t[n][i]) ) / (np.nanmax(difs_t[n][i]) - np.nanmin(difs_t[n][i]))
plt.figure()
plt.imshow( mark_boundaries(normm, wtssal_t[n][i], color=(1,1,1)) )
plt.show()

plt.figure()
# plt.imshow(difs_v[n][i])
plt.imshow(difs_t[n][i])
# plt.plot( np.where(skels[i])[1], np.where(skels[i])[0], 'k.', markersize=1 )
plt.show()
# plt.figure()
# plt.imshow(difs_v[n][i+1])
# plt.plot( np.where(skels[i+1])[1], np.where(skels[i+1])[0], 'k.', markersize=1 )
# plt.show()

#%%
n = 7
i = 80

t1 = time()
dife = np.copy( difs_v[n][i] )
dife[np.isnan(dife)] = -1000
mm3 = peak_local_max( -dife, labels=~np.isnan(difs_v[n][i]), min_distance=25 ) 

markr = np.zeros_like(difs_v[n][i])
markr[mm3[:,0],mm3[:,1]] = 1

dife = np.copy( difs_v[n][i] )
dife[np.isnan(dife)] = 1000

# wts = watershed(dife, markers=200, mask= ~np.isnan(difs_v[n][i]) )
wts = watershed(dife, markers=label(markr), mask= ~np.isnan(difs_v[n][i]) )

t2 = time()
print(t2-t1)


normm = ( difs_v[n][i] - np.nanmin(difs_v[n][i]) ) / (np.nanmax(difs_v[n][i]) - np.nanmin(difs_v[n][i]) )
normm[np.isnan(normm)] = 0.

plt.figure()
plt.imshow( mark_boundaries( normm  , wts ,color=(1,0,1)), cmap='viridis' )
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( wts )
plt.colorbar()
plt.show()


#%%

dife = np.copy( difs_v[n][i] )
dife[np.isnan(dife)] = -1000

t1=time()
mm2 = peak_local_max( -dife, labels=~np.isnan(difs_v[n][i]), footprint=disk(30) ) #, min_distance=10 ) # , footprint=np.ones((3, 3)))
t2 = time()
mm3 = peak_local_max( -dife, labels=~np.isnan(difs_v[n][i]), min_distance=30 ) # , footprint=np.ones((3, 3)))
t3 = time()
print(t2-t1,t3-t2)


plt.figure()
plt.imshow(difs_v[n][i])
plt.plot( mm3[:,1], mm3[:,0], 'ro' )
plt.plot( mm2[:,1], mm2[:,0], 'k.' )
plt.show()

#%%
n = 1
i = 100

t1 = time()
sobb = sobel(wtssal_v[n][i]) > 0
t2 = time()
print(t2-t1)

sky,skx = np.where(skels[i])
soy,sox = np.where(sobb)

plt.figure()
plt.imshow(difs_v[n][i])
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(difs_v[n][i])
plt.plot(skx,sky, 'k.', markersize=5)
# plt.plot(sox,soy, 'r.', markersize=3)
plt.show()
plt.figure()
plt.imshow(difs_v[n][i])
# plt.plot(skx,sky, 'k.', markersize=5)
plt.plot(sox,soy, 'r.', markersize=3)
plt.show()

# plt.figure()
# plt.imshow(skels[i])
# plt.show()

# plt.figure()
# plt.imshow( sobel(wtssal_v[n][i]) > 0 )
# plt.show()
#%%
n = 1
i = 60

dife = np.copy( difs_v[n][i] )
dife[np.isnan(dife)] = 1000

wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_v[n][i]) )
# wts = watershed( snd.maximum_filter(dife, size=(50,50)), mask= ~np.isnan(difs_v[n][i]) )

sobb = sobel(wts) > 0
soy,sox = np.where(sobb)



plt.figure()
plt.imshow(difs_v[n][i])
plt.plot(sox,soy,'k.', markersize=1)
plt.show()
#%%

t1 = time()
saa = sato( -difs_v[n][i] )
sk = thin( (saa / np.nanmax(saa)) > 0.15 )
t2=time()
# dife = np.copy(difs_v[n][i])
# dife[np.isnan(dife)] = 0.0
# hss = hessian( dife )
# # sk2 = thin( (saa / np.nanmax(saa)) > 0.15 )
# gas = gaussian( hss * saa, 5)
# sk2 = thin(gas>0.1)
t3=time()


print(t3-t2,t2-t1)
print(t3-t1)

s1y,s1x = np.where(sk)
# s2y,s2x = np.where(sk2)

kernel = np.ones((5,5))
# kerx,kery = np.cumsum(kernel,axis=1)-1, np.cumsum(kernel,axis=0)-1
kernel[0,:] = 0
kernel[-1,:] = 0
kernel[1,1:4] = 0
kernel[-2,1:4] = 0
# kernel = np.array([[1,0,1],[1,1,1],[1,0,1]]) * 1.

print(kernel)
# plt.figure()
# plt.imshow(difs_v[n][i])
# plt.plot(s1x,s1y,'k.')
# plt.show()

plt.figure()
plt.imshow(sk)
plt.show()
plt.figure()
plt.imshow( (snd.convolve(sk*1., kernel) * sk) > 2 )
plt.show()

# plt.figure()
# plt.imshow( sk2 )
# plt.show()
#%%
i = 50
plt.figure()
plt.imshow(difs_v[0][i])
plt.show()
plt.figure()
plt.imshow(difs_v[3][i])
plt.show()
plt.figure()
plt.imshow(difs_v[13][i])
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
#%%
# =============================================================================
# Watershed gaussian
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
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanstd(intensities[region]) #, ddof=0)
def image_minmax(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.nanmax(intensities[region]) - np.nanmin(intensities[region])  

# with finding minima + watershed
wtssal_v, propscal_v, totars_v = [],[], []
for n in tqdm(range(len(ds_v))):
    wats, scaprop = [], []
    for i in range(len(difs_v[n])):
        dife = np.copy( difs_v[n][i] )
        dife[np.isnan(dife)] = 1000
        
        wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_v[n][i]) )
        
        wats.append(wts)
        scaprop.append( regionprops(wts, intensity_image= gaussian(difs_v[n][i],sigma=7) , extra_properties=[image_stdev,image_minmax]) )
    
    xs,ys = xns_v[n], yns_v[n]
    area = np.trapz( np.trapz(~np.isnan(hints_v[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_v.append(area)
    wtssal_v.append(wats)
    propscal_v.append( scaprop )

wtssal_t, propscal_t, totars_t = [],[], []
for n in tqdm(range(len(ds_t))):
    wats, scaprop = [], []
    for i in range(len(difs_t[n])):
        dife = np.copy( difs_t[n][i] )
        dife[np.isnan(dife)] = 1000
        
        wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_t[n][i]) )
        
        wats.append(wts)
        scaprop.append( regionprops(wts, intensity_image= gaussian(difs_t[n][i],sigma=7) , extra_properties=[image_stdev,image_minmax]) )
    
    xs,ys = xns_t[n], yns_t[n]
    area = np.trapz( np.trapz(~np.isnan(hints_t[n]) * 1.0, x=xs[0,:], axis=2), x=-ys[:,0], axis=1)
        
    totars_t.append(area)
    wtssal_t.append(wats)
    propscal_t.append( scaprop )
#%%
nss = list(range(len(ds_v)))
sscas, nscas, nscafs, centss = [], [], [], []
eccens, oriens, stdevs, mimas = [],[], [], []
laxmas, laxmis, exts, cass = [], [], [], []
for n in tqdm(nss):
    ssca, nsca, nscaf, cents = [], [], [], []
    ecen,orien,stdev,mima = [],[], [], []
    laxma, laxmi, ext, cas = [],[], [], []
    scaprop = propscal_v[n]
    for i in range(len(scaprop)):
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        ec = np.array( [ scaprop[i][j].eccentricity for j in range(len(scaprop[i])) ] )
        ori = np.array( [ scaprop[i][j].orientation for j in range(len(scaprop[i])) ] )
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )
        ama = np.array( [ scaprop[i][j].axis_major_length for j in range(len(scaprop[i])) ] )
        ami = np.array( [ scaprop[i][j].axis_minor_length for j in range(len(scaprop[i])) ] )   
        ex = np.array( [ scaprop[i][j].extent for j in range(len(scaprop[i])) ] )   
                
        # if n==15 and i==50: print(len(ec), len(ama))
        fil = (scas>2000) * ( sd>0.3 )
    
        nsca.append( len(scaprop[i]) )
        nscaf.append( np.sum( fil ) )

        cents.append( cen[fil] )
        stdev.append( sd[fil] )
        mima.append( mm[fil] )
        
        ssca.append(  np.mean(scas[fil]) )
        ecen.append( ec[fil] )
        orien.append( ori[fil] )
        laxma.append( ama[fil] )
        laxmi.append( ami[fil] )
        ext.append( ex[fil] )
        cas.append( scas[fil] )

    sscas.append(ssca)
    nscas.append(np.array(nsca))
    nscafs.append(nscaf)
    centss.append(cents)
    eccens.append(ecen)
    oriens.append(orien)
    stdevs.append(stdev)
    mimas.append(mima)
    laxmas.append( laxma )
    laxmis.append( laxmi )
    exts.append( ext )
    cass.append( cas )

#%%
cols = np.linspace(0,35,256)
comap = np.array( [ 0.5 *np.ones_like(cols) , 1-(cols)/35 , cols/35 , 1*np.ones_like(cols) ] )
comap = comap.T
newcmp = ListedColormap(comap)

# fig,ax = plt.subplots()
# for l,n in enumerate(nss):
#     if salis_v[l] > 7 and salis_v[l] < 25:

#         ax.plot( np.array(nscafs[l]) / totars[l] *100 ,'.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))
#         # ax.plot( np.array(nscas[l]) / totars[l] *1000 ,'.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))
#         # plt.plot(nscas[l] - nscafs[l],'.-')

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# plt.show()


# fig,ax = plt.subplots()

# for l,n in enumerate(nss):
#     if salis_v[l] > 7 and salis_v[l] < 25: 
#         ax.plot( np.array(sscas[l]) /100 , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))
#         # ax.plot( np.mean( sscas[l]) / totars[l] ,'.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# plt.show()


fig,ax = plt.subplots()

for l,n in enumerate(nss):
    if salis_v[l] > -1 and salis_v[l] < 35: 
        msdvm, mmia = [], []
        for i in range(len(stdevs[l])):
            msdv.append(np.nanmean(stdevs[l][i]))
            mmia.append(np.nanmean(mimas[l][i]))
            
        # ax.plot( msdv , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))
        ax.plot( mmia , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35))

cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# cbar.ax.tick_params(labelsize=12)
cbar.set_label( label="S (g/kg)") #, size=12)

plt.show()

# plt.figure()
# l = 3
# for i in range(len(eccens[l])):
#     mec,sec = np.mean( eccens[l][i]), np.std( eccens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='k.' )
# l = 1
# for i in range(len(eccens[l])):
#     mec,sec = np.mean( eccens[l][i]), np.std( eccens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='r.' )
# plt.show()

# plt.figure()
# # for l,n in enumerate(nss):
# for i in range(len(oriens[l])):
#     # plt.plot( [i]*len(oriens[l][i]) , oriens[l][i],'.')
#     mec,sec = np.mean( oriens[l][i]), np.std( oriens[l][i])
#     plt.errorbar(i, mec, yerr=sec, fmt='k.' )
# plt.show()

#%%
n = 5
i = 60

sobb = sobel(wtssal_v[n][i]) > 0
soy,sox = np.where(sobb)

plt.figure()
plt.imshow(difs_v[n][i])
plt.plot(sox,soy, 'r.', markersize=3)
for j in range(len(stdevs[n][i])):
    # plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(cass[n][i][j]/1. ,2)) )
    plt.text(centss[n][i][j][1], centss[n][i][j][0] , str(round(stdevs[n][i][j]/1. ,2)) )
plt.show()


plt.figure()
plt.imshow(difs_v[n][i])
plt.show()


#%%



#%%


#%%
n = 1

t,x,y = ts_v[n], xns_v[n][0], yns_v[n][:,0]
gt,gy,gx = np.gradient(hints_v[n], t,y,x)
gty,gyy,gxy = np.gradient(gy, t,y,x)
gtx,gyx,gxx = np.gradient(gx, t,y,x)

curvatur = ( (1 + gyy**2) * gxx + (1 + gxx**2) * gyy - 2 * gx * gy * gxy ) / (1 + gx**2 + gy**2)**(3/2)
#%%
i = 69

plt.figure()
plt.imshow( hints_v[n][i] )
plt.show()


plt.figure()
plt.imshow(curvatur[i], vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.show()

#%%
# [7,6,5,28,23,8,9,15 ]
n = 7

t,x,y = ts_t[n], xns_t[n][0], yns_t[n][:,0]
gt,gy,gx = np.gradient(hints_t[n], t,y,x)
gty,gyy,gxy = np.gradient(gy, t,y,x)
gtx,gyx,gxx = np.gradient(gx, t,y,x)

curvatur = ( (1 + gyy**2) * gxx + (1 + gxx**2) * gyy - 2 * gx * gy * gxy ) / (1 + gx**2 + gy**2)**(3/2)
#%%
i = 69

plt.figure()
plt.imshow( hints_t[n][i] )
plt.show()


plt.figure()
plt.imshow( gaussian(curvatur[i],sigma=5), vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.show()
 
wts = watershed( -gaussian(curvatur[i],sigma=10), mask= ~np.isnan(difs_t[n][i]) )
sobb = sobel(wts) > 0
soy,sox = np.where(sobb)

plt.figure()
plt.imshow(difs_t[n][i])
plt.plot(sox,soy,'k.', markersize=1)
plt.show()

dife = np.copy( difs_t[n][i] )
dife[np.isnan(dife)] = 1000
wts = watershed( gaussian(dife,sigma=7), mask= ~np.isnan(difs_t[n][i]) )
sobb = sobel(wts) > 0
soy,sox = np.where(sobb)
 
plt.figure()
plt.imshow(difs_t[n][i])
plt.plot(sox,soy,'k.', markersize=1)
plt.show()
#%%


#%%


#%%
nss_v = list(range(len(ds_v)))
lxs_v,lys_v = [],[]
sscas_v, centss_v, centws_v, nscas_v, nscafs_v, labs_v = [], [], [], [], [], []
ssds_v, smms_v = [], []

for n in tqdm(nss_v):
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm = [], []

    scaprop = propscal_v[n]
    for i in range(len(scaprop)):
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        cenw = np.array( [ scaprop[i][j].centroid_weighted for j in range(len(scaprop[i])) ] )
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        slab = np.array( [ scaprop[i][j].label for j in range(len(scaprop[i])) ] )
        
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )

        nijs = np.array( [ (scaprop[i][j].moments_normalized).T for j in range(len(scaprop[i])) ] )

        bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
        hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        
        fil = (scas>2000) * ( sd>0.3 )
        
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
        
    sscas_v.append(ssca)
    centss_v.append(cents)
    centws_v.append(centws)
    labs_v.append(labe)
    nscas_v.append(nsca)
    nscafs_v.append(nscaf)
    ssds_v.append(ssd)
    smms_v.append(smm)
    lxs_v.append(lx)
    lys_v.append(ly)

    
nss_t = list(range(len(ds_t)))
lxs_t,lys_t = [],[]
sscas_t, centss_t, centws_t, nscas_t, nscafs_t, labs_t = [], [], [], [], [], []
ssds_t, smms_t = [], []

for n in tqdm(nss_t):
    lx,ly  = [],[]
    ssca,cents,centws,nsca,nscaf,labe = [], [], [], [], [], []
    ssd, smm = [], []

    scaprop = propscal_t[n]
    for i in range(len(scaprop)):
        cen = np.array( [ scaprop[i][j].centroid for j in range(len(scaprop[i])) ] )
        cenw = np.array( [ scaprop[i][j].centroid_weighted for j in range(len(scaprop[i])) ] )
        scas = np.array( [ scaprop[i][j].area for j in range(len(scaprop[i])) ] )
        slab = np.array( [ scaprop[i][j].label for j in range(len(scaprop[i])) ] )
        
        sd = np.array( [ scaprop[i][j].image_stdev for j in range(len(scaprop[i])) ] )
        mm = np.array( [ scaprop[i][j].image_minmax for j in range(len(scaprop[i])) ] )

        nijs = np.array( [ (scaprop[i][j].moments_normalized).T for j in range(len(scaprop[i])) ] )

        bn = (12 * scas**2)**(1/4) * (nijs[:,2,0]**3 / nijs[:,0,2] )**(1/8)
        hn = (12 * scas**2)**(1/4) * (nijs[:,0,2]**3 / nijs[:,2,0] )**(1/8)
        
        fil = (scas>2000) * ( sd>0.3 )
        
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
        
    sscas_t.append(ssca)
    centss_t.append(cents)
    centws_t.append(centws)
    labs_t.append(labe)
    nscas_t.append(nsca)
    nscafs_t.append(nscaf)
    ssds_t.append(ssd)
    smms_t.append(smm)
    lxs_t.append(lx)
    lys_t.append(ly)


#%%

n = 1
i = 0

sobb = thin(sobel(wtssal_v[n][i]) > 0)
soy,sox = np.where(sobb)

minx,maxx = np.min( xns_v[n] ), np.max( xns_v[n] )
miny,maxy = np.min( yns_v[n] ), np.max( yns_v[n] )
dx = xns_v[n][0,1] - xns_v[n][0,0]
dy = yns_v[n][0,0] - yns_v[n][1,0]

fig,axs = plt.subplots(1,3, figsize=(8,5), sharey=True)

ims1 = axs[1].imshow(difs_v[n][i], extent=(minx,maxx,miny,maxy))
axs[1].plot( xns_v[n][0,:][sox], yns_v[n][:,0][soy], 'k.', markersize=1)
for j in range(len(sscas_v[n][i])):
    # plt.text(centss_v[n][i][j][1], centss_v[n][i][j][0] , str(round(lys_v[n][i][j]/1. ,2)) )
    # plt.text(centss_v[n][i][j][1], centss_v[n][i][j][0] , str(round(sscas_v[n][i][j]/1. ,2)) )
    # plt.text(centss_v[n][i][j][1], centss_v[n][i][j][0] , str(round(ssds_v[n][i][j]/1. ,2)) )
    pass

axs[2].imshow(difs_v[n][i], extent=(minx,maxx,miny,maxy))
axs[2].plot( xns_v[n][0,:][sox], yns_v[n][:,0][soy], 'k.', markersize=1)
mask = np.zeros_like(wtssal_v[n][i])
for j in range(len(labs_v[n][i])):
    mask += wtssal_v[n][i] == labs_v[n][i][j]
mask += np.isnan(difs_v[n][i])
amask = np.ma.masked_where(mask, mask)
axs[2].imshow( amask, extent=(minx,maxx,miny,maxy), alpha = 0.5 )


ims2 = axs[0].imshow(difs_v[n][i], extent=(minx,maxx,miny,maxy)) #, vmin=-5)

for j in range(3):
    axs[j].plot([-30,20],[-188,-188],'k-', linewidth=3 )
    axs[j].text(-25, -209, '5 cm')
    axs[j].axis('off')
    # axs[j].set_ylim(top=125)
    # axs[j].set_xlim(-55,75)
    pass


fig.subplots_adjust(left=0.2)
cbar_ax = fig.add_axes([0.13, 0.23, 0.02, 0.59])
fig.colorbar(ims2, cax=cbar_ax, label='height (mm)', location='left')

# plt.tight_layout()
# plt.savefig('./Documents/watershed0.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

# sobb = sobel(wtssal_t[n][i]) > 0
# soy,sox = np.where(sobb)

# plt.figure()
# plt.imshow(difs_t[n][i])
# plt.plot(sox,soy, 'r.', markersize=3)
# for j in range(len(sscas_t[n][i])):
#     plt.text(centss_t[n][i][j][1], centss_t[n][i][j][0] , str(round(lys_t[n][i][j]/1. ,2)) )
#     # plt.text(centss_t[n][i][j][1], centss_t[n][i][j][0] , str(round(sscas_t[n][i][j]/1. ,2)) )
# plt.show()


# plt.figure()
# plt.imshow(difs_t[n][i])
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
# for l,n in enumerate(nss_v):
for l,n in enumerate([3,8,10,12,2]):
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
        # ax.errorbar(ts_v[n]/60, mly, yerr=sly, capsize=2, fmt='.-', errorevery=(l*2,20), \
        #              color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg' ) #, markersize=5, mfc='w' )

        # ax.plot( mlx , '.-', color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg')
        ax.errorbar(ts_v[n]/60, mlx, yerr=slx, capsize=2, fmt='.-', errorevery=(l*2,20), \
                     color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg' ) #, markersize=5, mfc='w' )



# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# ax.set_ylim(bottom=0)
# plt.show()

# fig,ax = plt.subplots()

# for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
# # for l,n in enumerate([5,9 ]):
#     dx = xns_t[n][0,1] - xns_t[n][0,0]
#     dy = yns_t[n][0,0] - yns_t[n][1,0]

#     mly, mlx = [], []
#     sly,slx = [], []
#     for i in range(len(lxs_t[n])):
#         mlx.append(np.nanmean(lxs_t[n][i]) * dx)
#         mly.append(np.nanmean(lys_t[n][i]) * dy)
#         slx.append(np.nanstd(lxs_t[n][i]) * dx)
#         sly.append(np.nanstd(lys_t[n][i]) * dy)
        
#     ax.plot( mly , '.-', color=np.array([0.5,salis_t[n]/26.2,1-salis_t[n]/26.2]) * (1 - (ang_t[n])/70 ), label=str(salis_t[n])+'; '+str(ang_t[n]))

#     # ax.errorbar(ts_t[l]/60, mly, yerr=sly, capsize=2, fmt='.-', label=str(salis_t[l])+' g/kg', \
#     #             errorevery=(l*2,20), color=np.array([0.5,salis_t[l]/35,1-salis_t[l]/35]) ) #, markersize=5, mfc='w' )
#     # ax.plot( mlx , '.-', color=(0.5,1-salis_t[n]/35,salis_t[n]/35))
#     # ax.plot( np.array(mlx) / np.array(mly) , '.-', color=(0.5,1-salis_t[n]/35,salis_t[n]/35))

# cbar = plt.colorbar( plt.cm.ScalarMappable(norm=Normalize(0, 35), cmap=newcmp), ax=ax)
# # cbar.ax.tick_params(labelsize=12)
# cbar.set_label( label="S (g/kg)") #, size=12)

# plt.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.set_xlabel('time (min)')
# ax.set_ylabel(r'$\lambda_y$ (mm)')
ax.set_ylabel(r'$\lambda_x$ (mm)')

ax.legend(loc='lower left')
# plt.savefig('./Documents/lamx_t.png',dpi=400, bbox_inches='tight', transparent=True)
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
        mly, mlx = [], []
        sly,slx = [], []
        for i in range(len(lxs_v[l])):
            mlx.append(np.nanmean(lxs_v[l][i]) * dx)
            mly.append(np.nanmean(lys_v[l][i]) * dy)
            slx.append(np.nanstd(lxs_v[l][i]) * dx)
            sly.append(np.nanstd(lys_v[l][i]) * dy)
            
        indy, indx = np.argmin(mly[30:]), np.argmin(mlx[30:])
        # ax.errorbar(salis_v[l], mly[indy+30], yerr=sly[indy+30], capsize=2, fmt='.-'  )

        mey, eey = np.nanmean(mly[30:]), np.nanmean(sly[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        li1y = ax[r'$a)$'].errorbar(salis_v[l], mey, yerr=eey, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1ys.append(li1y)

        mex, eex = np.nanmean(mlx[30:]), np.nanmean(slx[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
        li1x = ax[r'$b)$'].errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='o', markersize=5, \
                             color=((angys_v[l]+19)/70,0.5,1-(angys_v[l]+19)/70), mfc='w'  )        
        li1xs.append(li1x)
        # ax.errorbar(salis_v[l], mex, yerr=eex, capsize=2, fmt='b.', markersize=15, mfc='w', alpha=0.5, label=r'$\lambda_x$'  )

li2ys,li2xs = [],[]
for l,n in enumerate([7,6,5,28,23,8,9,15 ]):
    dx = xns_t[n][0,1] - xns_t[n][0,0]
    dy = yns_t[n][0,0] - yns_t[n][1,0]

    mly, mlx = [], []
    sly,slx = [], []
    for i in range(len(lxs_t[n])):
        mlx.append(np.nanmean(lxs_t[n][i]) * dx)
        mly.append(np.nanmean(lys_t[n][i]) * dy)
        slx.append(np.nanstd(lxs_t[n][i]) * dx)
        sly.append(np.nanstd(lys_t[n][i]) * dy)
        
    indy, indx = np.argmin(mly[30:]), np.argmin(mlx[30:])
    # ax.errorbar(salis_t[l], mly[indy+30], yerr=sly[indy+30], capsize=2, fmt='.-'  )

    mey, eey = np.nanmean(mly[30:]), np.nanmean(sly[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # li2y = ax[r'$a)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2y = ax[r'$a)$'].errorbar(salis_t[n], mey, yerr=eey, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2ys.append(li2y)

    mex, eex = np.nanmean(mlx[30:]), np.nanmean(slx[30:])  #np.sqrt(np.sum(np.array(sly[30:])**2)) #/ len(sly)
    # ax.errorbar(salis_t[l], mex, yerr=eex, capsize=2, fmt='b.', markersize=15, mfc='w', alpha=0.5, label=r'$\lambda_x$'  )
    # li2x = ax[r'$b)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+19)/70,0.5,1-(angys_t[n]+19)/70) )
    li2x = ax[r'$b)$'].errorbar(salis_t[n], mex, yerr=eex, capsize=2, fmt='o', markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2xs.append(li2x)


ax[r'$a)$'].set_ylim(0,55)
ax[r'$b)$'].set_ylim(0,55)

ax[r'$a)$'].set_ylabel(r'$\lambda_y$ (mm)')
ax[r'$b)$'].set_ylabel(r'$\lambda_x$ (mm)')
ax[r'$b)$'].set_xlabel('Salinity (g/kg)')


ax[r'$a)$'].legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.2), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/lamxy_s.png',dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%
# Area scallops (y quiza numero)

fig,ax = plt.subplots()

# for n in nss_v:
for l,n in enumerate([3,8,10,12,2]):
    if salis_v[n] > 7 and salis_v[n] < 25: 
        dx = xns_v[n][0,1] - xns_v[n][0,0]
        dy = yns_v[n][0,0] - yns_v[n][1,0]
        
        sarm,sars = [], []
        for i in range(len(sscas_v[n])):
            sarm.append( np.nanmean(sscas_v[n][i] * dx*dy) )
            sars.append( np.nanstd(sscas_v[n][i] * dx*dy) )

        plt.errorbar(ts_v[n]/60, np.array(sarm)/100, yerr=np.array(sars)/100,  capsize=2, fmt='.-', errorevery=(l*2,20), \
                     color=(0.5,1-salis_v[n]/35,salis_v[n]/35), label=r'$S = $'+str(salis_v[n])+' g/kg' )

ax.set_xlabel('time (min)')
ax.set_ylabel(r'Area (cm$^2$)')
ax.legend(bbox_to_anchor=(0.5,1.2), loc='upper center' , ncol=3)

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
    
        li1y = ax.errorbar(salis_v[n], np.nanmean(sarm[30:])/100, yerr=np.nanmean(sars[30:])/100, capsize=2, fmt='o', \
                     markersize=5, color=((angys_v[n]+19)/70,0.5,1-(angys_v[n]+19)/70), mfc='w' )
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
    li2y = ax.errorbar(salis_t[n], np.nanmean(sarm[30:])/100, yerr=np.nanmean(sars[30:])/100, capsize=2, fmt='o', \
                 markersize=5, color=((angys_t[n]+17)/47,0.5,1-(angys_t[n]+17)/47) )
    li2ys.append(li2y)

ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel(r'Area (cm$^2$)')
ax.legend([li1ys[-1],li2ys[-1],li2ys[3],li2ys[1],li2ys[2] ],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.12), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/ares.png',dpi=400, bbox_inches='tight', transparent=True)
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
for n in [0,3,8]:
    mimam, mimas = [],[]
    for i in range(len(smms_v[n])):
        mimam.append( np.nanmean(smms_v[n][i]) )
        mimas.append( np.nanstd(smms_v[n][i]) )
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
ax.set_ylabel('Amplitud (nm)')
ax.set_ylim(bottom=0)

ax.legend( [li1s[-1],li2s[-1],li2s[3],li2s[1],li2s[2]],['Set 1', 'Set 2',r'-$15°$',r'$15°$',r'$30°$'],\
                   bbox_to_anchor=(0.5,1.12), loc='upper center' , ncol=5 )

# plt.savefig('./Documents/amps.png', dpi=400, bbox_inches='tight', transparent=True)
plt.show()

#%%




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

    if safi: plt.savefig('./Documents/densityst.png', dpi=400, bbox_inches='tight', transparent=True)
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


density_fig()

#%%
# =============================================================================
# 3D graphs
# =============================================================================
n = 14
i = 18

# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan

halt = np.load('./Documents/Height profiles/ice_block_0_'+sal_v[n]+'.npy')
halg = nangauss(halt, 2)

# halt = np.load('./Documents/Height profiles/profile_s'+sal[n]+'_t'+inc[n]+'.npy')
# halg = nangauss(halt, 2)

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

ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.25)
ax.set_zlim(lzd-5,lzi+5)
ax.set_xlim(lxd+5,lxi-5)
ax.set_ylim(lyi-5,lyd+5)

plt.locator_params(axis='y', nbins=3)
# ax.text(30,40,180, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title( 't = '+str(0.5*n)+'s', fontsize=12)

# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
# ax.view_init(17,50)
ax.view_init(17,120)

# ax.text(-21,60,240, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title('t = '+str(0.5*n)+'min', fontsize=12)

# plt.axis('off')
# plt.savefig('./Documents/blocksurface3.png',dpi=192 * 5, transparent=True) # , bbox_inches='tight')

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

