#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:45:31 2024

@author: tomasferreyrahauchar
"""

import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.widgets import Button, Slider
# mpl.use('Agg') 

# from scipy import signal
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks, correlate, fftconvolve, peak_prominences, correlate2d, peak_prominences, savgol_filter
import scipy.ndimage as snd # from scipy.ndimage import rotate from scipy.ndimage import convolve
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline, Rbf, griddata, splrep, splev, splprep
from scipy.ndimage import maximum_filter

# import rawpy
import imageio
from tqdm import tqdm
from time import time
import h5py

from skimage.filters import gaussian, frangi, sato, hessian, meijering, sobel, rank, roberts #, gabor
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion, thin, skeletonize, remove_small_objects, binary_opening
from skimage.segmentation import felzenszwalb, mark_boundaries, watershed, chan_vese, slic, quickshift
from skimage.restoration import unwrap_phase as unwrap
from skimage.feature import peak_local_max, canny
from skimage.transform import rescale, resize, hough_circle, hough_circle_peaks
from sklearn.neighbors import NearestNeighbors

import networkx as nx

import uncertainties as un
from uncertainties import unumpy

from PIL import Image, ImageDraw
import io
import cv2
#%%
# file = ['/Volumes/Ice blocks/s0_t0/Hielo_s0_t0/Hielo_s0_t0','.tif']
file = ['/Volumes/Ice blocks/sc_s0_t0(2)/DSC_','.JPG']

imsm,ims = [],[]
for i in tqdm(range(6243,6336)):
    # ims.append(imageio.imread(file[0]+str(i)+file[1])[1000:6500,2000:4400,:])
    ims.append( rescale( imageio.imread(file[0]+str(i)+file[1])[1000:6500,2000:4400,:], scale=0.8, anti_aliasing=True) )
    # imsm.append(imageio.imread(file[0]+str(i)+file[1])[1808:5900,2000:4400,0])
    # imsm.append( rescale( imageio.imread(file[0]+str(i)+file[1])[5900:6500,2000:4400,:], scale=0.3, anti_aliasing=True)  )

imsm = np.array(imsm)
ims = np.array(ims)

# back = np.median(ims, axis=0)
#%%
file = ['/Volumes/Ice blocks/sc_s0_t0(2)/DSC_','.JPG']
# im = imageio.imread(file[0]+'2047'+file[1])[420:3400,300:1900,2]
im = imageio.imread(file[0]+'6243'+file[1])[1808:5900,2000:4400,:]
# back = imageio.imread(file[0]+'6335'+file[1])[1000:6500,2000:4400,0]
#%%


#%%

from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
#%%

training_labels = np.zeros( ims[0].shape[:2], dtype=np.uint8)
training_labels[800:1800,0:150] = 1
training_labels[1200:1600,0:150] = 1
training_labels[3200:00,1800:] = 1
training_labels[4200:,:250] = 2
training_labels[4210:,1750:] = 2
training_labels[:300,:200] = 2
training_labels[:200,1800:] = 2
training_labels[1000:3500,700:1600] = 3
# training_labels[200:500,800:1500] = 3
# training_labels[3850:4250,900:1500] = 3
training_labels[550:700,:200] = 4
training_labels[550:700,1800:] = 4


t1 = time()
sigma_min = 1
sigma_max = 16

features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=False,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    channel_axis=-1,
)
t1b = time()
features = features_func(ims[0][:,:,:1])
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)
t2 = time()
print(t2-t1, t1b-t1)


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow( ims[0][:,:,:1], cmap='gray'  )
ax[0].contour(training_labels)
# ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
# ax[1].set_title('Segmentation')
# fig.tight_layout()
plt.show()

plt.figure()
# plt.imshow( ims[0][:,:,:1], cmap='gray' )
# plt.contour(training_labels)
plt.imshow( result == 3 ) 
plt.show()
#%%
features = features_func(ims[60][:,:,:1])
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
clf = future.fit_segmenter(training_labels, features, clf)
result = future.predict_segmenter(features, clf)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow( ims[60][:,:,:1], cmap='gray'  )
# ax[0].contour(training_labels)
ax[1].imshow(result)
plt.show()

plt.figure()
# plt.imshow( imsm[0] )
# plt.contour(training_labels)
plt.imshow( result == 3 ) 
plt.show()

#%%
# =============================================================================
# With matlab clicking 
# =============================================================================
file = ['/Volumes/Ice blocks/sc_s0_t0(2)/DSC_','.JPG']

ims = []

for i in tqdm(range(6243,6336)):
# for i in tqdm(range(6243,6244)):
    ims.append(imageio.imread(file[0]+str(i)+file[1])[1000:6500,2000:4400,0])

ims = np.array(ims)
#%%
d = 5 / 83.51342674276873 # mm/px, error = 0.00036 mm/px

masx,masy = [],[]
fil = []
for i in range(5,98):
    # print('algo/{0:04}.csv'.format(i))
    mas = np.loadtxt('./Documents/MATLAB/s0_t0/{0:04}.csv'.format(i), delimiter=',').T
    masx.append( (mas[1]+200)*d )
    masy.append( (5500 - (5655-mas[0]))*d )

for i in range(len(masy)):
    fil.append( np.gradient( gaussian(masy[i], 20) ) > 0.05 ) # 0.04 )

ts = np.arange(len(masx)) * 30

xes, yes = masx[0][fil[0]], masy[0][fil[0]]
pixx = np.arange(len(yes)) 

lreg = linregress(pixx, yes)
ygr = pixx * lreg[0] + lreg[1]

xgr = []
xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))
for i in range(1,len(masx)):
    xes, yes = masx[i][fil[i]], masy[i][fil[i]]
    xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))
xgr = np.array(xgr)

pend, nflt = [],[]
for line in range(len(xgr.T)):
    flt = np.isnan(xgr[:,line])
    if np.sum(~flt) > 19:
        lreg = linregress(ts[~flt], xgr[:,line][~flt])
        pend.append(lreg[0])
        nflt.append(np.sum(~flt))
    else:
        pend.append(np.nan)
        nflt.append(np.sum(~flt))
pend = np.array(pend)

flt = ~np.isnan(pend)
print('melt rate =', np.trapz( -pend[flt], ygr[flt] ) / np.trapz( np.ones_like(pend[flt]), ygr[flt] ) , 'mm/s')
#%%

plt.figure()

for i in range(0,90,3):
# for i in [35]:
    plt.plot(masx[i], masy[i], 'b-')
    plt.plot(masx[i][fil[i]], masy[i][fil[i]], 'r-')

plt.ylabel('y (mm)')
plt.xlabel('x (mm)')
plt.axis('equal')
plt.grid()

plt.show()

plt.figure()
for i in range(0,91,5):
    xes, yes = masx[i][fil[i]], masy[i][fil[i]]
    plt.plot(xgr[i],ygr, '-'  )
plt.ylabel('y (mm)')
plt.xlabel('x (mm)')
plt.grid()

plt.show()

plt.figure()
plt.plot(ygr, -pend )
plt.xlabel('y (mm)')
plt.ylabel('melt rate (mm/s)')
plt.grid()
plt.show()



#%%
# =============================================================================
# Calibration
# =============================================================================
file = '/Volumes/Ice blocks/sc_s0_t0(2)/DSC_6336.JPG'

cal = np.array(imageio.imread(file)[1000:6500,2000:4400,0])
calt = np.array(imageio.imread(file)[:,:,0])

np.shape(cal)
#%%
plt.figure()
plt.imshow(cal)
plt.show()
# plt.figure()
# plt.imshow(calt)
# plt.show()
#%%
x1 = 1234.1
y1 = 3090.3

np.sqrt( (1150.7 - x1)**2 + (3090.5 - y1)**2 )

#%%
t1 = time()
cal1 = gaussian(cal,1) 
calg = gaussian(cal,50) 
mmm = np.max(cal1 - calg)

# calt = remove_small_objects( cal > 25, min_size=250) 
calt = remove_small_objects( ((cal1 - calg) / mmm) > 0.25 , min_size=250) 
kernel = np.zeros((11,11))
kernel[5] = 1
calb = remove_small_objects( binary_opening(calt, kernel ), min_size=100 )

leb = label( binary_closing( calb[:,25:-50], disk(10) ) )
ppro = regionprops(leb)

t2 = time()
print(t2-t1)
#%%
centsy = [(ppro[i].centroid)[0] for i in range(len(ppro)) ]
centsx = [(ppro[i].centroid)[1]+25 for i in range(len(ppro)) ]

# ygr = [79] * 28
# xgr = [83*i-3 for i in range(1,29)]


xgr = np.array( [ i/3.7 + j*83.3 + 62 for i in range(1,66) for j in range(0,28) ] ) 
ygr = np.array( [ 83.8*i - j/3 -4.8 for i in range(1,66) for j in range(0,28)] )

grid = np.zeros_like(xgr).reshape((65,28))

ocx, ocy = [], []
for i in range(len(xgr)):
        if i == 1189:
            ocx.append(np.nan)
            ocy.append(np.nan)
        else: 
            indx = np.argmin( (centsx - xgr[i])**2 + (centsy - ygr[i])**2  )
            ocx.append(centsx[indx])
            ocy.append(centsy[indx])
ocx, ocy = np.array(ocx), np.array(ocy)

oocy = np.array( [ ocy[(i*28)%(65*28) + int(i/65)] for i in range(len(ocy)) ])
oocx = np.array( [ ocx[(i*28)%(65*28) + int(i/65)] for i in range(len(ocy)) ])

#%%
fin = 1000
# p = 11 #1189
# q = 17

# plt.figure()
# # plt.imshow( cal1 , cmap='gray')

# # plt.plot(centsx[:fin], centsy[:fin], 'r.')
# # plt.plot(xgr[:fin], ygr[:fin], 'b.') #,alpha=0.5)

# # plt.plot(ocx[:fin], ocy[:fin], 'g.') #,alpha=0.5)
# plt.plot(oocx[:fin], oocy[:fin], 'b.') #,alpha=0.5)

# # plt.plot(xgr[p], ygr[p], 'g.')
# # plt.plot(centsx[q], centsy[q], 'r.')

# plt.grid()
# plt.ylim(-200,5700)
# plt.xlim(-50,2420)

# plt.show()


distx = np.sqrt((ocx[1:] - ocx[:-1])**2 + (ocy[1:] - ocy[:-1])**2)
disty = np.sqrt((oocx[1:] - oocx[:-1])**2 + (oocy[1:] - oocy[:-1])**2)

filx,fily = (distx>70) * (distx<90), (disty>70) * (disty<90) 


plt.figure()
plt.plot((ocx[1:] + ocx[:-1])[filx]/2, distx[filx]  ,'.')
plt.plot( (oocy[1:] + oocy[:-1])[fily]/2, disty[fily]  ,'.')
plt.ylim(80,87)
plt.show()

np.mean(np.concatenate((distx[filx],disty[fily]))), np.std(np.concatenate((distx[filx],disty[fily])))


#%%
# =============================================================================
# Second ice (30°C)
# =============================================================================
file = ['/Volumes/Ice blocks/sc_s0_t30/DSC_','.JPG']

ims = []

for i in tqdm(range(2951,3040)):
# for i in [2951]:
    
    ims.append(imageio.imread(file[0]+str(i)+file[1])[:,:,0])

ims = np.array(ims)
#%%
d = 5 / 42.89206598813636 # mm/px, error = 0.001609881121724237 mm/px

masx,masy = [],[]
fil = []
for i in range(1,90):
    # print('algo/{0:04}.csv'.format(i))
    mas = np.loadtxt('./Documents/MATLAB/s0_t30/{0:04}.csv'.format(i), delimiter=',').T
    masx.append( mas[1] *d )
    masy.append( mas[0] *d )
    
lre = linregress( masx[0][1600:4000], masy[0][1600:4000] )
angled = 90 - np.arctan(lre[0]) * 180/np.pi
angler = angled * np.pi/180

masxr = [ masx[i] * np.cos(angler) - masy[i] * np.sin(angler) for i in range(len(masx)) ]
masyr = [ masx[i] * np.sin(angler) + masy[i] * np.cos(angler) for i in range(len(masy)) ]

for i in range(len(masyr)):
    fil.append( np.gradient( gaussian(masyr[i], 20) ) > 0.08 ) 

ts = np.arange(len(masx)) * 30

xes, yes = masxr[0][fil[0]], masyr[0][fil[0]]
pixx = np.arange(len(yes)) 

lreg = linregress(pixx, yes)
ygr = pixx * lreg[0] + lreg[1]

xgr = []
xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))
for i in range(1,len(masx)):
    xes, yes = masxr[i][fil[i]], masyr[i][fil[i]]
    xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))
xgr = np.array(xgr)

pend2, pend1, nflt = [],[],[]
for line in range(len(xgr.T)):
    flt = np.isnan(xgr[:,line])
    if np.sum(~flt) > 19:
        # lreg = linregress(ts[~flt], xgr[:,line][~flt])
        lreg = np.polyfit(ts[~flt], xgr[:,line][~flt], 2)
        pend2.append(lreg[0])
        pend1.append(lreg[1])
        nflt.append(np.sum(~flt))
    else:
        pend2.append(np.nan)
        pend1.append(np.nan)
        nflt.append(np.sum(~flt))
pend2, pend1 = np.array(pend2), np.array(pend1)

flt = ~np.isnan(pend)
# print('melt rate =', np.trapz( -pend[flt], ygr[flt] ) / np.trapz( np.ones_like(pend[flt]), ygr[flt] ) , 'mm/s')
print('melt rate =', -np.trapz( pend2[flt] * ts[-1] + pend1[flt], ygr[flt] ) / np.trapz( np.ones_like(pend[flt]), ygr[flt] ) , 'mm/s' )
#%%
i = 50

# ini = 1600
# fin = 4000

# plt.figure()

# # plt.imshow( snd.rotate(ims[i],angled), extent=(0,2752,0,4128) )
# # plt.imshow( ims[i], extent=(0,2752,0,4128) )

# # plt.plot( masx[i], masy[i], 'r-' )
# # plt.plot( masx[i][fil[i]], masy[i][fil[i]], 'm-' )
# for i in range(0,50,5):
#     plt.plot( masxr[i], masyr[i], 'r-' )
#     plt.plot( masxr[i][fil[i]], masyr[i][fil[i]], 'g-' )

# plt.axis('equal')

# plt.show()

# plt.figure()
# for i in range(0,90,5):
#     xes, yes = masxr[i][fil[i]], masyr[i][fil[i]]
#     plt.plot(xgr[i], '-'  )
# plt.ylabel('y (mm)')
# plt.xlabel('x (mm)')
# plt.grid()

# line = 1000
# flt = np.isnan(xgr[:,line])
# lreg = linregress(ts[~flt], xgr[:,line][~flt])
# ll = np.polyfit(ts[~flt], xgr[:,line][~flt], 2)

# print(ll)
# print(lreg[0], lreg[1])

# plt.figure()
# plt.plot(ts/60, xgr[:, line], '.-' )
# # plt.plot(ts/60, ts * pend[line] + xgr[0,line] - 3, '.-' )
# plt.plot(ts/60, ts * lreg[0] + lreg[1], '.-' )
# plt.plot(ts/60, ts**2 * ll[0] + ts * ll[1] + ll[2], '.-' )
# plt.show()

plt.figure()
plt.plot(ygr, - (pend2 * ts[-1] + pend1) )
plt.xlabel('y (mm)')
plt.ylabel('melt rate (mm/s)')
plt.grid()
plt.show()

#%%
# =============================================================================
# Calibration
# =============================================================================
file = '/Volumes/Ice blocks/sc_s0_t30/DSC_3043.JPG'

cal = np.array(imageio.imread(file)[500:3500,400:2200,2])
# im = np.array(imageio.imread('/Volumes/Ice blocks/sc_s0_t30/DSC_2951.JPG')[:,:,2])

np.shape(cal)

plt.figure()
plt.imshow( cal )
plt.show()


#%%

t1 = time()
calc = rank.mean(cal, disk(10)) 
calt = remove_small_objects((calc - cal) > 100, min_size = 230)

kernel = np.zeros((11,11))
kernel[5] = 1
calb = remove_small_objects( binary_opening(calt, kernel ), min_size=100 )
calb = remove_small_objects( binary_opening(calb, kernel.T ), min_size=1 )
calb = remove_small_objects( binary_opening(calb, disk(5) ), min_size=1 )

leb = label( calb[:,:] )
ppro = regionprops(leb)

centsy = [(ppro[i].centroid)[0] for i in range(len(ppro)) ]
centsx = [(ppro[i].centroid)[1] for i in range(len(ppro)) ]

xgr = np.array( [ 42.6*i - 0.25*j + 44.5 for j in range(0,69) for i in range(42) ] ) 
ygr = np.array( [ 43*j + 25 for j in range(0,69) for i in range(42) ] )

ocx, ocy = [], []
for i in range(len(xgr)):
        if i == 1866:
            ocx.append(np.nan)
            ocy.append(np.nan)
        else: 
            indx = np.argmin( (centsx - xgr[i])**2 + (centsy - ygr[i])**2  )
            ocx.append(centsx[indx])
            ocy.append(centsy[indx])
ocx, ocy = np.array(ocx), np.array(ocy)

oocy = np.array( [ ocy[(i*42)%(69*42) + int(i/69)] for i in range(len(ocy)) ])
oocx = np.array( [ ocx[(i*42)%(69*42) + int(i/69)] for i in range(len(ocy)) ])


t2 = time()
print(t2-t1)
#%%
# plt.figure()
# plt.imshow( calb )
# plt.plot( centsx, centsy, 'r.', markersize=1 )
# plt.show()

# fin = 1866
# fig, ax = plt.subplots()


# line, = ax.plot( centsx, centsy, 'r.', markersize=3 )
# # line, = ax.plot( xgr[:fin], ygr[:fin], 'b.', markersize=3 )
# # line, = ax.plot( ocx[:fin], ocy[:fin], 'b.', markersize=3 )
# line, = ax.plot( oocx[:fin], oocy[:fin], 'b.', markersize=3 )
# # ax.set_ylim(0,3000)

# fig.subplots_adjust(left=0.25, bottom=0.25)
# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# slid = Slider( ax=axfreq, label='Fin', valmin=1, valmax=len(xgr), valinit=50, valstep=1)
# def update(val):
#     # line.set_ydata( ygr[:int(slid.val)] )
#     # line.set_xdata( xgr[:int(slid.val)] )
#     line.set_ydata( oocy[:int(slid.val)] )
#     line.set_xdata( oocx[:int(slid.val)] )
#     fig.canvas.draw_idle()
# slid.on_changed(update)

# plt.grid()
# plt.show()

distx = np.sqrt((ocx[1:] - ocx[:-1])**2 + (ocy[1:] - ocy[:-1])**2)
disty = np.sqrt((oocx[1:] - oocx[:-1])**2 + (oocy[1:] - oocy[:-1])**2)

filx,fily = (distx>30) * (distx<60), (disty>30) * (disty<60) 

plt.figure()
plt.plot(ocx[1:][filx] , distx[filx], '.' )
plt.plot(oocy[1:][fily] , disty[fily], '.' )
plt.show()



np.mean(np.concatenate((distx[filx],disty[fily]))), np.std(np.concatenate((distx[filx],disty[fily])))
# np.mean(distx[filx]),np.std(distx[filx]), np.mean(disty[fily]),np.std(disty[fily])

#%%
# =============================================================================
# Shadow opaque 30°
# =============================================================================

file = ['/Volumes/Ice blocks/so_s0_t30/_DSC','.JPG']

ims = []
for i in tqdm(range(1693,1698)):
    ims.append(imageio.imread(file[0]+str(i)+file[1])[400:4800,1400:6900,0])

back = np.mean(np.array(ims),axis=0)

ims = []
for i in tqdm(range(1590,1672)):
    ims.append(imageio.imread(file[0]+str(i)+file[1])[400:4800,1400:6900,0])

ims = np.array(ims)

#%%
d = 5 / 77.48104145562611  # mm/px, error = 0.0012417621082226739 mm/px

t1 = time()
imts = []
tramos = []
for i in tqdm(range(len(ims))):
    imt = sobel(ims[i] - back)
    lab = label(imt<1.5)
    num = np.median(lab[1600:2000,2200:2600])
    imb = remove_small_holes( binary_closing(lab==num, disk(10)), area_threshold=1e5)
    imts.append(imb)
    
    edd = imb * 1. - binary_erosion(imb) * 1.
    yed,xed = np.where(edd)

    edge = []
    for j in range(len(xed)):
        edge.append([xed[j],yed[j]])

    clf = NearestNeighbors(n_neighbors=2).fit(edge)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_array(G)
    order = list(nx.dfs_preorder_nodes(T, 0))
    tra_ord = np.array(edge)[order]
    tramos.append(tra_ord[50:-50,:])

t2 = time()
print(t2-t1)

#%%
xpi, ypi = [], [] 
for i in tqdm(range(0,len(tramos),1)):  #46,47

    yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]
    ay,ax = np.argmax(yed), np.argmax(xed)
    part = np.sort( [ay,ax] )
    
    if i == 46:
        ffax = np.concatenate( (xed[4930:5269:1], xed[9705:6250:-1]) )
        ffay = np.concatenate( (yed[4930:5269:1], yed[9705:6250:-1]) )
        
    elif i == 47:
        ffay, ffax = yed[5510:9280], xed[5510:9280]
        
    else:
        ffax, ffay = xed[part[0]:part[1]], yed[part[0]:part[1]]
        
    xpi.append(ffax)
    ypi.append(ffay)

lre = linregress( xpi[0], ypi[0] )
angled = 90 + np.arctan(lre[0]) * 180/np.pi
angler = angled * np.pi/180

masxr = [ xpi[i] * np.cos(angler) + ypi[i] * np.sin(angler) for i in range(len(xpi)) ]
masyr = [ -xpi[i] * np.sin(angler) + ypi[i] * np.cos(angler) for i in range(len(xpi)) ]


xes, yes = masxr[0], masyr[0]
pixx = np.arange(len(yes))

lreg = linregress(pixx, yes)
ygr = pixx * lreg[0] + lreg[1]

xgr = []
xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))
for i in range(1,len(xpi)):
    xes, yes = masxr[i], masyr[i]
    spe = int( np.sign( linregress(np.arange(len(yes)), yes)[0] ))
    xgr.append(np.interp(ygr, yes[::spe], xes[::spe], left=np.nan, right=np.nan))
xgr = np.array(xgr) * d

ts = np.arange(len(xgr)) * 30

pend2, pend1, nflt = [],[],[]
for line in range(len(xgr.T)):
    flt = np.isnan(xgr[:,line])
    if np.sum(~flt) > 19:
        # lreg = linregress(ts[~flt], xgr[:,line][~flt])
        lreg = np.polyfit(ts[~flt], xgr[:,line][~flt], 2)
        pend2.append(lreg[0])
        pend1.append(lreg[1])
        nflt.append(np.sum(~flt))
    else:
        pend2.append(np.nan)
        pend1.append(np.nan)
        nflt.append(np.sum(~flt))
pend2, pend1 = np.array(pend2), np.array(pend1)

flt = ~np.isnan(pend)
# print('melt rate =', np.trapz( -pend[flt], ygr[flt] ) / np.trapz( np.ones_like(pend[flt]), ygr[flt] ) , 'mm/s')
print('melt rate =', -np.trapz( pend2[flt] * ts[-1] + pend1[flt], ygr[flt] ) / np.trapz( np.ones_like(pend[flt]), ygr[flt] ) , 'mm/s' )

#%%
i = 53

yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]
ay,ax = np.argmax(yed), np.argmax(xed)

ini = 0
fin = 1

plt.figure()
# plt.imshow(imts[i])
plt.imshow( (ims[i].T)[::-1] )
# plt.plot( tramos[i][:,0], tramos[i][:,1],'r-' )
plt.plot( xed[ini:-fin],yed[ini:-fin], 'r-' )
plt.plot(xpi[i], ypi[i], 'k-' )

plt.show()



plt.figure()
for i in range(0,70,5):
    plt.plot(masxr[i],masyr[i], 'r.-')
    plt.plot(xgr[i], ygr, 'b.-')
plt.show()

plt.figure()

for i in range(0,70,5):
    plt.plot( xgr[i], ygr, 'r-' )
plt.plot(xgr[:, 1000], [ygr[1000]]*82, '.-'  )

# plt.plot(ts, xgr[:, 1000] , '.-'  )

plt.show()

#%%

# xgr = []
# xgr.append(np.interp(ygr, yes, xes, left=np.nan, right=np.nan))

i = 5
xes, yes = masxr[i], masyr[i]
spe = int( np.sign( linregress(np.arange(len(yes)), yes)[0] ))
print(spe)
xgr = np.interp(ygr, yes[::spe], xes[::spe], left=np.nan, right=np.nan)

# xgr

plt.figure()
plt.plot(xes,yes)
plt.plot(xgr,ygr)
# plt.plot(yes)
plt.show()

#%%
plt.figure()
for i in range(4,len(tramos),7):  #46,47
# for i in [46]:

    yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]
    ay,ax = np.argmax(yed), np.argmax(xed)
    part = np.sort( [ay,ax] )
    
    if i == 46:
        ffax = np.concatenate( (xed[4930:5269:1], xed[9705:6250:-1]) )
        ffay = np.concatenate( (yed[4930:5269:1], yed[9705:6250:-1]) )

        
    elif i == 47:
        ffay, ffax = yed[5510:9280], xed[5510:9280]
        
    else:
        ffax, ffay = xed[part[0]:part[1]], yed[part[0]:part[1]]
    
    plt.plot(xed, yed, 'g-' )    
    plt.plot(ffax,ffay, 'r-' )
    
    # plt.plot( yed, 'g-' )
    # plt.plot( xed, 'r-' )
    
    # plt.plot( np.gradient( gaussian(yed,0) ,gaussian(xed,0) ) ,'r-' )
    # plt.plot( np.gradient( gaussian(yed,20), gaussian(xed,20) ) ,'g-' )
plt.gca().invert_yaxis()
plt.show()


# plt.figure()
# # for i in range(0,82,5):
#     # yed, xed = tramos[i][:,0], tramos[i][:,1]
# #     plt.plot( xed,yed ,'r-' )
# plt.show()



#%%
i = 46

yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]

ini = 4930
fin = 5269

dex = np.concatenate( (xed[4930:5269:1], xed[9705:6250:-1]) )
dey = np.concatenate( (yed[4930:5269:1], yed[9705:6250:-1]) )

plt.figure()

plt.plot( xed, yed, 'r.' )
plt.plot( dex , dey, 'b.-' )
# plt.plot( xed[ini:fin], yed[ini:fin], 'g.-' )

plt.gca().invert_yaxis()
plt.show()

#%%
i = 47

yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]

ini = 5510
fin = 9280

plt.figure()

plt.plot( xed, yed, 'r.' )
# plt.plot( dex , dey, 'b.-' )
plt.plot( xed[ini:fin], yed[ini:fin], 'g.-' )

plt.gca().invert_yaxis()
plt.show()
#%%
# =============================================================================
# Calibration
# =============================================================================
file = '/Volumes/Ice blocks/so_s0_t30/_DSC1686.JPG'
# cal = imageio.imread(file)[800:4600,1200:7000,0] #[400:4800,1400:6900,0]
cal = imageio.imread(file)[400:4800,1400:6900,0]
#%%
t1 = time()
# calc = rank.otsu(cal, disk(15)) 
g3 = gaussian(cal,3)
g30 = gaussian(cal,30)
no = g3 - g30
norm = np.zeros_like(no)
norm[430:-250,40:-40] = (no[430:-250,40:-40] - np.min(no[430:-250,40:-40])) / (np.max(no[430:-250,40:-40]) - np.min(no[430:-250,40:-40])) 
imt = remove_small_objects(((norm > 0.4)), min_size=300)
calb = imt

leb = label( calb[:,:] )
ppro = regionprops(leb)

centsy = [(ppro[i].centroid)[0] for i in range(len(ppro)) ]
centsx = [(ppro[i].centroid)[1] for i in range(len(ppro)) ]

xgr = np.array( [ 78.05*i - 0.2*j + 105 for j in range(48) for i in range(69) ] ) 
ygr = np.array( [ 0.38*i + 76.7*j + 473 for j in range(48) for i in range(69) ] )

ocx, ocy = [], []
for i in range(len(xgr)):
        if i == 1681:
            ocx.append(np.nan)
            ocy.append(np.nan)
        else: 
            indx = np.argmin( (centsx - xgr[i])**2 + (centsy - ygr[i])**2  )
            ocx.append(centsx[indx])
            ocy.append(centsy[indx])
ocx, ocy = np.array(ocx), np.array(ocy)

oocy = np.array( [ ocy[(i*69)%(48*69) + int(i/48)] for i in range(len(ocy)) ])
oocx = np.array( [ ocx[(i*69)%(48*69) + int(i/48)] for i in range(len(ocy)) ])


t2 = time()
print(t2-t1)

#%%
# plt.figure()
# plt.imshow(calb)
# plt.plot(centsx, centsy, 'r.' )
# plt.show()

# xgr = np.array( [ 78.05*i - 0.2*j + 105 for j in range(48) for i in range(69) ] ) 
# ygr = np.array( [ 0.38*i + 76.7*j + 473 for j in range(48) for i in range(69) ] )

# fin = 1682
# plt.figure()
# plt.plot(centsx, centsy, 'r.' )
# plt.plot(xgr[:fin], ygr[:fin], 'b.' )
# # plt.plot(oocx[:fin], oocy[:fin], 'g.' )
# plt.show()

# fin = 1866
# fig, ax = plt.subplots()

# line, = ax.plot( centsx, centsy, 'r.', markersize=3 )
# # line, = ax.plot( xgr[:fin], ygr[:fin], 'b.', markersize=3 )
# # line, = ax.plot( ocx[:fin], ocy[:fin], 'b.', markersize=3 )
# line, = ax.plot( oocx[:fin], oocy[:fin], 'b.', markersize=3 )
# # ax.set_ylim(0,3000)

# fig.subplots_adjust(left=0.25, bottom=0.25)
# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# slid = Slider( ax=axfreq, label='Fin', valmin=1, valmax=len(xgr), valinit=50, valstep=1)
# def update(val):
#     # line.set_ydata( ygr[:int(slid.val)] )
#     # line.set_xdata( xgr[:int(slid.val)] )
#     line.set_ydata( oocy[:int(slid.val)] )
#     line.set_xdata( oocx[:int(slid.val)] )
#     fig.canvas.draw_idle()
# slid.on_changed(update)

# plt.grid()
# plt.show()

distx = np.sqrt((ocx[1:] - ocx[:-1])**2 + (ocy[1:] - ocy[:-1])**2)
disty = np.sqrt((oocx[1:] - oocx[:-1])**2 + (oocy[1:] - oocy[:-1])**2)

filx,fily = (distx>30) * (distx<100), (disty>30) * (disty<100) 

plt.figure()
plt.plot(ocx[1:][filx] , distx[filx], '.' )
# plt.plot(ocx[1:] , distx, '.' )
plt.plot(oocy[1:][fily] , disty[fily], '.' )
plt.show()



np.mean(np.concatenate((distx[filx],disty[fily]))), np.std(np.concatenate((distx[filx],disty[fily])))
# # np.mean(distx[filx]),np.std(distx[filx]), np.mean(disty[fily]),np.std(disty[fily])

# 5 / 76, 5 / 78 # mm/px, error = 0.001609881121724237 mm/px

#%%
# =============================================================================
# Shadow opaque 0°
# =============================================================================

file = ['/Volumes/Ice blocks/so_s0_t0/_DSC','.JPG']

ims = []
for i in tqdm([1818,1819,1820]):
    ims.append(imageio.imread(file[0]+str(i)+file[1]) [1100:3400,660:5800,0])

back = np.mean(np.array(ims),axis=0)

ims = []
for i in tqdm(range(1708,1809)):
    ims.append(imageio.imread(file[0]+str(i)+file[1]) [1100:3400,660:5800,0])

ims = np.array(ims)
#%%
d = 5 / 77.8438158305425 # mm/px, error = 0.0008711155203865149 mm/px

t1 = time()
imts = []
tramos = []
for i in tqdm(range(len(ims))):
    imt = sobel(ims[i] - back)
    lab = label(imt<1.0)
    num = np.median(lab[1100:1500,1400:2400])
    imb = remove_small_holes( binary_closing(lab==num, disk(15)), area_threshold=1e5)
    imts.append(imb)
    
    edd = imb * 1. - binary_erosion(imb) * 1.
    yed,xed = np.where(edd)

    edge = []
    for j in range(len(xed)):
        edge.append([xed[j],yed[j]])

    clf = NearestNeighbors(n_neighbors=2).fit(edge)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_array(G)
    order = list(nx.dfs_preorder_nodes(T, 0))
    tra_ord = np.array(edge)[order]
    tramos.append(tra_ord[50:-50,:])

t2 = time()
print(t2-t1)

#%%
i = 50

plt.figure()
plt.imshow(imts[i])
plt.show()

#%%
for i in range(60,80):

    yed, xed = 5140 - tramos[i][:,0], tramos[i][:,1]
    ay,ax = np.argmin(yed), np.argmax(yed) # np.argmax(xed)
    
    part = np.sort( [ay,ax] )
    ffax, ffay = xed[part[0]:part[1]], yed[part[0]:part[1]]

    
    ini = 0
    fin = 1
    plt.figure()
    # plt.imshow(imts[i])
    plt.imshow( (ims[i].T)[::-1] )
    # plt.plot( tramos[i][:,0], tramos[i][:,1],'r-' )
    plt.plot( xed[ini:-fin],yed[ini:-fin], 'r-' )
    plt.plot( ffax,ffay, 'g-' )
    plt.show()


# for i in tqdm(range(0,len(tramos),1)):  #46,47

#     yed, xed = 5500 - tramos[i][:,0], tramos[i][:,1]
#     ay,ax = np.argmax(yed), np.argmax(xed)
#     part = np.sort( [ay,ax] )
    
#     if i == 46:
#         ffax = np.concatenate( (xed[4930:5269:1], xed[9705:6250:-1]) )
#         ffay = np.concatenate( (yed[4930:5269:1], yed[9705:6250:-1]) )
        
#     elif i == 47:
#         ffay, ffax = yed[5510:9280], xed[5510:9280]
        
#     else:
#         ffax, ffay = xed[part[0]:part[1]], yed[part[0]:part[1]]
        
        

#%%
# =============================================================================
# Calibration
# =============================================================================
file = '/Volumes/Ice blocks/so_s0_t0/_DSC1813.JPG'
# cal = imageio.imread(file)[500:4250,800:6500,1] #[1100:3400,660:5800,0]
cal = imageio.imread(file)[1100:3400,660:5800,1]

#%%
t1 = time()
g3 = gaussian(cal,3)
g30 = gaussian(cal,30)
no = g3 - g30
norm = np.zeros_like(no)
norm[20:,140:] = (no[20:,140:] - np.min(no[20:,140:])) / (np.max(no[20:,140:]) - np.min(no[20:,140:])) 
imt = remove_small_objects(((norm > 0.4)), min_size=700)
calb = imt

leb = label( calb[:,:] )
ppro = regionprops(leb)

centsy = [(ppro[i].centroid)[0] for i in range(len(ppro)) ]
centsx = [(ppro[i].centroid)[1] for i in range(len(ppro)) ]

xgr = np.array( [ 78.2*i + 0.03*j + 169 for j in range(29) for i in range(64)  ] ) 
ygr = np.array( [ -0.01*i + 77.8*j + 82 for j in range(29) for i in range(64)  ] )

ocx, ocy = [], []
for i in range(len(xgr)):
        if i == 1052:
            ocx.append(np.nan)
            ocy.append(np.nan)
        else: 
            indx = np.argmin( (centsx - xgr[i])**2 + (centsy - ygr[i])**2  )
            ocx.append(centsx[indx])
            ocy.append(centsy[indx])
ocx, ocy = np.array(ocx), np.array(ocy)

oocy = np.array( [ ocy[(i*64)%(29*64) + int(i/29)] for i in range(len(ocy)) ])
oocx = np.array( [ ocx[(i*64)%(29*64) + int(i/29)] for i in range(len(ocy)) ])


t2 = time()
print(t2-t1)
#%%

# plt.figure()
# plt.imshow(calb)
# # plt.plot(centsx, centsy, 'r.' )
# plt.show()

# xgr = np.array( [ 78.2*i + 0.03*j + 169 for j in range(29) for i in range(64)  ] ) 
# ygr = np.array( [ -0.01*i + 77.8*j + 82 for j in range(29) for i in range(64)  ] )

# fin = 1053
# plt.figure()
# plt.plot(centsx, centsy, 'r.' )
# plt.plot(xgr[:fin], ygr[:fin], 'b.' )
# # plt.plot(oocx[:fin], oocy[:fin], 'g.' )
# plt.show()


# fin = 1369
# fig, ax = plt.subplots()

# line, = ax.plot( centsx, centsy, 'r.', markersize=3 )
# # line, = ax.plot( xgr[:fin], ygr[:fin], 'b.', markersize=3 )
# # line, = ax.plot( ocx[:fin], ocy[:fin], 'b.', markersize=3 )
# line, = ax.plot( oocx[:fin], oocy[:fin], 'b.', markersize=3 )
# # ax.set_ylim(0,3000)

# fig.subplots_adjust(left=0.25, bottom=0.25)
# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# slid = Slider( ax=axfreq, label='Fin', valmin=1, valmax=len(xgr), valinit=50, valstep=1)
# def update(val):
#     # line.set_ydata( ygr[:int(slid.val)] )
#     # line.set_xdata( xgr[:int(slid.val)] )
#     line.set_ydata( oocy[:int(slid.val)] )
#     line.set_xdata( oocx[:int(slid.val)] )
#     fig.canvas.draw_idle()
# slid.on_changed(update)

# plt.grid()
# plt.show()


distx = np.sqrt((ocx[1:] - ocx[:-1])**2 + (ocy[1:] - ocy[:-1])**2)
disty = np.sqrt((oocx[1:] - oocx[:-1])**2 + (oocy[1:] - oocy[:-1])**2)

filx,fily = (distx>30) * (distx<100), (disty>30) * (disty<100) 

plt.figure()
plt.plot(ocx[1:][filx] , distx[filx], '.' )
# plt.plot(ocx[1:] , distx, '.' )
plt.plot(oocy[1:][fily] , disty[fily], '.' )
plt.show()

np.mean(np.concatenate((distx[filx],disty[fily]))), np.std(np.concatenate((distx[filx],disty[fily])))
# np.mean(distx[filx]),np.std(distx[filx]), np.mean(disty[fily]),np.std(disty[fily])


#%%









#%%




