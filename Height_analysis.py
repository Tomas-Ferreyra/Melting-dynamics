#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:16:57 2023

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
import scipy.ndimage as snd # from scipy.ndimage import rotate from scipy.ndimage import convolve
from scipy.stats import linregress, skew
from scipy.interpolate import make_interp_spline
from scipy.ndimage import maximum_filter

# import rawpy
import imageio
from tqdm import tqdm
from time import time
import h5py

from skimage.filters import gaussian #, gabor
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes, binary_erosion, thin, skeletonize #, remove_small_objects
from skimage.segmentation import felzenszwalb, mark_boundaries, watershed
from skimage.restoration import unwrap_phase as unwrap
from skimage.feature import peak_local_max
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

def polyfit(n,halg,mmmm,xr,yr):
    haltura = (halg*mmmm)[n]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[n][~np.isnan(haltura)], yr[n][~np.isnan(haltura)]
    A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol2(coeff,xr,yr):
    return coeff[0] + coeff[1] * xr[0] + coeff[2] * yr[0] + coeff[3] * xr[0]**2 + coeff[4] * xr[0]*yr[0] + coeff[5] * yr[0]**2

def track(yys, promines, delim=4 ):
    parts = []
    tiemps = []
    promins = []
    for i in range(len(yys)):
        for j in range(len(yys[i])):
            pac = yys[i][j]
            for k in range(len(parts)):
                ptr = parts[k][-1]
                ttr = tiemps[k][-1]
                if np.abs(ptr - pac) < delim and (i-ttr)==1:
                    parts[k].append(pac)
                    tiemps[k].append(i)
                    promins[k].append( promines[i][j] )
                    break
            else: 
                parts.append([pac])
                tiemps.append([i])
                promins.append([promines[i][j]])
    return parts, promins     

def vels( t,y, halg, col=500, promin=1.4, delim=3.7, mlen=16 ):
    pekas = []
    promines = []
    for n in range(0,55):
        peka = find_peaks(halg[n,:,col], prominence=promin)[0]
        promine = peak_prominences(halg[n,:,col], peka, wlen=50)[0]
        pekas.append(y[peka])
        promines.append(promine)
    
    trayect, promins = track(pekas,promines,delim=delim)
    
    yini, slop, eslop, rval = [], [], [], []
    for i in range(len(trayect)):
        ltra = len(trayect[i]) 
        if ltra > mlen:
            yini.append(trayect[i][0])
            lnr = linregress(t[:ltra]/60, trayect[i])
            slop.append(lnr[0])
            eslop.append(lnr[4])
            rval.append(np.abs(lnr[2]))

    return yini, slop, promins, eslop, rval

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
    

def linfit(n,halg,mmmm,xr,yr):
    haltura = (halg*mmmm)[n]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[n][~np.isnan(haltura)], yr[n][~np.isnan(haltura)]
    A = np.array([xfit*0+1,xfit,yfit]).T
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol1(coeff,xr,yr):
    plane = coeff[0] + coeff[1] * xr[0] + coeff[2] * yr[0] 
    return plane

def resha(h, n=1024 ):
    nt = int(len(h) / n**2)
    return h.reshape((nt,n,n))


#%%


#%%




#%%


#%%
#%%
# =============================================================================
# 3D graph
# =============================================================================
n = 40
mkf = mkfin.copy() * 1.
mkf[ mkfin==False ] = np.nan


hice = halg[n]

az,al = 0 , 90
ver,fra = 0.1 , 100.

blue = np.array([1., 1., 1.])
rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

ls = LightSource()
illuminated_surface = ls.shade_rgb(rgb, hice)

ax.plot_surface(xr[n], hice * mkf[n], yr[n], ccount=300, rcount=300,
                antialiased=True,
                facecolors=illuminated_surface)

ax.set_box_aspect([2,1.6,4])
ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')

# ax.invert_zaxis()
ax.invert_xaxis()

ax.set_zlim(-160,180)
ax.set_xlim(60,-120)
ax.set_ylim(-35,25)

# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
ax.view_init(17,120)
# plt.savefig('./Documents/blocksurface.png',dpi=400, bbox_inches='tight')
plt.show()



#%%

#%%

#%%

#%%

#%%


#%%
# =============================================================================
# Pruebo PIV con las imagenes de 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from matplotlib.colors import LinearSegmentedColormap
from imageio import imread
# %matplotlib notebook
np.seterr(all='raise')

def interrogation_window(im,size):
    '''
    divides image into the different interrogation windows
    '''    
    ny,nx = np.shape(im)
    vspli = np.split(im,ny/size,axis=0)
    conc = np.concatenate(vspli,axis=-1)
    divi = np.hsplit(conc,nx*ny/size**2)   
    return np.array(divi)

def crosscorr(imad,imbd):
    '''
    calculates cross correlation between images
    '''    
    ny,nx = np.shape(imad[0])
    corr = fftconvolve(imbd,imad[:,::-1,::-1],axes=[1,2])
    nc,cnx,cny = np.shape(corr)
    d = corr.reshape([nc,cnx*cny]).argmax(axis=1)
    iy,ix = np.unravel_index(d, corr[0].shape)
    iyv, ixv = iy-(ny-1), ix-(ny-1)
    
    corr = np.pad(corr,1,mode='mean')[1:-1]
    
    ixs,iys = np.zeros_like(ix)*1., np.zeros_like(iy)*1.
    for i in range(len(iy)):
        ixi,iyi = ix[i]+1, iy[i]+1
        try:
            ixi,iyi = ix[i]+1, iy[i]+1 #+1 because of the padding
            ixs_v = 0.5 * (np.log(corr[i,iyi,ixi-1]) - np.log(corr[i,iyi,ixi+1])) / \
                (np.log(corr[i,iyi,ixi-1]) - 2.*np.log(corr[i,iyi,ixi]) + np.log(corr[i,iyi,ixi+1]))
            iys_v = 0.5 * (np.log(corr[i,iyi-1,ixi]) - np.log(corr[i,iyi+1,ixi])) / \
                (np.log(corr[i,iyi-1,ixi]) - 2.*np.log(corr[i,iyi,ixi]) + np.log(corr[i,iyi+1,ixi]))
            ixs[i], iys[i] = ixs_v,iys_v
        except FloatingPointError: 
            ixv[i], iyv[i] = 0,0
# We do this becuase some 0's dividing appear, so instead of getting a 'nan' we set those vectors to 0

    ixsv, iysv = ixv + ixs, iyv + iys
    return ixsv, iysv, corr, ixv, iyv

def crossco_peak(corr,imad):
    ny,nx = np.shape(imad[0])
    
    nc,cnx,cny = np.shape(corr)
    d = corr.reshape([nc,cnx*cny]).argmax(axis=1)
    iy,ix = np.unravel_index(d, corr[0].shape)
    iyv, ixv = iy-(ny-1), ix-(ny-1)
    
    corr = np.pad(corr,1,mode='mean')[1:-1]
    
    ixs,iys = np.zeros_like(ix)*1., np.zeros_like(iy)*1.
    for i in range(len(iy)):
        ixi,iyi = ix[i]+1, iy[i]+1
        try:
            ixi,iyi = ix[i]+1, iy[i]+1 #+1 because of the padding
            ixs_v = 0.5 * (np.log(corr[i,iyi,ixi-1]) - np.log(corr[i,iyi,ixi+1])) / \
                (np.log(corr[i,iyi,ixi-1]) - 2.*np.log(corr[i,iyi,ixi]) + np.log(corr[i,iyi,ixi+1]))
            iys_v = 0.5 * (np.log(corr[i,iyi-1,ixi]) - np.log(corr[i,iyi+1,ixi])) / \
                (np.log(corr[i,iyi-1,ixi]) - 2.*np.log(corr[i,iyi,ixi]) + np.log(corr[i,iyi+1,ixi]))
            ixs[i], iys[i] = ixs_v,iys_v
        except FloatingPointError: 
            ixv[i], iyv[i] = 0,0
# We do this becuase some 0's dividing appear, so instead of getting a 'nan' we set those vectors to 0

    ixsv, iysv = ixv + ixs, iyv + iys
    return ixsv, iysv, corr, ixv, iyv

def crosc1d(corr,imad,ax):
    ny,nx = np.shape(imad)
    
    cny,cnx = np.shape(corr)
    d = corr.argmax(axis=ax)
    iiv = d - (ny-1)
    
    iis = np.zeros_like(d)*1.
    for i in range(len(iiv)):
        try:
            if ax == 0: 
                iis_v = 0.5 * (np.log(corr[d[i]-1,i]) - np.log(corr[d[i]+1,i])) / \
                   (np.log(corr[d[i]-1,i]) - 2.*np.log(corr[d[i],i]) + np.log(corr[d[i]+1,i]))
                iis[i] = iis_v
            if ax == 1: 
                iis_v = 0.5 * (np.log(corr[i,d[i]-1]) - np.log(corr[i,d[i]+1])) / \
                   (np.log(corr[i,d[i]-1]) - 2.*np.log(corr[i,d[i]]) + np.log(corr[i,d[i]+1]))
                iis[i] = iis_v
        except FloatingPointError: 
            iiv[i] = 0
        except IndexError:
            iiv[i] = 0
        
    iisv = iiv + iis
    return iisv, iiv
#%%

size = 16
imad = interrogation_window(difes[40],size)
imbd = interrogation_window(difes[41],size)

ix,iy, _,_,_ = crosscorr(imad,imbd)

oo = [size*i+size/2 for i in range(int(len(halg[0])/size))]
ox,oy = np.meshgrid(oo,oo)
ox,oy = ox.flatten(), oy.flatten()

color = np.sqrt(ix**2 + iy**2)

plt.figure(figsize=(7,7))
plt.imshow(difes[30])
plt.quiver(ox,oy,ix,iy, angles='xy', scale_units='xy', scale=1e-2) #, cmap='viridis')
# plt.colorbar()
plt.gca().invert_yaxis()
plt.xlabel('x (px)')
plt.xlabel('y (py)')
plt.gca().invert_yaxis()
plt.show()

# plt.figure()
# plt.imshow(difes[30])
# plt.show()
# plt.figure()
# plt.imshow(difes[32])
# plt.show()
#%%
size = 64

corrs = []
corm = np.zeros( (int(1024**2/size**2),size*2-1,size*2-1) )
for i in range(5):
    imad = interrogation_window(difes[40+i],size)
    imbd = interrogation_window(difes[41+i],size)
    
    corr = fftconvolve(imbd,imad[:,::-1,::-1],axes=[1,2])
    corrs.append(corr)
    corm += corr

# for i in range(1):
#     ix,iy, _,_,_ = crossco_peak(corrs[i],imad)

#     oo = [size*i+size/2 for i in range(int(len(halg[0])/size))]
#     ox,oy = np.meshgrid(oo,oo)
#     ox,oy = ox.flatten(), oy.flatten()

#     color = np.sqrt(ix**2 + iy**2)

#     plt.figure(figsize=(7,7))
#     plt.imshow(difes[30])
#     plt.quiver(ox,oy,ix,iy, angles='xy', scale_units='xy', scale=1e-2) #, cmap='viridis')
#     # plt.colorbar()
#     plt.gca().invert_yaxis()
#     plt.xlabel('x (px)')
#     plt.xlabel('y (py)')
#     plt.gca().invert_yaxis()
#     plt.show()
    


ix,iy, _,_,_ = crossco_peak(corm,imad)

oo = [size*i+size/2 for i in range(int(len(halg[0])/size))]
ox,oy = np.meshgrid(oo,oo)
ox,oy = ox.flatten(), oy.flatten()

color = np.sqrt(ix**2 + iy**2)

plt.figure(figsize=(7,7))
plt.imshow(difes[45])
plt.quiver(ox,oy,ix,iy, angles='xy', scale_units='xy', scale=1.e-2) #, cmap='viridis')
# plt.colorbar()
plt.gca().invert_yaxis()
plt.xlabel('x (px)')
plt.xlabel('y (py)')
plt.xlim(260,540)
plt.ylim(130,770)
plt.gca().invert_yaxis()
plt.savefig('./Documents/piv_8.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
it = 0.5 * 1
plt.figure()
plt.hist( iy*d/it, bins=50, edgecolor='k' )
plt.grid()
plt.xlabel('Velocity (mm/min)')
plt.savefig('./Documents/vepihi_8.png',dpi=400, bbox_inches='tight')
plt.show()
plt.figure()
plt.hist( ix*d/it, bins=50, edgecolor='k' )
plt.grid()
plt.show()

print( np.nanmean(iy*d/it), np.nanstd(iy*d/it) )
print( np.nanmean(ix*d/it), np.nanstd(ix*d/it) )

#%%

plt.figure()
for n in range(42,44,1):
    plt.plot(y,difes[n,:,400])
plt.grid()
plt.show()


l1,l2 = difes[43,170:720,400], difes[43,170:720,400]

clin = fftconvolve(l1-np.mean(l1),l2[::-1]-np.mean(l2))
plt.figure()
plt.plot(clin,'.-')
plt.grid()
plt.show()


#%%

size = 16
imad = interrogation_window(difes[40],size)
imbd = interrogation_window(difes[42],size)

xvs,yvs = [],[]
for nu in range(len(imad)):
    im1, im2 = imad[nu],imbd[nu]
    
    my1,my2 = np.array([np.mean(im1,axis=0)]), np.array([np.mean(im2,axis=0)])
    cony = fftconvolve(im2-my2, (im1-my1)[::-1,::1], axes=0)
    yv, yyv = crosc1d(cony, im1, 0)
    
    mx1,mx2 = np.array([np.mean(im1,axis=1)]).T, np.array([np.mean(im2,axis=1)]).T
    conx = fftconvolve(im2-mx2, (im1-mx1)[::1,::-1], axes=1)
    xv, xxv = crosc1d(conx, im1, 1)
    
    xvs.append( np.median(xv) )
    yvs.append( np.median(yv) )
 
oo = [size*i+size/2 for i in range(int(len(halg[0])/size))]
ox,oy = np.meshgrid(oo,oo)
ox,oy = ox.flatten(), oy.flatten()

plt.figure()
plt.imshow(difes[40])
plt.quiver(ox,oy,xvs,yvs, angles='xy', scale_units='xy', scale=1e-2) #, cmap='viridis')
plt.show()


    
    
    
# nu = 85


# plt.figure()
# plt.imshow(im1)
# plt.show()
# plt.figure()
# plt.imshow(im2)
# plt.show()

# cwe = fftconvolve(im2 - np.mean(im2), im1[::-1,::-1] - np.mean(im1))
# plt.figure()
# # plt.imshow(cwe)
# plt.plot(cwe[:,63],'.-')
# plt.plot(cwe[63,:],'.-')
# plt.grid()
# plt.show()


# cony = fftconvolve(im2, im1[::-1,::1], axes=0)
# yv, yyv = crosc1d(cony, im1, 0)
# plt.figure()
# plt.plot(yv*d,'.-')
# plt.grid()
# plt.show()


# conx = fftconvolve(im2, im1[::1,::-1], axes=1)
# xv, xxv = crosc1d(conx, im1, 1)
# plt.figure()
# plt.plot(xv*d,'.-')
# plt.grid()
# plt.show()


# plt.figure()
# l1,l2 = im1[444,:], im2[444,:]
# plt.plot(l1,'-')
# plt.plot(l2,'--')
# plt.grid()
# plt.show()

# plt.close('all')

# fiy,fix = yv!=0, xv!=0

# print( np.median( yv[fiy] ), np.std( yv[fiy] )  )
# print( np.median( xv[fix] ), np.std( xv[fix] )  )

#%%
# =============================================================================
# Watershed
# =============================================================================
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

scallops = []
for n in tqdm(range(nt)):
    peks = peak_local_max( difes[n], threshold_abs=0, footprint = disk(30) )#, labels=mmmm[n]>0  )
    cmask = np.zeros_like( difes[n], dtype=bool )
    cmask[ tuple(peks.T) ] = True
    cord = label(cmask)
    
    scals = watershed(-difes[n], cord ,mask=mmmm[n]>0, compactness= 1e-2) #compact = 0.01 no esta tan mal
    scallops.append(scals)
    
scal_min = []
for n in tqdm(range(nt)):
    peks = peak_local_max( -difes[n], threshold_abs=0, footprint = disk(30) )#, labels=mmmm[n]>0  )
    cmask = np.zeros_like( difes[n], dtype=bool )
    cmask[ tuple(peks.T) ] = True
    cord = label(cmask)
    
    scals = watershed(difes[n], cord ,mask=mmmm[n]>0, compactness= 1e-2) #compact = 0.01 no esta tan mal
    scal_min.append(scals)
    
#%%
for n in range(40,60):
    aldif = ( difes[n] - np.nanmin(difes[n])) / np.nanmax( difes[n] - np.nanmin(difes[n]))
    aldif[ np.isnan(aldif) ] = 0
    bou = mark_boundaries(aldif , scal_min[n], color=(1,0,1))
    bou[ bou==0 ] = np.nan
    plt.figure(figsize=(15,15))
    plt.imshow( difes[n] )
    plt.imshow( bou[:,:,0] )
    # plt.imshow( mark_boundaries(aldif , scallops[n], color=(1,0,1)) )
    # plt.imshow( mark_boundaries(aldif , scal_min[n], color=(1,0,1)) )
    # plt.imshow(scallops[n])
    plt.xlim(260,550)
    plt.ylim(900,240)
    # plt.savefig('./Documents/watershed_15.png',dpi=400, bbox_inches='tight')
    plt.show()

# for n in range(50,40,-1):
#     aldif = difes[n]/np.nanmax(difes[n])
#     aldif[ np.isnan(aldif) ] = 0
    
#     plt.figure()
#     # plt.imshow( mark_boundaries(aldif , scallops[n], color=(1,0,1)) )
#     plt.imshow( mark_boundaries(aldif , scal_min[n], color=(1,0,1)) )
#     # plt.imshow(scallops[n])
#     plt.show()

# plt.figure()
# plt.imshow(scal_min[40] )
# plt.show()

# plt.figure()
# plt.plot(t, np.max(scallops,axis=(1,2)),'b.-')
# plt.plot(t, np.max(scal_min,axis=(1,2)),'g.-')
# plt.grid()
# plt.show()


#%%
cwss = []
for n in range(40,60): #range(nt):
    
    aldif = ( difes[n] - np.nanmin(difes[n])) / np.nanmax( difes[n] - np.nanmin(difes[n]))
    aldif[ np.isnan(aldif) ] = 0
    
    props = regionprops(scal_min[n], intensity_image=-aldif+1)
    # props = regionp

    cws = []
    for j in range(len(props)):
        cw = props[j].centroid_weighted
        cws.append(cw)
    cwss.append(np.array(cws))
cwss = np.array(cwss)

plt.figure()
# plt.imshow(difes[n])
# plt.imshow( mark_boundaries( aldif , scal_min[n], color=(1,0,1)), cmap='viridis' )
for n in range(20):
    plt.plot(cwss[n][:,1],cwss[n][:,0],'r.')
plt.axis('equal')  
# plt.gca().invert_yaxis()
plt.xlim(260,550)
plt.ylim(900,240)
plt.show()
#%%

hepe = []
for n in range(nt): #range(nt):
    
    aldif = difes[n]
    # aldif[ np.isnan(aldif) ] = 0

    props = regionprops(scallops[n], intensity_image=aldif)    
    # props = regionprops(scal_min[n], intensity_image=-aldif+1)
    # props = regionp

    cws = []
    for j in range(len(props)):
        cw = props[j].intensity_max
        cws.append(cw)
    hepe.append(np.array(cws))
hepe = np.array(hepe)

hepms, hepss = [],[]
for n in range(nt):
    hepm, heps = np.mean(hepe[n]), np.std(hepe[n])
    hepms.append(hepm)
    hepss.append(heps)


plt.figure()
plt.errorbar(t/60, hepms, yerr=hepss, fmt='ks', capsize=2)
# plt.plot(hepe[n],'.')
plt.xlabel('Time (min)')
plt.ylabel('Peak heights')
plt.grid()
# plt.savefig('./Documents/phei_15.png',dpi=400, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(t/60, hepss,'.-')
# plt.plot(hepe[n],'.')
plt.xlabel('Time (min)')
plt.ylabel('Spread peaks height')
plt.grid()
# plt.savefig('./Documents/pheisp_15.png',dpi=400, bbox_inches='tight')
plt.show()


#%%






#%%







#%%







#%%
#=============================================================================
# All at once
# =============================================================================
# halt = np.load('./Documents/Height profiles/ice_block_0_8(1).npy')
# d = 0.39250081308793844 #0_8(1)

sals = ['5','15','23','8(1)','0','2','4','6','8(3)','12','14','18','15(2)','22','27','10']
salis = [8,15.2,23.38,8,0,2.31,4.06,6.40,8.26,12.87,14.8,18.13,15.2,21.9,27.39,10.26]
ds = [ 0.37566904425667014, 0.39250081308793844, 0.3957774907299554, 0.39250081308793844, 0.3779326761033847, 0.37940313057152375,
       0.37917605048206754, 0.3780874065675317, 0.3773925197773744, 0.37804894658912125, 0.37799748108152204, 0.37903057740207474,
       0.3788519951979505, 0.3795787182927243, 0.3782083129763212, 0.3783574601007215 ]
temps = [ 20.69, 20.08, 21.40, 19.95, 21.00, 20.65, 19.6, 20.14, 19.13, 18.57, 20.04, 20.37, 19.46, 18.60, 17.76, 19.22 ]
#%%
halts, halgs = [], []
nts,nxs,nys = [],[],[]
trs,xrs,yrs = [],[],[]
ts,xs,ys = [],[],[]
gts,gys,gxs = [],[],[]

for i,ss in enumerate(sals):
    halt = np.load('./Documents/Height profiles/ice_block_0_'+ss+'.npy')
    
    halts.append(halt)
    
    d = ds[i]
    halg = nangauss(halt, 2)
    halgs.append(halg) 
    
    nt,ny,nx = np.shape(halt)
    nts.append(nt)
    nxs.append(nx)
    nys.append(ny)
    
    x,y = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d
    xs.append(x)
    ys.append(y)
    
    t = np.arange(nt) * 30
    ts.append(t)
    tr,yr,xr = np.meshgrid(t,y,x, indexing='ij')
    
    trs.append(tr)
    xrs.append(xr)
    yrs.append(yr)

    gt,gy,gx = np.gradient(halg, t,y,x)
    gts.append(gt)
    gxs.append(gx)
    gys.append(gy)
#%%
# =============================================================================
# Hdf5
# =============================================================================
with h5py.File('./Documents/Height profiles/npys/heigtarr.hdf5', 'w') as f:
        # h = f.create_group('h')
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    hh = f.create_dataset('h',(len(salis),),dtype=dt)
    xh = f.create_dataset('x',(len(salis),),dtype=dt)
    yh = f.create_dataset('y',(len(salis),),dtype=dt)
    th = f.create_dataset('t',(len(salis),),dtype=dt)
    mfh = f.create_dataset('mfin',(len(salis),),dtype=dt)
    mmh = f.create_dataset('mmm',(len(salis),),dtype=dt)
    
    for i,ss in tqdm(enumerate(sals)):
        halt = np.load('./Documents/Height profiles/ice_block_0_'+ss+'.npy')
        
        d = ds[i]
        halg = nangauss(halt, 2)
        
        hh[i] = halg.flatten()
        
        nt,ny,nx = np.shape(halt)
        x,y = (np.arange(0.5,nx+0.5) - nx/2) * d, (-np.arange(0.5,ny+0.5) + ny/2) * d        
        t = np.arange(nt) * 30
        tr,yr,xr = np.meshgrid(t,y,x, indexing='ij')
    
        xh[i] = xr.flatten()
        yh[i] = yr.flatten()
        th[i] = tr.flatten()
        
        mks = ~np.isnan(halg)    
        mkfin = binary_erosion( np.pad(mks, ((0,0),(1,1),(1,1))), np.ones((1,15,15)) )[:,1:-1,1:-1]
        mkse = binary_erosion( np.pad(mks, ((0,0),(1,1),(1,1))), np.ones((1,50,50)) )[:,1:-1,1:-1] #+ mks*1. 1,50,50, solia usar 1,100,100
        
        mmmm = mkse.copy() * 1. 
        mmmm[ mkse==False ] = np.nan
        
        mfh[i] = mkfin.flatten()
        mmh[i] = mmmm.flatten()
        
#%%
with h5py.File('./Documents/Height profiles/npys/heigtarr.hdf5', 'r') as f:
    for i in tqdm(range(len(salis))):
        hal = resha(f['h'][i])
        tr = resha(f['t'][i])
        yr = resha(f['y'][i])
        xr = resha(f['x'][i])
        mmmm = resha(f['mmm'][i])
        mfi = resha(f['mfin'][i])
    
    print( np.shape(hal), np.shape(xr), np.shape(yr), np.shape(tr) )
    
#%%
mkss, mkfins, mmms = [],[],[]

for i,ss in tqdm(enumerate(sals)):
    mks = ~np.isnan(halgs[i])
    
    # mkse = binary_erosion( np.pad(mks, ((0,0),(1,1),(1,1))), np.ones((1,100,100)) )[:,1:-1,1:-1] #+ mks*1. # takes a lot of time
    mkse = binary_erosion( np.pad(mks, ((0,0),(1,1),(1,1))), np.ones((1,50,50)) )[:,1:-1,1:-1] #+ mks*1.
    
    mkfin = binary_erosion( np.pad(mks, ((0,0),(1,1),(1,1))), np.ones((1,15,15)) )[:,1:-1,1:-1]

    mmmm = mkse.copy() * 1. 
    mmmm[ mkse==False ] = np.nan
    
    mkss.append(mkse)
    mkfins.append(mkfin)
    mmms.append( mmmm )
#%%
difs = []
for i,ss in enumerate(sals):
    difes = []
    for n in tqdm(range(nts[i])):
        coeff, r, rank, s = polyfit(n,halgs[i],mmms[i],xrs[i],yrs[i])
        cuapla = pol2(coeff,xrs[i],yrs[i]) 
        difes.append( (halgs[i][n]-cuapla)*mmms[i][n] )
    difes = np.array(difes)
    difs.append(difes)
#%%
# =============================================================================
# Melt rates
# =============================================================================
# salis = [8,15,23]

plt.figure()
# for i,ss in enumerate(salis):
for i in [0,3,8,1,10,12]:
    ss = salis[i]    
    megt = np.nanmean(gts[i], axis=(1,2)) 
    sdgt = np.nanstd(gts[i], axis=(1,2))
    
    # plt.errorbar( ts[i]/60, -megt*60, yerr=sdgt*60, fmt='.', capsize=2, label=str(ss)+' g\kg' )
    plt.plot( ts[i]/60, -megt*60, '-.', label=str(ss)+' g\kg' )
    
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Melt rate (mm/min)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()
# plt.savefig('./Documents/meanmelt.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
plt.figure()
# for i,ss in enumerate(salis):
for i in [0,3,8,1,10,12]:
    ss = salis[i]
    
    mee = np.nanmean(halts[i], axis=(1,2))
    
    plt.plot( ts[i]/60, mee - mee[0], '.-', label=str(ss)+' g\kg' )

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Mean height (mm)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()
# plt.savefig('./Documents/meanhei.png',dpi=400, bbox_inches='tight')
plt.show()

# plt.figure()
# for i,ss in enumerate(salis):
#     sdef = np.nanstd(difs[i], axis=(1,2))
    
#     plt.plot( ts[i]/60, sdef, '.-', label=str(ss)+' g\kg' )
    
# plt.legend()
# plt.grid()
# plt.show()
#%%

slps, sdsl = [],[]

plt.figure()
for i,ss in enumerate(salis):
# for i in [0,3,8]:
#     ss = salis[i]
    
    mee = np.nanmean(halts[i], axis=(1,2))
    # mee = halts[i][:,500,500]
    
    lr = linregress(ts[i][20:]/60, y=mee[20:] - mee[0])
    sl,inte,rv,pv,stdsl = lr[0], lr[1], lr[2], lr[3], lr[4]
    # print(sl,inte,rv,pv,stdsl)
    slps.append( sl )
    sdsl.append( stdsl )
    
    plt.plot( ts[i]/60, mee - mee[0], '.-', label=str(ss)+' g\kg' )
    plt.plot( ts[i]/60, sl*ts[i]/60+inte, '--' )

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Mean height (mm)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()
# plt.savefig('./Documents/meanhei.png',dpi=400, bbox_inches='tight')
plt.show()

plt.figure()
plt.errorbar( salis, -np.array(slps), yerr=sdsl, fmt='k.', capsize=2 )
# plt.errorbar( [8,8,8.2], -np.array(slps), yerr=sdsl, fmt='k.', capsize=2 )
plt.grid()
plt.show()
#%%
# =============================================================================
# Melt rates with slope in area
# =============================================================================

slps, sdsl = [],[]

plt.figure()

for i,ss in enumerate(salis):
    ss = salis[i]
    are = np.sum(mkfins[i], axis=(1,2)) * ds[i]**2
    lr = linregress(ts[i]/60, y=are - are[0])
    sl,inte,rv,pv,stdsl = lr[0], lr[1], lr[2], lr[3], lr[4]
    
    plt.plot(ts[i], are - are[0], '.-', label=ss)
    plt.plot( ts[i], sl*ts[i]/60+inte, '--' )
    
    slps.append( sl )
    sdsl.append( stdsl )

plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.errorbar( salis, np.array(slps) / slps[4], yerr=sdsl / -slps[4], fmt='k.', capsize=2 )
# plt.errorbar( [8,8,8.2], -np.array(slps), yerr=sdsl, fmt='k.', capsize=2 )
plt.grid()
plt.xlabel('S (g/kg)')
plt.ylabel(r'Melt rate ')
plt.show()
#%%
# time half area

tss = []
plt.figure()
for i,ss in enumerate(salis):
    ss = salis[i]
    are = np.sum(mkfins[i], axis=(1,2)) * ds[i]**2
    
    iha = np.argmin( np.abs( are - are[0]/2 ) )
    
    if i in [0,3]:
        plt.plot( ts[i], are, '.-' )
        plt.plot( ts[i][iha], are[0]/2, 'k.' )
    tss.append(ts[i][iha])
    
plt.show()

tss = np.array(tss)
f = 1/tss

plt.figure()
plt.plot( salis, f / f[4], '.' )
plt.plot( [8,8], [f[0]/f[4],f[3]/f[4]], 'kx' )
plt.grid()
plt.xlabel('S (g/kg)',fontsize=12)
plt.ylabel(r'Melt rate ',fontsize=12)
plt.savefig('./Documents/meltrate_time.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Wavelength (with distance between max)
# =============================================================================

lmeas, lmeds, lstds = [],[],[]
# for i,ss in enumerate(salis):
for i in [1,10,12]:
    ss = salis[i]

    lmea,lmed,lsd = [],[],[]
    for n in range(nts[i]):
        lons =[]
        for l in range(nxs[i]):
            line = difs[i][n,:,l]
            pek = find_peaks(line, prominence=0.5)[0]
            long = ys[i][pek][:-1] - ys[i][pek][1:]
            lons += list(long) 
        lmea.append( np.mean(lons) )
        lmed.append( np.median(lons) )
        lsd.append( np.std(lons) )
    lmeas.append(lmea)
    lmeds.append(lmed)
    lstds.append(lsd)


# plt.figure()
# # plt.plot( y, line )
# # plt.plot( y[pek], line[pek], 'k.' )
# # plt.axis('equal')
# plt.hist(lons,bins=50, edgecolor='k')
# plt.grid()
# plt.show()

# print('Mean =',np.mean(lons))
# print('Median =',np.median(lons))
# print('Std =',np.std(lons))

plt.figure()
# for i,ss in enumerate(salis):
for i,j in enumerate([1,10,12]):
    ss = salis[j]
    
    plt.errorbar(ts[j]/60, lmeas[i], yerr=lstds[i], capsize=2, fmt='.', label=str(ss)+' g/kg')

plt.grid()
plt.legend(fontsize=12)
plt.ylabel('Wavelenght (mm)', fontsize=12)
plt.xlabel('Time (min)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./Documents/wales15.png',dpi=400, bbox_inches='tight')
plt.show()
#%%

# plt.figure()

tss,lmm,lss = [],[],[]
# for i,ss in enumerate(salis):
for i,j in enumerate(range(len(salis))):
    
    ss = salis[j]
    
    imi = np.argmin( lmeas[i] )
    
    # print(    ss, ts[j][imi]/60., lmeas[i][imi], lstds[i][imi]    )
    print( '{:<5} {:<4} {:<20} {:<20f}'.format( ss, ts[j][imi]/60., lmeas[i][imi], lstds[i][imi] ) ) 
    # plt.plot( ts[j]/60, lmeas[i], label=ss )
    
    tss.append( ts[j][imi]/60. )
    lmm.append( lmeas[i][imi] )
    lss.append( lstds[i][imi] )
    
plt.figure()
# plt.errorbar( [salis[l] for l in [0,3,8,1,10,12]] , lmm, yerr=lss, capsize=2, fmt='k.')
plt.errorbar( salis , lmm, yerr=lss, capsize=2, fmt='k.')
# plt.legend()
plt.ylim(0,70)
plt.grid()
plt.ylabel('Wavelenght (mm)', fontsize=12)
plt.xlabel('S (g/kg)', fontsize=12)
# plt.savefig('./Documents/scallopswave_sal.png',dpi=400, bbox_inches='tight')
plt.show()

plt.figure()
# plt.errorbar( [salis[l] for l in [0,3,8,1,10,12]] , lmm, yerr=lss, capsize=2, fmt='k.')
plt.plot( salis, tss, 'k.' )
# plt.legend()
# plt.ylim(0,40)
plt.grid()
plt.ylabel('Time scallops (min)', fontsize=12)
plt.xlabel('S (g/kg)', fontsize=12)
plt.show()

#%%
# =============================================================================
#  Correlations
# =============================================================================

coexs,coeys = [],[]
for i,ss in enumerate(salis):
    gymas = ma.masked_invalid( gys[i] * mmms[i] )
    gxmas = ma.masked_invalid( gxs[i] * mmms[i] )
    gtmas = ma.masked_invalid( gts[i] * mmms[i] )
    
    coey, coex = [], []
    for n in tqdm(range(nts[i])):
        coey.append(ma.corrcoef( gymas[n].flatten(), gtmas[n].flatten() )[0,1] )
        coex.append(ma.corrcoef( gxmas[n].flatten(), gtmas[n].flatten() )[0,1] )
    coexs.append(coex)
    coeys.append(coey)
#%%
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #['b','orange','g']

plt.figure()
# for i,ss in enumerate(salis):
for i in [0,3,8]:
    ss = salis[i]
    plt.plot(ts[i]/60 ,coeys[i], '-', color=colors[i%10] , label=str(ss)+' g/kg')  #label='dh/dy')
    # plt.plot(ts[i]/60 ,coexs[i], '--', color=colors[i%10] )  #label='dh/dy')
    # plt.plot(t/60, coex, label='dh/dx')
    
plt.xlabel('time (min)',fontsize=12)
plt.ylabel('Correlation',fontsize=12)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
# plt.savefig('./Documents/corrs8.png',dpi=400, bbox_inches='tight')
plt.show()

# n = 40
# plt.figure()
# plt.scatter( gymas[n], gtmas[n], s=1 )
# plt.xlabel('dh/dy')
# plt.ylabel('melt rate')
# plt.grid()
# plt.xlim([-1.1,1.1])
# plt.show()
# plt.figure()
# plt.scatter( gxmas[n], gtmas[n], s=1 )
# plt.xlabel('dh/dx')
# plt.ylabel('melt rate')
# plt.grid()
# plt.xlim([-0.58,0.58])
# plt.show()
#%%
# =============================================================================
# Roughness
# =============================================================================
# rones = []
# for n in [0,3,8]: #1,10,12]:
#     rougs = []
#     for i in range(nts[n]):
#         roug = np.nanstd(difs[n][i])
#         rougs.append(roug)
#     rones.append(rougs)

# plt.figure()
# for i,j in enumerate([0,3,8]):
#     plt.plot(ts[j]/60, rones[i], label=salis[j])
# plt.grid()
# plt.legend()
# plt.show()

rones = []
for n in [0,3,8]:
    rougs = []
    for i in range(nts[n]):
        hhee = difs[n][i]
        roug = np.sqrt( np.nanmean( hhee**2 ) )
        rougs.append(roug)
    rones.append(rougs)

plt.figure()
for i,j in enumerate([0,3,8]):
    plt.plot(ts[j]/60, rones[i], label=salis[j])
plt.grid()
plt.legend()
plt.xlabel('Time (min)',fontsize=12)
plt.ylabel('Roughness',fontsize=12)
# plt.savefig('./Documents/rough8.png',dpi=400, bbox_inches='tight')
plt.show()


#%%

n = 12

rougs = []
for i in range(0,nts[n],1):
    heig = np.sort( (difs[n][i]).flatten() )
    heig = heig[ ~np.isnan(heig) ]
    pdh, beh = np.histogram( heig, bins=100 )
    cdh = np.cumsum( pdh ) / np.sum(pdh)
    
    xbe = (beh[1:] + beh[:-1] ) / 2
    
    gra = np.gradient(cdh, xbe)
    roug = np.max(gra)
    rougs.append(roug)

    # plt.plot(xbe, cdh, '.-', label=i )

plt.figure()
plt.plot(ts[n]/60, rougs )
plt.grid()
plt.legend()
plt.show()
#%%

# plt.figure()
# plt.imshow( halgs[n][50]  )
# plt.hist( (halgs[n][50]).flatten() , bins=50)
# plt.show()

plt.figure()
plt.imshow( halgs[0][50]  )
plt.show()
plt.figure()
plt.imshow( halgs[3][50]  )
plt.show()
plt.figure()
plt.imshow( halgs[8][50]  )
plt.show()




#%%
# =============================================================================
# Tracking maxima (to see scallops velocity)
# =============================================================================
myss, mxss, ttss = [],[],[]
for i,ss in enumerate(sals):
# for i in [0]:
#     ss = salis[i]

    iceh = halgs[i] * mmms[i]
    mys, mxs, tts = [],[],[]
    for n in tqdm(range(nts[i])):
        # my,mx = np.where( maximum_filter(iceh[n], footprint=disk(30)) == iceh[n] )
        my,mx = np.where( maximum_filter(-iceh[n], footprint=disk(30)) == -iceh[n] )
        mys = mys + list(ys[i][my])
        mxs = mxs + list(xs[i][mx])
        tts = tts + [ts[i][n]/60] * len(my)
    myss.append(mys)
    mxss.append(mxs)
    ttss.append(tts)
#%%
mve, msd = [],[]
# for i,ss in enumerate(salis):
for i in [0]:
    ss = salis[i]
    
    mxs,mys,tts = mxss[i], myss[i], ttss[i]
    
    colos = [ [i,1-i ,0]  for i in np.linspace(0.1,1.0,25) ]
    
    mxs,mys,tts = np.array(mxs), np.array(mys), np.array(tts)
    
    plt.figure()
    plt.imshow( halgs[i][30], extent=(np.min(xs[i]), np.max(xs[i]), np.min(ys[i]), np.max(ys[i]) )  )
    plt.scatter(  mxs[tts==30], mys[tts==30], s=2, c='k')
    plt.show()
    
    
    mint = 12
    fig,ax = plt.subplots()
    ppl = ax.scatter(mxs[tts>mint],mys[tts>mint], c=tts[tts>mint], cmap='viridis')
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.set_xlim([-53,53])
    # ax.set_ylim([170,-69])
    
    fig.colorbar(ppl,ax=ax,label='time (min)')
    
    plt.gca().invert_yaxis()
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.grid()
    plt.show()
    
    
    trax,tray,tiemps = track2d(mxs[tts>mint],mys[tts>mint],tts[tts>mint], delim=4, dtm=1)
    
    trxl, tryl, tiel = [],[],[]
    for i in range(len(trax)):
        if len(trax[i]) > 15: 
            trxl.append(trax[i])
            tryl.append(tray[i])
            tiel.append(tiemps[i])
    del trxl[1], tryl[1], tiel[1]
    
    fig,ax = plt.subplots()
    for i in range(len(trxl)):
        ppil = ax.plot(trxl[i],tryl[i],'.-', label=i)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    # ax.set_xlim([-53,53])
    # ax.set_ylim([170,-69])
    ax.legend( loc='upper center', bbox_to_anchor=(1.25, 0.8), ncol=1 )#, frame=False)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
    
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
    
    
    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].errorbar(np.arange(len(slops)),slops, yerr=eslo, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    ax[1].errorbar(np.arange(len(slpx)),slpx, yerr=eslx, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_ylabel('vel y (mm/min)', fontsize=12)
    ax[1].set_ylabel('vel x (mm/min)', fontsize=12)
    ax[1].set_xlabel('Scallop', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.savefig('./Documents/vels_'+ str(ss) +'.png',dpi=400, bbox_inches='tight')
    plt.show()
    
    # plt.figure()
    # plt.errorbar(slpx, slops, xerr=eslx, yerr=eslo, fmt='.')
    # plt.grid()
    # plt.show()
    
    print(ss, np.mean(slops), np.median(slops), np.std(slops))
    mve.append( np.mean(slops) )
    msd.append( np.std(slops) )
    
    # n = 30
    # plt.figure()
    # plt.imshow( iceh[n], extent=(x[0],x[-1],y[-1],y[0]) )
    # plt.plot(mxs[tts*2==n], mys[tts*2==n], 'k.')
    # # plt.scatter(mxs[tts>15],mys[tts>15], c=tts[tts>15], cmap='Reds')
    # plt.xlabel('x (mm)')
    # plt.ylabel('y (mm)')
    # plt.show()
#%%

plt.figure()
plt.errorbar(salis, mve, yerr=msd, fmt='.', elinewidth=1., capsize=1.)
plt.grid()
plt.xlabel('S (g/kg)')
plt.ylabel(r'$v_y$ (g/kg)')
plt.savefig('./Documents/velsvsal_min.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# 3D graph
# =============================================================================
i = 0
n = 50

# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan


hice = halgts[i][n]

az,al = 0 , 90
ver,fra = 0.1 , 100.

blue = np.array([1., 1., 1.])
rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')

ls = LightSource()
illuminated_surface = ls.shade_rgb(rgb, hice)

ax.plot_surface(xrts[i][n], hice , yrts[i][n], ccount=300, rcount=300, # * mkf[n], yrts[i][n], ccount=300, rcount=300,
                antialiased=True,
                facecolors=illuminated_surface, label='t = '+str(0.5*n)+'seg')

ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_zlabel('y (mm)')
# ax.invert_zaxis()
ax.invert_xaxis()

lxd, lxi = np.nanmax(xrts[i]), np.nanmin(xrts[i])
lzd, lzi = np.nanmin(yrts[i]), np.nanmax(yrts[i])
lyi, lyd = np.nanmin(halgts[i]), np.nanmax(halgts[i])

print( lzd,lzi )
print( lxd,lxi )
print( lyd,lyi )

# ax.set_box_aspect([2,1.6,4])
# ax.set_zlim(-180,140)
# ax.set_xlim(110,-70)
# ax.set_ylim(-60,0)

ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=1.3)
ax.set_zlim(lzd-5,lzi+5)
ax.set_xlim(lxd+5,lxi-5)
ax.set_ylim(lyi-5,lyd+5)

plt.locator_params(axis='y', nbins=3)
# ax.text(30,40,180, 't = '+str(0.5*n)+'min', fontsize=12)
# ax.set_title( 't = '+str(0.5*n)+'s', fontsize=12)

# ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
ax.view_init(17,120)
plt.axis('off')
plt.savefig('./Documents/blocksurface2.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Make gif
# =============================================================================
i = 12
lista_im = []
# mkf = mkfins[i].copy() * 1.
# mkf[ mkfins[i]==False ] = np.nan
for n in tqdm(range(len(halgts[i]))):
    hice = halgts[i][n]
    
    az,al = 0 , 90
    ver,fra = 0.1 , 100.
    
    blue = np.array([1., 1., 1.])
    rgb = np.tile(blue, (hice.shape[0], hice.shape[1], 1))

    plt.close('all')
    plt.ioff()

    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection='3d')
    
    ls = LightSource()
    illuminated_surface = ls.shade_rgb(rgb, hice)
    
    ax.plot_surface(xrts[i][n], hice, yrts[i][n], ccount=300, rcount=300,  # * mkf[n] , yrs[i][n], ccount=300, rcount=300,
                    antialiased=True,
                    facecolors=illuminated_surface)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    
    # ax.invert_zaxis()
    ax.invert_xaxis()
    
    # ax.set_box_aspect([2,1.6,4])    
    # ax.set_zlim(-180,140)
    # ax.set_xlim(110,-70)
    # ax.set_ylim(-60,0)
    
    ax.set_box_aspect([2, 2 * np.abs(lyi-lyd) / np.abs(lxi-lxd) ,  2 * np.abs(lzi-lzd) / np.abs(lxi-lxd)], zoom=0.85)
    ax.set_zlim(lzd-5,lzi+5)
    ax.set_xlim(lxd+5,lxi-5)
    ax.set_ylim(lyi-5,lyd+5)
    
    plt.locator_params(axis='y', nbins=2)
    ax.text(30,40,180, 't = '+str(0.5*n)+'min', fontsize=12)
    # ax.set_title( 't = '+str(0.5*n)+'s', fontsize=12)

    # ax.set_title('Azimutal = '+str(az)+'°, Altitud = '+str(al)+'°')
    ax.view_init(17,120)
    plt.savefig('imgifi.png',dpi=400, bbox_inches='tight')
    lista_im.append(imageio.imread('imgifi.png')[:,:,0])
imageio.mimsave('./Documents/Scallops_ice_15(2).gif', lista_im, fps=10, format='gif')



#%%


for i in [4,5,6,0]:
    plt.figure()
    for n in range(0,nts[i],3):
        hice = halgs[i][n]
        yyy = ys[i]
        aaa = plt.plot(yyy, hice[:,400] )
    plt.title(salis[i])
    plt.grid()
    plt.show()

#%%
# =============================================================================
# Watershed
# =============================================================================
sa = 1

scals = []
for n in tqdm(range(len(ts[sa]))):
    scal = watershed( difs[sa][n], mask=mmms[sa][n]>0, compactness=1e3)
    scals.append(scal)
scals = np.array(scals)

n = 50
hhee = difs[sa][n]
hhnn = ( difs[sa][n] - np.nanmin(difs[sa][n]) ) / np.nanmax( difs[sa][n] - np.nanmin(difs[sa][n]) )
plt.figure()
plt.imshow(hhnn, cmap='gray')
plt.show()
plt.figure()
plt.imshow( mark_boundaries(hhnn, scals[n]) )
plt.show()

#%%
sa = 1 #which experiment 

scallops = []
for n in tqdm(range(nts[i])):
    peks = peak_local_max( difs[sa][n], threshold_abs=0, footprint = disk(30) )#, labels=mmmm[n]>0  )
    cmask = np.zeros_like( difs[sa][n], dtype=bool )
    cmask[ tuple(peks.T) ] = True
    cord = label(cmask)
    
    scals = watershed(-difs[sa][n], cord ,mask=mmms[sa][n]>0, compactness= 1e-2) #compact = 0.01 no esta tan mal
    scallops.append(scals)
        
scallops = np.array(scallops)

# scal_min = []
# for n in tqdm(range(nt)):
#     peks = peak_local_max( -difs[i][n], threshold_abs=0, footprint = disk(30) )#, labels=mmmm[n]>0  )
#     cmask = np.zeros_like( difs[i][n], dtype=bool )
#     cmask[ tuple(peks.T) ] = True
#     cord = label(cmask)
    
#     scals = watershed(difs[i][n], cord ,mask=mmms[i][n]>0, compactness= 1e-2) #compact = 0.01 no esta tan mal
#     scal_min.append(scals)


#%%
cxs,cys,cts = [],[],[]
for n in tqdm(range(nts[sa])): #range(nt):
    
    aldif = ( difs[sa][n] - np.nanmin(difs[sa][n])) / np.nanmax( difs[sa][n] - np.nanmin(difs[sa][n]))
    aldif[ np.isnan(aldif) ] = 0
    
    props = regionprops(scallops[n], intensity_image=aldif)

    for j in range(len(props)):
        cw = props[j].centroid_weighted
        cys.append(cw[0])
        cxs.append(cw[1])
        cts.append(n/2)

cxs = np.array(cxs)*ds[sa] #no es propiamente la posicion en x (ver como hacer bien esto)
cys = np.array(cys)*ds[sa]
#%%
colos = [ [i,1-i ,0]  for i in np.linspace(0.1,1.0,25) ]

mxs,mys,tts = np.array(cxs), np.array(cys), np.array(cts)

mint = 12
fig,ax = plt.subplots()
ppl = ax.scatter(mxs[tts>mint],mys[tts>mint], c=tts[tts>mint], cmap='viridis')

ax.set_aspect('equal')
ax.invert_yaxis()
# ax.set_xlim([-53,53])
# ax.set_ylim([170,-69])

fig.colorbar(ppl,ax=ax,label='time (min)')

plt.gca().invert_yaxis()
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.grid()
plt.show()


trax,tray,tiemps = track2d(mxs[tts>mint],mys[tts>mint],tts[tts>mint], delim=4, dtm=1)

trxl, tryl, tiel = [],[],[]
for i in range(len(trax)):
    if len(trax[i]) > 15: 
        trxl.append(trax[i])
        tryl.append(tray[i])
        tiel.append(tiemps[i])
del trxl[1], tryl[1], tiel[1]

fig,ax = plt.subplots()
for i in range(len(trxl)):
    ppil = ax.plot(trxl[i],tryl[i],'.-', label=i)

ax.set_aspect('equal')
ax.invert_yaxis()
# ax.set_xlim([-53,53])
# ax.set_ylim([170,-69])
ax.legend( loc='upper center', bbox_to_anchor=(1.25, 0.8), ncol=1 )#, frame=False)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.gca().invert_yaxis()
plt.grid()
plt.show()

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


fig,ax = plt.subplots(2,1, sharex=True)
ax[0].errorbar(np.arange(len(slops)),slops, yerr=eslo, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
ax[1].errorbar(np.arange(len(slpx)),slpx, yerr=eslx, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel('vel y (mm/min)', fontsize=12)
ax[1].set_ylabel('vel x (mm/min)', fontsize=12)
ax[1].set_xlabel('Scallop', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/vels_'+ str(ss) +'.png',dpi=400, bbox_inches='tight')
plt.show()

# plt.figure()
# plt.errorbar(slpx, slops, xerr=eslx, yerr=eslo, fmt='.')
# plt.grid()
# plt.show()

print('y', np.mean(slops), np.median(slops), np.std(slops))
print('x', np.mean(slpx), np.median(slpx), np.std(slpx))

# n = 30
# plt.figure()
# plt.imshow( iceh[n], extent=(x[0],x[-1],y[-1],y[0]) )
# plt.plot(mxs[tts*2==n], mys[tts*2==n], 'k.')
# # plt.scatter(mxs[tts>15],mys[tts>15], c=tts[tts>15], cmap='Reds')
# plt.xlabel('x (mm)')
# plt.ylabel('y (mm)')
# plt.show()


#%%


#%%
sa=1
n = 40

# t1 = time()

# peks = peak_local_max( difs[sa][n], threshold_abs=0, footprint = disk(30) )

# t2 = time()

# peks2 = np.where( maximum_filter( difs[sa][n] , footprint=disk(30)) == difs[sa][n] )

t3 = time()

sder = np.abs(gxs[sa][n]) + np.abs(gys[sa][n])
mima = np.where( sder < 1e-2 )

t4 = time()

print(t4-t3,t3-t2, t2-t1)

plt.figure()
plt.plot(peks[:,0], peks[:,1],'bo')
plt.plot(peks2[0],peks2[1]+1,'rs',alpha=0.5)
plt.plot(mima[0],mima[1]+2,'go',alpha=0.5)
plt.show()

#%%

# a = np.array([[0,0,1],[0,1,0],[0,0,0]])
# np.where(a)

# plt.figure()
# plt.imshow(scallops[0])
# plt.show()

props = regionprops(scallops[0])


props[20].centroid[0]

#%%
# =============================================================================
# Maxima trails
# =============================================================================
sa = 1

skes = []

for n in tqdm(range(len(ts[sa]))):
    hhee = difs[sa][n]
    
    ksi = 60
    kernel = np.ones((ksi,ksi))
    difsm = snd.convolve(difs[sa][n], kernel) / ksi**2
    
    bina = difs[sa][n] > difsm*1.2
    ske =  skeletonize(bina)
    skes.append(ske)

    # plt.figure()
    # plt.imshow( difs[sa][n] )
    # plt.colorbar()
    # plt.show()

skes = np.array(skes)
#%%

for n in range(20,40,2):
    plt.figure()
    # plt.imshow( difs[sa][n] )
    # plt.plot(  np.where(skes[n]>0)[1], np.where(skes[n]>0)[0], 'k.'  )
    plt.imshow( skes[n] )
    plt.show()

nubo = np.sum( skes, axis=(1,2) )
are = np.sum(mkfins[sa], axis=(1,2))
ncon = []
for n in range(len(ts[sa])):
    ncon.append( np.max( label(skes[n]) ) )

plt.figure()
plt.plot( ncon, '.-' )
plt.grid()
plt.show()
plt.figure()
plt.plot( ncon / are, '.-' )
plt.grid()
plt.show()

plt.figure()
plt.plot( nubo / are, '.-' )
plt.grid()
plt.show()
#%%
n = 0

plt.figure()
plt.imshow( skes[n] )
plt.show()

wama = []
for l in range(1024):
    lin = skes[n][:,l]
    # lin = skes[n][l,:]
    vals = np.where(lin)[0]
    dima = vals[1:] - vals[:-1]
    wama += list( dima[dima>0] )

plt.figure()
plt.hist( wama, bins=50 )
plt.show()
#%%


#%%
a = np.ones((10,10))
a = np.pad(a,2)
# a[a==0] = np.nan

ker = np.array([[1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1],
                [1,1,1,1,1]])

# a
snd.convolve(a, ker) / 5**2



#%%
# =============================================================================
# Temperatures
# =============================================================================

fileu = ['/Volumes/Ice blocks/Ice_block_0','_dataup2.txt']
filed = ['/Volumes/Ice blocks/Ice_block_0','_datadown2.txt']
exps = ['(21)', '(20)', '(22)']
sass = [15.2,18.13,21.9]
start = [29,59,47]
finish = [2,2,2]

plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 

for i in range(3):
    tup = np.loadtxt( fileu[0]+exps[i]+fileu[1], delimiter=';', skiprows=1)
    tdo = np.loadtxt( filed[0]+exps[i]+filed[1], delimiter=';', skiprows=1)
    
    tu, td = (tup[start[i]:,0] - tup[0,0]) / (tup[1,0] - tup[0,0]) , (tdo[start[i]:,0] - tdo[0,0]) /  (tdo[1,0] - tdo[0,0])
    
    # plt.figure()
    # plt.plot(tup[start[i]:-2,1], '-', label=str(sass[i]) + 'g/kg' , color=colors[i])
    # plt.plot(tdo[start[i]:-2,1], '--', color=colors[i])
    
    plt.plot( tup[start[i]:-2-finish[i],1] - tdo[start[i]:-2,1], '-', label=str(sass[i]) + 'g/kg' , color=colors[i])
    

plt.grid()
plt.legend()
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel(r'Temperature diference ($^\circ$C)', fontsize=12)
plt.savefig('./Documents/diftemp.png',dpi=400, bbox_inches='tight')
plt.show()

#%%




#%%

def linfit(n,halg,mmmm,xr,yr):
    haltura = (halg*mmmm)[n]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[n][~np.isnan(haltura)], yr[n][~np.isnan(haltura)]
    A = np.array([xfit*0+1,xfit,yfit]).T
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol1(coeff,xr,yr):
    plane = coeff[0] + coeff[1] * xr[0] + coeff[2] * yr[0] 
    return plane
#%%
def rot1(x,y,z,mat):
    """
    Returns the rotated 2d arrays y and z using the rotation matrix from mrot().
    
    y: 2d-array
    z: 2d-array
    mat: 2x2 matrix from mrot()
    """
    # n1,n2 = np.shape(y)
    # yzf = np.zeros((n1*n2,2))
    # yzf[:,0], yzf[:,1] = y.flatten(), z.flatten()
    # yzr = np.dot(yzf,mat.T)
    # yr,zr = yzr[:,0].reshape(n1,n2), yzr[:,1].reshape(n1,n2)
    
    # n1,n2 = np.shape(y)
    # yzf = np.zeros((n1*n2,3))
    # yzf[:,0], yzf[:,1], yzf[:,2] = x.flatten(), y.flatten(), z.flatten()
    # yzr = np.dot(yzf,mat.T) 
    # xr,yr,zr = yzr[:,0].reshape(n1,n2), yzr[:,1].reshape(n1,n2), yzr[:,2].reshape(n1,n2)    

    n1,n2 = np.shape(y)
    yzf = np.zeros((3,n1*n2))
    yzf[0,:], yzf[1,:], yzf[2,:] = x.flatten(), y.flatten(), z.flatten()
    yzr = np.dot(mat, yzf)
    xr,yr,zr = yzr[0,:].reshape(n1,n2), yzr[1,:].reshape(n1,n2), yzr[2,:].reshape(n1,n2) 
    
    return xr,yr,zr

#%%

a = np.array([[1,2,3],
              [2,3,4],
              [3,4,5]])
# a = np.array([[1,1,1],
#               [2,2,2],
#               [3,3,3]])

x = np.array([[-1,0,1],
              [-1,0,1],
              [-1,0,1]])
y = np.array([[-1,-1,-1],
              [0,0,0],
              [1,1,1]])

A = np.array([x.flatten()*0+1,x.flatten(),y.flatten()]).T

coeff, r, rank, s = np.linalg.lstsq(A, a.flatten(), rcond=None)

plane = coeff[0] + coeff[1] * x + coeff[2] * y

# print( coeff, r, rank, s )
thx, thy = -np.arctan(coeff[1]), -np.arctan(coeff[2])

maty = np.array([[1, 0, 0],
                 [0,np.cos(thy), -np.sin(thy)],
                 [0,np.sin(thy),  np.cos(thy)]])

matx = np.array([[np.cos(thx), 0, -np.sin(thx)],
                 [0, 1, 0],
                 [np.sin(thx), 0, np.cos(thx)]])


matz = np.array([[np.cos(thy), -np.sin(thy), 0],
                  [np.sin(thy), np.cos(thy), 0],
                  [0, 0, 1]])

# xrot, arot = rot(x,a,matx)  
# yrot, arot = rot(y,arot,maty)
# xrot,yrot, arot = rot(x,y, a , np.dot(matx,maty))
xrot,yrot, arot = rot(x,y,a,maty) 
# xrot,yrot, arot = rot(xrot,yrot,arot,matx) 

a, arot #, xrot, yrot
#%%
thx, thy = np.arctan(0), np.arctan(2)

maty = np.array([[1, 0, 0],
                 [0,np.cos(thy), -np.sin(thy)],
                 [0,np.sin(thy),  np.cos(thy)]])

matx = np.array([[np.cos(thx), 0, -np.sin(thx)],
                 [0, 1, 0],
                 [np.sin(thx), 0, np.cos(thx)]])

xyxy = np.array( [ x.flatten(), y.flatten(), np.zeros((3*3))] )


np.dot( maty, xyxy ) 

#%%


# a = np.array([[1,2,3],
#               [2,3,4],
#               [3,4,5]])
a = np.ones( (3,3) )

x = np.array([[-1,0,1],
              [-1,0,1],
              [-1,0,1]])
y = np.array([[-1,-1,-1],
              [0,0,0],
              [1,1,1]])

# plt.figure()
# # plt.plot(x.T,a.T, '.' )
# plt.grid()
# plt.show()

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( x,a,y )
ax.set_xlim(-2,2)
ax.set_zlim(-2,2)
ax.set_ylim(-2,2)
plt.show()

# thx = np.arctan(0.5)
# matx = np.array([[np.cos(thx), -np.sin(thx)],
#                 [np.sin(thx),  np.cos(thx)]])
# thy = np.arctan(1)
# maty = np.array([[np.cos(thy), -np.sin(thy)],
#                 [np.sin(thy),  np.cos(thy)]])


# rx,ra = np.dot( matx, np.array([x.flatten(), a.flatten()]) )
# rx, ra = np.reshape(rx, x.shape), np.reshape(ra, a.shape)
# ry,ra = np.dot( maty, np.array([y.flatten(), ra.flatten()]) )
# ry, ra = np.reshape(ry, y.shape), np.reshape(ra, a.shape)

thx, thy = np.arctan(0.5), np.arctan(1)
maty = np.array([[1, 0, 0],
                 [0,np.cos(thy), -np.sin(thy)],
                 [0,np.sin(thy),  np.cos(thy)]])

matx = np.array([[np.cos(thx), 0, np.sin(thx)],
                 [0, 1, 0],
                 [-np.sin(thx), 0, np.cos(thx)]])

rx,ry,ra = np.dot( np.dot(matx,maty), np.array( [x.flatten(), y.flatten(), a.flatten()] ) )
rx, ry, ra = np.reshape(rx, x.shape), np.reshape(ry, y.shape), np.reshape(ra, a.shape)


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( rx,ra,ry )
ax.set_xlim(-2,2)
ax.set_zlim(-2,2)
ax.set_ylim(-2,2)
plt.show()

thx, thy = -np.arctan(0.5), -np.arctan(1)
matyi = np.array([[1, 0, 0],
                 [0,np.cos(thy), -np.sin(thy)],
                 [0,np.sin(thy),  np.cos(thy)]])

matxi = np.array([[np.cos(thx), 0, np.sin(thx)],
                 [0, 1, 0],
                 [-np.sin(thx), 0, np.cos(thx)]])

rrx,rry,rra = np.dot( np.dot(matyi,matxi), np.array( [rx.flatten(), ry.flatten(), ra.flatten()] ) )
rrx, rry, rra = np.reshape(rrx, x.shape), np.reshape(rry, y.shape), np.reshape(rra, a.shape)


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( rrx,rra,rry )
ax.set_xlim(-2,2)
ax.set_zlim(-2,2)
ax.set_ylim(-2,2)
plt.show()
#%%


Rt = np.dot(matx,maty)
Rti = np.dot(matyi,matxi)

np.dot(Rt,Rti)

#%%

A = np.array([x.flatten()*0+1,x.flatten(),y.flatten()]).T

coeff, r, rank, s = np.linalg.lstsq(A, ra.flatten(), rcond=None)

plane = coeff[0] + coeff[1] * x + coeff[2] * y

plane, ra, coeff

# thx, thy = -np.arctan(coeff[1]), -np.arctan(coeff[2])
thx, thy = -np.arctan(0.5), -np.arctan(1)

maty = np.array([[1, 0, 0],
                 [0,np.cos(thy), -np.sin(thy)],
                 [0,np.sin(thy),  np.cos(thy)]])

matx = np.array([[ np.cos(thx), 0, np.sin(thx)],
                 [0, 1, 0],
                 [-np.sin(thx), 0, np.cos(thx)]])


matz = np.array([[np.cos(thy), -np.sin(thy), 0],
                  [np.sin(thy), np.cos(thy), 0],
                  [0, 0, 1]])

xrot,yrot, arot = rot(x,y, a , np.dot(matx,maty))
# xrot,yrot, arot = rot(x,y,a,maty) 

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( x,ra,y )
ax.set_xlim(-2,2)
ax.set_zlim(-2,2)
ax.set_ylim(-2,2)
plt.show()


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( xrot,arot,yrot )
ax.set_xlim(-2,2)
ax.set_zlim(-2,2)
ax.set_ylim(-2,2)
plt.show()

ra, arot

#%%
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
#%%

a = np.array([[1,2,3],
              [2,3,4],
              [3,4,5]])

x = np.array([[-1,0,1],
              [-1,0,1],
              [-1,0,1]])
y = np.array([[-1,-1,-1],
              [0,0,0],
              [1,1,1]])


# parx,cov = curve_fit(lin, x.flatten(), a.flatten())
# pary,parx, cov
# parx,cov = curve_fit(lin, xr[200], phiy[200,:])

pary,cov = curve_fit(lin, y.flatten(), a.flatten())
m2 = mrot( -np.arctan(pary[0]) )
yrot, arot = rot(y,a,m2)

# print(yrot,'\n',arot, end='\n\n')

parx,cov = curve_fit(lin, x.flatten(), arot.flatten())
m2 = mrot( -np.arctan(parx[0]) )
xrot, arot = rot(x,arot,m2)

# print(xrot,'\n',arot)
print(parx,pary)



parx,cov = curve_fit(lin, x.flatten(), a.flatten())
m2 = mrot( -np.arctan(parx[0]) )
xrot, arot = rot(x,a,m2)

# print(xrot,'\n',arot, end='\n\n')

pary,cov = curve_fit(lin, y.flatten(), arot.flatten())
m2 = mrot( -np.arctan(pary[0]) )
yrot, arot = rot(y,arot,m2)

# print(xrot,'\n',arot)
print(parx,pary)


#%%

prfo = halgs[5][0] * mmms[5][0]
rrx, rry = xrs[5][0], yrs[5][0]

A = np.array([rrx.flatten()*0+1,rrx.flatten(),rry.flatten()]).T
finan = ~np.isnan(prfo.flatten())
coeff, r, rank, s = np.linalg.lstsq(A[finan] , (prfo.flatten())[finan], rcond=None)
plane = coeff[0] + coeff[1] * rrx + coeff[2] * rry
plane = plane * mmms[5][0]

print(coeff)

# plt.figure()
# plt.imshow( prfo )
# plt.show()
# plt.figure()
# plt.imshow( plane )
# plt.show()

# fig = plt.figure(figsize=(10,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface( rrx, prfo, rry )
# # ax.set_xlim(-2,2)
# # ax.set_zlim(-2,2)
# # ax.set_ylim(-2,2)
# plt.show()

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( rrx, plane, rry )
# ax.set_xlim(-2,2)
# ax.set_zlim(-2,2)
# ax.set_ylim(-2,2)
plt.show()


thx, thy = -np.arctan(coeff[1]), -np.arctan(coeff[2])
# thx, thy

maty = np.array([[1, 0, 0],
                  [0,np.cos(thy), -np.sin(thy)],
                  [0,np.sin(thy),  np.cos(thy)]])

matx = np.array([[ np.cos(thx), 0, -np.sin(thx)],
                  [0, 1, 0],
                  [np.sin(thx), 0, np.cos(thx)]])

xrot,yrot, prot = rot1( rrx, rry, plane , np.dot(matx,maty))


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( xrot, prot, yrot )
# ax.set_xlim(-2,2)
# ax.set_zlim(-2,2)
# ax.set_ylim(-2,2)
plt.show()


finan = ~np.isnan(prfo.flatten())
pary,cov = curve_fit(lin, (rry.flatten())[finan], (plane.flatten())[finan])
m2 = mrot( -np.arctan(pary[0]) )
yrot, prot = rot(rry,plane,m2)

# print(yrot,'\n',arot, end='\n\n')

parx,cov = curve_fit(lin, (rrx.flatten())[finan], (prot.flatten())[finan])
m2 = mrot( -np.arctan(parx[0]) )
xrot, prot = rot(rrx,prot,m2)

print(parx,pary)

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( xrot, prot, yrot )
# ax.set_xlim(-2,2)
# ax.set_zlim(-2,2)
# ax.set_ylim(-2,2)
plt.show()


finan = ~np.isnan(prfo.flatten())
parx,cov = curve_fit(lin, (rrx.flatten())[finan], (plane.flatten())[finan])
m2 = mrot( -np.arctan(parx[0]) )
xrot, prot = rot(rrx,plane,m2)

# print(yrot,'\n',arot, end='\n\n')

pary,cov = curve_fit(lin, (rry.flatten())[finan], (prot.flatten())[finan])
m2 = mrot( -np.arctan(pary[0]) )
yrot, prot = rot(rry,prot,m2)

print(parx,pary)

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface( xrot, prot, yrot )
# ax.set_xlim(-2,2)
# ax.set_zlim(-2,2)
# ax.set_ylim(-2,2)
plt.show()

#%%

prfo = halgs[5][0] * mmms[5][0]
rrx, rry = xrs[5][0], yrs[5][0]

A = np.array([rrx.flatten()*0+1,rrx.flatten(),rry.flatten()]).T
finan = ~np.isnan(prfo.flatten())
coeff, r, rank, s = np.linalg.lstsq(A[finan] , (prfo.flatten())[finan], rcond=None)
plane = coeff[0] + coeff[1] * rrx + coeff[2] * rry
plane = plane * mmms[5][0]

# fig = plt.figure(figsize=(10,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface( rrx, plane, rry )
# plt.show()

plt.figure()
plt.plot( rrx[600,:],plane[600,:], '-' )
plt.grid()
plt.show()

finan = ~np.isnan(prfo.flatten())
parx,cov = curve_fit(lin, (rrx.flatten())[finan], (plane.flatten())[finan])
m2 = mrot( -np.arctan(coeff[1]) )
# m2 = mrot( -np.arctan(parx[0]) )
xrot, prot = rot(rrx,plane,m2)

yrot = rry

# fig = plt.figure(figsize=(10,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface( xrot, prot, yrot )
# plt.show()

plt.figure()
plt.plot( xrot[600,:], prot[600,:], '-' )
plt.grid()
plt.show()

pary,cov = curve_fit(lin, (rry.flatten())[finan], (prot.flatten())[finan])
# m2 = mrot( -np.arctan(coeff[2]) )
m2 = mrot( -np.arctan(pary[0]) )
yrot, prot = rot(rry,prot,m2)

# fig = plt.figure(figsize=(10,7))
# ax = plt.axes(projection='3d')
# ax.plot_surface( xrot, prot, yrot )
# plt.show()

plt.figure()
plt.plot( xrot[:,600], prot[:,600], '-' )
plt.grid()
plt.show()

print(parx[0], pary[0])

#%%
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
        
    return uihalg, xrots, yrots, np.arctan(coeff[1]) * 180 / np.pi , np.arctan(pary[0]) * 180 / np.pi

def polyfit(n,halg,mmmm,xr,yr, order=2):
    haltura = (halg*mmmm)[n]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[n][~np.isnan(haltura)], yr[n][~np.isnan(haltura)]
    
    poli = []
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: poli.append( xfit**i * yfit**j )
    A = np.array(poli).T
            
    # A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol2(coeff,xr,yr,n,order=2):
    poo, cof = 0, 0
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: 
                poo += coeff[cof] * xr[n]**i * yr[n]**j
                cof += 1
    return poo
    # return coeff[0] + coeff[1] * xr[n] + coeff[2] * yr[n] + coeff[3] * xr[n]**2 + coeff[4] * xr[n]*yr[n] + coeff[5] * yr[n]**2

#%%
# =============================================================================
# Take out tilt
# =============================================================================
sals = ['5','15','23','8(1)','0','2','4','6','8(3)','12','14','18','15(2)','22','27','10']
salis = [8,15.2,23.38,8,0,2.31,4.06,6.40,8.26,12.87,14.8,18.13,15.2,21.9,27.39,10.26]
ds = [ 0.37566904425667014, 0.39250081308793844, 0.3957774907299554, 0.39250081308793844, 0.3779326761033847, 0.37940313057152375,
       0.37917605048206754, 0.3780874065675317, 0.3773925197773744, 0.37804894658912125, 0.37799748108152204, 0.37903057740207474,
       0.3788519951979505, 0.3795787182927243, 0.3782083129763212, 0.3783574601007215 ]
temps = [ 20.69, 20.08, 21.40, 19.95, 21.00, 20.65, 19.6, 20.14, 19.13, 18.57, 20.04, 20.37, 19.46, 18.60, 17.76, 19.22 ]

with h5py.File('./Documents/Height profiles/npys/heigtarr.hdf5', 'r') as f:

    halgts, xrts, yrts, angxs, angys, ts = [], [], [], [], [], []
    difs = []
    for n in tqdm(range(len(salis))):
        
        hal = resha(f['h'][n])
        tr = resha(f['t'][n])
        yr = resha(f['y'][n])
        xr = resha(f['x'][n])
        mmmm = resha(f['mmm'][n])
        # mmmm = resha(f['mfin'][n])
        

        halgt, xrt, yrt, angx, angy = untilt( hal, mmmm, xr, yr )
        halgts.append( np.array(halgt) )
        xrts.append( np.array(xrt) ) 
        yrts.append( np.array(yrt) )
        angxs.append( angx )
        angys.append( angy )
        ts.append( tr[:,0,0] )
             
        difes = []
        for i in range(len(hal)):
            coeff, r, rank, s = polyfit(i,halgt,mmmm,xrt,yrt, order=3)
            cuapla = pol2(coeff,xrt,yrt,i, order=3) 
            difes.append( (halgt[i]-cuapla)*mmmm[i] )
        difes = np.array(difes)
        difs.append(difes)

#%%

meses, xmes, ymes = [], [], []
    
for n in tqdm(range(len(salis))):
    mess, xes, yes = [], [], []
    for l1 in range(1024):
        for l2 in range(1024):
            mehe = halgts[n][:,l1,l2]
            if np.sum(~np.isnan(mehe)) > 5:
                me, _  = curve_fit(lin, (ts[n])[~np.isnan(mehe)]  , mehe[~np.isnan(mehe)] )
                mess.append(-me[0])
                xes.append( xrts[n][0,l1,l2] )
                yes.append( yrts[n][0,l1,l2] )
    meses.append( mess )
    xmes.append( xes )
    ymes.append( yes )

#%%
# =============================================================================
# Melt rate mean, normalized
# =============================================================================
meme, mesd = [], []
for i in range(len(meses)):
    meme.append( np.mean(meses[i]) )
    mesd.append( np.std(meses[i]) )
meme, mesd =np.array(meme), np.array(mesd)

memen = meme / meme[4]
mesdn = 1/meme[4] * np.sqrt( (mesd)**2 + (meme*mesd[4])**2 )

plt.figure()
# plt.errorbar(salis, meme, yerr=mesd, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
plt.errorbar(salis, memen, yerr=mesdn, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
plt.grid()
plt.xlabel('Salinity (g/kg)',fontsize=12)
plt.ylabel(r'$f/f_0$',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.savefig('./Documents/meratesa.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Melt rate heat map
# =============================================================================
for n in [4,-1]:
    plt.figure()
    # plt.imshow( meses[n] )
    plt.scatter(xmes[n], ymes[n], c=meses[n], s=0.5, vmax=0.02, vmin=0.0)
    plt.title( salis[n] )
    plt.colorbar()
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    # plt.axis('equal')
    # plt.xlim( np.min(xmes[n])-5, np.max(xmes[n])+5 )
    # plt.ylim( np.min(ymes[n])-5, np.max(ymes[n])+5 )
    plt.axis([np.min(xmes[n])-5, np.max(xmes[n])+5 , np.min(ymes[n])-5, np.max(ymes[n])+5])
    plt.gca().set_aspect("equal")
    plt.show()
#%%
meses, xmes, ymes = [], [], []
for n in tqdm(range(len(salis))):
    prop = regionprops( label(halgts[n][0] < 1000) )
    cen = prop[0].centroid
    cen = np.array(cen,dtype='int')
    exx, exy = 70, 100

    hice = halgts[n][:,cen[0]-exy:cen[0]+exy, cen[1]-exx:cen[1]+exx] 
    xice = xrts[n][:,cen[0]-exy:cen[0]+exy, cen[1]-exx:cen[1]+exx] 
    yice = yrts[n][:,cen[0]-exy:cen[0]+exy, cen[1]-exx:cen[1]+exx] 

    mess, xes, yes = [], [], []
    for l1 in range(exy*2):
        for l2 in range(exx*2):
            mehe = hice[:,l1,l2]
            if np.sum(~np.isnan(mehe)) > 5:
                me, _  = curve_fit(lin, (ts[n])[~np.isnan(mehe)]  , mehe[~np.isnan(mehe)] )
                mess.append(-me[0])
                xes.append( xice[0,l1,l2] )
                yes.append( yice[0,l1,l2] )
    meses.append( mess )
    xmes.append( xes )
    ymes.append( yes )

meme, mesd = [], []
for i in range(len(meses)):
    meme.append( np.mean(meses[i]) )
    mesd.append( np.std(meses[i]) )
meme, mesd =np.array(meme), np.array(mesd)

memen = meme / meme[4]
mesdn = 1/meme[4] * np.sqrt( (mesd)**2 + (meme*mesd[4])**2 )

plt.figure()
# plt.errorbar(salis, meme, yerr=mesd, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
plt.errorbar(salis, memen, yerr=mesdn, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
plt.grid()
plt.xlabel('Salinity (g/kg)',fontsize=12)
plt.ylabel(r'$f/f_0$',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.savefig('./Documents/meratesa.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# for n in range(16):
plt.figure()
plt.scatter(xice[0],yice[0],c=meses[-1])
plt.show()

#%%
# =============================================================================
# Melt rate half volume (the one I use)
# =============================================================================
th0 = 0.1 # m

plt.figure()
tmins = []
for n in tqdm(range(len(salis))):
# for n in [0,3,8]:

    xa,ya = np.arange(1024), -np.arange(1024)
    dxyp,dxxp = np.gradient(xrts[n][0] / 1000, ya,xa)
    dyyp,dyxp = np.gradient(yrts[n][0] / 1000, ya,xa)
    dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0
    jacob = dxxp * dyyp - dyxp * dxyp
    
    jacob[np.isnan(jacob)] = 0
    
    Ap0 = np.trapz( np.trapz( jacob, xa, axis=1 ), -ya, axis=0 ) #not sure why -ya but needed for Ap positive
    Vp0 = th0 * Ap0
    Vn, An = [1], [1]
    
    for i in range(1,len(ts[n])):
        dxyp,dxxp = np.gradient(xrts[n][i] / 1000, ya,xa)
        dyyp,dyxp = np.gradient(yrts[n][i] / 1000, ya,xa)
        dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0
        jacob = dxxp * dyyp - dyxp * dxyp
        
        # delh = ( halgts[n][i-1] - halgts[n][i] ) / 1000
        delh = ( halgts[n][0] - halgts[n][i] ) / 1000
        
        jacob[np.isnan(jacob)] = 0
        delh[np.isnan(delh)] = 0
        
        Ap = np.trapz( np.trapz( jacob, xa, axis=1 ), -ya, axis=0 )
        vi1 = np.trapz( np.trapz( delh * jacob, xa, axis=1 ), ya, axis=0 )
        
        Vp = th0 * Ap - 2 * vi1
        Vn.append(Vp / Vp0)
        An.append(Ap / Ap0)
        
    Vn = np.array(Vn)  #* 1000 #for liters

    p = np.polyfit(ts[n], Vn , 3)
    # volfit = np.polyval(p, ts[n])
    volfit = np.polyval(p, np.arange(0,4000,30))
    
    f0t = 0.6
    itm = np.argmin( np.abs( volfit - f0t  )  )
    # tmin = ts[n][itm]
    tmin = np.arange(0,4000,30)[itm]
    if Vn[-1] < Vn[0] * (f0t + 0.05):
        tmins.append( tmin )
    else:
        tmins.append(np.nan)
    
    plt.plot( ts[n], Vn, label=salis[n] )
    plt.plot(tmin, Vn[0] * f0t, 'k.' )
    # plt.plot( ts[n], np.polyval(p, ts[n]), '--' )
    # plt.plot( ts[n], An  )

# plt.plot(salis, tmins, 'k.')
plt.legend()
plt.grid()
plt.show()
#%%
from uncertainties import unumpy as unp

tme = unp.uarray( tmins, 30 )
fs = tme[4] / tme

# tmins = np.array(tmins)
# tmsd = 15 * tmins * np.sqrt( (tmins / tmins[4])**2 + 1) 
plt.figure()
# plt.plot(salis, tmins / tmins[4], 'k.')
# plt.errorbar(salis, unp.nominal_values(fs), yerr=unp.std_devs(fs), fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
plt.errorbar( salis , unp.nominal_values(fs), yerr=unp.std_devs(fs), capsize=2, fmt='k.', markersize=10)
plt.grid()
plt.xlabel('Salinity (g/kg)',fontsize=12)
plt.ylabel(r'$f/f_0$',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig('./Documents/mera60v.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Melt rate volume near center
# =============================================================================
# do it with label, find centroid and go from there (probably easier)



#%%
n,i = 8, 100

plt.figure()
plt.imshow(halgts[n][i])
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( difs[n][i])
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( halgts[n][i] - difs[n][i])
plt.colorbar()
plt.show()

# print( np.nanmean(halgts[n][i]), np.nanmean(difs[n][i]) )
#%%
# =============================================================================
# Roughness (with std)
# =============================================================================
# rones, romean = [], []
# for n in tqdm(range(len(salis))):
#     rougs, rom = [], []
#     for i in range(len(difs[n])):
#         hhee = difs[n][i]
#         # roug = np.sqrt( np.nanmean( hhee**2 ) )
#         roug = np.nanmean( np.abs(hhee) )  
#         rougs.append(roug)
#         rom.append( np.nanmean( hhee ) )
#     rones.append(rougs)
#     romean.append(rom)

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# for n in [1,10,12]:
#     ax1.plot(rones[n], '.-')
#     ax2.plot(romean[n], '--')
# plt.grid()
# plt.show()

vro = []
plt.figure()
for i in [1,10,12]: # range(len(salis)):
    plt.plot(ts[i]/60, rones[i], label=salis[i], c=(salis[i]/30,0,0))

    # grd = np.gradient(rones[i], ts[i]/60 )
    # gr = np.where( grd * np.roll(grd,1) <0)[0]
    # if np.sum(gr>20)>0:
    #     ind = gr[gr>20][0]
    #     plt.plot( ts[i][ind]/60, rones[i][ind], 'k.' )
    #     vro.append( rones[i][ind] )
    # # plt.plot(ts[i]/60, grd, label=salis[i])
    # else:
    #     vro.append(rones[i][-1])
    ind = np.argmax(rones[i])
    vro.append( rones[i][ind] )
        
    
plt.grid()
plt.legend()
plt.xlabel('Time (min)',fontsize=12)
plt.ylabel('Roughness (mm)',fontsize=12)
# plt.savefig('./Documents/rough8.png',dpi=400, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(salis, vro, '.')
plt.show()
#%%
# =============================================================================
# Roughness (with slope cdf)
# =============================================================================
rones2 = []
for n in tqdm(range(len(salis))): #[0,3,8]):
    rougs = []
    for i in range(0,len(difs[n]),1):
        heig = np.sort( (difs[n][i]).flatten() )
        heig = heig[ ~np.isnan(heig) ]
        pdh, beh = np.histogram( heig, bins=100 )
        cdh = np.cumsum( pdh ) / np.sum(pdh)
        
        xbe = (beh[1:] + beh[:-1] ) / 2
        
        gra = np.gradient(cdh, xbe)
        roug = np.max(gra)
        rougs.append(roug)
    rones2.append(rougs)

#     # plt.plot(xbe, cdh, '.-', label=i )

# vro2 = []
# for n in range(len(salis)):
#     vro2.append( np.min( rones2[n] ) )

vro2, vros2 = [], []
for n in range(len(salis)):
    vro2.append( np.mean(rones2[n][40:]) )
    vros2.append( np.std(rones2[n][40:]) )


plt.figure()
# plt.plot(ts[n]/60, rougs )
# plt.plot(xbe, cdh)

# for n in range(len(salis)):
#     plt.plot(ts[n]/60, rones2[n], label=salis[n] )

# plt.plot(salis, vro2, '.')
plt.errorbar(salis, vro2, yerr=vros2, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
    
plt.grid()
plt.legend()
plt.show()
#%%
rones4 = []
for n in tqdm(range(len(salis))): #[0,3,8]):
    rougs = []
    for i in range(0,len(difs[n]),1):
        heig = np.sort( (difs[n][i]).flatten() )
        heig = heig[ ~np.isnan(heig) ]
        pdh, beh = np.histogram( heig, bins=100, density=True )
        cdh = np.cumsum( pdh ) / np.sum(pdh)
        
        
        xbe = (beh[1:] + beh[:-1] ) / 2
        lnr = linregress(xbe[30:70],cdh[30:70])
        
        xmin = -lnr[1] / lnr[0]
        xmax = (1-lnr[1]) / lnr[0]
        rougs.append( xmax-xmin )
    rones4.append( rougs )
#%%
plt.figure()
for n in [1,10,12]:
    plt.plot(ts[n], rones4[n] )
plt.show()


#%%

for n in [8]: # tqdm(range(len(salis))): #[0,3,8]):
    rougs = []
    for i in range(0,len(difs[n]),1):
        heig = np.sort( (difs[n][i]).flatten() )
        heig = heig[ ~np.isnan(heig) ]
        pdh, beh = np.histogram( heig, bins=100, density=True)
        cdh = np.cumsum( pdh ) / np.sum(pdh)
        xbe = (beh[1:] + beh[:-1] ) / 2
        
        mee = (np.sum( xbe*pdh ) ) / len(xbe)
        sd = np.sqrt( (np.sum( (xbe-mee )**2*pdh )) / len(xbe) )
        
def gaussianf(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
plt.figure()
plt.plot(xbe, pdh)
plt.plot( xbe, gaussianf(xbe, np.mean(heig) , np.std(heig))  ) 
plt.plot( xbe, gaussianf(xbe, mee , sd)  ) 
plt.show()


#%%
# =============================================================================
# Roughness (with rms slope)
# =============================================================================
rones3 = []

plt.figure()

for n in tqdm([0,3,8]):
    slrms, slrsd = [], []
    gyme, gxme = [], []
    for i in range(len(difs[n])):
        xa,ya = np.arange(1024), np.arange(1024)
        
        gyyp,gxxp = np.gradient(difs[n][i], y,x) #yrts[n][i],xrts[n][i])
        dxyp,dxxp = np.gradient(xrts[n][i], y,x)
        dyyp,dyxp = np.gradient(yrts[n][i], y,x)
        gy, gx = gyyp / dyyp, gxxp / dxxp
    
        slo2 = gy**2 + gx**2
        slrms.append( np.sqrt(np.nanmean(slo2)) )
        # slrsd.append( np.sqrt(np.nanstd(slo2)) )
        gyme.append( np.sqrt( np.nanmean((gyyp / dyyp)**2) ) )
        gxme.append( np.sqrt( np.nanmean((gxxp / dxxp)**2) ) )
    
    # plt.plot(ts[n]/60, slrms, '.-', label=salis[n])
    plt.plot(ts[n]/60, gyme, '.-', label=salis[n])
    plt.plot(ts[n]/60, gxme, 's-', label=salis[n])
    # plt.errorbar(ts[n], slrms, yerr=slrsd, fmt='.-', elinewidth=1., capsize=1.)
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.imshow( slo2, vmin=0, vmax=1 )
plt.show()
 


#%%
# =============================================================================
# Rayleight and Nusselt
# =============================================================================
mu = 1.0016 #kg / ms
kt = 1.4e-7 # m^2 / s
ks = 1.4e-9 # m^2 / s
Cb = 0.011 # kg / m^3 K^2
b0 = 0.77 # kg^2 / m g
g = 9.81 # m / s^2
kappa = 0.598 # kg m / s^3 K
La = 334e3 # m^2 / s^2
delt = 30 # s
rhoi = 916.8 # kg / m^3

th0 = 0.1 # m

Rass,Rats, Nus = [], [], []
for n in tqdm(range(len(salis))):
    Rat, Ras, Nu1,Nu2 = [],[], [], []

    xa,ya = np.arange(1024), -np.arange(1024)
    dxyp,dxxp = np.gradient(xrts[n][0] / 1000, ya,xa)
    dyyp,dyxp = np.gradient(yrts[n][0] / 1000, ya,xa)
    dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0
    
    Ap = np.trapz( np.trapz( dxxp * dyyp, xa, axis=1 ), -ya, axis=0 ) #not sure why -ya but needed for Ap positive
    Vp = th0 * Ap
    Vn, An = [Vp], [Ap]
    
    for i in range(1,len(ts[n])):
        L = (np.nanmax( yrts[n][i] ) - np.nanmin( yrts[n][i] )) / 1000 # m
        Rat.append( g * Cb * (temps[n])**2 * L**3 / (mu * kt) )
        Ras.append( g * b0 * salis[n] * L**3 / (mu * ks) ) 
        
        # delh = np.nanmean( halgts[n][0] - halgts[n][i] ) / 1000 # m
        # Nu1.append( rhoi * La * delh * L / (delt*i * kappa * temps[n]) )
        
        dxyp,dxxp = np.gradient(xrts[n][i] / 1000, ya,xa)
        dyyp,dyxp = np.gradient(yrts[n][i] / 1000, ya,xa)
        dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0
        
        delh = ( halgts[n][i-1] - halgts[n][i] ) / 1000
        delh[np.isnan(delh)] = 0
        
        Ap = np.trapz( np.trapz( dxxp * dyyp, xa, axis=1 ), -ya, axis=0 )
        vi1 = np.trapz( np.trapz( delh * dxxp * dyyp, xa, axis=1 ), ya, axis=0 )
        
        Vp = th0 * Ap - 2 * vi1
        Vn.append(Vp)
        An.append(Ap)
        
        Nu1.append( rhoi * (Vn[-2]-Vn[-1]) / An[-1] * La  * L / (delt * kappa * temps[n]) )
        
        
    Rats.append(Rat)
    Rass.append(Ras)
    Nus.append(Nu1)
#%%
Num, Nusd = [],[]
Rasi, Rati = [], []
for n in range(len(salis)):
    Num.append( np.mean( Nus[n] ) )
    Nusd.append( np.std( Nus[n] ) )
    Rasi.append( Rass[n][0] )
    Rati.append( Rats[n][0] )


plt.figure()
# plt.plot(Rat)
# plt.plot(Ras)
# plt.plot(Nus[0])
# plt.plot(Nu2,'.-')

for n in [0,3,10]: # range(len(salis)):
    # plt.plot(Rats[n], Nus[n])
    plt.plot( Nus[n])

# plt.errorbar(Rati, Num, yerr=Nusd, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)
# plt.errorbar(Rasi, Num, yerr=Nusd, fmt='k.', ecolor='k', elinewidth=1., capsize=1.)

# plt.plot( Rati, Num, '.' )
# plt.plot( Rasi, Num, '.' )

plt.grid()
plt.show()

#%%

# halgts[n][0][500,500] - halgts[n][1][500,500]
np.nanmean( halgts[n][0] - halgts[n][1]  )


#%%
def polyfit(halg,mmmm,xr,yr, order=2):
    haltura = (halg*mmmm)
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[~np.isnan(haltura)], yr[~np.isnan(haltura)]
    
    poli = []
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: poli.append( xfit**i * yfit**j )
    A = np.array(poli).T
    print(len(poli))
            
    # A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol2(coeff,xr,yr,n,order=2):
    poo, cof = 0, 0
    for i in range(order+1):
        for j in range(order+1):
            if i+j<order: 
                poo += coeff[cof] * xr[n]**i * yr[n]**j
                cof += 1
    return poo
#%%

ua = np.arange(5)
u,v = np.meshgrid(ua,ua)
w = np.ones_like(u)

# z = 4 + 4 * u - 5 * v 
z = np.ones((5,5))
z[0,:], z[:,0] = np.nan, np.nan
z[-1,:], z[:,-1] = np.nan, np.nan

i1 = np.trapz(z, ua, axis=1)
i2 = np.trapz(i1, ua, axis=0)
i2, i1
#%%
n = 0
xa,ya = np.arange(1024), np.arange(1024)
dxyp,dxxp = np.gradient(xrts[n][0] , ya,xa)
dyyp,dyxp = np.gradient(yrts[n][0] , ya,xa)
dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)], dyxp[np.isnan(dyxp)], dxyp[np.isnan(dxyp)] = 0, 0, 0, 0
jacob = dxxp * dyyp - dyxp * dxyp

Ap = np.trapz( np.trapz( jacob, xa, axis=1 ), ya, axis=0 )
# plt.figure()
# plt.plot(np.trapz( jacob, -xa, axis=0 ))
# plt.show()
Ap / 100

# plt.figure()
# # plt.imshow(dyyp)
# # plt.plot(xa, xrts[n][0][400,:] )
# plt.plot(ya, yrts[n][0][:,400] )
# plt.grid()
# plt.show()
#%%

a,b = np.arange(21), np.arange(0,41,2)
aa,bb = np.meshgrid(a,b)

u,v = np.linspace(0,1,21), np.linspace(0,1,21)

aa1 = np.trapz( np.trapz( np.ones_like(aa)  , a, axis=1), b, axis=0 )

dav,dau = np.gradient( aa, v,u )
dbv,dbu = np.gradient( bb, v,u )
jacob = dau*dbv - dav*dbu
aa2 = np.trapz( np.trapz( jacob  , u, axis=1), v, axis=0 )

aa1, aa2


#%%

te = np.linspace(-5, 25, 1000)
sa = np.linspace( 0, 35, 1000 )
sa, te = np.meshgrid(sa,te)

rho = - 0.011/2 * (te - 4 + 0.25 * sa)**2 + 0.77 * sa

plt.figure()
plt.imshow( rho, extent=(0,35,-5,25) )
plt.colorbar()
plt.show()
#%%
# =============================================================================
# Nusselt and Rayleigh (functions of z or time)
# =============================================================================
mu = 1.0016 #kg / ms
nu = 1e-4 #m^2 / s
kt = 1.4e-7 # m^2 / s
ks = 1.4e-9 # m^2 / s
Cb = 0.011 # kg / m^3 K^2
b0 = 0.77 # kg^2 / m g
g = 9.81 # m / s^2
kappa = 0.598 # kg m / s^3 K
La = 334e3 # m^2 / s^2
delt = 30 # s
rhoi = 916.8 # kg / m^3

def rhots(T,Sk):
    S = Sk /1000
    a1, a2, a3, a4, a5 = 9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8
    b1, b2, b3, b4, b5 = 8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5
    rho = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4 + b1*S + b2*S*T + b3*S*T**2 + b4*S*T**3 + b5*S**2 * T**2
    return rho

Nuzn, Razn = [],[]
for n in range(len(salis)):

    xrn, yrn = xrts[n][0] / 1000, yrts[n][0] / 1000
    hn, vts = halgts[n], ts[n]
    pen2,pen1,ori,tini = np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan
    for i in tqdm(range(1024)):
        for j in range(1024):
            devh = hn[:,i,j] / 1000 # to converts from mm to m
            idx = ~np.isnan(devh)
            if np.sum(idx) > 5: 
                # pen2[i,j], pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 2)
                pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 1) #if I want average melt rate over time, pen1 = meltrate
                tini[i,j] = (vts[idx])[-1]
    
    rhoin, rhowa = rhots(temps[n], salis[n]), rhots(0, 0) 
    gp = g * (rhoin - rhowa) / rhoin
    
    Nuz = - pen1 * rhoi * La / ( kappa * temps[n] ) * (yrn - np.nanmin(yrn))
    Raz = np.abs(gp) / (kt * nu)  *  (yrn - np.nanmin(yrn))**3
    
    yrnma, yrnmi = np.nanmax(yrn), np.nanmin(yrn)
    valyr = np.linspace(yrnmi, yrnma, 100)
    
    nuz,raz = [], []
    for i in range(len(valyr)-1):
        filt = (yrn >=  valyr[i]) * (yrn <=  valyr[i+1])
        nuz.append( np.nanmean(Nuz[filt]) )
        raz.append( np.nanmean(Raz[filt]) )
   
    Nuzn.append( nuz )
    Razn.append( raz )
#%%
plt.figure()    
# for n,sal in enumerate(salis):
for n,sal in enumerate(salis[-2:-1]):
    plt.plot(Razn[n],Nuzn[n],'.-', c=(1 * sal/30, 1 * (1-sal/30), 0), label=sal)
plt.xscale('log')
plt.yscale('log')
# plt.legend()
plt.grid()

sss = np.logspace(1.4, 8.5, 10000)
plt.plot( sss, sss**(1/3)*1.7e-1, 'k--' )

sss = np.logspace(1.4, 8.5, 10000)
plt.plot( sss, sss**(1/4)*9e-1, 'b--' )

plt.show()
#%%

Nuzt, Razt = [],[]
for n in range(len(salis)):

    xrn, yrn = xrts[n][0] / 1000, yrts[n][0] / 1000
    hn, vts = halgts[n], ts[n]
    pen2,pen1,ori,tfin = np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan
    for i in tqdm(range(1024)):
        for j in range(1024):
            devh = hn[:,i,j] /1000
            idx = ~np.isnan(devh)
            if np.sum(idx) > 5: 
                pen2[i,j], pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 2)
                # pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 1) #if I want average melt rate over time, pen1 = meltrate
                tfin[i,j] = (vts[idx])[-1]
    
    rhoin, rhowa = rhots(temps[n], salis[n]), rhots(0, 0) 
    gp = g * (rhoin - rhowa) / rhoin
    
    nut, rat = [], []
    for i in range(len(vts)):
        filt = tfin >= vts[i]
        p1m, p2m = np.nanmean(pen1[filt]), np.nanmean(pen2[filt])
        
        nut.append( - (2*p2m*vts[i]+p1m) * rhoi * La / ( kappa * temps[n] ) * (np.nanmax(yrts[n][i] /1000) - np.nanmin(yrts[n][i] /1000)) )
        rat.append( np.abs(gp) / (kt * nu)  *  ( np.nanmax(yrts[n][i] /1000) - np.nanmin(yrts[n][i] /1000) )**3  )
        
    Nuzt.append( nut )
    Razt.append( rat )

#%%

plt.figure()
for n,sal in enumerate(salis):
    plt.plot(Razt[n],Nuzt[n],'.-', c=(1 * sal/30, 1 * (1-sal/30), 0), label=sal)
plt.xscale('log')
plt.yscale('log')
# plt.legend()
# plt.grid()

plt.show()
#%%

Nu, Ra = [],[]
memel = []
for n in range(len(salis)):

    xrn, yrn = xrts[n][0] / 1000, yrts[n][0] / 1000
    hn, vts = halgts[n], ts[n]
    pen2,pen1,ori,tfin = np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan, np.zeros((1024,1024))*np.nan
    for i in tqdm(range(1024)):
        for j in range(1024):
            devh = hn[:,i,j] /1000
            idx = ~np.isnan(devh)
            if np.sum(idx) > 5: 
                # pen2[i,j], pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 2)
                pen1[i,j], ori[i,j] = np.polyfit(vts[idx], devh[idx], 1) #if I want average melt rate over time, pen1 = meltrate
                tfin[i,j] = (vts[idx])[-1]
    
    rhoin, rhowa = rhots(temps[n], salis[n]), rhots(0, 0) 
    gp = g * (rhoin - rhowa) / rhoin

    p1m, p2m = np.nanmean(pen1[filt]), np.nanmean(pen2[filt])
    
    memel.append( - p1m )
        
    Nu.append( - p1m * rhoi * La / ( kappa * temps[n] ) * (np.nanmax(yrts[n][0] /1000) - np.nanmin(yrts[n][0] /1000)) )
    Ra.append( np.abs(gp) / (kt * nu)  *  ( np.nanmax(yrts[n][0] /1000) - np.nanmin(yrts[n][0] /1000) )**3  )
#%%        
plt.figure()
plt.plot(Ra,Nu,'.', c=(1 * sal/30, 1 * (1-sal/30), 0), label=sal)
plt.xscale('log')
plt.yscale('log')
# plt.legend()
# plt.grid()
plt.show()

plt.figure()
plt.plot(salis, memel, '.')
plt.show()

#%%
# plt.figure()
# plt.imshow(pen1)
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow(Raz)
# plt.colorbar()
# plt.show()

plt.figure()
plt.plot( Raz.flatten(), Nuz.flatten(), '.' )
plt.xscale('log')
plt.yscale('log')

sss = np.logspace(1, 7, 10000)
plt.plot( sss, sss**(1/3)*3e2, 'r-' )
plt.plot( sss, sss**(1/2)*1e2, 'g-' )

plt.show()
#%%
mu = 1.0016 #kg / ms
nu = 1e-4 #m^2 / s
kt = 1.4e-7 # m^2 / s
ks = 1.4e-9 # m^2 / s
Cb = 0.011 # kg / m^3 K^2
b0 = 0.77 # kg^2 / m g
g = 9.81 # m / s^2
kappa = 0.598 # kg m / s^3 K
La = 334e3 # m^2 / s^2
delt = 30 # s
rhoi = 916.8 # kg / m^3

def rhots(T,Sk):
    S = Sk /1000
    a1, a2, a3, a4, a5 = 9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8
    b1, b2, b3, b4, b5 = 8.020e2, -2.001, 1.677e-2, -3.060e-5, -1.613e-5
    rho = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4 + b1*S + b2*S*T + b3*S*T**2 + b4*S*T**3 + b5*S**2 * T**2
    return rho

asi = np.linspace(0,35,20)
rhoin, rhowa = rhots(20, asi),  rhots(0, 0) 
gp = g * (rhoin - rhowa) / rhoin
L = 0.35

Ra = np.abs(gp) / (kt * nu)  *  L**3

plt.figure()
plt.plot(asi,Ra,'.-')
plt.yscale('log')
plt.show()


#%%
# devh = hn.reshape( (len(hn), 1024*1024 ) )
# pen2, pen1, ori = np.polyfit(vts, devh, 2)
# pen2, pen1, ori = pen2.reshape((1024,1024)), pen1.reshape((1024,1024)), ori.reshape((1024,1024)) 

plt.figure()
plt.imshow(tini)
plt.show()
plt.figure()
plt.imshow(pen2)
plt.show()
plt.figure()
plt.imshow(pen1)
plt.show()

# tn = 50
# mera = (pen2 *2 * vts[tn] + pen1) * (tini >= vts[tn])
# plt.figure()
# plt.imshow(mera)
# plt.show()

#%%

a = np.array([0,1,2,3,4])
b = np.array([2,4,6,8, np.nan])

np.polyfit(a, b, 2)

#%%
mu = 1.0016 #kg / ms
kt = 1.4e-7 # m^2 / s
ks = 1.4e-9 # m^2 / s
Cb = 0.011 # kg / m^3 K^2
b0 = 0.77 # kg^2 / m g
g = 9.81 # m / s^2
kappa = 0.598 # kg m / s^3 K
La = 334e3 # m^2 / s^2
delt = 30 # s
rhoi = 916.8 # kg / m^3

th0 = 0.1 # m

# Rass,Rats, Nus = [], [], []
# for n in tqdm(range(len(salis))):
n = 10
    
Rat, Ras, Nu1,Nu2 = [],[], [], []

xa,ya = np.arange(1024), -np.arange(1024)
dxyp,dxxp = np.gradient(xrts[n][0] / 1000, ya,xa)
dyyp,dyxp = np.gradient(yrts[n][0] / 1000, ya,xa)
dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0

Ap = np.trapz( np.trapz( dxxp * dyyp, xa, axis=1 ), -ya, axis=0 ) #not sure why -ya but needed for Ap positive
Vp = th0 * Ap
Vn, An = [Vp], [Ap]

for i in range(1,len(ts[n])):
    L = (np.nanmax( yrts[n][i] ) - np.nanmin( yrts[n][i] )) / 1000 # m
    Rat.append( g * Cb * (temps[n])**2 * L**3 / (mu * kt) )
    Ras.append( g * b0 * salis[n] * L**3 / (mu * ks) ) 
    
    # delh = np.nanmean( halgts[n][0] - halgts[n][i] ) / 1000 # m
    # Nu1.append( rhoi * La * delh * L / (delt*i * kappa * temps[n]) )
    
    dxyp,dxxp = np.gradient(xrts[n][i] / 1000, ya,xa)
    dyyp,dyxp = np.gradient(yrts[n][i] / 1000, ya,xa)
    dxxp[np.isnan(dxxp)], dyyp[np.isnan(dyyp)] = 0, 0
    
    delh = ( halgts[n][i-1] - halgts[n][i] ) / 1000
    delh[np.isnan(delh)] = 0
    
    Ap = np.trapz( np.trapz( dxxp * dyyp, xa, axis=1 ), -ya, axis=0 )
    vi1 = np.trapz( np.trapz( delh * dxxp * dyyp, xa, axis=1 ), ya, axis=0 )
    
    Vp = th0 * Ap - 2 * vi1
    Vn.append(Vp)
    An.append(Ap)
    
    Nu1.append( rhoi * (Vn[-2]-Vn[-1]) / An[-1] * La  * L / (delt * kappa * temps[n]) )
    
    
Rats.append(Rat)
Rass.append(Ras)
Nus.append(Nu1)
#%%

plt.figure()
plt.plot(ts[n], np.array(Vn) * 10)
plt.plot(ts[n], An)
plt.show()

plt.figure()
plt.plot(Rat)
# plt.plot(Ras)
plt.show()



#%%
# =============================================================================
# correlations
# =============================================================================
coexs,coeys = [],[]
for i,ss in enumerate(salis):
    
    xa,ya = np.arange(1024), -np.arange(1024)
    gt,gy,gx = np.gradient(halgts[i], ts[i]/60,ya,xa)
    
    dxt,dxyp,dxxp = np.gradient(xrts[i] / 10, ts[i]/60,ya,xa)
    dyt,dyyp,dyxp = np.gradient(yrts[i] / 10, ts[i]/60,ya,xa)
    
    gymas = ma.masked_invalid( gy / dyyp)
    gxmas = ma.masked_invalid( gx / dxxp )
    gtmas = ma.masked_invalid( gt )
    
    coey, coex = [], []
    for n in tqdm(range(len(gt))):
        coey.append(ma.corrcoef( gymas[n].flatten(), gtmas[n].flatten() )[0,1] )
        coex.append(ma.corrcoef( gxmas[n].flatten(), gtmas[n].flatten() )[0,1] )
    coexs.append(coex)
    coeys.append(coey)
#%%
plt.figure()
ccm = []
for i,ss in enumerate(salis):
# for i in [1,12,10]:
    ss = salis[i]
    plt.plot(ts[i]/60 ,coeys[i], '-', c=(salis[i]/30,0,0), label=str(ss)+' g/kg')  #label='dh/dy')
    # plt.plot(ts[i]/60 ,coexs[i], '--', color=colors[i%10] )  #label='dh/dy')
    # plt.plot(t/60, coex, label='dh/dx')
    coei = coeys[i]
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
plt.savefig('./Documents/corrmax.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Correlation graphs
# =============================================================================
n = 12
i = 50

xa,ya = np.arange(1024), -np.arange(1024)
gt,gy,gx = np.gradient(halgts[n], ts[n]/60,ya,xa)

dxt,dxyp,dxxp = np.gradient(xrts[n] , ts[n]/60,ya,xa)
dyt,dyyp,dyxp = np.gradient(yrts[n] , ts[n]/60,ya,xa)

gymas = ma.masked_invalid( gy / dyyp)
gxmas = ma.masked_invalid( gx / dxxp )
gtmas = ma.masked_invalid( gt )
#%%
ylin = 570

plt.figure()
plt.scatter(xrts[n][i], yrts[n][i], c=halgts[n][i], s=0.5, cmap='gray')
plt.plot(xrts[n][i][:,ylin], yrts[n][i][:,ylin], 'r-' )
plt.xlabel('x (mm)', fontsize=12)
plt.ylabel('y (mm)', fontsize=12)
plt.axis([np.nanmin(xrts[n])-5, np.nanmax(xrts[n])+5 , np.nanmin(yrts[n])-5, np.nanmax(yrts[n])+5])
plt.gca().set_aspect("equal")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/profline.png',dpi=400, bbox_inches='tight')
plt.show()


# plt.figure()
# plt.plot(yrts[n][i][:,570], halgts[n][i][:,570],)
# plt.grid()
# plt.show()

# plt.figure()
# plt.plot(yrts[n][i][:,570], gymas[i][:,570] )
# plt.plot(yrts[n][i][:,570], gtmas[i][:,570] )
# # plt.imshow(gtmas[i])
# plt.show()

fig, ax1 = plt.subplots(figsize=(12,5))

color = 'tab:red'
ax1.set_xlabel('y (mm)', fontsize=12)
ax1.set_ylabel('dh/dy', color=color, fontsize=12)
# ax1.plot(y, halg[n,:,500], color=color)
ax1.plot( yrts[n][i][:,ylin], gymas[i][:,ylin], color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('dh/dt (mm/min)', color=color, fontsize=12)  # we already handled the x-label with ax1
ax2.plot( yrts[n][i][:,ylin], gtmas[i][:,ylin], color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

# ax1.invert_xaxis()
ax1.grid()
# plt.savefig('./Documents/gymelt.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Scallops movement graph
# =============================================================================
n = 12
i = 50

ylin = 550

plt.figure()
plt.scatter(xrts[n][i], yrts[n][i], c=difs[n][i], s=0.5, cmap='gray')
plt.plot(xrts[n][i][:,ylin], yrts[n][i][:,ylin], 'r-' )
plt.xlabel('x (mm)', fontsize=12)
plt.ylabel('y (mm)', fontsize=12)
plt.axis([np.nanmin(xrts[n])-5, np.nanmax(xrts[n])+5 , np.nanmin(yrts[n])-5, np.nanmax(yrts[n])+5])
plt.gca().set_aspect("equal")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/profline.png',dpi=400, bbox_inches='tight')
plt.show()


tyl = yrts[n][:,:,ylin]
thl = difs[n][:,:,ylin]

ttl = list(ts[n]) * 1024 
ttl = np.reshape( ttl, (1024,len(ts[n])) )

plt.figure()
# plt.scatter(ttl, tyl, c=thl, s=0.5, cmap='gray')
# plt.xlabel('t (seg)', fontsize=12)
# plt.ylabel('y (mm)', fontsize=12)
# # plt.axis([np.nanmin(xrts[n])-5, np.nanmax(xrts[n])+5 , np.nanmin(yrts[n])-5, np.nanmax(yrts[n])+5])
# # plt.gca().set_aspect("equal")
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.savefig('./Documents/profline.png',dpi=400, bbox_inches='tight')
plt.imshow( thl.T, extent=(0, ts[n][-1]/60, np.nanmin(yrts[n][0][:,ylin]), np.nanmax(yrts[n][0][:,ylin]) ), aspect=1/2, cmap='viridis'  )
plt.ylim(-156,85)

plt.xlabel('t (min)', fontsize=12)
plt.ylabel('y (mm)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/scamove18.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Scallops wavelength max distance
# =============================================================================
# n = 12
plt.figure()

lmeas, lmeds, lstds = [],[],[]
for n in tqdm(range(len(salis))):
    lmea,lmed,lsd = [],[],[]
    for i in range(len(ts[n])):
        lons =[]
        for l in range(1024):
            line = difs[n][i,:,l]
            pek = find_peaks(line, prominence=0.5)[0]
            long = yrts[n][i,:,l][pek][:-1] - yrts[n][i,:,l][pek][1:]
            lons += list(long) 
        lmea.append( np.mean(lons) )
        lmed.append( np.median(lons) )
        lsd.append( np.std(lons) )
    lmeas.append(lmea)
    lmeds.append(lmed)
    lstds.append(lsd)

    plt.errorbar(ts[n]/60, lmea, yerr=lsd, capsize=2, fmt='.')

plt.grid()
plt.ylabel('Wavelenght (mm)')
plt.xlabel('Time (min)')
# plt.savefig('./Documents/wale_23.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
tss,lmm,lss = [],[],[]
# for i,ss in enumerate(salis):
for i,j in enumerate(range(len(salis))):
    
    ss = salis[j]
    
    imi = np.argmin( lmeas[i] )
    
    # print(    ss, ts[j][imi]/60., lmeas[i][imi], lstds[i][imi]    )
    print( '{:<5} {:<4} {:<20} {:<20f}'.format( ss, ts[j][imi]/60., lmeas[i][imi], lstds[i][imi] ) ) 
    # plt.plot( ts[j]/60, lmeas[i], label=ss )
    
    tss.append( ts[j][imi]/60. )
    lmm.append( lmeas[i][imi] )
    lss.append( lstds[i][imi] )
    
dSs_un = np.array( [23.38, 15.20, 4.06, 6.40 , 8.26 , 12.87, 14.80, 18.13, 21.90, 27.39, 10.26, 11.22, 2.31 , 0.00 ] )
soin = np.argsort( dSs_un )
dSs = dSs_un[soin]
lambs = [15.56899116935976, 16.460074303488604, 18.09925904964528, 18.007158732094368, 19.64018037806087, 19.4887995962532, 20.267482589348546,
 20.850794145642915, 18.480350515471226, 19.725783416298484, 18.600138286902368, 22.704439216741143, 18.512399306214324, 7.8048467539888735]

plt.figure()
# plt.errorbar( [salis[l] for l in [0,3,8,1,10,12]] , lmm, yerr=lss, capsize=2, fmt='k.')
plt.errorbar( salis , lmm, yerr=lss, capsize=2, fmt='k.', markersize=10)
plt.plot( dSs, lambs, '.', markersize = 10 )
# plt.legend()
plt.ylim(0,70)
plt.grid()
plt.ylabel('Wavelenght (mm)', fontsize=12)
plt.xlabel('S (g/kg)', fontsize=12)
# plt.xlim(5,25)
plt.savefig('./Documents/com_expins.png',dpi=400, bbox_inches='tight')
plt.show()

plt.figure()
# plt.errorbar( [salis[l] for l in [0,3,8,1,10,12]] , lmm, yerr=lss, capsize=2, fmt='k.')
# plt.plot( salis, tss, 'k.' )
plt.errorbar( salis, tss, yerr=1,  capsize=2, fmt='k.', markersize=10)
# plt.legend()
# plt.ylim(0,40)
plt.grid()
plt.ylabel('Time scallops (min)', fontsize=12)
plt.xlabel('S (g/kg)', fontsize=12)
plt.xlim(5,25)
# plt.savefig('./Documents/scatime.png',dpi=400, bbox_inches='tight')
plt.show()
#%%
# =============================================================================
# Scallops vel
# =============================================================================
delim = 4

def track1d(yys, delim=2 ):
    parts = []
    tiemps = []
    for i in range(len(yys)):
        for j in range(len(yys[i])):
            pac = yys[i][j]
            for k in range(len(parts)):
                ptr = parts[k][-1]
                ttr = tiemps[k][-1]
                if np.abs(ptr - pac) < delim and (i-ttr)==1:
                    parts[k].append(pac)
                    tiemps[k].append(i)
                    break
            else: 
                parts.append([pac])
                tiemps.append([i])
    return parts

# velsam, velsas = [],[]
# for n in tqdm(range(len(salis))):
#     velis = []
#     for l in (range(1024)):
#         ype, tpe = [], []
#         for i in range(len(ts[n])):
#             line = difs[n][i,:,l]
#             pek = find_peaks(line, prominence=0.5)[0]
#             peky = yrts[n][i,:,l][pek]
#             pekt = [i] * len(pek)
#             # ype += list(peky)
#             # tpe += list(pekt)
#             ype.append(peky)
            
#         traye = track1d(ype,delim=2)

#         for j in range(len(traye)):
#             if len(traye[j]) > 19:
#                 tpe = np.arange(len(traye[j])) * 0.5
#                 lr = linregress(tpe, traye[j])
#                 velis.append(lr[0])

#     velsam.append(np.mean(velis))
#     velsas.append(np.std(velis))

plt.figure()
# plt.errorbar( [salis[l] for l in [0,3,8,1,10,12]] , lmm, yerr=lss, capsize=2, fmt='k.')
plt.errorbar( salis , velsam, yerr=velsas, capsize=2, fmt='k.', markersize=10)
# plt.legend()
# plt.ylim(0,70)
plt.grid()
plt.ylabel(r'$v_y$ (mm/min)', fontsize=12)
plt.xlabel('S (g/kg)', fontsize=12)
plt.xlim(5,25)
plt.savefig('./Documents/scavel.png',dpi=400, bbox_inches='tight')
plt.show()



#%%
# =============================================================================
# Curvature maps and PDF
# =============================================================================

curvats = []

for n in tqdm(range(16)):

    nt,ny,nx = np.shape(halgts[n])
    curvs = []

    for i in range(nt):
        xdy, xdx = np.gradient( xrts[n][i], 1,1 )
        ydy, ydx = np.gradient( yrts[n][i], 1,1 )
        xdxy, xdxx = np.gradient( xdx, 1,1 )
        ydyy, ydyx = np.gradient( ydy, 1,1 )
        
        hdy, hdx = np.gradient( halgts[n][i], 1,1  )
        hdyy, hdyx = np.gradient( hdy, 1,1 )
        hdxy, hdxx = np.gradient( hdx, 1,1 )
        
        hy, hx = hdy * ydy**-1, hdx * xdx**-1
        hyy = hdyy * ydy**-2 - hdy * ydyy * ydy**-3
        hxx = hdxx * xdx**-2 - hdx * xdxx * xdx**-3
        hxy = hdxy * ydy**-1 * xdx**-1 - hdy * ydyx * ydy**-2 * xdx**-1
        
        curv = 1/2 * ( (1 + hx**2) * hyy + (1+ hy**2) * hxx - 2 * hx * hy * hxy ) / (1 + hx**2 + hy**2)**(3/2) 
        curvs.append(curv)
    
    curvats.append(curvs)
#%%
n, i = 11, -20
plt.figure()
plt.imshow( halgts[n][i] )
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( curvats[n][i], vmin=-0.2, vmax=0.2 )
plt.colorbar()
plt.show()
#%%
def compg(hist,bine,curv):
    xg = np.linspace(bine[0],bine[-1],1000)
    curvc = curv[~np.isnan(curv)]
    mea,std = np.mean(curvc), np.std(curvc)
    print(mea, std)
    norm, argu = std * np.sqrt(2*np.pi), ((xg-mea)/std)**2
    return xg, 1/norm * np.exp(-1/2 * argu)

n = 4
plt.figure()
cmap,cmm = customcmap(0, 90*0.5)

for i in [10,30,50,70]:
# for i in [90,95,100]:
    curv = curvats[n][i]
    curvl = curv[~np.isnan(curv)]
    mee, sii = np.mean(curvl), np.std(curvl)
    
    # histn, bine = np.histogram( curvl, bins=150, density=True )
    histn, bine = np.histogram( (curvl-mee)/sii , bins=200, density=True )
    binc = (bine[1:] + bine[:-1]) / 2

    # if i == 90: xg, gg = compg(hist,bine,curv)

    plt.plot(binc, histn, label=str(i*0.5)+' min', c=cmap(i/2 / (90*0.5) ) )
    
xg = np.linspace(-10,10,1000)
gg = 1/np.sqrt(2*np.pi) * np.exp(-1/2 * xg**2)
plt.plot(xg, gg, '--')
plt.grid()
plt.legend()
plt.yscale('log')
# plt.xlim(-0.5,0.5)
plt.ylabel('PDF')
plt.xlabel(r'Curvature (mm$^{-1}$)')
plt.ylim(7e-5, 1)
plt.xlim(-7.5,7.5)
# plt.savefig('./Documents/pdfcurv.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
n = 4

nt,_,_ = np.shape(halgts[n])
skw = []
for i in range(nt):
    curv = curvats[n][i]
    cur =  curv[~np.isnan(curv)]
    fi = (cur > -0.4 ) * (cur < 0.4)
    skw.append( skew( cur[fi] ) ) 
    
    
plt.figure()
plt.plot( np.arange(0,nt)*0.5 , skw, '-' )
plt.grid()
plt.ylabel('Skewness')
plt.xlabel('Time (min)')
# plt.savefig('./Documents/skewn.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# Power spectrum
# =============================================================================
from largestinteriorrectangle import lir
# from skimage.transform import warp_polar
from skimage.filters import window
from math import ceil
from matplotlib import colors
from matplotlib.cm import ScalarMappable

def insr(halgts,n, i=-1):
    ha = halgts[n][i]
    bbo = (lir(ha<1e8)).astype('float32')
    bbd,bbl,bbu,bbr = int(bbo[1]+bbo[3]), int(bbo[0]) ,int(bbo[1]), int(bbo[0]+bbo[2])
    return bbu,bbd,bbl,bbr

def welch2d(ha, yh, xh, Ly, Lx, D=0.5, wind=('hann')):
    '''
    D: overlap (between 0 and 1) 
    wind: Tuple detailing the name of desired window and relevant value (if neccesary for the window)    
    
    Possible windows: hannig, hamming, blackman, kaiser, tukey 
    kasiser and tukey windows need the extra parameter

    Recommended: 
        wind = ('hann')
        D = 0.5
    '''
    
    ny, nx  = np.shape(ha)

    My, Mx = int( ny / (Ly + D*(1-Ly)) ), int( nx / (Lx + D*(1-Lx)) )
    ovy, ovx = ceil(My*D), ceil(Mx*D)

    haw = []
    for u in range(Ly):
        for v in range(Lx):
            haw.append( ha[ u*My - u*ovy : (u+1)*My - u*ovy, v*Mx - v*ovx : (v+1)*Mx - v*ovx ] )
            
    dyt = yh[:-1] - yh[1:]
    dy = np.nanmean(dyt)  #the average distance (in mm) of pixels in y direc (varies in 0.3% only)
    dxt = xh[:,1:] - xh[:,:-1]
    dx = np.nanmean(dxt)  #analogous to y direc        
    
    kfs, sfs = [], []
    for u in range(len(haw)):
        
        hw = haw[u] * window(wind, np.shape(haw[u]) )
        
        hwf = np.fft.fftshift( np.fft.fft2(hw) )
        ky = np.fft.fftfreq( np.shape(haw[u])[0], d=dy )
        kx = np.fft.fftfreq( np.shape(haw[u])[1], d=dx )
        kx,ky = np.meshgrid(kx,ky)
        kx,ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
        k = np.sqrt(ky**2 + kx**2)
        
        kfl, sfl = k.flatten(), np.abs(hwf.flatten())**2
        kfs.append(kfl)
        sfs.append(sfl)

    ks,sf = kfs[0], np.mean(sfs,axis=0)
            
    return ks, sf

def welch2d_2(ha, yh, xh, Ly, Lx, D=0.5, wind=('hann')):
    '''
    D: overlap (between 0 and 1) 
    wind: Tuple detailing the name of desired window and relevant value (if neccesary for the window)    
    
    Possible windows: hannig, hamming, blackman, kaiser, tukey 
    kasiser and tukey windows need the extra parameter

    Recommended: 
        wind = ('hann')
        D = 0.5
    '''
    
    ny, nx  = np.shape(ha)

    My, Mx = int( ny / (Ly + D*(1-Ly)) ), int( nx / (Lx + D*(1-Lx)) )
    ovy, ovx = ceil(My*D), ceil(Mx*D)

    haw = []
    for u in range(Ly):
        for v in range(Lx):
            haw.append( ha[ u*My - u*ovy : (u+1)*My - u*ovy, v*Mx - v*ovx : (v+1)*Mx - v*ovx ] )
            
    dyt = yh[:-1] - yh[1:]
    dy = np.nanmean(dyt)  #the average distance (in mm) of pixels in y direc (varies in 0.3% only)
    dxt = xh[:,1:] - xh[:,:-1]
    dx = np.nanmean(dxt)  #analogous to y direc        
    
    kfs, tfs, sfs = [], [], []
    for u in range(len(haw)):
        
        hw = haw[u] * window(wind, np.shape(haw[u]) )
        
        hwf = np.fft.fftshift( np.fft.fft2(hw) )
        ky = np.fft.fftfreq( np.shape(haw[u])[0], d=dy )
        kx = np.fft.fftfreq( np.shape(haw[u])[1], d=dx )
        kx,ky = np.meshgrid(kx,ky)
        kx,ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
        
        k = np.sqrt(ky**2 + kx**2)
        the = np.arctan2(ky,kx)
        Sn = np.abs( hwf )**2
                
        kfs.append(k)
        tfs.append(the)
        sfs.append(Sn)

    # ks,sf = kfs[0], np.mean(sfs,axis=0)
    return kfs, tfs, sfs

def customcmap(vmin,vmax):
    cdict = {'red':   [(0.0,  0.0, 1.0),
                       (0.5,  0.5, 0.5),
                       (1.0,  0.0, 1.0)],

             'green': [(0.0,  0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (1.0,  0.0, 0.0)],

             'blue':  [(0.0,  0.0, 0.0),
                       (0.5,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}

    cmap = colors.LinearSegmentedColormap('custom', cdict)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmm = ScalarMappable(norm=norm, cmap=cmap)
    return cmap, cmm

def polyfit2(halg,mmmm,xr,yr, order=2):
    haltura = (halg*mmmm)
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[~np.isnan(haltura)], yr[~np.isnan(haltura)]
    
    poli = []
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: poli.append( xfit**i * yfit**j )
    A = np.array(poli).T
    # print(len(poli))
    # A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol22(coeff,xr,yr,order=2):
    poo, cof = 0, 0
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: 
                poo += coeff[cof] * xr**i * yr**j
                cof += 1
    return poo

def polyfit(n,halg,mmmm,xr,yr, order=2):
    haltura = (halg*mmmm)[n]
    hfit,xfit,yfit = haltura[~np.isnan(haltura)], xr[n][~np.isnan(haltura)], yr[n][~np.isnan(haltura)]
    
    poli = []
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: poli.append( xfit**i * yfit**j )
    A = np.array(poli).T
            
    # A = np.array([xfit*0+1,xfit,yfit,xfit**2,xfit*yfit,yfit**2]).T
    
    coeff, r, rank, s = np.linalg.lstsq(A, hfit, rcond=None)
    return coeff, r, rank, s  

def pol2(coeff,xr,yr,n,order=2):
    poo, cof = 0, 0
    for i in range(order+1):
        for j in range(order+1):
            if i+j<=order: 
                poo += coeff[cof] * xr[n]**i * yr[n]**j
                cof += 1
    return poo

#%%
n = 12
i = 60

bbu,bbd,bbl,bbr = insr(halgts, n, i=-2)
ha = halgts[n][i][bbu:bbd,bbl:bbr]

yh = yrts[n][i]
dyt = yh[:-1] - yh[1:]
dy = np.nanmean(dyt)  #the average distance (in mm) of pixels in y direc (varies in 0.3% only)

xh = xrts[n][i]
dxt = xh[:,1:] - xh[:,:-1]
dx = np.nanmean(dxt)  #analogous to y direc

hf = np.fft.fftshift( np.fft.fft2(ha) )
ky = np.fft.fftfreq( np.shape(ha)[0], d=dy  )
kx = np.fft.fftfreq( np.shape(ha)[1], d=dx  )
kx,ky = np.meshgrid(kx,ky)
kx,ky = np.fft.fftshift(kx), np.fft.fftshift(ky)
k = np.sqrt(ky**2 + kx**2)

# hfp = warp_polar(np.abs(hf))


plt.figure()
plt.imshow( ha )
plt.show()
plt.figure()
plt.imshow( k )
plt.show()
plt.figure()
plt.imshow( np.log(np.abs(hf)) )
plt.show()
#%%
lar = 200
kb = np.linspace(0, np.max(k), lar)
dk = kb[1]-kb[0]

sms = []
for i in range(lar):
    filt = (k < kb[i]+dk/2) * (k > kb[i]-dk/2)
    sm = np.mean( np.abs(hf[filt])**2 )
    sms.append(sm)


plt.figure()
plt.plot(kb[1:], sms[1:], '.-')
plt.plot(kb[1:], kb[1:]**-4 * 300, 'g-' )
plt.plot(kb[1:], kb[1:]**-3 * 8000, 'r-')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

# plt.figure()
# plt.scatter(k, np.abs(hf)**2  )
# plt.plot(k, k**-4 * algo, 'b-')
# plt.yscale('log')
# # plt.xscale('log')
# plt.grid()
# plt.show()


# plt.figure()
# plt.imshow(k)
# plt.show()
# plt.figure()
# plt.imshow(filt)
# plt.show()

#%%
n = 0
i = 60

bbu,bbd,bbl,bbr = insr(halgts, n, i=-2)
ha = halgts[n][i][bbu:bbd,bbl:bbr]

yh = yrts[n][i]
dyt = yh[:-1] - yh[1:]
dy = np.nanmean(dyt)  #the average distance (in mm) of pixels in y direc (varies in 0.3% only)

xh = xrts[n][i]
dxt = xh[:,1:] - xh[:,:-1]
dx = np.nanmean(dxt)  #analogous to y direc

hfy = np.fft.fftshift( np.fft.fft(ha, axis=0), axes=0 )
hfym = np.mean(np.abs(hfy)**2, axis=1)
ky = np.fft.fftshift( np.fft.fftfreq( np.shape(ha)[0], d=dy ) )

hfx = np.fft.fftshift( np.fft.fft(ha, axis=1), axes=1 )
hfxm = np.mean(np.abs(hfx)**2, axis=0)
kx = np.fft.fftshift( np.fft.fftfreq( np.shape(ha)[1], d=dx ) )


plt.figure()
# plt.plot(ky, np.abs(hfy)**2 )
plt.plot(ky, hfym)
plt.plot(kx, hfxm)

plt.plot(ky, ky**-2  )

plt.yscale('log')
plt.xscale('log')
plt.show()

#%%
n = 10
i = 30

hag = halgts[n][i]
xr, yr = xrts[n][i], yrts[n][i]
mms = hag > -1e5

coef, a1,a2,a3 = polyfit2( hag, mms, xr, yr, order=2)
poh = pol22(coef,xr,yr,order=2)


# print(coef1 - coef2)

plt.figure()
plt.imshow(hag)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(poh)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(hag - poh)
plt.colorbar()
plt.show()

#%%

#%%
# =============================================================================
# power law with welch 
# =============================================================================
n = 11
nt = np.shape( halgts[n] )[0] - 1

plt.figure()
cmap,cmm = customcmap(0, nt*0.5)

for i in tqdm(range(0,nt,5)):
    bbu,bbd,bbl,bbr = insr(halgts, n, i=i)
    hao = halgts[n][i][bbu:bbd,bbl:bbr]
    yh = yrts[n][i][bbu:bbd,bbl:bbr]
    xh = xrts[n][i][bbu:bbd,bbl:bbr]
    
    mms = hao > -1e5

    coef, a1,a2,a3 = polyfit2( hao, mms, xh, yh, order=2)
    poh = pol22(coef,xh,yh,order=2)
    ha = hao - poh
    
    D = 0.5
    ks, sf = welch2d(ha,yh,xh, 3, 2)
    
    lar = 250
    kb = np.linspace(0, np.max(ks), lar)
    dk = kb[1]-kb[0]
    
    sms = []
    for j in range(lar):
        filt = (ks < kb[j]+dk/2) * (ks > kb[j]-dk/2)
        sm = np.mean( sf[filt] )
        sms.append(sm)

    # plt.plot(ks,sf,'.')
    plt.plot(kb,sms,'-', c=cmap(i/nt))
    
plt.plot(kb[:55],kb[:55]**-4*1e-1,'--')
# plt.plot(kb[:55],kb[:55]**-5*1e-2,'--')
plt.grid()
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.colorbar(cmm)
plt.show()

#%%
n = 3
nt = np.shape( halgts[n] )[0] - 1

plt.figure()
cmap,cmm = customcmap(0, nt*0.5)

for i in tqdm([0,25,55]): # tqdm(range(0,nt,5)):
    bbu,bbd,bbl,bbr = insr(halgts, n, i=i)
    hao = halgts[n][i][bbu:bbd,bbl:bbr]
    yh = yrts[n][i][bbu:bbd,bbl:bbr]
    xh = xrts[n][i][bbu:bbd,bbl:bbr]
    
    mms = hao > -1e5

    coef, a1,a2,a3 = polyfit2( hao, mms, xh, yh, order=2)
    poh = pol22(coef,xh,yh,order=2)
    ha = hao - poh

    kfs, tfs, sfs = welch2d_2(ha,yh,xh, 3, 2)   
    
    Sns, sks = [], []
    for l in range(len(kfs)):
    
        indf = np.argsort(kfs[l].flatten())
        flk = (kfs[l].flatten())[indf]
        flt = (tfs[l].flatten())[indf]
        fls = (sfs[l].flatten())[indf]
    
        ints, kss = [], []
        kig, tig, sig = [flk[0]], [flt[0]], [fls[0]]
        for j in range(1,len(flk)):
            if flk[j] == kig[-1]:
                kig.append(flk[j])
                tig.append(flt[j])
                sig.append(fls[j])
            else:
                indt = np.argsort(tig)
                ski, sti, ssi = np.array(kig)[indt], np.array(tig)[indt], np.array(sig)[indt]
                
                ints.append( np.trapz(ssi * ski, sti) )
                kss.append( ski[-1] )
                
                kig, tig, sig = [flk[j]], [flt[j]], [fls[j]]
                
        Sns.append(ints)
        sks.append(kss)
        
    ks, Sn = np.array(sks[0]), np.mean(Sns, axis=0)
    
    lar = 250
    kb = np.linspace(0, np.max(ks), lar)
    dk = kb[1]-kb[0]
    
    sms = []
    for j in range(lar):
        filt = (ks < kb[j]+dk/2) * (ks > kb[j]-dk/2)
        sm = np.median( Sn[filt] )
        sms.append(sm)

    # plt.plot(ks,sf,'.')
    # plt.plot(ks / 2/np.pi,Sn,'.', c=cmap(i/nt))
    plt.plot(kb /2/np.pi,sms,'-', c=cmap(i/nt), label=str(i/2)+' min')
    
plt.plot(kb[8:80] / 2/np.pi ,(kb[8:80]/ 2/np.pi)**-4*1e-2,'--', label=r'k$^{-4}$')
plt.grid()
plt.xscale('log')
plt.yscale('log')
# cba = plt.colorbar(cmm)
# cba.set_label('Time (min)', rotation=270)
plt.xlabel(r'$k/2\pi$ (mm$^{-1}$)')
plt.ylabel(r'$S_{\eta}$ (mm$^3$) ')
# plt.savefig('./Documents/powspec.png',dpi=400, bbox_inches='tight')
plt.legend()
plt.show()

#%%

a = np.array([0,1,3])
a[[0]]

#%%

#posibles windows:
# rectangular
# triangular
# hannig
# hamming
# blackman
# kaiser
# tukey (agregue yo)
# gaussian (agregue yo)

# plt.figure()
# plt.imshow(window(('hann'), np.shape(haw[u]) ))
# plt.show()

#%%
# =============================================================================
# Normal melt rate
# =============================================================================
def normelt(xr1,yr1,h1, xr2,yr2,h2, ven=20 ):
    ny,nx = np.shape(h1)

    xdy, xdx = np.gradient( xr1, 1,1 )
    ydy, ydx = np.gradient( yr1, 1,1 )
    hdy, hdx = np.gradient( h1, 1,1  )
    
    hy, hx = hdy * ydy**-1, hdx * xdx**-1
    
    mes = np.zeros_like(h1)
    t21, t32, t43 = 0,0,0
    for i in tqdm(range(nx)):
        for j in range(ny):
            
            if np.isnan(h1[j,i]): continue
            
            t1 = time()
            
            xv, yv, hv = xr2[j-ven:j+ven, i-ven:i+ven], yr2[j-ven:j+ven, i-ven:i+ven], h2[j-ven:j+ven, i-ven:i+ven]
    
            resth = h1[j,i] - hv        
            dist2 = (xv - xr1[j,i] - hx[j,i] * (resth))**2 + (yv - yr1[j,i] - hy[j,i] * (resth))**2
            
            t2 = time()
            t21 += t2-t1
    
            jmi, imi = np.unravel_index(dist2.argmin(), dist2.shape)
            jmi = jmi + j-ven
            imi = imi + i-ven
    
            t3 = time()
            t32 += t3-t2
    
            mera = np.sqrt( (xr1[j,i] - xr2[jmi,imi])**2 + (yr1[j,i] - yr2[jmi,imi])**2 + (h1[j,i] - h2[jmi,imi])**2 )
            mes[j,i] = mera
    
            t4 = time()
            t43 += t4-t3

    print('\n',t21, t32, t43)
    return mes


#%%
n = -2
xa,ya = np.arange(1024), -np.arange(1024)
gt,gy,gx = np.gradient(halgts[n], ts[n]/60,ya,xa)

dxt,dxyp,dxxp = np.gradient(xrts[n] , ts[n]/60,ya,xa)
dyt,dyyp,dyxp = np.gradient(yrts[n] , ts[n]/60,ya,xa)

gymas = gy / dyyp
gxmas = gx / dxxp 
gtmas = gt 
#%%
i = 70

xr1,yr1,h1 = xrts[n][i], yrts[n][i], halgts[n][i]
xr2,yr2,h2 = xrts[n][i+1], yrts[n][i+1], halgts[n][i+1]

mera = normelt(xr1,yr1,h1, xr2,yr2,h2, ven=20 ) / 0.5
#%%
mera[ mera==0 ] = np.nan

plt.figure()
plt.imshow( h1 )
plt.colorbar()
plt.show()        

plt.figure()
plt.imshow( -gtmas[i] )
plt.colorbar()
plt.show()        

plt.figure()
plt.imshow( mera, vmax = 1.6, vmin=0.6 )
plt.colorbar()
plt.show()      

plt.figure()
plt.imshow( -gtmas[i] - mera, vmin=-0.1, vmax=0.4)
plt.colorbar()
plt.show()        
#%%
plt.figure()
plt.hist( (-gtmas[i]).flatten(), bins=50)
plt.show()
plt.figure()
plt.hist( mera.flatten(), bins=50)
plt.show()
plt.figure()
plt.hist( (-gtmas[i] - mera).flatten(), bins=50)
plt.show()
#%%
# plt.figure()
# plt.plot( xr1[500,:], h1[500,:] )
# plt.plot( xr2[500,:], h2[500,:] )
# plt.grid()
# plt.show()    

plt.figure()
# plt.plot( xr1[500,:], h1[500,:] )
# plt.plot( xr1[500,:], -gtmas[i][500,:] /2  )
# plt.plot( xr2[500,:], mera[500,:] /2 * 10 )
plt.grid()
plt.show()    

plt.figure()
plt.plot( yr1[:,500], gymas[i][:,500]  )
plt.plot( yr1[:,500], -gtmas[i][:,500] /2  )
plt.plot( yr1[:,500], mera[:,500] /2  )
plt.grid()
plt.show()
#%%
x,dx = np.linspace(-10,10,1000, retstep=True)
y,dy = np.linspace(-10,10,1000, retstep=True)
x, y = np.meshgrid(x,y)

h1 = 5 * np.cos(x * y / 20)
hy, hx = np.gradient(h1, dy,dx)
norm = np.array([hx, hy, -np.ones_like(hx)])

h2 = 5 * np.cos(x * y / 20) - 0.5

ven = 15
mes = np.zeros_like(h1)
t21, t32, t43 = 0,0,0
for i in tqdm(range(51,949)):
    for j in range(51,949):

        t1 = time()
        
        xv, yv, hv = x[j-ven:j+ven, i-ven:i+ven], y[j-ven:j+ven, i-ven:i+ven], h2[j-ven:j+ven, i-ven:i+ven]

        resth = h1[j,i] - hv        
        dist2 = (xv - x[j,i] - hx[j,i] * (resth))**2 + (yv - y[j,i] - hy[j,i] * (resth))**2
        
        t2 = time()
        t21 += t2-t1

        jmi, imi = np.unravel_index(dist2.argmin(), dist2.shape)
        jmi = jmi + j-ven
        imi = imi + i-ven

        t3 = time()
        t32 += t3-t2

        mera = np.sqrt( (x[j,i] - x[jmi,imi])**2 + (y[j,i] - y[jmi,imi])**2 + (h1[j,i] - h2[jmi,imi])**2 )
        mes[j,i] = mera

        t4 = time()
        t43 += t4-t3

print(imi,jmi)
print(t21, t32, t43)
#%%

for i in range(10):
    # print(i)
    if i%2 == 0: continue
    print(i)



#%%
# =============================================================================
# other thing
# =============================================================================
te = np.linspace(0, 25, 1000)
a1 = -3.983035
a2 = 301.797
a3 = 522528.9
a4 = 69.34881
a5 = 999.974950


rho = a5 * (1 - (te + a1)**2 * (te+a2) / (a3*(te+a4)) )

plt.figure()
plt.plot( te, rho )
plt.grid()
plt.xlabel('Temperature (°C)',fontsize=12)
plt.ylabel(r'Density (kg m$^-3$)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('./Documents/densano.png',dpi=400, bbox_inches='tight')
plt.show()

#%%
from matplotlib import colors
from matplotlib.cm import ScalarMappable
fig, ax = plt.subplots(figsize=(6, 6))

cdict = {'red':   [(0.0,  0.0, 1.0),
                   (0.5,  0.5, 0.5),
                   (1.0,  0.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0,  0.0, 0.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.0, 0.0)]}

cmap = colors.LinearSegmentedColormap('custom', cdict)
norm = colors.Normalize(vmin=0, vmax=4)
cmm = ScalarMappable(norm=norm, cmap=cmap)

for i in np.linspace(0, 4,10):
    # Plot 50 lines, from y = 0 to y = 1, taking a corresponding value from the cmap
    ax.plot([-1, 1], [i, i], c=cmap(i/4), label=i)

plt.colorbar(cmm)
plt.show()

#%%






 