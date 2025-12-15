#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:48:51 2025

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt


import imageio.v2 as imageio
from tqdm import tqdm
from time import time

from scipy.optimize import least_squares
from scipy.stats import linregress

from skimage import io
from skimage.filters import gaussian, roberts, frangi, sato
from skimage.morphology import remove_small_objects, binary_dilation, disk, skeletonize, binary_closing, binary_erosion
from skimage.measure import label, regionprops
from skimage.segmentation import felzenszwalb, mark_boundaries

#%%
def order_points( centsx, centsy, max_iter=100 ):
    puntos = np.hstack((np.array([centsx]).T, np.array([centsy]).T, np.array([[0]*len(centsy)]).T ))
    mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
    or_puntos = np.empty((0,3), int)

    itere = 0
    while len(puntos) > 0 or itere > max_iter:
        dists = []
        p1, p2 = puntos[mima][0], puntos[mima][1] 
        for i in range(len(puntos)):
            p3 = puntos[i]
            dist = np.linalg.norm( np.cross(p2-p1, p1-p3) ) / np.linalg.norm(p2-p1)
            dists.append(dist)
        dists = np.array(dists)
        fil = dists < 20
        orde = np.argsort( puntos[fil][:,0] )
        
        or_puntos = np.vstack((or_puntos, puntos[fil][orde])) 
        
        puntos = puntos[~fil]
        itere += 1

    if itere > max_iter: print('Maximum iterations reached')
    return or_puntos

def calibration_first(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] 
    ytop = v[3]*x + v[4]*y + v[5] 
    bot = v[6]*x + v[7]*y + 1 

    X,Y = xtop / bot, ytop / bot
    return X,Y

def calibration_second(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] + v[3]*x**2 +  v[4]*y**2 +  v[5]*x*y
    ytop = v[6]*x + v[7]*y + v[8] + v[9]*x**2 + v[10]*y**2 + v[11]*x*y
    bot = v[12]*x + v[13]*y + 1 + v[14]*x**2 + v[15]*y**2 + v[16]*x*y

    X,Y = xtop / bot, ytop / bot
    return X,Y
#%%
# =============================================================================
# Calibration
# =============================================================================
t1 = time()

cal = imageio.imread('/Volumes/Ice blocks/Test channel laser/25-03-12/DSC_3206.JPG' )[40:5475,2460:2940,1]
calg =  gaussian(cal,2) - gaussian(cal,30)
caln = (calg - np.min(calg)) / (np.max(calg)- np.min(calg))
calb = caln < 0.5
calr = binary_dilation( remove_small_objects(calb, min_size=500), disk(20) ) 
cals = remove_small_objects(calb, min_size=25)
cald = (calr + cals)*1. - remove_small_objects(calr + cals, min_size=300)*1.
labc = label( cald > 0 )
props = regionprops(labc)

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 40
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 2460

sor_points = order_points(centsx, centsy) # en px
spx, spy = sor_points[:,0], sor_points[:,1]

nx,ny = 25, 77

px, py =  np.array([i%nx for i in range(nx*ny)]), np.array([int(i/nx) for i in range(nx*ny)])[::-1]
px, py = px*3, (np.max(py)-py)*6 # en mm

t2 = time()
t2-t1

#%%
t1 = time()

def residual_first(v):
    xtop = v[0]*spx + v[1]*spy + v[2]
    ytop = v[3]*spx + v[4]*spy + v[5]
    bot = v[6]*spx + v[7]*spy + 1 

    X,Y = xtop / bot, ytop / bot
    return (X - px)**2 + (Y - py)**2

ls1 = least_squares(residual_first , [1,1,1,1,1,1,0,0], method='lm')

def residual_second(v):
    xtop = v[0]*spx + v[1]*spy + v[2] + v[3]*spx**2 +  v[4]*spy**2 +  v[5]*spx*spy
    ytop = v[6]*spx + v[7]*spy + v[8] + v[9]*spx**2 + v[10]*spy**2 + v[11]*spx*spy
    bot = v[12]*spx + v[13]*spy + 1 + v[14]*spx**2 + v[15]*spy**2 + v[16]*spx*spy

    X,Y = xtop / bot, ytop / bot
    return (X - px)**2 + (Y - py)**2

ls0 = [0.]*17
ls0[:3], ls0[6:9], ls0[12:14] = ls1.x[:3], ls1.x[3:6], ls1.x[6:]

ls2 = least_squares(residual_second, ls0, method='lm')

t2 = time()
t2-t1, ls1, ls2

#%%
cals = imageio.imread('/Volumes/Ice blocks/Test channel laser/25-03-12/DSC_3206.JPG' ) #[40:5475,2460:2940,1]


plt.figure()
plt.imshow(cal, cmap='gray')
# plt.plot( spx, spy, 'r.', markersize=1 )
# plt.plot( px, py, '.' )
# plt.plot( X, Y, '.' )
plt.xlabel('x (px)')
plt.ylabel('y (px)')
# plt.gca().invert_yaxis()
# plt.axis('equal')
# plt.savefig('./Documents/im1.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

plt.figure()
plt.imshow(cals, cmap='gray')
plt.plot( spx, spy, 'r.', markersize=1 )
# plt.plot( px, py, '.' )
# plt.plot( X, Y, '.' )
plt.xlabel('x (px)')
plt.ylabel('y (px)')
# plt.gca().invert_yaxis()
# plt.axis('equal')
# plt.savefig('./Documents/im2.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

X,Y = calibration_first(ls1.x, spx, spy)

plt.figure()
# plt.plot( spx, spy, '.' )
plt.plot( px, py, '.' )
plt.plot( X, Y, '.' )
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
# plt.savefig('./Documents/im3.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

X,Y = calibration_second(ls2.x, spx, spy)

plt.figure()
# plt.plot( spx, spy, '.' )
plt.plot( px, py, '.' )
plt.plot( X, Y, '.' )
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
# plt.savefig('./Documents/im4.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()
#%%
# =============================================================================
# Laser end
# =============================================================================
xis, yis = [], []
for n in tqdm(range(3211,3295)):

    im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n)+'.JPG' )[1100:3500,2700:3250,1] 
    # im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3211.JPG' )[1100:3500,2700:3250,1] 
    # im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3289.JPG')[1100:3500,2700:3250,1] 
    
    img = gaussian(im,2)
    fim = frangi(np.gradient(img,axis=1))
    imb = (fim / np.max(fim)) > 0.03 
    
    if n not in 3211 + np.array([33,47,48,49,58,70,80,81,82]): imr = remove_small_objects(imb,min_size=100)
    elif n in 3211 + np.array([48,49,58,80,81,82]): imr = remove_small_objects(imb,min_size=200)
    elif n in 3211 + np.array([33,47,70]): imr = remove_small_objects(imb,min_size=300)
        
    pop = np.where(imr)
    xma,xmi = np.max(pop[1]), np.min(pop[1])
    yma,ymi = np.max(pop[0]), np.min(pop[0])
    
    dif = 60
    # omc = binary_closing(imr , disk(40))
    omc = binary_closing(imr[ymi-dif:yma+dif,xmi-dif:xma+dif] , disk(dif-20))
    ske = skeletonize(omc)
    
    xis.append( np.where(ske)[1] + xmi - dif )
    yis.append( np.where(ske)[0] + ymi - dif )

t2 = time()
t2-t1
#%%

n = 80
im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n+3211)+'.JPG' )[1100:3500,2200:3250,:]

plt.figure()
plt.imshow( im  )
# plt.plot( xis[n] + 500, yis[n], 'b-' )
# plt.savefig('./Documents/im7.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# plt.figure()
# for n in range(len(xis)):
# # for n in range(80,84):
#     plt.plot( xis[n], yis[n], '-', label=n, color=(n/84,0,1-n/84) )
    
#     # X,Y = calibration_second(ls2.x, xis[n], yis[n])
#     # plt.plot( X, Y, '-', label=n, color=(n/84,0,1-n/84) )
    
# # plt.legend()
# plt.gca().invert_yaxis()
# # plt.axis('equal')
# plt.xlabel('x (px)')
# plt.ylabel('y (px)')
# # plt.savefig('./Documents/im9.png',dpi=400, bbox_inches='tight', transparent=False)
# plt.show()


x14 = []
for n in range(len(xis)):
    inn = np.where(yis[n] == 1300)
    X,Y = calibration_second(ls2.x, xis[n], yis[n])
    # x14.append( xis[n][inn] )
    x14.append( X[inn] )
    

plt.figure()
# plt.plot(x14, np.arange(len(x14))/2, '.-')
plt.plot(np.arange(len(x14))/2, x14, '.-')
plt.ylabel('x (px)')
plt.xlabel('t (min)')
# plt.gca().invert_yaxis()
# plt.axis('equal')
# plt.savefig('./Documents/im10.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


#%%
t1 = time()
xas, yas = [], []
for n in tqdm(range(3211,3295)):

    im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n)+'.JPG' )[1100:3500,2700:3250,1] 
    # im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3211.JPG' )[1100:3500,2700:3250,1] 
    # im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3289.JPG')[1100:3500,2700:3250,1] 
    
    img = gaussian(im,5)
    imga = np.gradient(img,axis=1)
    
    xpi, ypi = np.argmin( imga, axis=1 ), np.arange(np.shape(imga)[0]) 
    imm = np.zeros_like(imga)
    imm[ypi,xpi] = 1
    imm = remove_small_objects(imm>0,connectivity=2,min_size=100)
    miy,may = np.min(np.where(imm)[0]), np.max(np.where(imm)[0])
    
    lenx = np.shape(imga)[1]
    in_n, in_p, in_ = np.abs(imga[ypi, xpi-1]), np.abs(imga[ypi, (xpi+1)%lenx]), np.abs(imga[ypi, xpi])
    xav = xpi + 0.5 * ( np.log(in_n) - np.log(in_p) ) / ( np.log(in_n) + np.log(in_p) - 2*np.log(in_))
    xav, yav = xav[miy:may], ypi[miy:may]
    
    xas.append( xav )
    yas.append( yav )

t2 = time()
t2-t1
#%%
n = 80
im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n+3211)+'.JPG' )[1100:3500,2700:3250,:]

# plt.figure()
# plt.imshow( im  )
# plt.plot( xas[n], yas[n], 'b-' )
# # plt.savefig('./Documents/im7.png',dpi=400, bbox_inches='tight', transparent=False)
# plt.show()

plt.figure()
for n in range(len(xis)):
# for n in range(80,84):
    # plt.plot( xas[n], yas[n], '-', label=n, color=(n/84,0,1-n/84) )
    
    X,Y = calibration_second(ls2.x, xas[n], yas[n])
    plt.plot( X, Y, '-', label=n, color=(n/84,0,1-n/84) )
    
# plt.legend()
plt.gca().invert_yaxis()
# plt.axis('equal')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
# plt.savefig('./Documents/profiles.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


x14 = []
for n in range(len(xas)):
    inn = np.where(yas[n] == 1300)
    X,Y = calibration_second(ls2.x, xas[n], yas[n])
    # x14.append( xis[n][inn] )
    x14.append( X[inn] )
    

plt.figure()
# plt.plot(x14, np.arange(len(x14))/2, '.-')
plt.plot(np.arange(len(x14))/2, x14, '.-')
plt.ylabel('x (mm)')
plt.xlabel('t (min)')
# plt.gca().invert_yaxis()
# plt.axis('equal')
# plt.savefig('./Documents/timeev.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

#%%

n = 0
im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n+3211)+'.JPG' )[1100:3500,2200:3250,:]

plt.figure()
plt.imshow( im  )
plt.axis('off')
plt.savefig('./Documents/im0.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

plt.figure()
plt.imshow( im  )
plt.plot(xas[n]+500,yas[n],'b-')
plt.axis('off')
plt.savefig('./Documents/im0_p.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


#%%

n = 3211 + 1

im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_'+str(n)+'.JPG' )[1100:3500,2700:3250,1] 
# im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3211.JPG' )[1100:3500,2700:3250,1] 
# im = imageio.imread('/Volumes/Ice blocks/Test channel laser/12-03-25/DSC_3289.JPG')[1100:3500,2700:3250,1] 

img = gaussian(im,5)
imga = np.gradient(img,axis=1)

xpi, ypi = np.argmin( imga, axis=1 ), np.arange(np.shape(imga)[0]) 
imm = np.zeros_like(imga)
imm[ypi,xpi] = 1
imm = remove_small_objects(imm>0,connectivity=2,min_size=100)
miy,may = np.min(np.where(imm)[0]), np.max(np.where(imm)[0])

lenx = np.shape(imga)[1]
in_n, in_p, in_ = np.abs(imga[ypi, xpi-1]), np.abs(imga[ypi, (xpi+1)%lenx]), np.abs(imga[ypi, xpi])
xav = xpi + 0.5 * ( np.log(in_n) - np.log(in_p) ) / ( np.log(in_n) + np.log(in_p) - 2*np.log(in_))
xav, yav = xav[miy:may], ypi[miy:may]


plt.figure()
plt.imshow( imm )
plt.show()
plt.figure()
plt.imshow( imga )
plt.plot( xpi, ypi, 'r.' )
plt.show()

# plt.figure()
# plt.plot( imga[600,:], '.-' )
# plt.show()

plt.figure()
plt.plot( xpi, ypi, 'r-' )
plt.plot( xav, yav, 'b-' )
plt.show()

#%%
# =============================================================================
# Water channel parameters (temperature and velocity)
# =============================================================================

data = np.loadtxt('/Volumes/Ice blocks/Test channel laser/12-03-25/250312_140133.txt', skiprows=9, usecols=(0,2,3,4,5,6,19,20,21,22,23))

fin = 24450 
t = data[:fin,0]
vel = data[:fin,2]
puper = data[:fin,3]
tamb = data[:fin,6]
tin = data[:fin,7]
tout = data[:fin,8]
settchil = data[:fin,9]
readtchil = data[:fin,10]


plt.figure()
plt.plot( t/60, tin, '-', label=r'$T_{in}$' )
plt.plot( t/60, tout, '-', label=r'$T_{out}$' )
# plt.plot( t/60, tamb, '-', label=r'$T_{amb}$' )
plt.legend()
plt.xlabel('t (min)')
plt.ylabel('T (°C)')
plt.savefig('./Documents/tempcha.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()


plt.figure()
plt.plot( t/60, vel, '-', label=r'$v$' )
# plt.plot( t/60, puper, '-', label=r'% pump' )
# plt.legend()
plt.xlabel('t (min)')
plt.ylabel('v (cm/s)')
plt.savefig('./Documents/velcha.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()





#%%
from scipy.ndimage import rotate
from skimage.color import rgb2gray
#%%
# =============================================================================
# Scan?
# =============================================================================


# vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/20-05-25/DSC_4488.MP4', 'ffmpeg') # 1599 last frame
# vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/22-05-25/DSC_4491.MP4', 'ffmpeg') # 1599 last frame
# vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/22-05-25/DSC_4489.MP4', 'ffmpeg') # 1599 last frame
# vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/22-05-25/DSC_4492.MP4', 'ffmpeg') # 1599 last frame
# vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/22-05-25/DSC_4493.MP4', 'ffmpeg') # 1599 last frame

vid = imageio.get_reader('/Volumes/Ice blocks/Test channel laser/22-05-25/DSC_4499.MP4', 'ffmpeg') # 1599 last frame

#%%
# for i in range(20,40):
#     im = vid.get_data(i)
#     plt.figure()
#     plt.imshow( im )
#     plt.title(i)
#     plt.show()
    
ims = []
for i in tqdm(range(5)):
    im = vid.get_data(i)[:,:,:]
    # im = rgb2gray(im)
    # ims.append( rotate(im, 90) )
    plt.figure()
    plt.imshow( rotate(im, 90), cmap='gray' )
    plt.title(i)
    plt.show()
    
# imm = np.mean( ims, axis=0 )

#%%
 
plt.figure()
plt.imshow( imm, cmap='gray' )
plt.show()

#%%
# =============================================================================
# Angles grid
# =============================================================================
i90 = io.imread('/Volumes/Ice blocks/Test channel laser/25-06-12/DSC_0445.NEF')[:,:,0]
#%%

cuerda = i90[:,250:450]
cord = remove_small_objects( (gaussian(cuerda,0.1) - gaussian(cuerda,25)) < -0.05, min_size=200)

yco, xco = np.where( cord )
linc = linregress(xco, yco)

grid = i90[:,1850:2150]

ygr, xgr = np.where( grid < 40 )
ling = linregress(xgr, ygr)

xc = np.linspace(82,158,100) #+ 250 
xg = np.linspace(68,224,100) #+ 1850

plt.figure()
plt.imshow( i90, cmap='gray' )
plt.plot(xc+250, linc[0]*xc+linc[1], 'r-')
plt.plot(xg+1850, ling[0]*xg+ling[1], 'r-')
plt.show()
# plt.figure()
# plt.imshow( cuerda, cmap='gray' )
# plt.plot(x, linr[0]*x+linr[1], 'r-')
# plt.show()

#%%
c90 = io.imread('/Volumes/Ice blocks/Test channel laser/25-06-12/DSC_0134.NEF')[:5230,4000:5200,0]

xc = np.linspace(36.5,50.5,100)
xp = np.linspace(331.3, 354.8, 100)

plt.figure()
plt.imshow( c90 )
plt.plot( xc+260, -373.5 * (xc - 50.5), 'r-' )
plt.plot( xp, -223.239 * xp + 79201.772 , 'r-' )
plt.show()



#%%

m1, m2 = linc[0], ling[0]
# print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) ) #* 180/np.pi )
print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) * 180/np.pi )

m1, m2 = -373.5, -223.239
# print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) ) #* 180/np.pi )
print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) * 180/np.pi )



#%%
# =============================================================================
# Angles grid (not rotated)
# =============================================================================
iside = io.imread('/Volumes/Ice blocks/Test channel laser/25-06-12/DSC_0135.NEF')[:,4000:5000,0]
iback = io.imread('/Volumes/Ice blocks/Test channel laser/25-06-12/DSC_0443.NEF')[:,:,0]
#%%

xs = np.linspace(470.8, 476.5, 100 )
ys = -946.55 * xs + 451028.91

xc = np.linspace(293.2, 309, 100 )
yc = -342.58 * xc + 105857.42

plt.figure()
plt.imshow( iside, cmap='gray' )
plt.plot(xs, ys, 'r-')
plt.plot(xc, yc, 'r-')
plt.show()

#%%

# left: 323, right: 360 -> xt: 341.5 , yt: 0
# left: 260, right: 311 -> xt: 285.5 , yt: 7270

xg = np.linspace(1348.5, 1498, 100 )
yg = 51.29 * xg - 69106.83

xc = np.linspace(281, 341.5, 100 )
yc = -129.82 * xc + 44334.02

plt.figure()
plt.imshow( iback, cmap='gray' )
plt.plot( xg, yg, 'r-' )
plt.plot( xc, yc, 'r-' )
plt.show()
#%%

m1, m2 = -946.55, -342.58
# print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) ) #* 180/np.pi )
print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) * 180/np.pi )

m1, m2 = 51.29, -129.82
# print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) ) #* 180/np.pi )
print( np.arctan( np.abs( (m1-m2)/(1+m1*m2) ) ) * 180/np.pi )



#%%
 
#







#%%
 







#%%
 







#%%
 







#%%
 







#%%
 
