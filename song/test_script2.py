#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:10:05 2016

@author: cham
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:37:24 2016

@author: cham
"""

#%%
import os
import re
import numpy as np
import ccdproc

from collections import OrderedDict
from astropy.table import Table, Column, Row
from hrs.logtable import log2table
from hrs.aperture import find_apertures, combine_apertures, group_apertures, extract_1dspec, combine_flat, substract_scattered_light
from hrs.calibration import fix_thar_sat_neg, thar_corr2d, interpolate_wavelength_shift, interpolate_order_shift, load_thar_temp
import hrs
#from hrs import *
import hrs
reload(hrs)
from joblib import Parallel, delayed


from scipy.interpolate import interp1d
from skimage.filters import gaussian

os.chdir('/hydrogen/song/star_spec/20161114/night/raw/')

%matplotlib qt

import glob
from astropy.io import fits
rcParams.update({'font.size':20})
#%%

""" scan fits header """
def scan_fits_header(fps):
    
    for i, fp in enumerate(fps):
        print i, len(fps)
        # read fits
        hl = fits.open(fp)
        hdr_dict = dict(hl[0].header)
        # initiate table
        if i == 0:
            t = Table()
            for k, v in hdr_dict.items():
#                print k, v
                if type(v) is str:
                    t.add_column(Column(data=np.array([]), dtype=object, name=k))
                else:
                    t.add_column(Column(data=np.array([]), dtype=type(v), name=k))
        # add rows
        t.add_row(hdr_dict.values())
    t.add_column(Column(fps, 'fps'), 0)
    return t
    

#fps_all = glob.glob('/hydrogen/song/star_spec/20161114/night/raw/*.fits')

fps_all = glob.glob('/hydrogen/song/star_spec/20161206/test/*.fits')
fps_all = glob.glob('/hydrogen/song/star_spec/20161206/night/raw/*.fits')
fps_all.sort()
t = scan_fits_header(fps_all)

t['FRAME'].data


#%% vis corr
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

vx = np.arange(-20, 21)
vy = np.arange(-50, 51)
mx, my = np.meshgrid(vx, vy)

ax.plot_surface(mx, my, corr[1])

#%% vis thar
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

vx = np.arange(2048)
vy = np.arange(2048)
mx, my = np.meshgrid(vx, vy)

ax.plot_surface(mx, my, thar0)


from astropy.time import Time
t.add_column(Column(Time(t['FRAME'].data.astype(str)).mjd, 'MJD'))

bias = fits.open('/hydrogen/song/star_spec/20161206/night/raw/s2_2016-12-07T00-06-00.fits')[0].data
i1=0
thar1 = (fits.open(fps[i1])[0].data.reshape(2048, 2048)).astype(float)


def par_corr2(thar1, i2, fps):
    thar2 = (fits.open(fps[i2])[0].data.reshape(2048, 2048)).astype(float)
    corr = thar_corr2d(thar0, thar1, xtrim=(512, 1536), ytrim=(512, 1536), x_shiftmax=30, y_shiftmax=30)
    return corr
    
corr_list = Parallel(n_jobs=20, verbose=10)(delayed(par_corr2)(thar1, i2, fps_all) for i2 in range(len(fps_all)))

xshift=np.array([_[[0][0][0]] for _ in corr_list], dtype=int)
yshift=np.array([_[[0][1][0]] for _ in corr_list], dtype=int)

print("%s: %s, %s" % (fps[i2], corr[0][0][0], corr[0][1][0]))

imshow(corr[1], interpolation='nearest')
imshow(log10(thar1), interpolation='nearest')



t.show_in_browser()

plot(t['MJD-MID'],t['BVC'], 'o-')




print t['IMAGETYP','EXPTIME','OBJECT','OBJ-RA','OBJ-DEC','I2POS']
print np.unique(t['IMAGETYP'])
np.count_nonzero





ind_bias = t['IMAGETYP'] == 'BIAS'
print("%s BIAS found" % np.sum(ind_bias))
bias = ccdproc.combine(','.join(t['fps'][ind_bias]), unit='adu', method='average')


ind_flat = t['IMAGETYP'] == 'FLAT'
print("%s FLAT found" % np.sum(ind_flat))
flat = ccdproc.combine(','.join(t['fps'][ind_flat]), unit='adu', method='median')


ind_thar = t['IMAGETYP'] == 'THAR'
print("%s THAR found" % np.sum(ind_thar))
thar = ccdproc.combine(','.join(t['fps'][ind_thar]), unit='adu', method='median')

""" rotation """
def ccd_rot90(ccd, k=1):
    return ccdproc.CCDData(np.rot90(ccd, k=k), unit=ccd.unit)

bias_rot = ccd_rot90(bias)
flat_rot = ccd_rot90(flat)
thar_rot = ccd_rot90(thar)

""" bias substraction """
flat_rot_bias = ccdproc.subtract_bias(flat_rot, bias_rot)
thar_rot_bias = ccdproc.subtract_bias(thar_rot, bias_rot)
    
    
#figure();imshow(bias)
#figure();imshow(flat_rot)
#figure();imshow(np.log10(thar))
#figure();imshow(np.rot90(np.log10(thar)))
#           
#type(t['IMAGETYP'][0])

""" combine flat """
flat_list = [flat_rot_bias]

# find & combine & group apertures
find_aps_param_dict = dict(start_col=200, max_drift=8, max_apwidth=12, n_pix_goodap=1000)
ap_comb = hrs.combine_apertures(flat_list, n_jobs=10, find_aps_param_dict=find_aps_param_dict)
cheb_coefs, ap_uorder_interp = group_apertures(ap_comb, start_col=1000, order_dist=10)

# combine flat
#flat_comb, flat_origin = combine_flat(flat_list, ap_uorder_interp, sat_count=45000, p=95)
#flat_comb = ccdproc.CCDData(flat_comb, unit='adu')
flat_comb = flat_rot_bias

# scattered light substraction
flat_comb_sl = substract_scattered_light(flat_comb, ap_uorder_interp, ap_width=13, shrink=.85)
flat1d = extract_1dspec(flat_comb_sl, ap_uorder_interp, ap_width=10)[0]

thar1d = extract_1dspec(thar_rot_bias, ap_uorder_interp, ap_width=10)[0]

fig=figure()
ax = fig.add_subplot(121)
ax.imshow(np.log10(flat_rot_bias), interpolation='nearest')
ax.plot(ap_uorder_interp.T, 'k')
ax = fig.add_subplot(122)
ax.imshow(np.log10(star_rot_bias), interpolation='nearest')
ax.plot(ap_uorder_interp.T, 'k')
ax.contour(np.log10(flat_rot_bias), c='w')


fig=figure()
imshow(np.log10(star_rot_bias), interpolation='nearest', cmap=cm.gray)
plot(ap_uorder_interp.T, 'k')
contour(np.log10(flat_rot_bias), c='k')

figure()
plot(flat_rot_bias[:, 1000])
plot(star_rot_bias[:, 1000])


figure()
plot(flat1d.T)


figure()
icol=1000
plot(flat_comb[:, icol])
plot((flat_comb-flat_comb_sl)[:, icol])


figure()
for i in range(thar1d.shape[0]):
    plot(thar1d[i]/np.max(thar1d[i])+i)
#%%
fits.HDUList([fits.PrimaryHDU(data=thar1d)]).writeto('../thar1d.fits', clobber=True)

""" star """
ind_star = t['IMAGETYP']=='STAR'
n_star = np.sum(ind_star)

for i in range(np.sum(ind_star)):
#for i in range(10):
    star = ccdproc.CCDData.read(t['fps'][np.where(ind_star)[0][i]], unit='adu')
    # rotation
    star_rot = ccd_rot90(star)
    
    fig = figure(figsize=(8,8))
    plt.imshow(np.log10(star_rot), interpolation='nearest')
    plt.xlim(-0.5, 100.5)
    plt.ylim(100.5, -0.5)
    plt.grid(c='w',lw=1)
    fig.savefig("../figs/%04d.png" % i)
    plt.close(fig)
    
    print(i)

    
def compute_shifty(ccd, ap_uorder_interp, test_drift = np.arange(-12, 8)):
    corr_drift = np.zeros_like(test_drift)
    xcoord = np.arange(ccd.shape[1])
    for i_drift,drift_ in enumerate(test_drift):
        ap_drift = (ap_uorder_interp+drift_).astype(np.int)
        this_corr = 0
        for ap_drift_ in ap_drift:
            this_corr += np.sum(star_rot[ap_drift_, xcoord])
        corr_drift[i_drift] = this_corr
    return corr_drift

    
drifts_star = np.zeros((n_star))
for i in range(np.sum(ind_star)):
    #i = 1
    star = ccdproc.CCDData.read(t['fps'][np.where(ind_star)[0][i]], unit='adu')
    # rotation
    star_rot = ccd_rot90(star)
    
    corr_drift = compute_shifty(star_rot, ap_uorder_interp, test_drift = np.arange(-12, 8))
    #smooth
    corr_drift = np.convolve(corr_drift, [0.3, 0.4, 0.3], 'same')
    
    this_drift = test_drift[np.argmax(corr_drift)]
    drifts_star[i] = this_drift
    print ("@Cham: drift of this image is ", this_drift)

    
n_flat = np.sum(ind_flat)
drifts_flat = np.zeros((n_flat))
for i in range(np.sum(ind_flat)):
    #i = 1
    star = ccdproc.CCDData.read(t['fps'][np.where(ind_flat)[0][i]], unit='adu')
    # rotation
    star_rot = ccd_rot90(star)
    
    corr_drift = compute_shifty(star_rot, ap_uorder_interp, test_drift = np.arange(-12, 8))
    #smooth
    corr_drift = np.convolve(corr_drift, [0.3, 0.4, 0.3], 'same')
    
    this_drift = test_drift[np.argmax(corr_drift)]
    drifts_flat[i] = this_drift
    print ("@Cham: drift of this image is ", this_drift)

    
#figure(); plot(corr_drift)
figure();
ls, = plot(t['MJD-MID'][ind_star], drifts_star, 'bo-', ms=5)
lf, = plot(t['MJD-MID'][ind_flat], drifts_flat, 'o', mfc='None', mec='r', ms=10, lw=2)
legend((ls, lf), ['STAR', 'FLAT'], loc=2)
xlabel('MJD-MID')
ylabel("DRIFT/pixel (relative to stacked FLAT)")


figure();
plot(t['MJD-MID'][ind_bias], 'go-')
plot(t['MJD-MID'][ind_flat], 'ro-')
plot(t['MJD-MID'][ind_star], 'bo-')


#%%
reload(hrs)
fit_init = np.loadtxt('../database/fit_init.dat')

lc_order = fit_init[:, 0]+85
lc_coord = fit_init[:, 2]
lc_thar = fit_init[:, 4]

popt=np.repeat(np.array([[0,0,.1, 0.5, .5]]), len(lc_order),axis=0)
pcov=np.repeat(np.array([[0,0,.1, 0.5, .5]]), len(lc_order),axis=0)
# fit grating function
poly_order = (3, 5)
x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.fit_grating_equation(
    lc_coord, lc_order, lc_thar, popt, pcov, poly_order=poly_order, max_dev_threshold=1., iter=False)

# recover the wavelength (grid)
grid_coord = np.repeat(np.arange(thar1d.shape[1]).reshape(1,-1), thar1d.shape[0], axis=0)
grid_order = np.repeat(np.arange(thar1d.shape[0]).reshape(-1,1), thar1d.shape[1], axis=1)+85
wave_fitted =  hrs.calibration.grating_equation_predict(grid_coord, grid_order, x_mini_lsq, poly_order, scaler_coord, scaler_order, scaler_ml)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.plot(lc_coord, lc_order, lc_thar*lc_order, '.')
ax.plot_surface(grid_coord, grid_order, wave_fitted*grid_order)


#%%
""" a better estimation of wavelength """
reload(hrs)

thar_list = np.loadtxt('/home/cham/PycharmProjects/hrs/hrs/calibration/iraf/thar.dat')

#xcoord = np.repeat(np.arange(thar1d.shape[1]).reshape(1, -1), thar1d.shape[0], axis=0)
wave_init = np.fliplr(np.loadtxt('../lfitr.csv', delimiter=','))




#%%
#wave_init = wave_fitted-10

order_init = np.repeat(np.arange(thar1d.shape[0]).reshape(-1, 1), thar1d.shape[1], axis=1)+86

# refine thar positions
lc_coord, lc_order, lc_thar, popt, pcov, tcks = \
    hrs.refine_thar_positions(wave_init, order_init, thar1d, thar_list, n_jobs=20, verbose=10)

#figure();imshow(thar1d,aspect='auto')
print np.sum(np.isfinite(lc_coord))  


#%% 
# fit grating function
poly_order = (3, 5)
x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.calibration.fit_grating_equation(
    lc_coord, lc_order, lc_thar, popt, pcov, poly_order=poly_order, max_dev_threshold=1., iter=False)
#x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.calibration.fit_grating_equation(
#    lc_coord[ind_good_thar], lc_order[ind_good_thar], lc_thar[ind_good_thar], popt, pcov, poly_order=poly_order, max_dev_threshold=2., iter=False)
    
# recover the wavelength (grid)
grid_coord = np.repeat(np.arange(thar1d.shape[1]).reshape(1,-1), thar1d.shape[0], axis=0)
grid_order = order_init
wave_fitted =  hrs.calibration.grating_equation_predict(grid_coord, grid_order, x_mini_lsq, poly_order, scaler_coord, scaler_order, scaler_ml)

# recover the wavelength (thar)
lc_fitted = hrs.grating_equation_predict(lc_coord, lc_order, x_mini_lsq, poly_order, scaler_coord, scaler_order, scaler_ml)
lc_fitted_diff = lc_fitted - lc_thar
ind_good_thar = np.abs(lc_fitted_diff)<1

np.sqrt(np.mean(np.square(lc_fitted_diff[ind_good_thar])))
#%%
""" residual check """
fig = figure(figsize=(16,8));
ax = fig.add_subplot(221)
ax.plot(lc_order,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_order[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("Order")
axis([61, 161, -5.2, 5.2])

ax = fig.add_subplot(222)
ax.plot(lc_coord,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_coord[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("CCD X Coordinate")
xticks(np.arange(5)*1024)
axis([0, 4096, -5.2, 5.2])

ax = fig.add_subplot(212)
ax.plot(lc_thar,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_thar[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("True ThAr wavelength")
axis([3700, 10000, -5.2, 5.2])

fig.tight_layout()


figure()
for i in range(wave_fitted.shape[0]): 
    plot(wave_fitted[i], thar1d[i]/np.max(thar1d[i]))

#%%
i = 0
star = ccdproc.CCDData.read(t['fps'][np.where(ind_star)[0][i]], unit='adu')
# rotation
star_rot = ccd_rot90(star)
# bias substraction
star_rot_bias = ccdproc.subtract_bias(star_rot, bias_rot)
# scatterd light
star_comb_sl = substract_scattered_light(star_rot_bias, ap_uorder_interp+drifts_star[i], ap_width=8, shrink=.85)
# extract 1D
star1d = extract_1dspec(star_comb_sl, ap_uorder_interp+drifts_star[i], ap_width=10)[0]
# de-blaze
star1d_db = star1d/flat1d
# scale
star1d_dbsc = star1d_db/np.median(star1d_db, axis=1)[:, None]/2

figure()
imshow(star_comb_sl)
plot(ap_uorder_interp.T, 'k')

figure()
for i in range(star1d_dbsc.shape[0]):
    plot(star1d_dbsc[i]+i)

figure()
plot(star1d_db.T)
plot(flat1d.T)

figure()
imshow(star_rot_bias)

#%%
%%time
""" bias """
ind_bias = logt['obj']=='bias'
bias = ccdproc.combine(','.join(logt['filename'][ind_bias]), unit='adu', method='median')

""" flat """
ind_flat1_1 = (logt['obj']=='flat')*(logt['exp_time']==8)
ind_flat1_2 = (logt['obj']=='flat')*(logt['exp_time']==9)
ind_flat2_1 = (logt['obj']=='flat')*(logt['exp_time']==32)
ind_flat2_2 = (logt['obj']=='flat')*(logt['exp_time']==36)
ind_flat3 = (logt['obj']=='flat')*(logt['exp_time']==144)

flat1_1 = ccdproc.combine(','.join(logt['filename'][ind_flat1_1]), unit='adu', method='median')
flat1_2 = ccdproc.combine(','.join(logt['filename'][ind_flat1_2]), unit='adu', method='median')
flat2_1 = ccdproc.combine(','.join(logt['filename'][ind_flat2_1]), unit='adu', method='median')
flat2_2 = ccdproc.combine(','.join(logt['filename'][ind_flat2_2]), unit='adu', method='median')
flat3 = ccdproc.combine(','.join(logt['filename'][ind_flat3]), unit='adu', method='median')
flat1 = ccdproc.combine([flat1_1.multiply(9.).divide(8.), flat1_2], method='average')
flat2 = ccdproc.combine([flat2_1.multiply(36.).divide(32.), flat2_2], method='average')

flat1_bias = ccdproc.subtract_bias(flat1, bias)
flat2_bias = ccdproc.subtract_bias(flat2, bias)
flat3_bias = ccdproc.subtract_bias(flat3, bias)

flat1_bias_trim = ccdproc.trim_image(flat1_bias[:, :4096])
flat2_bias_trim = ccdproc.trim_image(flat2_bias[:, :4096])
flat3_bias_trim = ccdproc.trim_image(flat3_bias[:, :4096])


""" combine flat """
flat_list = [flat1_bias_trim, flat2_bias_trim, flat3_bias_trim]

# find & combine & group apertures
ap_comb = combine_apertures(flat_list, n_jobs=10)
cheb_coefs, ap_uorder_interp = group_apertures(ap_comb, start_col=2100, order_dist=10)

# combine flat
flat_comb, flat_origin = combine_flat(flat_list, ap_uorder_interp, sat_count=45000, p=95)
flat_comb = ccdproc.CCDData(flat_comb, unit='adu')

# scattered light substraction
flat_comb_sl = substract_scattered_light(flat_comb, ap_uorder_interp, ap_width=10, shrink=.85)
flat1d = extract_1dspec(flat_comb_sl, ap_uorder_interp, ap_width=7)[0]


""" thar """
ind_thar1 = (logt['obj']=='thar')*(logt['exp_time']==30)
ind_thar2 = (logt['obj']=='thar')*(logt['exp_time']==60)
ind_thar3 = (logt['obj']=='thar')*(logt['exp_time']==120)

thar1 = ccdproc.combine(','.join(logt['filename'][ind_thar1]), unit='adu', method='average')
thar2 = ccdproc.combine(','.join(logt['filename'][ind_thar2]), unit='adu', method='average')
thar3 = ccdproc.CCDData.read(','.join(logt['filename'][ind_thar3]), unit='adu', method='average')

thar1_bias = ccdproc.subtract_bias(thar1, bias)
thar2_bias = ccdproc.subtract_bias(thar2, bias)
thar3_bias = ccdproc.subtract_bias(thar3, bias)

thar1_bias_trim = ccdproc.trim_image(thar1_bias[:, :4096])
thar2_bias_trim = ccdproc.trim_image(thar2_bias[:, :4096])
thar3_bias_trim = ccdproc.trim_image(thar3_bias[:, :4096])

""" combine thar """
thar_list = [thar1_bias_trim, thar2_bias_trim, thar3_bias_trim]
thar_comb, thar_origin = combine_flat(thar_list, ap_uorder_interp, sat_count=45000, p=100)
thar_comb = ccdproc.CCDData(thar_comb, unit='adu')

""" extract 1d thar """
#thar_comb_sl = substract_scattered_light(thar_comb, ap_uorder_interp, ap_width=10, shrink=.5)
thar1d = extract_1dspec(thar_comb, ap_uorder_interp, ap_width=7)[0]
thar1d_scaled = thar1d/flat1d


#%%
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

fig = figure(figsize=(22, 6))
imshow(log10(thar1d), vmin=2., vmax=4.5, aspect='auto', interpolation='nearest')
xlabel('CCD X Coordinate')
ylabel('Extracted Order Number')
colorbar()
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/data/visualization_thar1d.pdf')
plt.close(fig)

fig = figure(figsize=(22, 6))
imshow(spec1d_nm, vmin=0.85, vmax=1.05, aspect='auto', interpolation='nearest')
xlabel('CCD X Coordinate')
ylabel('Extracted Order Number')
colorbar()
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/data/visualization_spec1d.pdf')
plt.close(fig)

#%%
""" wavelength calibration """


# laod template thar
thar_temp_path = '/home/cham/PycharmProjects/hrs/hrs/calibration/thar_template/thar_temp_w20160120022t.fits'
wave_temp, thar_temp, order_temp = load_thar_temp(thar_temp_path)

wave_temp = np.loadtxt('/hydrogen/song/20150903/night/calib_master/wave.dat')
thar_temp = np.loadtxt('/hydrogen/song/20150903/night/calib_master/thar.dat')
wave_temp = np.rot90(wave_temp, 2)
thar_temp = np.rot90(thar_temp, 2)
thar_temp = np.fliplr(thar_temp)

order_temp = np.repeat(np.arange(51).reshape(-1, 1), 2048, axis=1)+86


# fix thar
#thar1d_fixed = fix_thar_sat_neg(thar1d, arm=30, sat_count=500000)
#thar_temp_fixed = fix_thar_sat_neg(thar_temp, arm=30, sat_count=500000)


""" initial estimation of wavelength """
shift, corr2d = thar_corr2d(thar1d, thar_temp, xtrim=(512, 512*3), ytrim=(20, 40), y_shiftmax = 1, x_shiftmax=50)
print ("@Cham: the shift is ", shift)
figure(); imshow(corr2d, interpolation='nearest', aspect='auto')
figure(); imshow(np.log10(thar1d), interpolation='nearest', aspect='auto')
figure(); imshow(np.log10(thar_temp), interpolation='nearest', aspect='auto')
figure(); plot(thar1d[1]), plot(thar_temp[1][::-1])
figure(); plot(wave_temp[0])


wave_init = interpolate_wavelength_shift(wave_temp, shift, thar_temp, thar1d)
order_init = interpolate_order_shift(order_temp, shift, thar1d)



#%%
fig=figure(figsize=(12, 3))
imshow(corr2d, interpolation='nearest', aspect='auto', extent=(-20.5, 20.5, -3.5, 3.5))
xlabel("X shift")
ylabel("Y shift")
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_corr2d.pdf')

    
#%%
""" a better estimation of wavelength """

thar_list = np.loadtxt('/home/cham/PycharmProjects/hrs/hrs/calibration/iraf/thar.dat')

# refine thar positions
lc_coord, lc_order, lc_thar, popt, pcov,tcks = \
    hrs.calibration.refine_thar_positions(wave_init, order_init, thar1d, thar_list, n_jobs=20, verbose=10)
#x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.calibration.fit_grating_equation(
#    lc_coord[ind_good_thar], lc_order[ind_good_thar], lc_thar[ind_good_thar], popt, pcov, poly_order=poly_order, max_dev_threshold=2., iter=False)
#figure();hist(popt[:,3]-lc_thar, np.arange(-10, 10, .1))
#figure()
#plot(lc_thar-popt[:, 3], popt[:, 4], '.')
#ylim(-1, 5)
# fit grating function
poly_order = (3,4)
x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.calibration.fit_grating_equation(
    lc_coord, lc_order, lc_thar, popt, pcov, poly_order=poly_order, max_dev_threshold=3., iter=False)
#x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = hrs.calibration.fit_grating_equation(
#    lc_coord[ind_good_thar], lc_order[ind_good_thar], lc_thar[ind_good_thar], popt, pcov, poly_order=poly_order, max_dev_threshold=2., iter=False)    

# recover the wavelength (grid)
grid_coord = np.repeat(np.arange(wave_init.shape[1]).reshape(1,-1), wave_init.shape[0], axis=0)
grid_order = order_init
wave_fitted =  hrs.calibration.grating_equation_predict(grid_coord, grid_order, x_mini_lsq, poly_order, scaler_coord, scaler_order, scaler_ml)

# recover the wavelength (thar)
lc_fitted = hrs.grating_equation_predict(lc_coord, lc_order, x_mini_lsq, poly_order, scaler_coord, scaler_order, scaler_ml)
lc_fitted_diff = lc_fitted - lc_thar

ind_good_thar = np.abs(lc_fitted_diff)<2
#%%
figure()
plot(wave_fitted.T, wave_init.T-wave_fitted.T, '.')

figure()
plot(wave_init.T, thar1d.T, 'b')
plot(wave_temp.T, thar_temp.T, 'r')
plot(wave_fitted.T, thar1d.T+30000, 'c')


figure()
plot(wave_fitted.T, star1d_dbsc.T, '-')
plot(wave_init.T, spec1d_nm.T+1, '-')

#%%
""" figures """


""" fitted gaussian superposition """
ww = np.arange(3000,10600,.01)
ff = np.zeros_like(ww)
for i,popt_ in enumerate(popt):
#    if np.all(np.isfinite(popt_)):
    if ind_good_thar[i]:
        g = hrs.calibration.gauss_poly1(ww, 0., 0., popt_[2], popt_[3], popt_[4])
        ff = ff+g
        print i, ff[0], np.sum(g)

figure(); plot(ww, ff,'r'); plot(wave_init.T, thar1d_fixed.T, 'b')
figure(); plot(popt[:, 2], popt[:, 4], '.')


""" line consistency check """
matplotlib.rcParams.update({'font.size': 20})
fig=figure(figsize=(8,6));
l1 = plot(wave_init.reshape(104, 4096).T, spec1d_nm.T+.3, 'b')
l2 = plot(gfitted_wave.reshape(104, 4096).T, spec1d_nm.T, 'r')
xlim(5894.2,5896.5)
ylim(0.05, 1.5)
xlabel("Wavelength (A)")

l = legend((l1[0], l2[0]), ['Initial guess', 'Refined'], loc=9)
l.draw_frame=False
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_refine_thar_positions.pdf')


""" residual check """
fig = figure(figsize=(16,8));
ax = fig.add_subplot(221)
ax.plot(lc_order,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_order[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("Order")
axis([61, 161, -5.2, 5.2])

ax = fig.add_subplot(222)
ax.plot(lc_coord,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_coord[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("CCD X Coordinate")
xticks(np.arange(5)*1024)
axis([0, 4096, -5.2, 5.2])

ax = fig.add_subplot(212)
ax.plot(lc_thar,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_thar[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("True ThAr wavelength")
axis([3700, 10000, -5.2, 5.2])

fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_calibration_residuals.pdf')

""" residual check zoom in """
fig = figure(figsize=(16,8));
ax = fig.add_subplot(221)
ax.plot(lc_order,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_order[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("Order")
axis([61, 161, -0.1, 0.1])

ax = fig.add_subplot(222)
ax.plot(lc_coord,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_coord[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("CCD X Coordinate")
xticks(np.arange(5)*1024)
axis([0, 4096, -0.1, 0.1])

ax = fig.add_subplot(212)
ax.plot(lc_thar,lc_fitted-lc_thar, 'b+', alpha=.8)
ax.plot(lc_thar[ind_good_thar],lc_fitted[ind_good_thar]-lc_thar[ind_good_thar], 'r+', alpha=.8)
xlabel("True ThAr wavelength")
axis([3700, 10000, -0.1, 0.1])

fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_calibration_residuals_zoomin.pdf')


""" wavelength chagne 3D """
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid_coord, grid_order, wave_fitted-wave_init, rstride=10, cstride=256, cmap=cm.jet)
ax.set_xlim(-1, 4096)
ax.set_xticks(np.arange(5)*1024)
ax.set_xlabel("CCD X Coordinate")
ax.set_ylabel("Order")
ax.set_zlabel("Wavelength change (A)")
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_calibration_wchange_3d.pdf')


""" wavelength change 2D """
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(111)
ax.plot(wave_fitted.T, wave_fitted.T-wave_init.T, 'b-')
ax.set_xlim(3700, 10000)
#ax.set_xticks(np.arange(5)*1024)
ax.set_ylabel("Wavelength Chagne (A)")
ax.set_xlabel("Refined Wavelength (A)")
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_calibration_wchange_2d.pdf')


#%%
%%time
""" read science images """
# ########################################################################### #
#    CPU times: user 58.7 s, sys: 4.05 s, total: 1min 2s
#    Wall time: 59 s
# ########################################################################### #

"""
Master variables:
    bias
    ap_uorder_interp
    flat1d
    wave_fitted
"""

# 1. read
sci = ccdproc.CCDData.read(logt['filename'][41], unit='adu')

# 2. BIAS correction
sci_bias = ccdproc.subtract_bias(sci, bias)

# 3. trim 
sci_bias_trim = ccdproc.trim_image(sci_bias[:, :4096])

# 4. scattered light correction
sci_bias_trim_sl = substract_scattered_light(sci_bias_trim, ap_uorder_interp, ap_width=10, shrink=.85)

# 5. extract 1D spectra
spec1d = extract_1dspec(sci_bias_trim_sl, ap_uorder_interp, ap_width=7)[0]

# 6. de-blaze
spec1d_db = spec1d/flat1d

# 7. roughly scale to 1.
spec1d_nm = spec1d_db/np.percentile(spec1d_db, 95, axis=1)[:, None]

figure(); plot(wave_fitted.T, spec1d_nm.T)

#figure();plot(wave_fitted.T, np.sqrt(spec1d).T)
#figure();plot(wave_fitted.T, np.sqrt(flat1d).T)
#%%
scattered_light = sci_bias_trim-sci_bias_trim_sl
fig = figure(figsize=(16, 8))
plot(sci_bias_trim[:, 2100])
plot(scattered_light[:, 2100], 'r')
xlabel("CCD Y coordinate")
ylabel("CCD counts")
ylim(-100, 21000)
xlim(0, 4096)
xticks(np.arange(5)*1024)
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_scattered_background.pdf')



wave_fitted_masked = wave_fitted
wave_fitted_masked = np.where(np.log10(flat1d)<3.2, np.nan, wave_fitted_masked)

fig = figure(figsize=(20, 5))
plot(wave_fitted_masked.T, spec1d_nm.T)
xlim(6500, 6700)
ylim(-0.05, 1.15)
xlabel("Wavelength (A)")
ylabel("De-blazed & scaled flux")
fig.tight_layout()
fig.savefig('/home/cham/PycharmProjects/hrs/hrs/figures/visualization_deblazed_spectrum.pdf')

#%%
figure(); imshow(flat1d, aspect='auto')
figure();  imshow(log10(flat1d), aspect='auto', interpolation='nearest')

#%%

figure();  imshow(log10(flat1d)>3.5, aspect='auto', interpolation='nearest')
#%%