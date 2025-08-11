# GPLv3
#
# The Developers, 21st Century
import logging
import random

import numpy as num

from pyrocko import orthodrome
from pyrocko import moment_tensor as pmt

logger = logging.getLogger('ewricagm.util')

####
# Read in ensemble file
####


###############
# smth else
###############
def calc_source_width_length(magnitude, mode='Blaser', typ='scr', rake=0.):

    if mode == 'WC':
        # wells and copersmith 1994
        WD = 10**((magnitude - 4.06) / 2.25)
        LN = 10**((magnitude - 4.38) / 1.49)

    elif mode == 'Blaser':
        # Blaser et al., 2010

        if rake < 135 and rake > 45:
            # print('reverse')
            ws = 0.17
            wcov = num.matrix([[27.47, -3.77], [-3.77, 0.52]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.86, 0.46], wcov)

            ls = 0.18
            lcov = num.matrix([[26.14, -3.67], [-3.67, 0.52]]) * 1e-5
            la, lb = num.random.multivariate_normal([-2.37, 0.57], lcov)

        elif rake > -135 and rake < -45:
            # print('normal')
            ws = 0.16
            wcov = num.matrix([[264.18, -42.02], [-42.02, 6.73]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.20, 0.36], wcov)

            ls = 0.18
            lcov = num.matrix([[222.24, -32.34], [-32.34, 4.75]]) * 1e-5
            la, lb = num.random.multivariate_normal([-1.91, 0.52], lcov)

        else:
            # print('strike-slip')
            ws = 0.15
            wcov = num.matrix([[13.48, -2.18], [-2.18, 0.36]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.12, 0.33], wcov)

            ls = 0.18
            lcov = num.matrix([[12.37, -1.94], [-1.94, 0.31]]) * 1e-5
            la, lb = num.random.multivariate_normal([-2.69, 0.64], lcov)

        LN = 10**(num.random.normal(magnitude * lb + la, ls**2))
        WD = 10**(num.random.normal(magnitude * wb + wa, ws**2))

    elif mode == 'Leonard':
        # Leonard 2010/2014
        m0 = pmt.magnitude_to_moment(magnitude)

        mu = 3.3 * 1e10

        if typ == 'interplate':
            if rake < 135 and rake > 45 or rake > -135 and rake < -45:
                # dip-slip
                c1 = num.random.uniform(12, 25)
                c2 = num.random.uniform(1.5, 12)

            else:
                # strike-slip
                c1 = num.random.uniform(11, 20)
                c2 = num.random.uniform(1.5, 9.0)
        elif typ in ['scr', 'stabel continental regions']:
            if rake < 135 and rake > 45 or rake > -135 and rake < -45:
                # dip-slip
                c1 = num.random.uniform(10, 17)
                c2 = num.random.uniform(5.0, 10.0)
            else:
                # strike-slip
                c1 = num.random.uniform(9, 15)
                c2 = num.random.uniform(3.0, 11)

        c2 = c2 * 1e-5

        LN = 10**(
            (num.log10(m0) - (1.5 * num.log10(c1)) - num.log10(c2 * mu)) / 2.5)
        WD = 10**(
            (num.log10(m0) + (2.25 * num.log10(c1)) - num.log10(c2 * mu)) /
            3.75)

        LN = LN / 1000
        WD = WD / 1000

    else:
        raise ValueError('Wrong Scaling-Relation Mode. Choose between:'
                         'WC, Blaser and Leonard.')

    return WD, LN


def calc_rupture_duration(
        source=None, mag=None, moment=None,
        vr=None, WD=None, LN=None, nucx=None, nucy=None,
        mode='uncertain'):

    if source:
        if (hasattr(source, 'rupture_velocity') and
                source.rupture_velocity is not None and
                source.rupture_velocity not in [-99.0, 0.0, 999.0]):

            vr = source.rupture_velocity
        elif hasattr(source, 'velocity'):
            vr = source.velocity
        else:
            vs = 5.9 / num.sqrt(3)  # extract from velocity model for crust?
            vr = 0.8 * (vs)
        logger.info('Rupture velocity [ms]:', vr)

        if source.length and source.width:
            WD = source.width
            LN = source.length
        else:
            WD, LN = calc_source_width_length(
                source.magnitude, rake=source.rake)

        nucx = source.nucleation_x
        nucy = source.nucleation_y
    elif vr is not None and WD is not None and LN is not None:
        if nucx is None:
            nucx = 0
        if nucy is None:
            nucy = 0

    if mode == 'own':
        eLN = LN * (0.5 + 0.5 * abs(nucx))
        eWD = WD * (0.5 + 0.5 * abs(nucy))
        diag = num.sqrt((eLN)**2 + (eWD)**2)

        maxlen = float(max(eLN, eWD, diag))
        duration = (maxlen / vr)

    elif mode == 'uncertain':
        eLN = LN * 0.5
        eWD = WD * 0.5
        diag = num.sqrt((eLN)**2 + (eWD)**2)

        maxlen = float(max(eLN, eWD, diag))
        dur = (maxlen / vr)  # Duration from middle
        duration = float(num.random.uniform(dur, 2 * dur))  # Uncertainty

    elif mode == 'pub':
        if mag is not None or moment is not None:
            if mag is not None:
                moment = pmt.magnitude_to_moment(mag)
            # duration = num.sqrt(moment) / 0.5e9
            duration = num.power(moment, 0.33) / 0.25e6
        else:
            raise ValueError('Magnitude or moment missing.')

    else:
        raise ValueError('Wrong Rupture Duration mode %s' % (mode))

    return duration


def calc_rise_time(source=None, mag=None, fac=0.8):
    '''
    Rise time after Chen Ji 2021; Two Empirical Double-Corner-Frequency Source
    Spectra and Their Physical Implications
    '''
    if mag is None:
        if source is None:
            raise ValueError('Need to select either source or magnitude')

        mag = source.magnitude

    fc2 = 10**(3.250 - (0.5 * mag))
    rise_time = fac / fc2

    if rise_time < 1.0:
        rise_time = 1.0

    return rise_time


#############################
'''
Mapping functions

Create grid coordinates around the (Hypo)center of the given source.
'''
#############################

def random_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.0):
    coords = []

    lats = num.random.uniform(
        source.lat - mapextent[1],
        source.lat + mapextent[1],
        ncoords**2)
    lons = num.random.uniform(
        source.lon - mapextent[0],
        source.lon + mapextent[0],
        ncoords**2)

    for lon, lat in zip(lons, lats):
        dist = orthodrome.distance_accurate50m_numpy(
            source.lat, source.lon, lat, lon) / 1000.
        if abs(dist) > rmin:
            coords.append([lon, lat])
        else:
            pass

    coords = num.array(coords)
    return coords


def rectangular_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.0):
    coords = []

    lats = num.linspace(source.lat - mapextent[1],
                        source.lat + mapextent[1], ncoords)
    lons = num.linspace(source.lon - mapextent[0],
                        source.lon + mapextent[0], ncoords)

    for lon in lons:
        for lat in lats:
            dist = orthodrome.distance_accurate50m_numpy(
                source.lat, source.lon, lat, lon) / 1000.
            if abs(dist) > rmin:
                coords.append([lon, lat])
            else:
                pass

    coords = num.array(coords)
    return coords


def circular_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.05):
    coords = []
    dcor = min(mapextent[1], mapextent[0])

    r = num.logspace(num.log10(rmin), num.log10(dcor), int(ncoords / 1.25))
    theta = num.linspace(0, 2 * num.pi, int(ncoords * 1.25)) \
        + random.random() * 2 * num.pi

    R, Theta = num.meshgrid(r, theta)
    lons = R * num.cos(Theta) + source.lon
    lats = R * num.sin(Theta) + source.lat

    for lon, lat in zip(lons.flatten(), lats.flatten()):
        coords.append([lon, lat])

    coords = num.array(coords)

    return coords
