# GPLv3
#
# The Developers, 21st Century
import logging
import traceback
import math

import numpy as num
import geopandas as gpd

from scipy.integrate import cumulative_trapezoid as cumtrapz

import eqsig
from pyrocko import orthodrome, gf, trace
from pyrocko.guts import (
    Object, StringChoice, Float, String, List, Dict, Choice)

from openquake.hazardlib.geo import geodetic, Point, Mesh, PlanarSurface

logger = logging.getLogger('ewricagm.gm.sources')

#############################
# Constants
#############################
deg = 111.19
G = 9.81  # m/s*s


#############################
# Classes
#############################
class Cloneable(object):

    def __iter__(self):
        return iter(self.T.propnames)

    def __getitem__(self, k):
        if k not in self.keys():
            raise KeyError(k)

        return getattr(self, k)

    def __setitem__(self, k, v):
        if k not in self.keys():
            raise KeyError(k)

        return setattr(self, k, v)

    def clone(self, **kwargs):
        '''
        Make a copy of the object.

        A new object of the same class is created and initialized with the
        parameters of the object on which this method is called on. If
        ``kwargs`` are given, these are used to override any of the
        initialization parameters.
        '''

        d = dict(self)
        for k in d:
            v = d[k]
            if isinstance(v, Cloneable):
                d[k] = v.clone()

        d.update(kwargs)
        return self.__class__(**d)

    @classmethod
    def keys(cls):
        '''
        Get list of the source model's parameter names.
        '''

        return cls.T.propnames


class SourceClass(Object, Cloneable):
    name = String.T()
    depth = Float.T()
    lon = Float.T()
    lat = Float.T()
    magnitude = Float.T()
    time = Float.T(optional=True)

    form = StringChoice.T(
        choices=['point', 'rectangular', 'pdr'],
        default='point')
    nucleation_x = Float.T(default=0.)
    nucleation_y = Float.T(default=0.)

    # optional
    strike = Float.T(optional=True)
    dip = Float.T(optional=True)
    rake = Float.T(optional=True)
    width = Float.T(optional=True)
    length = Float.T(optional=True)
    duration = Float.T(optional=True)
    rupture = Dict.T(optional=True)
    nucleationCoords = List.T(optional=True)
    moment = Float.T(optional=True)
    slip = Float.T(optional=True)
    tensor = Dict.T(optional=True)
    ztor = Float.T(optional=True)

    risetime = Float.T(optional=True)
    rupture_velocity = Float.T(optional=True)

    pyrockoSource = gf.Source.T(optional=True)

    def update(self, **kwargs):
        for (k, v) in kwargs.items():
            self[k] = v

    def create_synthetic_rupture_plain_from_hypo(self):
        nucx_fac2 = (self.nucleation_x + 1) / 2  # convert to 0-1 range
        nucx_fac1 = 1 - nucx_fac2
        nucy_fac = ((self.nucleation_y + 1) / 2)  # convert to 0-1 range

        surface_width = self.width * num.cos(
            num.radians(self.dip))  # length of width projected to surface
        depth_range = self.width * num.sin(
            num.radians(self.dip))  # length of depth along Z

        dist1 = num.sqrt(
            (nucy_fac * surface_width)**2 +
            (self.length * nucx_fac1)**2)
        dist2 = num.sqrt(
            (nucy_fac * surface_width)**2 +
            (self.length * nucx_fac2)**2)

        if nucx_fac2 <= 0:
            nucx_fac2 = 0.001

        if nucx_fac1 <= 0:
            nucx_fac1 = 0.001

        azi1 = self.strike - (
            num.arctan(
                (nucy_fac * surface_width) / (self.length * nucx_fac1)) *
            (180. / num.pi))
        azi2 = self.strike - 180. + (
            num.arctan(
                (nucy_fac * surface_width) / (self.length * nucx_fac2)) *
            (180 / num.pi))

        depth1 = self.depth - nucy_fac * depth_range
        depth2 = self.depth - nucy_fac * depth_range

        lat1, lon1 = orthodrome.azidist_to_latlon(self.lat, self.lon,
                                                  azi1, dist1 / deg)
        lat2, lon2 = orthodrome.azidist_to_latlon(self.lat, self.lon,
                                                  azi2, dist2 / deg)
        p1 = Point(lon1, lat1, depth1)
        p2 = Point(lon2, lat2, depth2)
        p3 = p1.point_at(surface_width, depth_range, self.strike + 90)
        p4 = p2.point_at(surface_width, depth_range, self.strike + 90)

        self.rupture = {'UR': [p1.longitude, p1.latitude, p1.depth],
                        'UL': [p2.longitude, p2.latitude, p2.depth],
                        'LL': [p4.longitude, p4.latitude, p4.depth],
                        'LR': [p3.longitude, p3.latitude, p3.depth]}

        self.form = 'rectangular'

    def create_rupture_surface(self):
        if self.form == 'point':
            surface = Mesh(num.array([self.lon]),
                           num.array([self.lat]),
                           num.array([self.depth])
                           )
            self.surface = surface

        elif self.form == 'rectangular':
            '''
            To do:
            - check if the strike is always correct calculated
            '''

            p1 = Point(self.rupture['UR'][0], self.rupture['UR'][1],
                       self.rupture['UR'][2])
            p2 = Point(self.rupture['UL'][0], self.rupture['UL'][1],
                       self.rupture['UL'][2])
            p3 = Point(self.rupture['LL'][0], self.rupture['LL'][1],
                       self.rupture['LL'][2])
            p4 = Point(self.rupture['LR'][0], self.rupture['LR'][1],
                       self.rupture['LR'][2])

            if p1.depth <= 0:
                p1.depth = 0.01
                logger.warning(
                    'Point 1 above ground. Set rupture top depth to 10m.')
            if p2.depth <= 0:
                p2.depth = 0.01
                logger.warning(
                    'Point 2 above ground. Set rupture top depth to 10m.')

            surface = PlanarSurface.from_corner_points(
                top_left=p2, top_right=p1, bottom_left=p3, bottom_right=p4)

            testdist = geodetic.geodetic_distance(
                p1.longitude, p1.latitude, p4.longitude, p4.latitude)
            p5 = geodetic.point_at(
                p1.longitude, p1.latitude,
                surface.get_strike() + 90., testdist)
            valdist = geodetic.geodetic_distance(
                p5[0], p5[1], p4.longitude, p4.latitude)

            if valdist > 1:
                surface = PlanarSurface.from_corner_points(
                    top_left=p1, top_right=p2, bottom_left=p4, bottom_right=p3)

                testdist = geodetic.geodetic_distance(
                    p1.longitude, p1.latitude, p4.longitude, p4.latitude)
                p5 = geodetic.point_at(
                    p1.longitude, p1.latitude,
                    surface.get_strike() + 90., testdist)
                valdist = geodetic.geodetic_distance(
                    p5[0], p5[1], p4.longitude, p4.latitude)

                if valdist > 1:
                    raise ValueError('''Check rupture plane:
Strike: {:.1f}; dip: {:.1f}; rake: {:.1f}
corner point 1: {}
corner point 2: {}
corner point 3: {}
corner point 4: {}
'''.format(surface.get_strike(), surface.get_dip(), self.rake, p1, p2, p3, p4))

                else:
                    logger.warning('''Switched Rupture points to get:
Strike: {:.1f}; dip: {:.1f}; rake: {:.1f}'''.format(
                        surface.get_strike(), surface.get_dip(), self.rake))

            self.surface = surface
            self.strike = float(self.surface.get_strike())
            self.dip = float(self.surface.get_dip())

            if self.rake is None:
                self.rake = 0.0
                logger.warning('Set rake manually to 0')

            self.ztor = float(self.surface.get_top_edge_depth())

            self.area = float(self.surface.get_area())
            self.width = float(self.surface.get_width())
            self.length = self.area / self.width

            # Convert nucleation coordinates
            if self.nucleationCoords:
                nuc_lat = self.nucleationCoords[1]
                nuc_lon = self.nucleationCoords[0]
                nuc_depth = self.nucleationCoords[2]
            else:
                nuc_lat = self.lat
                nuc_lon = self.lon
                nuc_depth = self.depth

            depth_upper = self.ztor
            depth_lower = surface.bottom_right.depth

            depth_mid = (depth_upper + depth_lower) / 2.
            self.nucleation_y = float(
                (nuc_depth - depth_mid) / (depth_lower - depth_mid))

            d1 = orthodrome.distance_accurate50m_numpy(
                nuc_lat, nuc_lon,
                surface.top_left.latitude, surface.top_left.longitude)
            d2 = orthodrome.distance_accurate50m_numpy(
                nuc_lat, nuc_lon,
                surface.top_left.latitude, surface.top_left.longitude)
            d3 = orthodrome.distance_accurate50m_numpy(
                surface.top_left.latitude, surface.top_left.longitude,
                surface.top_right.latitude, surface.top_right.longitude)

            nuc_x = ((d1**2 - d2**2 - d3**2) / (-2 * d3**2))[0]
            self.nucleation_x = float(-((nuc_x * 2) - 1))

        return

    def calc_distance(self, lons, lats, dist_type='rhypo'):

        loc_points = []
        dists = []
        for lon, lat in zip(lons, lats):
            if dist_type == 'rhypo':
                dists.append(geodetic.distance(
                    self.lon, self.lat, self.depth, lon, lat, 0.))

            else:
                loc_points.append(Point(lon, lat))

        if hasattr(self, 'surface'):
            if dist_type == 'rjb':
                dists = self.surface.get_joyner_boore_distance(
                    Mesh.from_points_list(loc_points))

            elif dist_type == 'ry0':
                if self.refSource.form == 'rectangular':
                    dists = self.surface.get_ry0_distance(
                        Mesh.from_points_list(loc_points))
                else:
                    dists = num.zeros(len(loc_points)) * num.nan

            elif dist_type == 'rx':
                if self.refSource.form == 'rectangular':
                    dists = self.surface.get_rx_distance(
                        Mesh.from_points_list(loc_points))
                else:
                    dists = num.zeros(len(loc_points)) * num.nan

            elif dist_type == 'rrup':
                dists = self.surface.get_min_distance(
                    Mesh.from_points_list(loc_points))

        return dists

    def calc_azimuth(self, lons, lats):

        azimuths = geodetic.azimuth(self.lon, self.lat, lons, lats)
        return azimuths

    def calc_rupture_azimuth(self, lons, lats):

        mesh = Mesh(num.array(lons), num.array(lats))

        rup_azimuth = self.surface.get_azimuth_of_closest_point(mesh)
        centre_azimuth = self.surface.get_azimuth(mesh)

        return rup_azimuth, centre_azimuth


class GMClass(Object, Cloneable):
    name = String.T()
    value = Float.T()
    unit = String.T(optional=True)


class ComponentGMClass(Object, Cloneable):
    component = String.T()
    gms = Dict.T(
        optional=True,
        content_t=GMClass.T())

    traces = Dict.T()


class StationGMClass(Object, Cloneable):
    network = String.T()
    station = String.T()
    lon = Float.T()
    lat = Float.T()

    components = Dict.T(
        optional=True,
        content_t=ComponentGMClass.T())

    azimuth = Float.T(optional=True)
    rup_azimuth = Float.T(optional=True)
    centre_azimuth = Float.T(optional=True)
    rhypo = Float.T(optional=True)
    rjb = Float.T(optional=True)
    ry0 = Float.T(optional=True)
    rx = Float.T(optional=True)
    rrup = Float.T(optional=True)
    vs30 = Float.T(optional=True)

    azimuth = Float.T(optional=True)
    rupture_azimuth = Float.T(optional=True)
    centre_azimuth = Float.T(optional=True)


class StationContainer(Object, Cloneable):
    stations = Dict.T(
        content_t=StationGMClass.T())

    refSource = Choice.T([
        SourceClass.T()],
        optional=True)

    def calc_distances(self, dist_types=['rhypo', 'rjb', 'rx', 'ry0', 'rrup']):
        loc_points = []
        for ii in self.stations:
            if 'rhypo' in dist_types:
                rhypo = geodetic.distance(
                    self.refSource.lon, self.refSource.lat,
                    self.refSource.depth,
                    self.stations[ii].lon, self.stations[ii].lat,
                    0.)
                self.stations[ii].rhypo = float(rhypo)

            loc_points.append(
                Point(self.stations[ii].lon, self.stations[ii].lat))

        if hasattr(self.refSource, 'surface'):

            if 'rjb' in dist_types:
                rjbs = self.refSource.surface.get_joyner_boore_distance(
                    Mesh.from_points_list(loc_points))

            if 'ry0' in dist_types:
                if self.refSource.form == 'rectangular':
                    ry0s = self.refSource.surface.get_ry0_distance(
                        Mesh.from_points_list(loc_points))
                else:
                    ry0s = num.zeros(len(loc_points)) * num.nan

            if 'rx' in dist_types:
                if self.refSource.form == 'rectangular':
                    rxs = self.refSource.surface.get_rx_distance(
                        Mesh.from_points_list(loc_points))
                else:
                    rxs = num.zeros(len(loc_points)) * num.nan

            if 'rrup' in dist_types:
                rrups = self.refSource.surface.get_min_distance(
                    Mesh.from_points_list(loc_points))

            for nn, ii in enumerate(self.stations):
                if 'rrup' in dist_types:
                    self.stations[ii].rrup = float(rrups[nn])
                if 'rx' in dist_types:
                    self.stations[ii].rx = float(rxs[nn])
                if 'ry0' in dist_types:
                    self.stations[ii].ry0 = float(ry0s[nn])
                if 'rjb' in dist_types:
                    self.stations[ii].rjb = float(rjbs[nn])

        return

    def get_distances(self, dist_types=['rhypo', 'rjb', 'rx', 'ry0', 'rrup']):
        dist_dict = {}
        for sta, staDict in self.stations.items():
            for disttyp in dist_types:
                if disttyp not in dist_dict:
                    dist_dict[disttyp] = []
                dist_dict[disttyp].append(staDict[disttyp])

        return dist_dict

    def calc_azimuths(
            self,
            aziTypes=['azimuth', 'rupture_azimuth', 'centre_azimuth']):

        lons = []
        lats = []
        for ii in self.stations:
            lons.append(self.stations[ii].lon)
            lats.append(self.stations[ii].lat)

        if 'azimuth' in aziTypes:
            azimuth = geodetic.azimuth(
                self.refSource.lon, self.refSource.lat, lons, lats)

        mesh = Mesh(num.array(lons), num.array(lats))
        if hasattr(self.refSource, 'surface'):
            if 'rupture_azimuth' in aziTypes:
                rup_azis = self.refSource.surface.get_azimuth_of_closest_point(
                    mesh)

            if 'centre_azimuth' in aziTypes:
                centroid_azis = self.refSource.surface.get_azimuth(mesh)

        for nn, ii in enumerate(self.stations):
            if 'azimuth' in aziTypes:
                self.stations[ii].azimuth = float(azimuth[nn])
            if hasattr(self.refSource, 'surface'):
                if 'rupture_azimuth' in aziTypes:
                    self.stations[ii].rupture_azimuth = float(rup_azis[nn])
                if 'centre_azimuth' in aziTypes:
                    self.stations[ii].centre_azimuth = float(centroid_azis[nn])

        return

    def get_azimuths(
            self,
            aziTypes=['azimuth', 'rupture_azimuth', 'centre_azimuth']):
        azi_dict = {}
        for sta, staDict in self.stations.items():
            for azitype in aziTypes:
                if azitype not in azi_dict:
                    azi_dict[azitype] = []
                azi_dict[azitype].append(staDict[azitype])

        return azi_dict

    def get_gm_values(self):
        gmdict = {}
        for sta in self.stations:
            for comp in self.stations[sta].components:
                for gm in self.stations[sta].components[comp].gms:
                    ims = '%s_%s' % (comp, gm)
                    if ims not in gmdict:
                        gmdict[ims] = []
                    val = self.stations[sta].components[comp].gms[gm].value
                    gmdict[ims].append(val)

        return gmdict

    def get_gm_from_wv(
            self,
            imts=['pga', 'pgv'],
            freqs=[0.3, 1.0, 3.0],
            H2=False,
            delete=False,
            deleteWvData=True):
        '''
        To do:
        - clean
        - stuetzstellen einfuegen mit sinc-interpolation
        (hochsamplen /resamplen)

        see https://github.com/emolch/wafe/blob/master/src/measure.py
        '''

        tfade = 2

        nfreqs = []
        for freq in freqs:
            nfreqs.append(float('%.3f' % float(freq)))
        freqs = num.array(nfreqs)

        for sta in self.stations:

            # Horizontal Vector Sum component
            if H2:
                cha = 'H'

                if 'E' in self.stations[sta].components.keys() \
                        and 'N' in self.stations[sta].components.keys():

                    traces_e = self.stations[sta].components['E'].traces
                    tr_de = traces_e['disp'].copy()
                    tr_ve = traces_e['vel'].copy()
                    tr_ae = traces_e['acc'].copy()

                    traces_n = self.stations[sta].components['N'].traces
                    tr_dn = traces_n['disp'].copy()
                    tr_vn = traces_n['vel'].copy()
                    tr_an = traces_n['acc'].copy()

                    comp = ComponentGMClass(component=cha, gms={})
                    flagshort = False

                    if len(tr_ae.ydata) < 500. or len(tr_an.ydata) < 500.:
                        logger.warning(
                            'One of the horizontal components is too short:\n'
                            'E: {}\nN: {}'.format(
                                len(tr_ae.ydata), len(tr_an.ydata)))
                        flagshort = True

                    else:
                        tmin = max(tr_ae.tmin, tr_an.tmin)
                        tmax = min(tr_ae.tmax, tr_an.tmax)
                        tr_ae.chop(tmin, tmax, include_last=True)
                        tr_an.chop(tmin, tmax, include_last=True)

                        tmin = max(tr_ve.tmin, tr_vn.tmin)
                        tmax = min(tr_ve.tmax, tr_vn.tmax)
                        tr_ve.chop(tmin, tmax, include_last=True)
                        tr_vn.chop(tmin, tmax, include_last=True)

                        tmin = max(tr_de.tmin, tr_dn.tmin)
                        tmax = min(tr_de.tmax, tr_dn.tmax)
                        tr_de.chop(tmin, tmax, include_last=True)
                        tr_dn.chop(tmin, tmax, include_last=True)

                        deltat = tr_ae.deltat
                        data_ah = num.abs(
                            num.sqrt(tr_ae.ydata**2 + tr_an.ydata**2))
                        data_vh = num.abs(
                            num.sqrt(tr_ve.ydata**2 + tr_vn.ydata**2))
                        data_dh = num.abs(
                            num.sqrt(tr_de.ydata**2 + tr_dn.ydata**2))

                    sa_freqs = []
                    for gm in imts:
                        if flagshort:
                            val = 0.000001
                            unit = 'NaN'

                        elif 'sigdur' == gm:
                            dur = eqsig.im.calc_sig_dur_vals(
                                data_ah, dt=deltat)
                            val = dur
                            unit = 's'

                        elif 'ai' == gm:
                            ai = arias_intensity(data_ah, dt=deltat)
                            val = num.log10(ai.max())
                            unit = 'm/s'

                        elif 'pga' == gm:
                            pga = num.abs(data_ah).max()
                            val = num.log10((pga / G) * 100.)  # in g%
                            unit = 'g%'

                        elif 'vi' == gm:
                            vi = integrated_velocity(data_vh, dt=deltat)
                            val = num.log10(vi.max())
                            unit = 'm'

                        elif 'pgv' == gm:
                            pgv = num.abs(data_vh).max()
                            val = num.log10(pgv * 100.)  # in cm/s
                            unit = 'cm/s'

                        elif 'pgd' == gm:
                            pgd = num.abs(data_dh).max()
                            val = num.log10(pgd * 100.)  # in cm
                            unit = 'cm'

                        elif 'SA' in gm:
                            sa_freqs.append(float(gm.rsplit('_')[-1]))
                            continue

                        GM = GMClass(
                            name=gm,
                            value=float(val),
                            unit=unit)

                        comp.gms[GM.name] = GM

                    if len(sa_freqs) > 0:
                        tr_acc = eqsig.single.AccSignal(data_ah, deltat)
                        tr_acc.generate_response_spectrum(sa_freqs)
                        for f, sa in zip(sa_freqs, tr_acc.s_a):
                            val = num.log10((sa / G) * 100.)
                            unit = 'UKN'
                            GM = GMClass(
                                name='SA_%s' % f,
                                value=float(val),
                                unit=unit)
                            comp.gms[GM.name] = GM

                    if freqs.size != 0:
                        if flagshort:
                            for frq in freqs:
                                val = num.nan
                                unit = 'NaN'

                                GM = GMClass(
                                    name='f_%s' % frq,
                                    value=float(val),
                                    unit='nan')

                                comp.gms[GM.name] = GM

                        else:
                            spec = get_spectra(data_ah, deltat, tfade)
                            vals = eqsig.functions.calc_smooth_fa_spectrum(
                                spec[1], spec[0], num.array(freqs), band=40)

                            for frq, val in zip(freqs, vals):
                                GM = GMClass(
                                    name='f_%s' % frq,
                                    value=float(num.log10(val)),
                                    unit='m/s?')

                                comp.gms[GM.name] = GM

                    self.stations[sta].components[comp.component] = comp

                    if delete:
                        del self.stations[sta].components['E']
                        del self.stations[sta].components['N']

            # Standard components
            for cha in self.stations[sta].components:
                if cha == 'H':
                    continue

                comp = ComponentGMClass(
                    component=cha,
                    gms={})

                traces = self.stations[sta].components[cha].traces
                trD = traces['disp'].copy()
                trV = traces['vel'].copy()
                trA = traces['acc'].copy()

                deltat = trA.deltat
                dataA = trA.ydata
                dataV = trV.ydata
                dataD = trD.ydata

                sa_freqs = []
                for gm in imts:
                    if 'sigdur' == gm:
                        dur = eqsig.im.calc_sig_dur_vals(dataA, dt=deltat)
                        val = dur
                        unit = 's'

                    elif 'ai' == gm:
                        ai = arias_intensity(dataA, deltat)
                        val = num.log10(ai.max())
                        unit = 'm/s'

                    elif 'pga' == gm:
                        pga = num.abs(dataA).max()
                        val = num.log10((pga / G) * 100.)
                        unit = 'g%'

                    elif 'vi' == gm:
                        vi = integrated_velocity(dataV, deltat)
                        val = num.log10(vi.max())
                        unit = 'm'

                    elif 'pgv' == gm:
                        pgv = num.abs(dataV).max()
                        val = num.log10(pgv * 100.)
                        unit = 'cm/s'

                    elif 'pgd' == gm:
                        pgd = num.abs(dataD).max()
                        val = num.log10(pgd * 100.)
                        unit = 'cm'

                    elif 'SA' in gm:
                        sa_freqs.append(float(gm.rsplit('_')[-1]))
                        continue

                    GM = GMClass(
                        name=gm,
                        value=float(val),
                        unit=unit)
                    comp.gms[GM.name] = GM

                if len(sa_freqs) > 0:
                    tr_acc = eqsig.single.AccSignal(dataA, deltat)
                    tr_acc.generate_response_spectrum(sa_freqs)
                    for f, sa in zip(sa_freqs, tr_acc.s_a):
                        val = num.log10((sa / G) * 100.)
                        unit = 'UKN'
                        GM = GMClass(
                            name='SA_%s' % f,
                            value=float(val),
                            unit=unit)
                        comp.gms[GM.name] = GM

                if freqs.size != 0:
                    spec = get_spectra(dataA, deltat, tfade)
                    vals = eqsig.functions.calc_smooth_fa_spectrum(
                        spec[1], spec[0], num.array(freqs), band=40)
                    for frq, val in zip(freqs, vals):
                        GM = GMClass(
                            name='f_%s' % frq,
                            value=float(num.log10(val)),
                            unit='m/s?')

                        comp.gms[GM.name] = GM

                if deleteWvData:
                    pass
                else:
                    comp.traces = self.stations[sta].components[cha].traces

                self.stations[sta].components[comp.component] = comp

        return

    def to_dictionary(self):
        Dict = {}
        for sta, staDict in self.stations.items():
            for comp, compDict in staDict.components.items():
                if len(comp) > 3:
                    pass
                else:
                    comp = comp[-1]

                for gm, gmDict in compDict.gms.items():
                    if gm not in Dict:
                        Dict[gm] = {}

                    if comp not in Dict[gm]:
                        Dict[gm][comp] = {'vals': [], 'lons': [], 'lats': []}

                    Dict[gm][comp]['vals'].append(gmDict['value'])
                    Dict[gm][comp]['lons'].append(staDict['lon'])
                    Dict[gm][comp]['lats'].append(staDict['lat'])

        return Dict

    def to_geodataframe(self):
        Dict = {}
        lons = []
        lats = []
        for sta, staDict in self.stations.items():
            lons.append(staDict['lon'])
            lats.append(staDict['lat'])
            for comp, compDict in staDict.components.items():
                if len(comp) > 3:
                    pass
                else:
                    comp = comp[-1]

                for gm, gmDict in compDict.gms.items():
                    chagm = '%s_%s' % (comp, gm)
                    if chagm not in Dict:
                        Dict[chagm] = []

                    Dict[chagm].append(gmDict['value'])

        Dict['st_lon'] = lons
        Dict['st_lat'] = lats

        gdf = gpd.GeoDataFrame(Dict, geometry=gpd.points_from_xy(lons, lats))

        return gdf

    def create_all_waveforms_synth(self, disp=True, vel=True, acc=True):

        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                trD = self.stations[sta].components[comp].trace.copy()

                del self.stations[sta].components[comp].trace

                if disp:
                    self.stations[sta].components[comp].traces['disp'] = trD

                if vel:
                    trV = own_differentation(trD, 1)
                    self.stations[sta].components[comp].traces['vel'] = trV

                if acc:
                    trA = own_differentation(trD, 2)
                    self.stations[sta].components[comp].traces['acc'] = trA

    def resample_waveform(self, resample_f=200, resample_fac=1.):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                for key, reftr in \
                        self.stations[sta].components[comp].traces.items():
                    tr = reftr.copy()

                    # tr.ydata -= num.mean(tr.ydata)
                    if resample_f:
                        if (1 / resample_f) != tr.deltat:
                            tr.resample(1 / resample_f)
                    elif resample_fac != 1:
                        tr.resample(tr.deltat * resample_fac)

                    # cut 1.5s before and after, due to artifacts in clean way
                    tr.chop(tr.tmin + 1.5, tr.tmax - 1.5, include_last=True)
                    self.stations[sta].components[comp].traces[key] = tr


#############################
# Support functions
#############################
def merge_StationContainer(staCont1, staCont2):
    if staCont1.refSource != staCont2.refSource:
        raise ValueError(
            'EXIT\nSources are not the same for both Containers:\n'
            'Source1: {}\nSource2: {}'.format(
                staCont1.refSource.__str__,
                staCont2.refSource.__str__))

    newStaCont = StationContainer(
        refSource=staCont1.refSource,
        stations={})

    for sta1 in staCont1.stations:
        if sta1 not in staCont2.stations:
            newStaCont.stations[sta1] = staCont1.stations[sta1]

    for sta2 in staCont2.stations:
        if sta2 not in staCont1.stations:
            newStaCont.stations[sta2] = staCont2.stations[sta2]

    for sta1 in staCont1.stations:
        for sta2 in staCont2.stations:
            if sta1 != sta2:
                continue

            newStaCont.stations[sta1] = staCont1.stations[sta1]

            cmps_sc2 = staCont2.stations[sta2].components
            cmps_nsc = newStaCont.stations[sta1].components

            for comp2 in cmps_sc2:
                if comp2 in newStaCont.stations[sta1]:
                    for gm2 in cmps_sc2[comp2]:
                        if gm2 in cmps_nsc[comp2].gms:
                            if cmps_nsc[comp2].gms[gm2] == \
                                    cmps_sc2[comp2].gms[gm2]:
                                continue
                            else:
                                cmps_nsc[comp2].gms[str(gm2) + '_2'] = \
                                    cmps_sc2[comp2].gms[gm2]
                        else:
                            cmps_nsc[comp2].gms[gm2] = cmps_sc2[comp2].gms[gm2]
                else:
                    cmps_nsc[comp2] = cmps_sc2[comp2]

    return newStaCont


def integrated_velocity(array, dt):
    ai = cumtrapz(array ** 2, dx=dt, initial=0)
    return ai


def arias_intensity(array, dt):
    '''
    # from eqsig copied
    '''
    ai = num.pi / (2 * 9.81) * cumtrapz(
        array ** 2,
        dx=dt,
        initial=0)
    return ai


def create_stationdict_synthetic(traces, wvtargets):
    staDict = {}
    for tr, wvtarget in zip(traces, wvtargets):
        ns = '%s.%s' % (tr.network, tr.station)

        cha = tr.channel
        COMP = ComponentGMClass(
            component=cha)
        COMP.trace = tr

        if ns not in staDict:
            STA = StationGMClass(
                network=tr.network,
                station=tr.station,
                lat=float(wvtarget.lat),
                lon=float(wvtarget.lon),
                components={})

            STA.components[COMP.component] = COMP
            staDict[ns] = STA

        else:
            staDict[ns].components[COMP.component] = COMP

    return staDict


def get_spectra(acc, dt, tfade, minlen=200.):
    acc -= num.mean(acc)

    ndata = acc.size
    acc = acc * costaper(
        0., tfade,
        dt * (ndata - 1) - tfade,
        dt * ndata, ndata, dt)

    tlen = len(acc)
    minlen = minlen / dt
    if tlen < minlen:
        tdiff = minlen - tlen
        acc = num.pad(
            acc, (int(tdiff / 2), int(tdiff / 2)),
            'linear_ramp', end_values=(0, 0))

    points = int(len(acc) / 2)
    fa = num.fft.fft(acc)
    famp = fa[range(points)] * dt  # * Tlen
    freqs = num.arange(points) / (2 * points * dt)

    return abs(famp), freqs


def own_differentation(intr, intval=1, chop=True, transfercut=1):
    tr = intr.copy()

    vtr = tr.transfer(
        transfercut,
        transfer_function=trace.DifferentiationResponse(intval))

    if chop:
        vtr.chop(vtr.tmin + 1, vtr.tmax - 1, include_last=True)  # cleaner way

    # überarbeiten!!
    # fint = int(3 / tr.deltat)
    # lint = -fint
    # vtr.ydata = vtr.ydata[fint:lint]
    return vtr


def get_distances(lons, lats, source, dist_type='hypo'):

    hypoLat = source.lat
    hypoLon = source.lon
    hypoDepth = source.depth / 1000.

    if dist_type == 'hypo':
        dists = geodetic.distance(hypoLon, hypoLat, hypoDepth,
                                 lons, lats, 0.)
    else:
        points = []
        for ii in range(len(lons)):
            points.append(Point(lons[ii], lats[ii]))

        surface = source.surface

        if dist_type == 'rjb':
            dists = surface.get_joyner_boore_distance(
                Mesh.from_points_list(points))
        elif dist_type == 'ry0':
            dists = surface.get_ry0_distance(Mesh.from_points_list(points))
        elif dist_type == 'rx':
            dists = surface.get_rx_distance(Mesh.from_points_list(points))
        elif dist_type == 'rrup':
            dists = surface.get_min_distance(Mesh.from_points_list(points))
        else:
            raise ValueError('Wrong dist_type. Choose between:'
                             'hypo, rjb, ry0, rx or rrup.')

    dists = num.array(dists)

    return dists


def get_azimuths(lons, lats, source, aziType='hypo'):

    if aziType == 'hypo':
        hypoLat = source.lat
        hypoLon = source.lon
        azimuths = orthodrome.azimuth_numpy(
            num.array(hypoLat), num.array(hypoLon),
            num.array(lats), num.array(lons))

    elif aziType == 'rup':
        mesh = Mesh(num.array(lons), num.array(lats))
        azimuths = source.surface.get_azimuth_of_closest_point(mesh)

    elif aziType == 'centre':
        mesh = Mesh(num.array(lons), num.array(lats))
        azimuths = source.surface.get_azimuth(mesh)

    azimuths[azimuths > 180] = azimuths[azimuths > 180] - 360.

    return azimuths


def from_mtsource_to_own_source(src):
    from pyrocko import moment_tensor as pmt
    mt = pmt.MomentTensor(
        mnn=src.mnn, mee=src.mee, mdd=src.mdd,
        mne=src.mne, mnd=src.mnd, med=src.med)

    source = SourceClass(
        name='Synthetic',
        form='point',
        time=src.time,
        lon=float(src.lon),  # hypo
        lat=float(src.lat),  # hypo
        depth=float(src.depth),  # hypo
        magnitude=float(mt.moment_magnitude()),
        strike=float(mt.strike1),
        dip=float(mt.dip1),
        rake=float(mt.rake1),
        tensor=dict(
            mnn=src.mnn, mee=src.mee, mdd=src.mdd,
            mne=src.mne, mnd=src.mnd, med=src.med))

    return source


def from_rectsource_to_own_source(src):
    cp = src.outline(cs='latlondepth')

    rupture = {'UR': [cp[1][1], cp[1][0], cp[1][2] / 1000.],
               'UL': [cp[0][1], cp[0][0], cp[0][2] / 1000.],
               'LL': [cp[3][1], cp[3][0], cp[3][2] / 1000.],
               'LR': [cp[2][1], cp[2][0], cp[2][2] / 1000.]}

    # get nucleation point full coordinates
    p2 = Point(rupture['UL'][0], rupture['UL'][1], rupture['UL'][2])

    nucXfac = (src.nucleation_x + 1) / 2  # convert to 0-1 range
    nucYfac = (src.nucleation_y + 1) / 2  # convert to 0-1 range: 0 top; 1 down

    surface_width = (src.width / 1000.) * num.cos(
        num.radians(src.dip))  # length of width projected to surface
    depth_range = (src.width / 1000.) * num.sin(
        num.radians(src.dip))  # length of width projected to vertical
    plm = p2.point_at(
        surface_width * nucYfac, depth_range * nucYfac, src.strike + 90)
    phypo = plm.point_at((src.length / 1000.) * nucXfac, 0, src.strike)

    hypLon = phypo.longitude
    hypLat = phypo.latitude
    hypDepth = phypo.depth

    ownSource = SourceClass(
        name='Synthetic',
        form='rectangular',
        lon=float(hypLon),  # hypo
        lat=float(hypLat),  # hypo
        depth=float(hypDepth),  # hypo
        magnitude=float(src.magnitude),
        nucleation_x=float(src.nucleation_x),  # (-1:left edge, +1:right edge)
        nucleation_y=float(src.nucleation_y),  # (-1:upper edge, +1:lower edge)
        strike=float(src.strike),
        dip=float(src.dip),
        rake=float(src.rake),
        rupture=rupture,
        width=float(src.width) / 1000.,
        length=float(src.length) / 1000.,
        time=src.time)

    return ownSource


##############################
# Misc
#############################
def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0, min(int(snapfun(x / delta)), nmax))
    return snap


def costaper(a, b, c, d, nfreqs, deltaf):
    # from pyrocko trace
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 - 0.5 * num.cos(
        (deltaf * num.arange(hi(a), hi(b)) - a) / (b - a) * num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 + 0.5 * num.cos(
        (deltaf * num.arange(hi(c), hi(d)) - c) / (d - c) * num.pi)

    return tap


def get_gdf_Pyrocko(ii, args, src, engine, waveform_targets, srcsDict):

    logger.info('Starting: %s' % (ii))
    try:
        #############################
        # Get GMs for every Source
        #############################
        response = engine.process(src, waveform_targets)
        synthTraces = response.pyrocko_traces()

        synthStaDict = create_stationdict_synthetic(
            synthTraces, waveform_targets)
        pyrockoCont = StationContainer(refSource=src, stations=synthStaDict)
        pyrockoCont.create_all_waveforms_synth()
        pyrockoCont.resample_waveform(resample_f=20)

        pyrockoCont.get_gm_from_wv(
            imts=args.imts, freqs=args.freqs,
            H2=args.rotd100, delete=True,
            deleteWvData=True)

        gdf = pyrockoCont.to_geodataframe()

        srcsDict[ii] = gdf

        logger.info('Finished: %s' % (ii))

    except Exception:
        traceback.print_exc()
        return (ii, traceback.format_exc())
