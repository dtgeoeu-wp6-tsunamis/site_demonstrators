# GPLv3
#
# The Developers, 21st Century
import logging
import copy

import numpy as num

from pyrocko import gf, moment_tensor as pmt
from pyrocko.orthodrome import ne_to_latlon
from pyrocko.guts import (
    Bool, Dict, Object, Float, List, String, StringChoice, Timestamp, Int,
    load_all, load_all_xml, dump_all, dump_all_xml)
from pyrocko.guts_array import Array


logger = logging.getLogger('siria.sources')


d2r = num.pi / 180.


class SiriaSourceWithMagnitude(gf.SourceWithMagnitude):
    '''Simple point source characterized by location and magnitude
    '''

    misfit = Float.T(
        default=0.0,
        optional=True,
        help='Misfit value of the source.')

    probability = Float.T(
        default=0.0,
        optional=True,
        help='Probability of the occurence of the source.')


class SiriaDCSource(gf.DCSource):
    '''Double couple point source
    '''
    misfit = Float.T(
        default=0.0,
        optional=True,
        help='Misfit value of the source.')

    probability = Float.T(
        default=0.0,
        optional=True,
        help='Probability of the occurence of the source.')


class SiriaMTSource(gf.MTSource):
    '''Full moment tensor (MT) point source
    '''

    strike1__ = Float.T(
        optional=True,
        help='Strike of the primary nodal plane.')

    dip1__ = Float.T(
        optional=True,
        help='Dip of the primary nodal plane.')

    rake1__ = Float.T(
        optional=True,
        help='Rake of the primary nodal plane.')

    strike2__ = Float.T(
        optional=True,
        help='Strike of the secondary nodal plane.')

    dip2__ = Float.T(
        optional=True,
        help='Dip of the secondary nodal plane.')

    rake2__ = Float.T(
        optional=True,
        help='Rake of the secondary nodal plane.')

    misfit = Float.T(
        default=0.0,
        optional=True,
        help='Misfit value of the source.')

    probability = Float.T(
        default=0.0,
        optional=True,
        help='Probability of the occurence of the source.')

    @property
    def strike1(self):
        return self.strike1__

    @strike1.setter
    def strike1(self, strike1):
        self.strike1__ = self.pyrocko_moment_tensor().strike1

    @property
    def dip1(self):
        return self.dip1__

    @dip1.setter
    def dip1(self, dip1):
        self.dip1__ = self.pyrocko_moment_tensor().dip1

    @property
    def rake1(self):
        return self.rake1__

    @rake1.setter
    def rake1(self, rake1):
        self.rake1__ = self.pyrocko_moment_tensor().rake1

    @property
    def strike2(self):
        return self.strike2__

    @strike2.setter
    def strike2(self, strike2):
        self.strike2__ = self.pyrocko_moment_tensor().strike2

    @property
    def dip2(self):
        return self.dip2__

    @dip2.setter
    def dip2(self, dip2):
        self.dip2__ = self.pyrocko_moment_tensor().dip2

    @property
    def rake2(self):
        return self.rake2__

    @rake2.setter
    def rake2(self, rake2):
        self.rake2__ = self.pyrocko_moment_tensor().rake2

    def base_key(self):
        return gf.MTSource.base_key(self) + (
            self.strike1__, self.dip1__, self.rake1__,
            self.strike2__, self.dip2__, self.rake2__)


class SiriaRectangularSource(gf.RectangularSource):
    '''
    Rectangular source model (Heimann, 2011)

    Simplified constant slip rectangular fault model.
    '''
    misfit = Float.T(
        default=0.0,
        optional=True,
        help='Misfit value of the source.')

    probability = Float.T(
        default=0.0,
        optional=True,
        help='Probability of the occurence of the source.')


try:
    class SiriaPseudoDynamicRupture(gf.PseudoDynamicRupture):
        '''Eikonal quasi dynamic rupture model (Dahm et al., 2020)
        '''
        misfit = Float.T(
            default=0.0,
            optional=True,
            help='Misfit value of the source.')

        probability = Float.T(
            default=0.0,
            optional=True,
            help='Probability of the occurence of the source.')

except AttributeError:
    pass


class SiriaIDSLog(Object, gf.seismosizer.Cloneable):
    '''IDS calculation and misfit informations
    '''
    stop_cause = StringChoice.T(
        choices=['no_more_subfault_signal', 'no_more_subevent_found',
                 'abnormal_scaling_result', 'max._iterations_reached',
                 'convergence', 'no_more_useful_signal'],
        optional=True,
        help='Reason for IDS to stop.')

    misfit_all = Array.T(
        default=num.array([]),
        dtype=num.float64,
        # shape=(None,),
        serialize_as='list',
        help='Iterative normalized residual variance (nrv) based on weighted '
             'nrv of the included datasets.')

    misfit_subsets = Dict.T(
        StringChoice.T(
            optional=True,
            choices=['waveform', 'sm_offset', 'gnss', 'insar'],
            help='Subset ID.'),
        Array.T(
            dtype=num.float64,
            # shape=(None,),
            serialize_as='list',
            optional=True,
            help='Subset normalized residual variance.'),
        help='Iterative normalized residual variance for the included data '
             'subsets.')

    smoothing = Array.T(
        dtype=num.int64,
        # shape=(None,),
        serialize_as='list',
        optional=True,
        help='Applied smoothing.')

    max_slip = Array.T(
        dtype=num.float64,
        # shape=(None,),
        serialize_as='list',
        optional=True,
        help='Iterative maximum slip.')

    magnitude = Array.T(
        dtype=num.float64,
        # shape=(None,),
        serialize_as='list',
        optional=True,
        help='Iterative moment magnitude Mw as in [Hanks and Kanamori, 1979].')

    forward_modeling = Bool.T(
        default=False,
        help='If ``True``, waveform forward modelling was done based on a '
             'given slip and rupture propagation pattern, else ``False``.')

    def _validate_niterations(self):
        n_iterations = self.misfit_all.shape[0]

        for ar in self.smoothing, self.max_slip, self.magnitude:
            if ar is None:
                continue

            if ar.shape[0] != n_iterations:
                raise IndexError('Arrays in {} do not have the same '
                                 'length'.format(self.T.classname))

        for k, v in self.misfit_subsets.items():
            if v is None:
                continue

            if v.shape[0] != n_iterations:
                raise IndexError('Arrays in {} do not have the same '
                                 'length'.format(self.T.classname))

    @property
    def n_iterations(self):
        '''Number of IDS iterations
        '''
        if self.forward_modeling:
            return 0

        self._validate_niterations()

        return self.misfit_all.shape[0]

    @property
    def misfit_subsets_ordered(self):
        from collections import OrderedDict
        return OrderedDict(
            (k, self.misfit_subsets[k])
            for k in ['insar', 'gnss', 'sm_offset', 'waveform']
            if k in self.misfit_subsets)


class SiriaIDSSTF(Object, gf.seismosizer.Cloneable):
    '''Source time function (moment rate) of an IDSSource (or IDSPatch)
    '''

    deltat = Float.T(
        default=1.,
        help='Sampling interval in [s] of the source time function.')

    t_min = Float.T(
        default=0.,
        help='Time of first stf sample relative to the origin time of the '
             'source in [s].')

    amplitudes = Array.T(
        default=num.array([0.]),
        # shape=(None,),
        dtype=num.float64,
        serialize_as='list',
        help='Source time function (moment rate) in [Nm/s]. Describes the '
             'moment release.')

    @property
    def n_samples(self):
        '''Number of amplitude samples
        '''
        return self.amplitudes.shape[0]

    @property
    def times(self):
        '''Time of the amplitudes samples based on `t_min` and `n_samples`
        '''
        return num.arange(self.n_samples) * self.deltat + self.t_min

    @property
    def duration(self):
        return self.deltat * (self.n_samples - 1)

    @property
    def t_max(self):
        return self.times.max()

    @property
    def effective_duration(self):
        return self.duration

    @property
    def moment_rate_function(self):
        return self.amplitudes

    @property
    def cumulative_moment_function(self):
        '''Convert moment rate amplitudes to cumulative moment (time)

        :returns: cumulative seismic moment in [Nm] per time interval between
            t_min and t_min + deltat * (n_samples - 1)
        :rtype: :py:class:`numpy.ndarray`, shape ``(n_samples,)``
        '''
        return num.cumsum(self.amplitudes * self.deltat)

    @property
    def moment(self):
        '''Convert moment rate amplitudes to cumulative total seismic moment

        :returns: final total seismic moment in [Nm]
        :rtype: float
        '''
        return float(num.sum(self.amplitudes * self.deltat))

    @property
    def magnitude(self):
        '''Convert moment rate amplitudes to moment magnitude Mw

        :returns: moment magnitude Mw
        :rtype: float
        '''
        moment = self.moment

        if moment > 0.:
            return float(pmt.moment_to_magnitude(moment))
        else:
            return 0.

    def merge(self, stf, in_place=False):
        '''
        Add another source time function for merging
        '''
        from scipy.interpolate import interp1d

        times2, amps2 = stf.times, stf.amplitudes

        interpolator = interp1d(
            times2,
            amps2,
            fill_value=0.,
            bounds_error=False)

        times_interp = num.arange(
            self.t_min,
            times2.max() + self.deltat,
            self.deltat)

        amps_interp = interpolator(times_interp)
        amps_interp[:self.n_samples] += self.amplitudes

        if in_place:
            self.amplitudes = amps_interp

        else:
            return SiriaIDSSTF(
                t_min=self.t_min, amplitudes=amps_interp, deltat=self.deltat)


class SiriaIDSPatch(gf.Location, gf.seismosizer.Cloneable):
    '''Container for IDS subfault / patch results
    '''

    time = Timestamp.T(
        optional=True,
        help='Rupture front arrival/rupture origin time')

    magnitude__ = Float.T(
        default=None,
        optional=True,
        help='Moment magnitude Mw as in [Hanks and Kanamori, 1979].')

    moment__ = Float.T(
        default=None,
        optional=True,
        help='Seismic moment M0 in [Nm].')

    stf__ = SiriaIDSSTF.T(
        default=None,
        optional=True,
        help='Source time function (moment release) of the patch in [Nm/s].')

    slip = Float.T(
        optional=True,
        help='Length of the slip vector in [m].')

    strike = Float.T(
        default=0.0,
        help='Strike direction in [deg], measured clockwise from north.')

    dip = Float.T(
        default=90.0,
        help='Dip angle in [deg], measured downward from horizontal.')

    rake = Float.T(
        default=0.0,
        help='Rake angle in [deg], '
             'measured counter-clockwise from right-horizontal '
             'in on-plane view.')

    length = Float.T(
        default=0.,
        help='Length of rectangular source area [m].')

    width = Float.T(
        default=0.,
        help='Width of rectangular source area [m].')

    x_coordinate = Float.T(
        optional=True,
        help='Coordinate along strike relative to the source anchor in [m].')

    y_coordinate = Float.T(
        optional=True,
        help='Coordinate down dip relative to the source anchor in [m].')

    anchor = StringChoice.T(
        choices=['top', 'top_left', 'top_right', 'center', 'bottom',
                 'bottom_left', 'bottom_right'],
        default='center',
        optional=True,
        help='Anchor point for positioning the plane, can be: top, center or'
             'bottom and also top_left, top_right, bottom_left,'
             'bottom_right, center_left and center_right.')

    def __init__(self, stf=None, moment=None, magnitude=None, **kwargs):
        gf.Location.__init__(self, **kwargs)

        if stf is None:
            self.stf__ = stf
        else:
            self.stf = stf

        if moment is None:
            self.moment__ = moment
        else:
            self.moment = float(moment)

        if magnitude is None:
            self.magnitude__ = magnitude
        else:
            self.magnitude = float(magnitude)

    @property
    def merge_keys(self):
        # Meter accuracy
        return (
            num.round(self.lat, decimals=6),
            num.round(self.lon, decimals=6),
            num.round(self.north_shift, decimals=1),
            num.round(self.east_shift, decimals=1),
            num.round(self.depth, decimals=1),
            num.round(self.length, decimals=1),
            num.round(self.width, decimals=1),
            num.round(self.strike, decimals=1),
            num.round(self.dip, decimals=1),
            num.round(self.rake, decimals=1))

    @property
    def stf(self):
        return self.stf__

    @stf.setter
    def stf(self, stf):
        if stf is None:
            return

        self.moment__ = stf.moment
        self.magnitude__ = stf.magnitude
        self.stf__ = stf

    @property
    def moment_rate_function(self):
        return self.stf.moment_rate_function

    @property
    def cumulative_moment_function(self):
        return self.stf.cumulative_moment_function

    @property
    def moment(self):
        return self.moment__

    @moment.setter
    def moment(self, moment):
        if moment is None:
            return

        if self.stf is not None and isinstance(self.stf, SiriaIDSSTF):
            if self.stf.moment != moment:
                logger.warn(
                    'Attribute collision: a source time function defining the '
                    'cumulative moment is set already. No explicit moment '
                    'needed. Moment is ignored.')

        self.moment__ = moment
        self.magnitude__ = pmt.moment_to_magnitude(moment)

    @property
    def magnitude(self):
        return self.magnitude__

    @magnitude.setter
    def magnitude(self, magnitude):
        if magnitude is None:
            return

        if self.stf is not None and isinstance(self.stf, SiriaIDSSTF):
            if self.stf.magnitude != magnitude:
                logger.warn(
                    'Attribute collision: a source time function defining the '
                    'magnitude is set already. No explicit magnitude needed. '
                    'Magnitude is ignored.')

        self.magnitude__ = magnitude
        self.moment__ = pmt.magnitude_to_moment(magnitude)

    @property
    def slip_function(self):
        '''BETA version Scale final slip using the cumulative moment function
        '''
        if self.moment is not None and self.moment != 0.:
            return self.cumulative_moment_function * self.slip / self.moment
        else:
            return num.zeros_like(self.cumulative_moment_function)

    @property
    def slip_rate_function(self):
        if self.moment != 0.:
            return self.moment_rate_function * self.slip / self.moment
        else:
            return num.zeros_like(self.moment_rate_function)

    def outline(self, cs='xyz'):
        points = gf.seismosizer.outline_rect_source(
            self.strike, self.dip, self.length,
            self.width, self.anchor)

        points[:, 0] += self.north_shift
        points[:, 1] += self.east_shift
        points[:, 2] += self.depth

        if cs == 'xyz':
            return points
        elif cs == 'xy':
            return points[:, :2]
        elif cs in ('latlon', 'lonlat'):
            latlon = ne_to_latlon(
                self.lat, self.lon, points[:, 0], points[:, 1])

            latlon = num.array(latlon).T
            if cs == 'latlon':
                return latlon
            else:
                return latlon[:, ::-1]


class SiriaIDSSource(SiriaIDSPatch):
    '''Container for IDS finite fault model results (all subfaults/patches)
    '''

    name = String.T(
        default='',
        optional=True,
        help='Name of the source/corresponding event')

    patches__ = List.T(
        SiriaIDSPatch.T(),
        optional=True,
        help='List of all sub faults/fault patches. All patches need to have '
             'the same starttime and sampling interval of their source time '
             'function (if given).')

    nucleation_x = Float.T(
        optional=True,
        help='Horizontal position of rupture nucleation in normalized fault '
             'plane coordinates (-1 = left edge, +1 = right edge).')

    nucleation_y = Float.T(
        optional=True,
        help='Down-dip position of rupture nucleation in normalized fault '
             'plane coordinates (-1 = upper edge, +1 = lower edge).')

    centroid_x = Float.T(
        optional=True,
        help='Horizontal position of rupture centroid in normalized fault '
             'plane coordinates (-1 = left edge, +1 = right edge).')

    centroid_y = Float.T(
        optional=True,
        help='Down-dip position of rupture centroid in normalized fault '
             'plane coordinates (-1 = upper edge, +1 = lower edge).')

    nx = Int.T(
        optional=True,
        help='Number of patches along strike.')

    ny = Int.T(
        optional=True,
        help='Number of patches down dip.')

    velocity = Float.T(
        optional=True,
        default=3500.,
        help='Average speed of rupture front [m/s].')

    misfit_log = SiriaIDSLog.T(
        optional=True,
        help='Compressed misfit and logging information of the IDS run.')

    probability = Float.T(
        default=0.0,
        optional=True,
        help='Probability of the occurence of the source.')

    curved = Bool.T(
        default=False,
        help='If True, fault patches do not need to be aligned within a '
             'rectangular source plane. Location, geometry and depth might be'
             'misleading and erroerous then.')

    def __init__(
            self, patches=None, stf=None, moment=None, magnitude=None,
            misfit_log=None,
            **kwargs):

        gf.Location.__init__(self, **kwargs)

        if patches is not None:
            self.patches = patches

        if stf is not None:
            self.stf = stf

        if moment is not None:
            self.moment = moment

        if magnitude is not None:
            self.magnitude = magnitude

        if misfit_log is not None:
            self.misfit_log = misfit_log

    def _patches_to_stf(self, patches):
        '''Calculate source time function (stf) based on sum of patches stf

        :returns: cumulative source time function summed over all patches for
            the whole source
        :rtype: :py:class:`siria.sources.SiriaIDSSTF`
        '''

        if num.unique([p.stf.deltat for p in patches]).shape[0] > 1:
            raise ValueError('Patch sampling intervals need to be equal.')

        if num.unique([p.stf.t_min for p in patches]).shape[0] > 1:
            raise ValueError('Patch start times need to be equal.')

        return SiriaIDSSTF(
            deltat=patches[0].stf.deltat,
            t_min=patches[0].stf.t_min,
            amplitudes=num.sum(
                [p.stf.amplitudes for p in patches], axis=0))

    def _patches_to_moment(self, patches):
        '''Calculate seismic moment based on patches source time function

        :returns: cumulative seismic moment M0 summed over all patches for
            the whole source
        :rtype: float
        '''

        stf = self._patches_to_stf(patches)
        return stf.moment

    def _patches_to_magnitude(self, patches):
        '''Calculate moment magnitude Mw based on patches source time function

        :returns: cumulative moment magnitude Mw summed over all patches for
            the whole source
        :rtype: float
        '''

        stf = self._patches_to_stf(patches)
        return stf.magnitude

    @property
    def misfit(self):
        '''Final misfit value (normalized residual variance) of the source.
        '''
        if self.misfit_log is None:
            return AttributeError('No misfit log given.')
        return self.misfit_log.misfit_all[-1]

    @property
    def patches(self):
        return self.patches__

    @patches.setter
    def patches(self, patches):
        if patches is None:
            return

        if all(p.stf is not None for p in patches):

            self.stf__ = self._patches_to_stf(patches)

            self.moment__ = self.stf__.moment
            self.magnitude__ = self.stf__.magnitude

        else:
            self.moment__ = num.sum([p.moment for p in patches])
            self.magnitude__ = num.sum([p.magnitude for p in patches])

        self.patches__ = patches

    @property
    def n_patches(self):
        if self.patches is not None:
            return len(self.patches)
        else:
            return 0

    @property
    def stf(self):
        return self.stf__

    @stf.setter
    def stf(self, stf):
        if stf is None:
            return

        if self.patches is not None and self.stf__ is not None:
            if self._patches_to_stf(
                    self.patches).T.properties != self.stf__.T.properties:

                logger.warn(
                    'Attribute collision: source time function already '
                    'defined based on the patch source time functions. '
                    'Source time function is ignored.')

        self.moment__ = stf.moment
        self.magnitude__ = stf.magnitude
        self.stf__ = stf

    @property
    def moment_rate_function(self):
        return self.stf.moment_rate_function

    @property
    def cumulative_moment_function(self):
        return self.stf.cumulative_moment_function

    @property
    def moment(self):
        return self.moment__

    @moment.setter
    def moment(self, moment):
        if moment is None:
            return

        if self.patches is not None and isinstance(self.patches, list):
            if all(p.moment is not None for p in self.patches) and (
                    self._patches_to_moment(self.patches) != moment):

                logger.warn(
                    'Attribute collision: the cumulative moment of the given '
                    'patches is set already. No explicit moment needed. '
                    'Moment is ignored.')

        if self.stf is not None:
            if self.stf.moment != moment:
                logger.warn(
                    'Attribute collision: a source time function defining the '
                    'cumulative moment is set already. No explicit moment '
                    'needed. Moment is ignored.')

        self.moment__ = moment
        self.magnitude__ = pmt.moment_to_magnitude(moment)

    @property
    def magnitude(self):
        return self.magnitude__

    @magnitude.setter
    def magnitude(self, magnitude):
        if magnitude is None:
            return

        if self.patches is not None and isinstance(self.patches, list):
            if all(p.magnitude is not None for p in self.patches) and (
                    self._patches_to_magnitude(self.patches) != magnitude):
                logger.warn(
                    'Attribute collision: the cumulative moment magnitude of '
                    'the given patches is set already. No explicit magnitude '
                    'needed. Magnitude is ignored')

        if self.stf is not None and isinstance(self.stf, SiriaIDSSTF):
            if self.stf.magnitude != magnitude:
                logger.warn(
                    'Attribute collision: a source time function defining the '
                    'magnitude is given already. No magnitude needed. '
                    'Magnitude is ignored.')

        self.magnitude__ = magnitude
        self.moment__ = pmt.magnitude_to_moment(magnitude)

    @classmethod
    def primitive_merge(cls, sources):
        logger.warning('Merge of patch slip could be non-accurate!')

        niter = len(sources) - 1

        # Sort sources by times
        times = num.array([s.time for s in sources])
        idcs = num.argsort(times)

        sources = [sources[idx] for idx in idcs]

        source0 = sources[0]
        if niter == 0:
            return source0

        # Check same patches
        patches = copy.copy(source0.patches)

        for iiter in range(niter):
            src = sources[iiter+1]
            merge_keys = [p.merge_keys for p in src.patches]
            merge_keys_rake = [p.merge_keys[:-1] for p in src.patches]

            for ip, p in enumerate(patches):
                try:
                    idx = merge_keys.index(p.merge_keys)
                    rake1, rake2 = p.rake, p.rake
                except ValueError:
                    try:
                        idx = merge_keys_rake.index(p.merge_keys[:-1])
                        rake1, rake2 = p.rake, src.patches[idx].rake
                        logger.warning(
                            'Rakes are different. Slip and rake are rescaled '
                            'based on given rakes.')
                    except ValueError:
                        continue

                stf_init = p.stf
                stf_add = src.patches[idx].stf
                stf_add.t_min = src.time - source0.time

                p.stf = stf_init.merge(stf_add, in_place=False)

                if p.slip and src.patches[idx].slip:
                    slip_strike = num.cos(rake1 * d2r) * p.slip + \
                        num.cos(rake2 * d2r) * src.patches[idx].slip
                    slip_dip = num.sin(rake1 * d2r) * p.slip + \
                        num.sin(rake2 * d2r) * src.patches[idx].slip

                    p.slip = num.linalg.norm([slip_strike, slip_dip])
                    p.rake = num.arcsin(slip_dip / p.slip) * d2r
                elif src.patches[idx].slip:
                    p.slip = src.patches[idx].slip
                    p.rake = src.patches[idx].rake

                patches[ip] = p

        curved = False
        if any([s.curved for s in sources]):
            curved = True

        return cls(
            curved=curved,
            lat=source0.lat,
            lon=source0.lon,
            depth=source0.depth,
            anchor=source0.anchor,
            length=source0.length if not curved else None,
            width=source0.width if not curved else None,
            strike=source0.strike if not curved else None,
            dip=source0.dip if not curved else None,
            rake=source0.dip if not curved else None,
            nucleation_x=source0.nucleation_x,
            nucleation_y=source0.nucleation_y,
            velocity=num.mean([s.velocity for s in sources]),
            patches=patches)

    def pyrocko_event(self, store=None, target=None, **kwargs):
        mt = None
        if all([i is not None for i in (self.strike, self.dip, self.rake)]):
            mt = pmt.MomentTensor(
                strike=self.strike, dip=self.dip, rake=self.rake,
                magnitude=self.magnitude)

        return gf.Source.pyrocko_event(
            self, store, target,
            moment_tensor=mt,
            magnitude=self.magnitude,
            **kwargs)

    def get_patch_attribute(self, attribute):
        if self.patches is not None:
            return num.array([getattr(p, attribute) for p in self.patches])

    def moment_centroid(self):
        pass

    def get_sparrow_geometry(self):
        try:
            from pyrocko.model import geometry
            from pyrocko import table
        except ImportError:
            raise ImportError(
                'This function is currently just available on Pyrockos '
                '"sparrow" branch.')

        if self.patches is None:
            raise ValueError('No source patches found.')

        geom = geometry.Geometry()

        vertices = num.zeros((self.n_patches*4, 5))
        vertices[:, 0] = self.lat
        vertices[:, 1] = self.lon

        for ip, p in enumerate(self.patches):
            vertices[ip*4:(ip+1)*4, 2:] = p.outline(cs='xyz')[:-1, :]

        geom.set_vertices(vertices)

        faces = num.zeros((self.n_patches, 5), dtype=num.int64)
        faces[:, :-1] = num.array([
            num.arange(ip, ip+4) for ip in range(0, self.n_patches * 4, 4)])
        faces[:, -1] = faces[:, 0]

        geom.set_faces(faces)

        geom.times = self.stf.times

        geom.event = self.pyrocko_event()

        props = [
            'slip',
            'slip_function',
            'moment_rate_function',
            'moment',
            'cumulative_moment_function']

        for prop in props:
            val = self.get_patch_attribute(prop)

            if num.all([v is None for v in val]):
                continue

            if val.ndim == 2:
                sub_headers = ['' for i in range(geom.times.shape[0])]
            else:
                sub_headers = ['']

            prop = prop.replace('_', ' ')
            prop = prop.replace('function', '(t)')
            prop = prop.replace('cumulative ', '')

            geom.add_property(
                name=table.Header(
                    name=prop,
                    label=prop,
                    sub_headers=sub_headers),
                values=val)

        return geom


class SourceHeader(Object):
    '''Gather for all guts property help strings of a source class
    '''

    help_string = String.T(
        default='',
        optional=True,
        help='Help string for the source class')

    def __init__(self, source_class=None, **kwargs):
        Object.__init__(self, **kwargs)

        if source_class is not None:
            self.set_help_string(source_class)

    def set_help_string(self, source_class):
        ''' Gather the property helps for a source class in self.help_string

        :param source_class: source class object which help functions are
            gathered in the help string
        :type source_class:
            :py:class:`siria.sources.SiriaSourceWithMagnitude` or
            :py:class:`siria.sources.SiriaDCSource` or
            :py:class:`siria.sources.SiriaMTSource` or
            :py:class:`siria.sources.SiriaRectangularSource` or
            :py:class:`siria.sources.SiriaPseudoDynamicRupture` or
            :py:class:`siria.sources.SiriaIDSSource`
        '''

        s = '%s; ' % source_class.T.classname
        s += '%s; ' % source_class.T.class_help_string()

        s += '; '.join(['%s: %s' % (p, source_class.T.get_property(p).help)
                        for p in source_class.T.propnames])

        self.help_string = s


def _source_and_header_list_from_load(sh_list, load_header):
    header = None

    if isinstance(sh_list[0], SourceHeader):
        header = sh_list[0]
        sources = sh_list[1:]

    if load_header:
        return header, sources
    else:
        return sources


def _source_and_header_list_to_dump(sources, dump_header, header):
    if not isinstance(sources, list):
        sources = [sources]

    if dump_header:
        if header is None:
            header = SourceHeader(source_class=sources[0])
    else:
        header = None

    return [header] + sources


def load_sources(load_header=False, **kwargs):
    ''' Load source ensemble yaml file with optional preceding header object

    :param load_header: If a header (:py:class:`siria.sources.SourceHeader`)
        exists, return it (True) or not (False)
    :type load_header: optional, boolean

    keyword arguments for :py:func:`pyrocko.guts.load_all` as filename

    :returns: the source header with the help string (optional) and a list of
        sources
    :rtype: :py:class:`siria.sources.SourceHeader` (optional) and list of
        :py:class:`siria.sources.SiriaSourceWithMagnitude` or
        :py:class:`siria.sources.SiriaDCSource` or
        :py:class:`siria.sources.SiriaMTSource` or
        :py:class:`siria.sources.SiriaRectangularSource` or
        :py:class:`siria.sources.SiriaPseudoDynamicRupture` or
        :py:class:`siria.sources.SiriaIDSSource`
    '''
    return _source_and_header_list_from_load(
        load_all(**kwargs), load_header=load_header)


def load_one_source(*args, **kwargs):
    return load_sources(*args, **kwargs)[0]


def dump_sources(sources, dump_header=True, header=None, **kwargs):
    ''' Dump source list with preceding header to source ensemble yaml file

    :param sources: list of sources to be dumped
    :type sources: list of
        :py:class:`siria.sources.SiriaSourceWithMagnitude` or
        :py:class:`siria.sources.SiriaDCSource` or
        :py:class:`siria.sources.SiriaMTSource` or
        :py:class:`siria.sources.SiriaRectangularSource` or
        :py:class:`siria.sources.SiriaPseudoDynamicRupture` or
        :py:class:`siria.sources.SiriaIDSSource`
    :param dump_header: If a header (:py:class:`siria.sources.SourceHeader`)
        is dumped preceding the sources (True) or not (False). If header is not
        given, it gets automatically generated based on the first element of
        the sources list.
    :type dump_header: optional, boolean
    :param header: If given, this header is used instead of an automatically
        generated one.
    :type header: optional, :py:class:`siria.sources.SourceHeader`

    keyword arguments for :py:func:`pyrocko.guts.dump_all` as filename
    '''

    dump_all(_source_and_header_list_to_dump(
        sources, dump_header=dump_header, header=header), **kwargs)


def load_sources_xml(load_header=False, **kwargs):
    ''' Load source ensemble xml file with optional preceding header object

    :param load_header: If a header (:py:class:`siria.sources.SourceHeader`)
        exists, return it (True) or not (False)
    :type load_header: optional, boolean

    keyword arguments for :py:func:`pyrocko.guts.load_all_xml` as filename

    :returns: the source header with the help string (optional) and a list of
        sources
    :rtype: :py:class:`siria.sources.SourceHeader` (optional) and list of
        :py:class:`SiriaSourceWithMagnitude` or
        :py:class:`SiriaDCSource` or
        :py:class:`SiriaMTSource` or
        :py:class:`SiriaRectangularSource` or
        :py:class:`SiriaPseudoDynamicRupture` or
        :py:class:`SiriaIDSSource`
    '''

    # ToDo still buggy! Do not use
    return _source_and_header_list_from_load(
        load_all_xml(**kwargs), load_header=load_header)


def dump_sources_xml(sources, dump_header=True, header=None, **kwargs):
    ''' Dump source list with preceding header to source ensemble xml file

    :param sources: list of sources to be dumped
    :type sources: list of
        :py:class:`siria.sources.SiriaSourceWithMagnitude` or
        :py:class:`siria.sources.SiriaDCSource` or
        :py:class:`siria.sources.SiriaMTSource` or
        :py:class:`siria.sources.SiriaRectangularSource` or
        :py:class:`siria.sources.SiriaPseudoDynamicRupture` or
        :py:class:`siria.sources.SiriaIDSSource`
    :param dump_header: If a header (:py:class:`siria.sources.SourceHeader`)
        is dumped preceding the sources (True) or not (False). If header is not
        given, it gets automatically generated based on the first element of
        the sources list.
    :type dump_header: optional, boolean
    :param header: If given, this header is used instead of an automatically
        generated one.
    :type header: optional, :py:class:`siria.sources.SourceHeader`

    keyword arguments for :py:func:`pyrocko.guts.dump_all` as filename
    '''

    dump_all_xml(_source_and_header_list_to_dump(
        sources, dump_header=dump_header, header=header), **kwargs)


source_classes = [
    SiriaSourceWithMagnitude,
    SiriaDCSource,
    SiriaMTSource,
    SiriaRectangularSource,
    SiriaIDSSource]


try:
    source_classes.append(SiriaPseudoDynamicRupture)
except NameError:
    pass


__all__ = [S.__name__ for S in source_classes] + '''
SiriaIDSPatch
SiriaIDSSTF
SourceHeader
dump_sources
dump_sources_xml
load_sources
load_sources_xml
source_classes
'''.split()
