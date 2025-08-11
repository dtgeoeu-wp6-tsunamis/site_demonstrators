from pyrocko.guts import Float, String, Bool, List, Int, load,\
    StringChoice, Object

import logging
logger = logging.getLogger('ewrica.gm.config')


def get_config(*args, **kwargs):
    return GroundMotionConfig(*args, **kwargs)


class GroundMotionConfig(Object):
    config_path = String.T(
        default='',
        help='Path to the ground motion configuration file.')

    method = StringChoice.T(
        choices=['Pyrocko', 'NN'],
        default='NN',
        help='Method of the forward modeling for the ground motion values.')

    comps = List.T(
        String.T(),
        default=['Z', 'E', 'N'],
        optional=True,
        help='List of components to calculate the ground motions values for: '
             'e.g. Z, N, E.')

    gf = String.T(
        default='',
        optional=True,
        help='Path to the Green\'s functions store.')

    rotd100 = Bool.T(
        default=True,
        help='Enables the option to calculate the vector-sum of both '
             'horizontal, known as the RotD100 (BooreRef).')

    nndir = String.T(
        default='',
        optional=True,
        help='Directory where the Neuronal Networks are stored in.')

    nntype = String.T(
        default='',
        optional=True,
        help='Usage of precalculated Neuronal Networks, in the named '
             'subdirectory. (Default: Either MT or RS.)')

    imts = List.T(
        String.T(),
        default=['pga', 'pgv', 'pgd'],
        optional=True,
        help='Intensity measures to calculate.')

    freqs = List.T(
        String.T(),
        default=[],
        optional=True,
        help='Frequencies of spectral acceleration or Fourier spectrum to '
             'calculate.')

    mappoints = Int.T(
        default=10,
        help='SQRT of Number of points/locations to calculate imts for.')

    mapextent = List.T(
        Float.T(),
        default=[1., 1.],
        help='List of two numbers which define the map size in degree around '
             'the hypocenter.')

    mapmode = StringChoice.T(
        choices=['rectangular', 'circular'],
        default='rectangular',
        help='Mode of the map to be calculated for.')

    mp = Int.T(
        default=0,
        help='''
        Enables multiprocessing for the number of cores, if value > 1.
        Only valid for method: pyrocko.
        ''')

    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        logger.info('Read config from {} ...'.format(self.config_path))

        config = load(filename=self.config_path)
        self.config__ = config

        return config
