# GPLv3
#
# The Developers, 21st Century
import logging

import pickle
import os
#============================
# 0 = all messages are logged (default)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#===========================
import time

import numpy as num
import pandas as pd
import geopandas as gpd

from pyrocko import orthodrome
from pyrocko import moment_tensor as pmt

from openquake.hazardlib.geo import geodetic

import tensorflow as tf

import ewricagm.util as GMu
import ewricagm.sources as GMs


logger = logging.getLogger('gm.nn')


#####
# General
#####

def setup_dataframe(src, scaling_dict, inputcols,
                    azis, r_hypos, rrups, rupazis, lenfac):
    data = {}
    for params in scaling_dict.keys():
        if params == 'azistrike':
            data['azimuth'] = azis
            data['strike'] = [float(src.strike)] * lenfac

        if params == 'rhypo':
            data[params] = r_hypos

        if params == 'src_duration':
            try:
                data[params] = [float(src.stf.duration)] * lenfac
            except AttributeError:
                ########## only valid for MT source?
                logger.warning('IF RS source is used this needs to be updated!! (duration calc)')
                logger.warning(
                    'Source STF has no duration (Needs to be added).')
                # dur = GMu.calc_rupture_duration(source=src, mode='uncertain')
                dur = GMu.calc_rise_time(source=src)
                logger.info('Calculated STF duration of {} s'.format(dur))
                data[params] = [float(dur)] * lenfac

        if params == 'ev_depth':
            data[params] = [float(src.depth) / 1000.] * lenfac

        if params == 'dip':
            data[params] = [float(src.dip)] * lenfac

        if params == 'rake':
            data[params] = [float(src.rake)] * lenfac

        if params == 'magnitude':
            try:
                data[params] = [float(src.magnitude)] * lenfac
            except AttributeError:
                data[params] = [float(src.moment_magnitude)] * lenfac

        if params == 'length':
            try:
                data['length'] = [float(src.length)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no length information')

        if params == 'width':
            try:
                data['width'] = [float(src.width)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no width information')

        if params == 'nucleation_x':
            try:
                data['nucleation_x'] = [float(src.nucleation_x)] * lenfac
                data['nucleation_y'] = [float(src.nucleation_y)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no nucleation information')

        if params == 'rrup':
            data['rrup'] = rrups

        if params == 'rup_azimuth' or params == 'rup_azistrike':
            data['rup_azimuth'] = rupazis

    ## convert to azistrike
    data = calc_azistrike_numpy(data, strikecol='strike',
        azimuthcol='azimuth', azistrikecol='azistrike', delete=False)
    dropcols = ['azimuth', 'strike']

    if 'rup_azimuth' in data:
        data = calc_azistrike_numpy(data, strikecol='strike',
            azimuthcol='rup_azimuth', azistrikecol='rup_azistrike', delete=False)
        dropcols.append('rup_azimuth')
    for dcol in dropcols:
        del data[dcol]

    data = convert_distances(data)
    data = normalize(scaling_dict, data, mode='forward')

    data = pd.DataFrame(data)
    if inputcols is not None:
        cols = inputcols
    else:
        cols = [col for col in scaling_dict.keys() if col in data.columns]
    data = data[cols]

    return data


def model_predict(model, data, batchsize=100):
    return model.predict(data, batch_size=batchsize).T


def get_predict_df(model, data, targets, batchsize=100):

    pred = model_predict(model, data, batchsize=batchsize)

    predDict = {}
    for ii in range(len(pred)):
        target = targets[ii]
        predDict[target] = pred[ii]
    predDF = pd.DataFrame(predDict)

    return predDF


def get_gdf_NN_together(srcs, args, lons, lats):

    model, scaling_dict, targets, inputcols = select_NN_model(srcs[0], args)

    datas = []
    dabs_ref_time = time.time()
    for src in srcs:
        r_hypos = GMs.get_distances(lons, lats, src, dist_type='hypo')
        azis = GMs.get_azimuths(lons, lats, src, aziType='hypo')

        if args.nntype == 'MT':
            mt = src.pyrocko_moment_tensor()

            src.magnitude = mt.moment_magnitude()
            src.rake = mt.rake1
            src.dip = mt.dip1
            src.strike = mt.strike1

            rrups = None
            rupazis = None

        elif args.nntype == 'RS':

            ownsrc = GMs.from_rectsource_to_own_source(src)
            ownsrc.create_rupture_surface()

            rrups = GMs.get_distances(lons, lats, ownsrc, dist_type='rrup')
            rupazis = GMs.get_azimuths(lons, lats, ownsrc, aziType='rup')

        lenfac = len(lons)
        
        data = setup_dataframe(src, scaling_dict, inputcols,
            azis, r_hypos, rrups, rupazis, lenfac)

        # alldata = pd.concat([alldata, data], ignore_index=True)
        datas.append(data)
        # alldata.append(data)
    alldata = pd.concat(datas, ignore_index=True)
    logger.info('Finished data preparation: in %s s' % (time.time() - dabs_ref_time))

    dabs_ref_time = time.time()
    preddf = get_predict_df(model, alldata, targets, batchsize=10000)
    preddf = scale(scaling_dict, preddf, mode='inverse')
    logger.info('Finished predicting: in %s s' % (time.time() - dabs_ref_time))

    return preddf


def select_NN_model(source, args):
    if not os.path.exists(args.nndir):
        raise ValueError('NN-Directory does not exist: \'%s\'' % (args.nndir))

    # alternative work with: source.base_key()[6]
    filecore = '%s/%s/' % (args.nndir, args.nntype)

    suppfile = filecore + 'scalingdict.bin'
    # future problem with input order
    scaling_dict, targets, inputcols = pickle.load(open(suppfile, 'rb'))

    ending = 'model.h5'
    modelfile = filecore + ending
    model = tf.keras.models.load_model(modelfile)

    return model, scaling_dict, targets, inputcols


def calc_azistrike_numpy(data, strikecol='strike', azimuthcol='azimuth', azistrikecol='azistrike', delete=True):
    data[azimuthcol][data[azimuthcol] < 0] += 360
    data[azistrikecol] = data[azimuthcol] - data[strikecol]
    data[azistrikecol][data[azistrikecol] < 0] += 360

    if delete:
        del data[strikecol]
        del data[azimuthcol]

    return data


def convert_distances(data, mode=False):
    # for col in data.columns:
    for col in data.keys():

        if mode == 'inverse':
            if col in ['hypodist', 'rupdist', 'rjb', 'rrup', 'rhypo']:
                data[col] = 10 ** data[col]
        else:
            if col in ['hypodist', 'rupdist', 'rjb', 'rrup', 'rhypo']:
                data[col] = num.where(data[col] == 0.0, 0.01, data[col])
                data[col] = num.log10(data[col])

    return data


def convert_magnitude_to_moment(data):
    '''
    Convert magnitude to moment following Hanks and Kanamori (1979).
    '''
    from pyrocko.moment_tensor import magnitude_to_moment

    data['moment'] = magnitude_to_moment(data['magnitude'])

    return data


def standardize_data(data, sortcol='', targets=''):
    scaling_dict = {}
    for col in data.columns:
        if col in targets:
            continue
        if col in sortcol or col == sortcol:
            continue

        mean = num.mean(data[col])
        std = num.std(data[col])
        data[col] = (data[col] - mean) / std
        scaling_dict[col] = {'mean': mean, 'std': std}

    return data, scaling_dict


def standardize(scaling_dict, data, mode='forward'):
    ndata = {}
    for col in scaling_dict.keys():
        if col not in data:
            logger.warning(col, 'not in data')
            continue

        mean = scaling_dict[col]['mean']
        std = scaling_dict[col]['std']

        if mode == 'forward':
            ndata[col] = (data[col] - mean) / std

        elif mode == 'inverse':
            ndata[col] = mean + (data[col] * std)

        else:
            raise ValueError('Wrong scaling mode. Choose between forward or '
                             'inverse')

    return pd.DataFrame(ndata)


def standardize_column(scalingDict, data, col, mode='forward', verbose=False):

    mean = scalingDict[col]['mean']
    std = scalingDict[col]['std']

    if mode == 'forward':
        ndata = (data[col] - mean) / std

    elif mode == 'inverse':
        ndata = mean + (data[col] * std)

    else:
        logger.warning('Wrong scaling mode')
        exit()

    return ndata


def normalize_data(data, sortcol='', targets='', extra=None):
    scaling_dict = {}
    skipcol = []
    for col in data.columns:
        if col in targets:
            continue
        if col in sortcol or col == sortcol:
            continue

        valmin = num.min(data[col])
        valmax = num.max(data[col])

        if extra is not None:

            if extra in col:
                continue

            extracol = '%s%s' % (col, extra)
            logger.debug('Extracol: {}'.format(extracol))

            if extracol in data:
                valmin = min(num.min(data[extracol]), valmin)
                valmax = max(num.max(data[extracol]), valmax)

                logger.debug('Col: {}, vmin: {}, vmax: {}'.format(
                    col, valmin, valmax))
                logger.debug('Extracol: {}, vmin: {}, vmax: {}'.format(
                    extracol,
                    num.min(data[extracol]),
                    num.max(data[extracol])))
                logger.debug(
                    'Col: {}, Extracol: {}, vmin: {}, vmax: {}'.format(
                        col, extracol, valmin, valmax))

                data[extracol] = (data[extracol] - valmin) / (valmax - valmin)
                scaling_dict[extracol] = {'min': valmin, 'max': valmax}

                skipcol.append(extracol)

        data[col] = (data[col] - valmin) / (valmax - valmin)
        scaling_dict[col] = {'min': valmin, 'max': valmax}

    return data, scaling_dict


def normalize(scaling_dict, data, mode='forward', verbose=False):
    ndata = {}

    for col in scaling_dict.keys():
        if col not in data:
            if verbose is not False:
                logger.warning(col, 'not in Data')
            continue

        valmin = scaling_dict[col]['min']
        valmax = scaling_dict[col]['max']

        if mode == 'forward':
            ndata[col] = (data[col] - valmin) / (valmax - valmin)

        elif mode in ['inverse', 'reverse']:
            ndata[col] = ((valmax - valmin) * data[col]) + valmin

        else:
            raise ValueError('Wrong scaling mode. Choose between forward or '
                             'inverse')
    return ndata


def normalize_column(scalingDict, data, col, mode='forward', verbose=False):
    valmin = scalingDict[col]['min']
    valmax = scalingDict[col]['max']

    if mode == 'forward':
        ndata = (data[col] - valmin) / (valmax - valmin)

    elif mode in ['inverse', 'reverse']:
        ndata = ((valmax - valmin) * data[col]) + valmin
    else:
        logger.warning('Wrong scaling mode')
        exit()

    return ndata


def scale(scalingDict, data, mode='forward', verbose=False):
    ndata = {}

    for col in scalingDict.keys():
        if col not in data:
            if verbose is not False:
                logger.info(col, 'not in Data')
            continue

        if 'min' in scalingDict[col] and 'max' in scalingDict[col]:
            ndata[col] = normalize_column(scalingDict, data, col, mode)

        elif 'mean' in scalingDict[col] and 'std' in scalingDict[col]:
            ndata[col] = standardize_column(scalingDict, data, col, mode)
        else:
            logger.warning('Scaling dict contains nothing that can be used for scaling.')
            exit()

    return pd.DataFrame(ndata)
