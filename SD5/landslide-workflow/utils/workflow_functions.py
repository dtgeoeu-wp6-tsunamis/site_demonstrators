import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyrocko import moment_tensor as pmt

############################################
### Functions for step 2 of the workflow ###
############################################

def convert_csv_to_yaml(csv_file_path):
    """
    Converts a text data file into a YAML file for SiriaMTSource.

    csv_file_path: Path to the input text file.
    return: Path to the created YAML file.
    """
    if '_data.csv' in csv_file_path:
        output_yaml = csv_file_path.replace('_data.csv', '_sources.yaml')
    else:
        base_path = os.path.splitext(csv_file_path)[0]
        output_yaml = base_path + '.yaml'

    # List to store sources
    sources = []

    # Default parameters
    default_time = '1908-12-28 05:20:23.000000'
    default_stf = {'duration': 37.0, 'anchor': 0.0}
    default_stf_mode = 'post'
    default_misfit = 0.9
    default_probability = 1.0
    default_north_shift = 0.0
    default_east_shift = 0.0
    default_elevation = 0.0

    # Reading the text file
    with open(csv_file_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            # Extracting parameters (ignoring the first two columns: ID, Region)
            magnitude = values[2]
            lon = values[3]
            lat = values[4]
            depth_top = values[5] * 1000  # km -> m, depth of the top edge
            strike = values[6]
            dip = values[7]
            rake = values[8]
            length = values[9] * 1000  # km -> m
            area = values[10] * 1e6    # km² -> m²
            slip = values[11]
            id = int(values[0])

            # Calculating the fault width
            width = area / length  # m

            # Recalculating the center depth
            dip_rad = math.radians(dip)  # Converting dip to radians
            depth_center = depth_top + (width * math.sin(dip_rad)) / 2  # Center depth

            # Calculating the seismic moment from magnitude
            m0 = pmt.magnitude_to_moment(magnitude)

            # Creating the moment tensor
            mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=m0)

            # Forming the SiriaMTSource object
            source = {
                '!siria.sources.SiriaMTSource': {
                    'lat': lat,
                    'lon': lon,
                    'north_shift': default_north_shift,
                    'east_shift': default_east_shift,
                    'elevation': default_elevation,
                    'depth': depth_center,  # Using the center depth
                    'name': f'source_{id}',
                    'time': default_time,
                    'stf': '!pf.HalfSinusoidSTF',  # Direct specification of the tag
                    'duration': default_stf['duration'],
                    'anchor': default_stf['anchor'],
                    'stf_mode': default_stf_mode,
                    'mnn': mt.mnn,
                    'mee': mt.mee,
                    'mdd': mt.mdd,
                    'mne': mt.mne,
                    'mnd': mt.mnd,
                    'med': mt.med,
                    'misfit': default_misfit,
                    'probability': default_probability
                }
            }
            sources.append(source)

    # Creating the header
    header = {
        '!siria.sources.SourceHeader': {
            'help_string': (
                "SiriaMTSource; Full moment tensor (MT) point source;\n "
                " lat: latitude of reference point [deg];\n "
                " lon: longitude of reference point [deg];\n "
                " north_shift: northward cartesian offset from reference point [m];\n "
                " east_shift: eastward cartesian offset from reference point [m];\n"
                " elevation: surface elevation, above sea level [m];\n "
                " depth: depth, below surface [m];\n "
                " name: None;\n "
                " time: source origin time.;\n "
                " stf: source time function.;\n "
                " stf_mode: whether to apply source time function in pre or post-processing.;\n "
                " mnn: north-north component of moment tensor in [Nm];\n "
                " mee: east-east component of moment tensor in [Nm];\n "
                " mdd: down-down component of moment tensor in [Nm];\n "
                " mne: north-east component of moment tensor in [Nm];\n "
                " mnd: north-down component of moment tensor in [Nm];\n "
                " med: east-down component of moment tensor in [Nm];\n "
                " misfit: Misfit value of the source;\n "
                " probability: Probability of the occurrence of the source"
            )
        }
    }

    # Forming the YAML structure
    with open(output_yaml, 'w') as f:
        f.write('%YAML 1.1\n')
        f.write('--- !ewricagm.siria.sources.SourceHeader\n')
        f.write('help_string: "' + header['!siria.sources.SourceHeader']['help_string'] + '"\n')
        for source in sources:
            f.write('--- ')
            # Custom formatting for stf
            src_data = source['!siria.sources.SiriaMTSource']
            f.write('!ewricagm.siria.sources.SiriaMTSource\n')
            f.write(f'lat: {src_data["lat"]}\n')
            f.write(f'lon: {src_data["lon"]}\n')
            f.write(f'depth: {src_data["depth"]}\n')
            f.write(f'name: {src_data["name"]}\n')
            f.write(f'time: {src_data["time"]}\n')
            f.write(f'stf: {src_data["stf"]}\n')
            f.write(f'  duration: {src_data["duration"]}\n')
            f.write(f'  anchor: {src_data["anchor"]}\n')
            f.write(f'stf_mode: {src_data["stf_mode"]}\n')
            f.write(f'mnn: {src_data["mnn"]:.3e}\n')
            f.write(f'mee: {src_data["mee"]:.3e}\n')
            f.write(f'mdd: {src_data["mdd"]:.3e}\n')
            f.write(f'mne: {src_data["mne"]:.3e}\n')
            f.write(f'mnd: {src_data["mnd"]:.3e}\n')
            f.write(f'med: {src_data["med"]:.3e}\n')
            f.write(f'misfit: {src_data["misfit"]}\n')

    print(f"Created YAML file: {output_yaml}")
    return output_yaml

def plot_pga_shakemaps(data, index_to_plot):
    """
    Plot shakemaps (i.e., vertical and horizontal component of the Peak Ground Acceleration) for each PTF scenario.
    
    data: dictionary containing 'lon', 'lat', 'Z_pga', and 'H_pga' for each scenario.
    index_to_plot: index of the scenario to plot.
    """
    # Create a figure and a set of subplots with a geographic projection
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot Z_pga values at the specified scenario index
    ax_z = axes[0]
    ax_z.set_title(f'PGA_Z values for the scenario {index_to_plot}')
    scatter_z = ax_z.scatter([d['lon'] for d in data],
                             [d['lat'] for d in data],
                             c=[d['Z_pga'][index_to_plot] for d in data if d['Z_pga'] and len(d['Z_pga']) > index_to_plot],
                             s=50, marker = 's', cmap='coolwarm', transform=ccrs.Geodetic())
    ax_z.coastlines()
    ax_z.add_feature(cfeature.BORDERS)
    ax_z.add_feature(cfeature.STATES, linestyle=':')
    ax_z.set_extent([min([d['lon'] for d in data if d['lon'] is not None]) -.15,
                     max([d['lon'] for d in data if d['lon'] is not None]) +.15,
                     min([d['lat'] for d in data if d['lat'] is not None]) -.15,
                     max([d['lat'] for d in data if d['lat'] is not None]) +.15])

    # Add colorbar for Z_pga
    cbar_z = fig.colorbar(scatter_z, ax=ax_z, orientation='vertical', label='PGA_Z, log_10(g)', shrink = 0.85)

    # Plot H_pga values for the specified scenario index
    ax_h = axes[1]
    ax_h.set_title(f'PGA_H values for the scenario {index_to_plot}')
    scatter_h = ax_h.scatter([d['lon'] for d in data],
                             [d['lat'] for d in data],
                             c=[d['H_pga'][index_to_plot] for d in data if d['H_pga'] and len(d['H_pga']) > index_to_plot],
                             s=50, marker = 's', cmap='coolwarm', transform=ccrs.Geodetic())
    ax_h.coastlines()
    ax_h.add_feature(cfeature.BORDERS)
    ax_h.add_feature(cfeature.STATES, linestyle=':')
    ax_h.set_extent([min([d['lon'] for d in data if d['lon'] is not None]) -.15,
                     max([d['lon'] for d in data if d['lon'] is not None]) +.15,
                     min([d['lat'] for d in data if d['lat'] is not None]) -.15,
                     max([d['lat'] for d in data if d['lat'] is not None]) +.15])

    # Add colorbar for H_pga
    cbar_h = fig.colorbar(scatter_h, ax=ax_h, orientation='vertical', label='PGA_H, log_10(g)', shrink = 0.85)

    plt.tight_layout()
    plt.show()

############################################
### Functions for step 4 of the workflow ###
############################################
def exceedance_probability(probs, value, thresholds, weights):
    """
    Vectorized calculation of exceedance probabilities for an array of thresholds.
    Returns a numpy array of weighted exceedance probabilities for each threshold.

    probs:  2D numpy array of shape (num_clusters, num_scenarios) Probability of release by cluster and scenario.
    value: 1D numpy array of shape (num_clusters,) Value to be exceeded by cluster.
    thresholds: 1D numpy array of shape (num_thresholds,) Thresholds for exceedance.
    weights: 1D numpy array of shape (num_scenarios,) Weights for each scenario.
    """
    # Create a mask matrix: shape (num_thresholds, num_clusters)
    mask = value[None, :] >= thresholds[:, None]  # shape (T, C)

    # For each threshold, mask out clusters not exceeding the threshold
    # Set probabilities for clusters not exceeding threshold to 0 (so they don't affect the product)
    masked_probs = np.where(mask[:, :, None], probs[None, :, :], 0.0)  # shape (T, C, S)

    # Compute product over clusters (axis=1), for each threshold and scenario
    prod = np.prod(1 - masked_probs, axis=1)  # shape (T, S)
    exceed_by_scenario = 1 - prod  # shape (T, S)

    # Weighted sum over scenarios for each threshold
    weighted = np.dot(exceed_by_scenario, weights)  # shape (T,)

    return weighted

def plot_mih_percentiles(pois_coords, runup_data, mih, pois_to_plot, ax, fortitle):
    """
    Plot the Maximum Inundation Height (MIH) percentiles against the latitude of Points of Interest (POIs).
    This function assumes that the percentiles array is [0.99 0.95 0.85 0.15 0.05 0.01 0.  ]

    pois_coords: 2D numpy array of shape (num_pois, 2) with POI coordinates (longitude, latitude).
    runup_data: dictionary with coordinates ("lon" and "lat") and mih value ("h") of historical runup data.
    mih: 2D numpy array of shape (num_pois, num_percentiles)
    pois_to_plot: list of indices of POIs to plot.
    ax: matplotlib Axes object to plot on.
    fortitle: string to use in the plot title.
    """
    # Plot the maximum MIH value obtained for each specified POIs
    ax.plot(pois_coords[pois_to_plot, 1], mih[pois_to_plot, -1], 'ko', label = "Max MIH from simulations")

    # Plot the runup data
    i=0
    for lon1, lat1, h1 in zip(runup_data["lon"], runup_data["lat"], runup_data["h"]):
        if lat1 < np.max(pois_coords[pois_to_plot, 1]) and lat1 > np.min(pois_coords[pois_to_plot, 1]) and lon1 < 15.6:
            ax.plot(lat1, h1, 's', color='orange', label=' Runup Data' if i == 0 else "")
            i=i+1

    # Plot MIH at different percentiles ranges
    ax.fill_between(pois_coords[pois_to_plot, 1], mih[pois_to_plot, 0], mih[pois_to_plot, -2],
                        color='gray', alpha=0.3, label='1-99p')

    ax.fill_between(pois_coords[pois_to_plot, 1], mih[pois_to_plot, 1], mih[pois_to_plot, -3],
                        color='gray', alpha=0.5, label='5-95p')

    ax.fill_between(pois_coords[pois_to_plot, 1], mih[pois_to_plot, 2], mih[pois_to_plot, -4],
                        color='gray', alpha=0.7, label='15-85p')

    # Plot settings
    ax.set_title('MIH Data vs Simulations - ' + fortitle)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('MIH (m)')
    ax.legend(loc='upper left')

