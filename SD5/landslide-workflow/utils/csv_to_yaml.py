#Converting the *.csv data file into *.yaml

import yaml
import math
import os
from pyrocko import moment_tensor as pmt
from pyrocko.util import time_to_str

def convert_csv_to_yaml(csv_file_path):
    """
    Converts a CSV data file into a YAML file for SiriaMTSource.

    :param csv_file_path: Path to the input CSV file.
    :return: Path to the created YAML file.
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