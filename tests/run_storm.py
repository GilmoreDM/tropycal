#!/usr/bin/env python3

import sys
import os
import tropycal.tracks as tracks
import datetime as dt

try:
    track_file = sys.argv[1]
    image_path = sys.argv[2]
    tc_name = sys.argv[3]
    basin = sys.argv[4]
except:
    print("Usage: $ run_storm.py <datafile_with_path> <image_output_path> <storm_name> <storm_basin>")
    raise SystemExit

"""
if track_file[-1] == 'L':
    storm_num = track_file[:track_file.index('L')]
    try:
        nrldat_atl = tracks.NRLTrackDataset(basin='north_atlantic',source='CTAZ',include_btk=False,storm=f'{storm_num}L',image_path=image_path,tc_name=tc_name)
    except tracks.nrl_dataset.MissingData:
        print(f"Could not find data for storm {track_file}L")
        raise SystemExit
elif not track_file.isnumeric():
"""
if os.path.exists(track_file):
    try:
        nrldat_atl = tracks.NRLTrackDataset(basin=f'{basin}',source='CTAZ',include_btk=False,datafile=f'{track_file}',image_path=image_path,tc_name=tc_name)
    except tracks.nrl_dataset.MissingData:
        print(f"Could not use data file {track_file} for storm")
        raise SystemExit
    storm_num = track_file[track_file.index('_')+1:track_file.index('L.')]
else:
    print("Could not find datafile for storm")
    raise SystemExit


if isinstance(nrldat_atl,tracks.NRLTrackDataset):
    storm = nrldat_atl.get_storm((f'AL{storm_num}2020'))
    storm.plot_nrl_forecast(forecast=dt.datetime(2020, 8, 25, 0, 0),image_path=image_path)
else:
    print(f"Could not produce cone for {storm_num}L using dataset {track_file}")

