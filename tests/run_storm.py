#!/usr/bin/env python3

import sys
import tropycal.tracks as tracks
import datetime as dt

tstorm = sys.argv[1]
if 'L' in tstorm:
    tstorm = tstorm[:tstorm.index('L')]

if not tstorm.isnumeric():
    print("Usage: $ run_storm.py nn\nwhere 'nn' is the storm number")
    raise SystemExit

try:
    nrldat_atl = tracks.NRLTrackDataset(basin='north_atlantic',source='nrlcotc',include_btk=False,storm=f'{tstorm}L')
except tracks.nrl_dataset.MissingData:
    print(f"Could not find data for storm {tstorm}L")
    raise SystemExit

if isinstance(nrldat_atl,tracks.NRLTrackDataset):
    storm = nrldat_atl.get_storm((f'AL{tstorm}2020'))
    storm.plot_nrl_forecast(forecast=dt.datetime(2020, 8, 25, 0, 0))
else:
    print(f"Could not find dataset for {tstorm}L")

