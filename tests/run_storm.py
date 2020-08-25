#!/usr/bin/env python3

import sys
import os
import tropycal.tracks as tracks
import datetime as dt

tstorm = sys.argv[1]
image_path = sys.argv[2]
tc_name = sys.argv[3]

if tstorm[-1] == 'L':
    tstorm_num = tstorm[:tstorm.index('L')]
    try:
        nrldat_atl = tracks.NRLTrackDataset(basin='north_atlantic',source='nrlcotc',include_btk=False,storm=f'{tstorm_num}L',image_path=image_path,tc_name=tc_name)
    except tracks.nrl_dataset.MissingData:
        print(f"Could not find data for storm {tstorm}L")
        raise SystemExit
elif not tstorm.isnumeric():
    if os.path.exists(tstorm):
        try:
            nrldat_atl = tracks.NRLTrackDataset(basin='north_atlantic',source='nrlcotc',include_btk=False,datafile=f'{tstorm}',image_path=image_path,tc_name=tc_name)
        except tracks.nrl_dataset.MissingData:
            print(f"Could not use data file {tstorm} for storm")
            raise SystemExit
        tstorm_num = tstorm[tstorm.index('_')+1:tstorm.index('L.')]
    else:
        print("Could not find datafile for storm")
        raise SystemExit


if isinstance(nrldat_atl,tracks.NRLTrackDataset):
    storm = nrldat_atl.get_storm((f'AL{tstorm_num}2020'))
    storm.plot_nrl_forecast(forecast=dt.datetime(2020, 8, 25, 0, 0),image_path=image_path)
else:
    print(f"Could not find dataset for {tstorm}L")

