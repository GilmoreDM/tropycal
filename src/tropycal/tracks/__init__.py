r"""Functionality for reading and analyzing storm tracks."""

from .dataset import TrackDataset
from .storm import Storm
from .season import Season
from .nrl_storm import NRLStorm
from .nrl_dataset import NRLTrackDataset

import sys
if 'sphinx' not in sys.modules:
    from .plot import TrackPlot

from .nrl_plot import NRLTrackPlot
