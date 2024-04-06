#%% Imports
# Commonly used built-in imports
import datetime as dt
from functools import reduce
from glob import glob
import itertools as it
import os, sys
from pathlib import Path
import warnings

# Commonly used external imports
import contextily as ctx
import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
from geopandas import GeoSeries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as Arr
import pandas as pd
from pandas import DataFrame as Pdf
from pandas import Series
from pyarrow.parquet import read_schema
# from pyspark.sql import DataFrame as Sdf
import seaborn as sns
from tqdm.notebook import tqdm

# Mobiquity imports
import mobiquity.utils as U
from mobiquity.geo import CRS_DEG, CRS_M

#%% Important paths
# data folder for this package's projects
DATA = Path('../../data').resolve()

#%% Aliases of classes and data types
D = dict # Python dictionary
CAT = 'category'
I8, I16, I32, I64 = np.int8, np.int16, np.int32, np.int64
F16, F32, F64 = np.float16, np.float32, np.float64

#%% Constants
SEED = 1234 # common random state initializer for all main random operations
EPS = 1e-6 # small value to add to prevent DivisionByZeroError

#%% Custom settings for projects
# ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# set the newer pyogrio engine for geopandas for faster operations
gpd.options.io_engine = 'pyogrio'

# default plot settings
plt.rcParams.update(U.MPL_RCPARAMS)

# add the `disp` method to pandas and geopandas series & DF classes
pd.DataFrame.disp = U.disp
gpd.GeoDataFrame.disp = U.disp
gpd.GeoSeries.disp = U.disp
pd.Series.disp = U.disp
