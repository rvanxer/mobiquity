"""
Miscellaneous utility functions for file handling, plotting, display, etc.

@created March 31, 2024
"""
from datetime import datetime as dt
from functools import reduce
from pathlib import Path
import re

import contextily as ctx
import geopandas as gpd
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema


# Unit conversion factors
M2FT = 3.28084 # meter to feet
FT2M = 1 / M2FT
MI2M = 1609.34  # mile to meter
M2MI = 1 / MI2M
MI2KM = 1.60934  # mile to kilometer
KM2MI = 1 / MI2KM
SQMI2SQM = 2.59e6  # sq. mile to sq. meter
SQM2SQMI = 1 / SQMI2SQM # sq. m. to sq. mi.
MPS2MPH = 2.2369363 # meters per second to miles per hr
MPH2MPS = 1 / MPS2MPH # miles per hr to meters per second

# default plot settings
MPL_RCPARAMS = {
    'axes.edgecolor': 'k',
    'axes.edgecolor': 'k',
    'axes.formatter.use_mathtext': True,
    'axes.grid': True,
    'axes.labelcolor': 'k',
    'axes.labelsize': 13,
    'axes.linewidth': 0.5,
    'axes.titlesize': 15,
    'figure.dpi': 150,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Serif'],
    'grid.alpha': 0.15,
    'grid.color': 'k',
    'grid.linewidth': 0.5,
    'legend.edgecolor': 'none',
    'legend.facecolor': '.9',
    'legend.fontsize': 11,
    'legend.framealpha': 0.5,
    'legend.labelcolor': 'k',
    'legend.title_fontsize': 13,
    'mathtext.fontset': 'cm',
    'text.color': 'k',
    'text.color': 'k',
    # 'text.usetex': True,
    'xtick.bottom': True,
    'xtick.color': 'k',
    'xtick.labelsize': 10,
    'xtick.minor.visible': True,
    'ytick.color': 'k',
    'ytick.labelsize': 10,
    'ytick.left': True,
    'ytick.minor.visible': True,
}


def mkdir(path):
    """Shorthand for making a folder if it does not exist.

    Parameters
    ----------
    path : str | Path
        Folder to be created (if it does not exist).
    
    Returns
    -------
    Path
        Same path as input but converted to a PosixPath.
    """
    assert isinstance(path, str) or isinstance(path, Path)
    Path(path).mkdir(exist_ok=True, parents=True)
    return Path(path)


def mkfile(path):
    """Shorthand for making the base folder of the given path.
    
    Parameters
    ----------
    path : str | Path
        Path of the file to be created.
        
    Returns
    -------
    Path
        Same path as input but converted to PosixPath.
    """
    assert isinstance(path, str) or isinstance(path, Path)
    path = Path(path)
    return mkdir(path.parent) / path.name


def checkfile(path, overwrite=False, read_func=None, **read_kws):
    """Return a pandas dataframe from the provided path if it is a 
    valid CSV or parquet file.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        if read_func:
            return read_func(path, **read_kws)
        return load(path, **read_kws)


def load(path, geom=True, **kwargs):
    """Load a pandas or geopandas dataframe from path.
    
    Parameters
    ----------
    path : str | Path
        Path of the file to be created.
    geom : bool
        Whether read the geometry column (if exists) as geometry type.
    kwargs : dict
        Additional keywords for the reader functions.
        
    Returns
    -------
    df : pd.DataFrame
        Loaded data frame.
    """
    path = Path(path)
    ext = path.suffix[1:]
    if ext == 'csv':
        df = pd.read_csv(path, **kwargs)
    if ext in ['pickle', 'pkl']:
        df = pd.read_pickle(path, **kwargs)
    if ext == 'parquet':
        if geom and 'geometry' in read_schema(path).names:
            df = gpd.read_parquet(path, **kwargs)
        else:
            df = pd.read_parquet(path, **kwargs)
    return df


def normalize(x, vmin=None, vmax=None):
    """Normalize an array of values to fit in the range [0, 1].
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    vmin = vmin or np.min(x)
    vmax = vmax or np.max(x)
    return (x - vmin) / (vmax - vmin)


def standardize(x, err=1e-6):
    """Standardize an array of values (i.e., get the z-scores).
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    return (x - x.mean()) / (x.std() + err)


def factor(x):
    """Create a categorical series with categories in the original order."""
    cats = pd.Series(x).drop_duplicates()
    return pd.Categorical(x, categories=cats)


def filt(df, reset=True, **kwargs):
    """Filter a dataframe with keyword arguments."""
    masks = [(df[k] == v) for k, v in kwargs.items()]
    mask = reduce(pd.Series.mul, masks, pd.Series(True, df.index))
    df = df[mask].drop(columns=list(kwargs), errors='ignore')
    if reset:
        df = df.reset_index(drop=True)
    return df


def pdf2gdf(df, x='lon', y='lat', crs=None):
    """Convert a pandas DataFrame to a geopandas GeoDataFrame by creating 
    point geometry from the dataframes x & y columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be converted.
    x, y : str
        Column names corresponding to X (longitude) & Y (latitude) coordinates.
    crs : str | int | None
        Coordinate reference system of the converted dataframe.
    Returns
    -------
    gpd.GeoDataFrame
        Converted GDF.
    """
    geom = gpd.points_from_xy(df[x], df[y], crs=crs)
    return gpd.GeoDataFrame(df, geometry=geom)


def plot(ax=None, fig=None, size=None, dpi=None, title=None, xlab=None,
         ylab=None, xlim=None, ylim=None, titlesize=None, xlabsize=None,
         ylabsize=None, xeng=False, yeng=False, xticks=None, yticks=None,
         xticks_rotate=None, yticks_rotate=None, xlog=False, ylog=False,
         xminor=True, yminor=True, axoff=False, gridcolor=None,
         bordercolor=None, save=False, path=None):
    """Custom handler for matplotlib plotting options.

    Parameters
    ----------
    ax : plt.Axes
    fig : plt.Figure
    size : tuple[float, float]
        Figure size (width x height) (in inches).
    dpi : float
        Figure resolution measured in dots per inch (DPI).
    title, xlab, ylab : str
        Axes title, x-axis label and y-axis label.
    titlesize, xlabsize, ylabsize : float
        Font size of the axes title, xlabel, and ylabel.
    xlim, ylim : tuple[float, float]
        X-axis and y-axis lower and upper limits.
    xeng, yeng : bool
        Whether x/y-axis ticks are to be displayed in engineering format.
    xtime : bool
        Whether x-axis is to be displayed as time series.
    xticks, yticks: list-like
        Tick markers.
    xticks_rotate, yticks_rotate : float
        Extent of rotation of xticks/yticks (in degrees).
    xlog, ylog : bool
        Whether x/y-axis is to be displayed on log_10 scale.
    xminor, yminor : bool
        Whether show x.y-axis ticks.
    axoff : bool
        Whether turn off the axis boundary.
    gridcolor : str
        Color of the gridlines if a grid is shown.
    framebordercolor : str
        Color of the plotting frame's border.
    save : bool
        Whether the plotted image is to be saved to disk.
    path : str
        Path where the image is to be saved.

    Returns
    -------
    ax : plt.Axes
    """
    if isinstance(size, tuple) and fig is None:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
    ax = ax or plt.gca()
    ax.set_title(title, fontsize=titlesize or mpl.rcParams['axes.titlesize'])
    ax.set_xlabel(xlab, fontsize=xlabsize or mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(ylab, fontsize=ylabsize or mpl.rcParams['axes.labelsize'])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if xeng:
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    if yeng:
        ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xticks_rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticks_rotate)
    if yticks_rotate:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yticks_rotate)
    if xminor:
        ax.tick_params(which='minor', bottom=True)
    else:
        ax.tick_params(which='minor', bottom=False)
    if yminor:
        ax.tick_params(which='minor', left=True)
    else:
        ax.tick_params(which='minor', left=False)
    if axoff:
        ax.axis('off')
    if gridcolor:
        ax.grid(color=gridcolor)
    if bordercolor:
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_color(bordercolor)
    fig = fig or plt.gcf()
    auto_title = 'Untitled-' + dt.now().isoformat().replace(':', '-')
    if save:
        imsave(title or auto_title, root=mkdir(path), fig=fig)
    return ax


def maplot(df: gpd.GeoDataFrame, col=None, ax=None,
           size=(6, 6), dpi=150, title=None,
           cmap='rainbow', vmin=None, vmax=None,
           shrink=0.5, label=None, vert=True, scalebar=0.2,
           basemap=ctx.providers.OpenStreetMap.Mapnik, **kwargs):
    """Custom map plot for geopandas dataframes."""
    ax = ax or plot(size=size, dpi=dpi, title=title)
    orient = 'vertical' if vert else 'horizontal'
    kwds = dict(shrink=shrink, label=label, orientation=orient)
    df.plot(col, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
            legend=True, legend_kwds=kwds, **kwargs)
    if scalebar is not None:
        ax.add_artist(ScaleBar(scalebar))
    if basemap is not None:
        ctx.add_basemap(ax=ax, crs=df.crs, source=basemap)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def imsave(title=None, fig=None, ax=None, dpi=300,
           root='./fig', ext='png', opaque=True):
    """Save the current matplotlib figure to disk.
    
    Parameters
    ----------
    title : str
        Title of the filename (may contain special characters that will be
        removed).
    root : Path | str
        Folder path where the image is to be saved.
    fig : plt.Figure
        The figure which needs to be saved. If None, use the current figure.
    ax : plt.Axes
        Axes object of interest.
    dpi : int
        Dots per inch (quality) of the output image.
    ext : str
        File extension: One of supported types like 'png' and 'jpg'.
    opaque : bool
        Whether the output is to be opaque (if extension supports transparency).

    Returns
    -------
    None
    """
    fig = fig or plt.gcf()
    ax = ax or fig.axes[0]
    title = title or fig._suptitle or ax.get_title() or 'Untitled {}'.format(
        dt.now().strftime('%Y-%m-%d_%H-%m-%S'))
    title = re.sub(r'[^A-Za-z\s\d,.-]', '_', title)
    fig.savefig(f'{mkdir(root)}/{title}.{ext}', dpi=dpi, bbox_inches='tight',
                transparent=not opaque, facecolor='white' if opaque else 'auto')


def disp(x, top=1, mem=True, vert=False):
    """Custom display for pandas and geopandas dataframe and series objects 
    in Jupyter. This is a combination of methods like `head`, `dtypes`, and
    `memory_usage`.

    Parameters
    ----------
    x : pd.DataFrame | pd.Series | gpd.GeoDataFrame | gpd.GeoSeries |
        pyspark.sql.DataFrame
        Data object to be pretty displayed.
    top : int
        No. of first rows to be displayed.
    mem : bool
        Whether show size of memory usage of the object (in MiB). May be turned
        off for heavy objects whose memory consumption computation may take time.
    vert : bool-like
        Whether use vertical layout for pyspark.sql.DataFrame display.
    
    Returns
    -------
    The input object as it is.
    """
    def f(tabular: bool, crs: bool, mem=mem):
        """
        tabular : Is the object `x` a 2D (matrix or table-like) structure?
        crs : Does the object `x` have a CRS, i.e., is it geographic/geometric?
        """
        shape = ('{:,} rows x {:,} cols'.format(*x.shape) if tabular
                 else f'{x.size:,} rows')
        mem = x.memory_usage(deep=True) / (1024 ** 2) if mem else None
        memory = f'Memory: {(mem.sum() if tabular else mem):.1f} MiB'
        crs = repr(x.crs).split('\n')[0] if crs else ''
        print(shape + '; ' + memory + ('; ' + crs if crs else ''))
        if tabular:
            types = {x.index.name or '': '<' + x.dtypes.astype(str) + '>'}
            types = pd.DataFrame(types).T
            display(pd.concat([types, x.head(top).astype({'geometry': str}) 
                               if crs else x.head(top)]))
        else:
            print(x.head(top))

    if isinstance(x, gpd.GeoDataFrame):
        f(True, True)
    elif isinstance(x, pd.DataFrame):
        f(True, False)
    elif isinstance(x, gpd.GeoSeries):
        f(False, True)
    elif isinstance(x, pd.Series):
        f(False, False)
    # elif isinstance(x, pyspark.sql.DataFrame):
    #     if top == 0:
    #         x.printSchema()
    #     else:
    #         x.show(top, vertical=bool(vert))
    return x

