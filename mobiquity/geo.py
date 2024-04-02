#%% Imports
import geopandas as gpd
import pandas as pd
from scipy.spatial import KDTree

from mobiquity.utils import filt

#%% Constants
CRS_DEG = 'EPSG:4326' # geographical CRS (unit: degree)
CRS_M = 'EPSG:3857' # spatial CRS (unit: meter)

#%% Functions
def get_tiger_shp(scale, year, state_code=None, crs=CRS_DEG,
                  columns=('geoid', 'name', 'aland', 'awater', 'geometry')):
    """Download or load the boundary shapefile of all the US 
    Census regions at the given scale and the given year.
    """
    scale = scale.lower()
    assert scale in ['bg', 'cbsa', 'county', 'place', 'state', 'tract'], scale
    year_ = max(year, 2011) # to fix the 2010 vs 2011 issue
    url = f'https://www2.census.gov/geo/tiger/TIGER{year_}/{scale.upper()}'
    # for scales available only for the entire US
    if scale in ['cbsa', 'county', 'state']:
        files = [f'tl_{year_}_us_{scale}.zip']
    else: # for scales available only at the state level
        if isinstance(state_code, str):
            state_code = [state_code]
        files = [f'tl_{year_}_{int(code):02}_{scale}.zip'
                 for code in state_code]
    df = pd.concat([gpd.read_file(f'{url}/{f}') for f in files])
    df = df.to_crs(crs).rename(columns=str.lower)
    if 'name' not in df.columns:
        df['name'] = ''
    df = df.sort_values('geoid').reset_index(drop=True)[list(columns)]
    return df


def get_urban_areas():
    """Download the census block-level boundaries of urban areas (only 
    available for 2020) from the Census website and aggregate at three 
    scales: Block group, Tract, and County.
    
    Returns
    -------
    df : pd.DataFrame
        ...
    """
    url = 'https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt'
    df = pd.read_csv(url, sep='|', encoding_errors='replace')
    df = df.rename(columns={'2020_UA_NAME': 'urba', 'GEOID': 'geoid'})
    df.geoid = df.geoid.astype(str).str.zfill(15)
    res = []
    for scale, nchar in [('Tract', 11), ('BG', 12)]:
        d = df.assign(geoid=df.geoid.str[:nchar])
        d = d.groupby(['urba', 'geoid']).size().rename('n').reset_index()
        d = d.sort_values('n', ascending=0).drop_duplicates('geoid')
        res.append(d.assign(scale=scale).sort_values('geoid'))
    df = pd.concat(res).reset_index(drop=1)
    df = df.astype({'urba': 'category', 'scale': 'category'})
    df = df[['urba', 'scale', 'geoid']]
    return df


def get_census_crosstab_2010_2020():
    """Create the relationship crosswalk matrix file between the 
    years 2010 and 2020 for the given state at the given scale.
    
    Returns
    -------
    df : pd.DataFrame
        ...
    """
    base_url = 'https://www2.census.gov/geo/docs/maps-data/data/rel2020'
    res = []
    for scale, key, nchar in [('Tract', 'tract', 11), ('BG', 'blkgrp', 12)]:
        url = f'{base_url}/{key}/tab20_{key}20_{key}10_natl.txt'
        df = pd.read_csv(url, sep='|').rename(
            columns=lambda x: x.lower().replace('_' + key, '')
            .replace('_10', '10').replace('_20', '20')
            .replace('arealand', 'aland').replace('areawater', 'awater'))
        df.geoid10 = df.geoid10.astype(str).str.zfill(nchar)
        df.geoid20 = df.geoid20.astype(str).str.zfill(nchar)
        df = df[['geoid10', 'geoid20', 'aland10', 'aland20', 'aland_part',
                 'awater10', 'awater20', 'awater_part']]
        df.insert(0, 'scale', scale)
        res.append(df)
    df = pd.concat(res).reset_index(drop=1).astype({'scale': 'category'})
    return df


def get_zone_map10to20(zones1, zones2):
    """
    
    Parameters
    ----------
    zones1, zones2 : gpd.GeoDataFrame
    
    Returns
    -------
    df : pd.DataFrame
        ...
    """
    xy1 = zones1.assign(geometry=zones1.centroid).get_coordinates()
    xy2 = zones2.assign(geometry=zones2.centroid).get_coordinates()
    zones1 = pd.concat([zones1[['geoid', 'scale']], xy1], axis=1)
    zones2 = pd.concat([zones2[['geoid', 'scale']], xy2], axis=1)
    res = []
    for scale in ['County', 'Tract', 'BG']:
        z1 = filt(zones1, scale=scale).set_index('geoid')
        z2 = filt(zones2, scale=scale).set_index('geoid')
        tree = KDTree(z1[['x', 'y']])
        _, idx = tree.query(z2[['x', 'y']])
        df = pd.DataFrame(dict(geoid1=z1.index[idx], geoid2=z2.index))
        res.append(df.assign(scale=scale))
    df = pd.concat(res).reset_index(drop=1)
    return df

