#%% Imports
import argparse
from datetime import datetime as dt
from itertools import product
from pathlib import Path
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

#%% Main routine
def main(level, rgn, scale, mode, nmax, ip, year=2020, 
         max_time=90 * 60, outdir='dist', overwrite=False):
    """Compute the all-to-all distance/travel time OD matrix for all the zones 
    of the given region at the given scale.

    Parameters
    ----------
    leve : str
        Type of input region; one of 'state' (US state) & 'msa' (MSA).
    rgn : str
        Key of the target region in the format `{FIPS code}-{region-name}`.
    scale : str
        Spatial scale of analysis; one of {county, tract, bg}.
    mode : str
        Travel mode; one of {drive, walk, bike}.
    nmax : int
        Maximum batch size of the OSRM server, used to partition the zones.
    ip : str
        IP address of the local OSRM server along with the port number.
    year : int
        Year of the zones shapefile; one of {2010, 2020}.
    outdir : str | pathlib.Path
        Root folder where the output files/folders of the distances/
        travel times will be created.
    overwrite : bool
        Whether overwrite the output file if it exists.

    Returns
    -------
    None
    """
    level, rgn, mode, scale = [x.lower() for x in [level, rgn, mode, scale]]
    assert level in ['state', 'msa'], level
    assert mode in ['drive', 'bike', 'walk'], mode
    assert year in [2010, 2020], year
    label = f'{scale}_{mode}_{year}'
    outpath = Path(outdir) / f'{level}/{rgn}/{label}.parquet'
    pts = pd.read_parquet(f'zones/{level}_{year}.parquet')
    if outpath.exists() and not overwrite:
        print(f'Skipping {rgn} {label}')
        return
    tstart = dt.now()
    print(f'{tstart}: Processing {rgn} {label}')
    pts = pts[(pts['scale'] == scale) & (pts[level] == rgn)]
    geoids = pts.reset_index(drop=1)['geoid'].astype('category')
    xy = ';'.join([f'{x:.6f},{y:.6f}' for x, y in 
                   zip(pts['centerx'], pts['centery'])])
    mode_lab = dict(bike='bike', drive='driving', walk='walk')[mode]
    breaks = list(np.arange(0, len(pts), nmax)) + [len(pts)]
    parts = np.split(np.arange(0, len(pts)), breaks)[1:-1]
    od = []
    for ix, iy in tqdm(product(parts, parts), total=len(parts) ** 2):
        src = 'sources=' + ';'.join(ix.astype(str))
        trg = 'destinations=' + ';'.join(iy.astype(str))
        annot = 'annotations=distance,duration'
        url = f'{ip}/table/v1/{mode_lab}/{xy}?{annot}'
        if len(parts) > 1:
            url += f'&{src}&{trg}'
        data = requests.get(url).json()
        dist = (pd.DataFrame(data['distances'], index=ix, columns=iy)
                .reset_index().melt('index'))
        dist.columns = ['src', 'trg', 'distance']
        dur = (pd.DataFrame(data['durations'], index=ix, columns=iy)
               .reset_index().melt('index', value_name='duration'))
        df = pd.concat([dist, dur['duration']], axis=1).astype(dict(
            src=np.int32, trg=np.int32, distance=np.float32,
            duration=np.float32)).query(f'0 <= duration <= {max_time}')
        od.append(df)
    od = (pd.concat(od).reset_index(drop=1)
          .merge(geoids.rename('src_geoid'), left_on='src', right_index=True)
          .merge(geoids.rename('trg_geoid'), left_on='trg', right_index=True)
          .astype(dict(src_geoid='category', trg_geoid='category'))
          [['src_geoid', 'trg_geoid', 'distance', 'duration']])
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    od.to_parquet(outpath)
    tend = dt.now()
    print(f'{tend}: Runtime for {label}: {tend - tstart}')

#%% Script run
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-l', '--level') # level of target regions ('state' or 'msa')
    arg('-r', '--regions') # names of target regions, separated by '_'
    arg('-m', '--mode') # travel mode
    arg('-s', '--scales', default='county-tract-bg') # spatial scales
    arg('-n', '--nmax', type=int, default=3000) # max batch size
    arg('-y', '--year', type=int, default=2020) # year of zones
    arg('-d', '--domain', default='0.0.0.0') # domain of the local OSRM server
    arg('-p', '--port', type=int, default=5108) # port number
    arg('-w', '--overwrite', action=argparse.BooleanOptionalAction)
    kw = parser.parse_args() # keyword arguments
    ip = f'http://{kw.domain}:{kw.port}' # IP address of server
    for rgn in kw.regions.split('_'):
        for scale in kw.scales.split('-'):
            print(kw.level, rgn, scale, kw.mode, kw.nmax, kw.year, ip)
            try:
                main(kw.level, rgn, scale, kw.mode, kw.nmax, year=kw.year,
                    ip=ip, outdir='dist', overwrite=kw.overwrite)
            except Exception as e:
                print('ERROR:', kw.level, rgn, scale, kw.mode, kw.year, e)

#%%
