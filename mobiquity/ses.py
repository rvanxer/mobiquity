import itertools as it
import re
import requests

import pandas as pd

# base URL for the census data API
ACS_BASE_URL = 'https://api.census.gov/data'

# Census table subjects, obtained from
# https://www.census.gov/acs/www/data/data-tables-and-tools/subject-tables
ACS_SUBJECTS = { # subject_id: label
    '01': 'Age; Sex',
    '02': 'Race',
    '03': 'Hispanic or Latino Origin',
    '04': 'Ancestry',
    '05': 'Citizenship Status; Year of Entry; Foreign Born Place of Birth',
    '06': 'Place of Birth',
    '07': 'Migration/Residence 1 Year Ago',
    '08': 'Commuting (Journey to Work); Place of Work',
    '09': 'Relationship to Householder',
    '10': 'Grandparents and Grandchildren Characteristics',
    '11': 'Household Type; Family Type; Subfamilies',
    '12': 'Marital Status; Marital History',
    '13': 'Fertility',
    '14': 'School Enrollment',
    '15': 'Educational Attainment; Undergraduate Field of Degree',
    '16': 'Language Spoken at Home',
    '17': 'Poverty Status',
    '18': 'Disability Status',
    '19': 'Income',
    '20': 'Earnings',
    '21': 'Veteran Status; Period of Military Service',
    '22': 'Food Stamps/Supplemental Nutrition Assistance Program (SNAP)',
    '23': 'Employment Status; Work Status Last Year',
    '24': 'Industry, Occupation, and Class of Worker',
    '25': 'Housing Characteristics',
    '26': 'Group Quarters',
    '27': 'Health Insurance Coverage',
    '28': 'Computer and Internet Use',
    '29': 'Citizen Voting-Age Population',
    '98': 'Quality Measures',
    '99': 'Allocation Table for Any Subject'
}


def get_acs_subdivs(geo, src, year, key=None):
    """Get the list of political subdivisions and their FIPS codes for US regions
    using the US Census/ACS API.

    Parameters
    ----------
    geo : list[tuple[str, str]]
        Geography specification of the region(s) of interest.
        Examples:
            >> [('state', '18'), ('county', '*')] # all counties in Indiana
            >> [('state', '18'), ('tract', '*')] # all tracts in Indiana
    src : str
        Data source: either decennial census summary table ('sf1') or one of
        ACS datasets ('acs1', 'acs3', or 'acs5')
    year : int
        Year of the data.
    key : str
        Census data API key to be registered by each user.

    Returns
    -------
    pd.DataFrame | requests.Response
        If the request is successful, this function should return a table
        containing the names of the political subdivisions along with their
        FIPS codes.
        Example:
            >> name                     state    county
            >> ----                     -----    ------
            >> White County, Indiana       18       181
            >> ...
    """
    assert src in ['acs1', 'acs3', 'acs5', 'sf1'], f'Incorrect `src`: {src}'
    geo_ = [(x[0].replace(' ', '+'), x[1]) for x in geo]
    for_ = '&for=' + ':'.join(geo_.pop(-1))
    in_ = '&in=' + '+'.join(':'.join(x) for x in geo_)
    params = 'get=NAME' + for_ + (in_ if len(geo_) > 0 else '')
    params += (f'&key={key}' if isinstance(key, str) else '')
    pre_src = 'acs' if 'acs' in src else 'dec'
    url = f'{ACS_BASE_URL}/{year}/{pre_src}/{src}?{params}'
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list):
            cols = [x.lower() for x in data.pop(0)]
            return pd.DataFrame(data[1:], columns=cols)
    return resp


def get_acs_fields(year, src, table_type='detail'):
    """Fetch the ACS dataset description using the US Census API. This is mainly
    used to get the description of the variables in the different ACS tables.

    Parameters
    ----------
    year : int
        Year of the dataset.
    src : str
        Source of the dataset: must be one of {'acs1', 'acs5', 'acsse'}.
    table_type : str
        Table type as defined by the US Census. Allowed values are the 1 or
        2-character codes as defined in the `id` column of `mk.acs.TABLE_TYPES`.

    Returns
    -------
    pd.DataFrame
        A table containing the ACS fields with their details. Columns:
            id              Unique field ID.
            label           Description of the field.
            concept         Label of the table containing this field.
            predicateType   Data type of the output (predicate)
            group           ID of the table containing this field.
            limit           ???
            attributes      Other related fields to the estimate, e.g., margin
                            of error.
            required        ???
    """
    table_type = '/' + table_type if table_type != 'detail' else ''
    url = f'{ACS_BASE_URL}/{year}/acs/{src}{table_type}/variables.json'
    resp = requests.get(url)
    if resp.status_code == 200:
        vars_ = list(resp.json()['variables'].items())
        fields = pd.DataFrame(dict(vars_[3:])).T.rename_axis('id').reset_index()
        return fields
    elif resp.status_code == 404:
        raise ValueError('Error 404: ' + url)
    else:
        raise ValueError('Non-404 error: ' + url)


def process_acs_fields(orig_fields):
    """Clean the ACS fields table obtained from `mk.acs.fetch_fields()` so
    that the information of that table is easy to read and use.

    Parameters
    ----------
    orig_fields : pd.DataFrame
        The result table of `mk.acs.fetch_fields()`.

    Returns
    -------
    pd.DataFrame
        Cleaned table with the following columns (at least for ACS5 2020 data):
            type            Table type, defined in `mk.acs.TABLE_TYPES`.
            subject_id      2-digit character code of the table subject.
            subject         Label of the table subject.
            table_id        Unique table ID, e.g., 'B02001A'.
            number          The table number within a particular subject.
            table           Name/label of the table.
            race_id         Character code of the race identifier, if any.
            race_label      Name of the race identifier in the table, if any.
            is_PR           Whether this table is for Puerto Rico.
            field_id        Unique field ID, e.g., 'B02001A_002E'.
            field           Name/label of the field, separated in scope by `__`.
            dtype           Data type of the field output (predicate).
            L0, L1,...,L7   Columns representing the hierarchy of the field
                            name, obtained as `field.split('__')`.
    """
    # Fields
    fields = (
        orig_fields.drop(columns=['concept', 'limit', 'required', 'attributes'])
        .rename(columns={'predicateType': 'dtype', 'group': 'table_id'})
        .query('label != "Geography"')
        .assign(label=lambda df: [x.replace('!!', '__').replace(':', '')
                                  for x in df['label']]))
    expanded = (pd.DataFrame(list(zip(*it.zip_longest(
        *fields['label'].apply(lambda x: x.split('__')).tolist(), fillvalue=''
    ))), index=fields['id']).rename(columns=lambda x: f'L{x}').reset_index())
    fields = (fields.merge(expanded, on='id').sort_values('id')
              .rename(columns={'id': 'field_id', 'label': 'field'}))
    # Tables
    tab = (orig_fields.groupby(['group', 'concept'])
           .size().reset_index().drop(columns=0)
           .rename(columns={'group': 'table_id', 'concept': 'table'}))
    tab['table'] = tab['table'].str.title()
    tab['type'] = tab['table_id'].str.slice(0, 1)
    tab['subject'] = tab['table_id'].str.slice(1, 3)
    tab['number'] = tab['table_id'].str.slice(3, 6)
    tab['is_PR'] = tab['table_id'].str.endswith('PR')
    tab['race_id'] = [re.sub(r'\d', '', x[-1]) for x in
                      tab['table_id'].replace(r'PR$', '', regex=True)]
    tab['race_label'] = tab['table'].apply(
        lambda x: [y.strip() for y in x.replace(')', '').split('(')][-1])
    tab.loc[tab['race_id'] == '', 'race_label'] = ''
    tab.loc[tab['race_id'] != '', 'table'] = tab['table'].apply(
        lambda x: x.split('(')[0].strip())
    tab = tab.rename(columns={'subject': 'subject_id'})[
        ['table_id', 'type', 'subject_id', 'number', 'race_id',
         'is_PR', 'table', 'race_label']]
    # combine the subject, censustable, and field tables into one table
    subj = (pd.Series(ACS_SUBJECTS).rename('subject').rename_axis('subject_id')
            .reset_index())
    # subj = SUBJECTS.rename(columns=dict(id='subject_id', label='subject'))
    acs = (tab.merge(subj, on='subject_id')
           .merge(fields, on='table_id')
           [['type', 'subject_id', 'subject', 'table_id', 'number', 'table',
             'race_id', 'race_label', 'is_PR', 'field_id', 'field', 'dtype'] +
            list(fields.filter(regex='^L').columns)]
           .applymap(lambda x: x.lower() if type(x) == str else x))
    return acs


def search_field(fields, **params):
    """Search for the ID of a particular ACS field in the processed `fields`
    table using different search terms.

    Parameters
    ----------
    fields : pd.DataFrame
        Result of `mk.acs.process_fields()`.
    params : dict
        ...

    Returns
    -------
    str | pd.DataFrame
        Either the ID of the desired field (if one search result is found) or a
        table containing all the approximate matches.
    """
    func = params.pop('func', None)
    if 'race_id' not in params:
        params.update({'race_id': ''})
    params = list({k: v.lower() for k, v in params.items()}.items())
    query = ' and '.join(['{} == "{}"'.format(*x) for x in params])
    res = fields.query(query)
    if func is not None:
        res = res.pipe(func)
    if len(res) == 1:
        return res.iloc[0]['field_id'].upper()
    return res


def get_acs_fields(year):
    """Download the entire fields table of the ACS for a given year.
    
    Parameters
    ----------
    year : int
    
    Returns
    -------
    pd.DataFrame
        ...
    """
    url = f'https://api.census.gov/data/{year}/acs/acs5/variables'
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Couldn't fetch data: {resp.status_code}")
    df = pd.DataFrame(resp.json()[4:], columns=['fid', 'fname', 'tname'])
    df = df[df.fid.str[0].isin(['B', 'C'])].dropna(subset='tname')
    df.fname = [x.lower().replace('!!', '__').replace(
        ':', '').replace('estimate__', '') for x in df.fname]
    df.tname = [x.lower().replace('in the past 12 months', 'last year')
                .replace('in the past year', 'last year')
                .replace(f'in {year} inflation-adjusted dollars', 'in IA$')
                .replace(' years and over', '+ yr') for x in df.tname]
    df['tid'] = df.fid.str.split('_').str[0]
    df['fnum'] = df.fid.str.split('_').str[1].str[:-1]
    # filter only the given base tables
    df = df[df.tid.str[0] == 'B']
    # remove derivative tables (that have suffixes in their IDs)
    df = df[df.tid.str[1:].str.isdigit()]
    # remove tables related to survey data quality
    df = df[df.tid.str[1:3] <= '90']
    # remove detailed tables (they contain way too many variables!)
    df = df[~df.tname.str.startswith('detailed')]
    df = df.set_index('fid').sort_index()
    return df


def make_acs_field_tree(cols, outpath):
    """Export the input ACS fields dataframe into a hierarchical YAML 
    file structure for better navigating and exploring fields of 
    interest for analysis.
    
    Parameters
    ----------
    cols : pd.DataFrame
        ...
    outpath : str | pathlib.Path
        Path of the output YAML file (must end in '.yml' or '.yaml').
    
    Returns
    -------
    None
    """
    cols = cols.copy()
    cols['indent'] = cols['fname'].str.split('__').str.len()
    cols['leaf'] = list(cols['indent'].diff()[1:] <= 0) + [True]
    res = []
    for (tid, tname), df in cols.groupby(['tid', 'tname']):
        res.append(f'<{tid}> {tname}:')
        for _, r in df.iterrows():
            pfx = '  ' * r['indent'] + f'- <{r["fnum"]}>'
            name = r['fname'].split('__')[-1]
            sfx = '' if r['leaf'] else ':'
            res.append(f'{pfx} {name}{sfx}')
    with open(outpath, 'w') as f:
        f.write('\n'.join(res))


def get_acs_data(year, fields, state_codes, src='acs5',
                 key=None, chunksize=49):
    """Download the data of the given ACS fields for a given 
    region at the block group level.
    """
    scales = ('state', 'county', 'tract', 'block group')
    if isinstance(state_codes, int):
        state_codes = [state_codes]
    df_all = []
    for fips in state_codes:
        geos = list(zip(scales, [f'{fips:02}', '*', '*', '*']))
        dfState = pd.DataFrame()
        for i in range(0, len(fields), chunksize):
            cols = fields[i : (i + chunksize)]
            url = (f'https://api.census.gov/data/{year}/acs/{src}?' +
                   f'get={",".join(cols)}&for={":".join(geos[-1])}' +
                   f'&in={"+".join(":".join(x) for x in geos[:-1])}' +
                   (f'&key={key}' if isinstance(key, str) else ''))
            resp = requests.get(url.replace(' ', '+'))
            try:
                data = resp.json()
                df = pd.DataFrame(data[1:], columns=data[0])
                idx = pd.Index([''.join(x) for x in zip(*[
                    df.pop(x) for x in scales])], name='geoid')
                dfState = pd.concat([dfState, df.set_index(idx)], axis=1)
            except Exception as e:
                print('Failed fetching', cols, e)
        df_all.append(dfState.astype(float))
    df = pd.concat(df_all).reset_index()
    return df

