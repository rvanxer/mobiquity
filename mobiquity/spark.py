#%% Imports
import os, sys
from pathlib import Path

import pyspark
import pyspark.sql.functions as F

#%% Settings and configuration
# Make sure the same Python interpreter is used for all pyspark operations
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Set memory size and no. of cores (configured by default for UMNI lab servers)
_NODE = os.uname().nodename.split('.')[0]
_MEMORY = {
    'tnet1': '200g',
    'umni1': '36g',
    'umni2': '36g',
    'umni5': '160g',
}.get(_NODE, '16g') # for other computers, use 16 GB memory
_CORES = {
    'tnet1': 16,
    'umni1': 20,
    'umni2': 20,
    'umni5': 32,
}.get(_NODE, 8) # for other computers, use 8 virtual cores

# Default configuration of the pyspark session
# (can be overridden in SparkSession.__init__())
_DEFAULT_CONFIG = {
    'sql.shuffle.partitions': 40,
    'driver.maxResultSize': 0,
    'executor.memory': '36g',
    'executor.cores': 10,
    'cores.max': 10,
    'driver.memory': '36g',
    'default.parallelism': 12,
    'sql.session.timeZone': 'GMT',
    'sql.debug.maxToStringFields': 100,
    'sql.execution.arrow.pyspark.enabled': 'true',
    'spark.executor.memory': _MEMORY,
    'spark.driver.memory': _MEMORY,
    'spark.default.parallelism': _CORES,
}

#%% Class: Pyspark data types
class T:
    """Aliases for common pyspark data types."""
    _ = pyspark.sql.types
    
    # Simple types
    null = _.NullType()
    bool = _.BooleanType()
    int = _.IntegerType()
    int8 = _.ByteType()
    int16 = _.ShortType()
    int32 = _.IntegerType()
    int64 = _.LongType()
    float = _.FloatType()
    double = _.DoubleType()
    time = _.TimestampType()
    date = _.DateType()
    str = _.StringType()
    binary = _.BinaryType()
    
    # Callable/composite types
    array = _.ArrayType
    map = _.MapType
    field = _.StructField
    
    @staticmethod
    def matrix(dtype):
        return T.array(T.array(dtype))
    
    @staticmethod
    def schema(cols):
        """Build a pyspark schema with nullable fields from the given mapping of
        column names and data types.

        Parameters
        ----------
        cols : dict[str, <name in spark.py>]
            Columns keyed by column name. The value can either be a string
            exactly the same as one of the attribute names of class `T` or an
            instance of the attribute of class `T` directly. E.g., it can be
            either "int8" or `T.int8`.

        Returns
        -------
        pyspark.sql.types.StructField
            The desired pyspark schema object.
        """
        fields = [T.field(k, v, nullable=True) for k, v in cols.items()]
        return pyspark.sql.types.StructType(fields)


#%% Class: Spark session handler
class SparkSession:
    """A custom pyspark session handler to help with pyspark operations."""
    def __init__(self, config=None, log_level='WARN', start=True):
        """
        Parameters
        ----------
        config : dict[str, any]
            Custom configuration parameters in addition to the ones in
            `Spark.default_config`, by default listed in `project.yaml` ->
            `spark_config`.
        log_level : str
            Logging level for the project, taken from `logging` package.
        start : bool
            Whether start the pyspark session while constructing the object.
        """
        # take union of default config dictionary with given config
        config = _DEFAULT_CONFIG | (config or {})
        self.config = {'spark.' + k: v for k, v in config.items()}
        self.log_level = log_level
        self.context = None
        self.session = None
        if start:
            self.start()

    def start(self):
        """Start pyspark session and store relevant objects."""
        if not self.context and not self.session:
            # set the configuration
            self.config = pyspark.SparkConf().setAll(list(self.config.items()))
            # create the context
            self.context = pyspark.SparkContext(conf=self.config)
            # start the session and set its log level
            self.session = pyspark.sql.SparkSession(self.context)
            self.session.sparkContext.setLogLevel(self.log_level)

    def empty_df(self, cols):
        """Create an empty dataframe with the given schema.

        Parameters
        ----------
        cols : dict[str, type]
            Mapping of column names with their target pyspark data types.

        Returns
        -------
        df : pyspark.sql.DataFrame
            Empty dataframe with the given schema.
        """
        df = self.context.emptyRDD()
        return self.session.createDataFrame(df, schema=T.schema(cols))

    def read_csv(self, paths, schema=None, header=False, **kwargs):
        """Read CSV files and container folders as pyspark dataframes.

        Parameters
        ----------
        paths : str | list[str]
            Path(s) to one or more CSV file(s) or CSV-containing folder(s).
        schema : pyspark.sql.types.StructField
            Target schema of the dataframe.
        header : bool
            Whether read the first row of the file as columns.

        Returns
        -------
        pyspark.sql.DataFrame
            The loaded dataframe.
        """
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        reader = self.session.read.option('header', header)
        for k, v in kwargs.items():
            reader = reader.option(k, v)
        df = reader.csv(str(paths.pop(0)), schema)
        schema_ = df.schema
        for path in paths:
            df = df.union((reader.csv(str(path), schema_)))
        return df

    def read_parquet(self, paths):
        """Read parquet files and container folders as pyspark dataframes.

        Parameters
        ----------
        paths : str | list[str]
            Path(s) to one or more parquet file(s) or parquet-containing
            folder(s).

        Returns
        -------
        pyspark.sql.DataFrame
            The loaded dataframe.
        """
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        return self.session.read.parquet(*[str(p) for p in paths])

    def pdf2sdf(self, df):
        """Convert a Pandas dataframe to a Pyspark dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.

        Returns
        -------
        df : pyspark.sql.DataFrame
            Output dataframe (same schema as the input dataframe).
        """
        return self.session.createDataFrame(df)


