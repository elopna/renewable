import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pvlib
from pvlib import location
from pvlib import irradiance
import phik
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)

def parse_config(config):
    """
    Access the configuration values from the provided config file.

    Args:
        config (ConfigParser): A ConfigParser object containing the configuration settings.

    Returns:
        tuple: A tuple containing the extracted values for weather_data_file_name,
               production_data_file_name, num_samples, wind_features, solar_features,
               wind_target, and solar_target.
    """
    weather_data_file_name = config['data']['weather_data_file_name']
    production_data_file_name = config['data']['production_data_file_name']
    num_samples = int(config['data']['num_samples'])
    horizon = int(config['data']['horizon'])
    rolling_features = config['data']['rolling_features']
    rolling_features = [item.strip() for item in rolling_features.split(',')]
    quantile_fitering_columns = config['data']['quantile_fitering_columns']
    quantile_fitering_columns = [item.strip() for item in quantile_fitering_columns.split(',')]
    lower_quantile = float(config['data']['lower_quantile'])
    upper_quantile = float(config['data']['upper_quantile'])
    tune_hyperparams = bool(config['optuna']['tune_hyperparams'].strip().lower() == 'true')

    feature_sets = defaultdict(dict)
    for section in config.sections():
        if section.startswith('feature_set_'):
            numeric_features = config[section]['numeric_features']
            feature_sets[section]['numeric_features'] = [item.strip() for item in numeric_features.split(',')]
            categorical_features = config[section]['categorical_features']
            feature_sets[section]['categorical_features'] = [item.strip() for item in categorical_features.split(',')]
            target = config[section]['target']
            feature_sets[section]['target'] = target

    return weather_data_file_name, production_data_file_name, num_samples, horizon, rolling_features, \
        quantile_fitering_columns, lower_quantile, upper_quantile, feature_sets, tune_hyperparams

def quantile_filter(df, columns, lower_quantile=0.001, upper_quantile=0.999):
    """
    Applies a quantile filter to the specified columns of a dataframe.
    
    Args:
    - df: pandas DataFrame. The input dataframe.
    - columns: list. The columns names to use for filtering.
    - lower_quantile: float. The lower quantile to use for filtering.
    - upper_quantile: float. The upper quantile to use for filtering.

    Returns:
    - df_filtered: pandas DataFrame. The dataframe after applying the quantile filter.
    """
    df_filtered = df.copy()
    columns = [col for col in columns if col in df.columns]

    for col in columns:
        q_low = df[col].quantile(lower_quantile)
        q_high = df[col].quantile(upper_quantile)
        df_filtered = df_filtered[(df[col] >= q_low) & (df[col] <= q_high)]
    
    print('Dataframe length before quantile filtering:', len(df), 'and after:', len(df_filtered))
    return df_filtered


def create_time_features(df):
    """
    Create time-related features from a DataFrame with a datetime index.

    Args:
        df (pd.DataFrame): A DataFrame with a datetime index.

    Returns:
        pd.DataFrame: A DataFrame with additional time-related features.
    """
    df['timestamp'] = pd.to_datetime(df.index)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    df['season'] = (df.index.month%12 + 3)//3  # 1:Winter, 2:Spring, 3:Summer, 4:Fall
    return df

def create_lag_features(df, lag_features, horizon):
    for f in lag_features:
        df[f'{f}_prev_1'] = df[f].shift(horizon+1)
        df[f'{f}_prev_2'] = df[f].shift(horizon+2)
    df['yesterday_power'] = df[f].shift(24)
    return df

def compute_rolling_stats(df, columns, window, horizon):
    """
    Computes rolling mean and standard deviation on a given column.

    Args:
    df (pandas.DataFrame): input data
    column (str): column name on which to compute the rolling statistics
    window (int): number of periods to include in the rolling window
    horizon (int): forecasting horizon

    Returns:
    pandas.DataFrame: data with new features
    """
    shift_size = window + horizon
    for column in columns:
        df[f'{column}_rolling_mean_{window}'] = df[column].shift(shift_size).rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].shift(shift_size).rolling(window=window).std()
        df[f'{column}_rolling_min_{window}'] = df[column].shift(shift_size).rolling(window=window).min()
        df[f'{column}_rolling_max_{window}'] = df[column].shift(shift_size).rolling(window=window).max()
        df[f'{column}_rolling_sum_{window}'] = df[column].shift(shift_size).rolling(window=window).sum()

    return df

def compute_wind_components(df, speed_col, direction_col):
    """
    Computes the northward and eastward components of the wind.

    Args:
    df (pandas.DataFrame): input data
    speed_col (str): wind speed column name
    direction_col (str): wind direction column name

    Returns:
    pandas.DataFrame: data with new features
    """
    # Convert the wind direction degrees to radians
    wind_dir_rad = np.radians(df[direction_col])

    # Compute the northward and eastward components
    df['wind_component_N'] = df[speed_col] * np.cos(wind_dir_rad)
    df['wind_component_E'] = df[speed_col] * np.sin(wind_dir_rad)

    return df


def get_cluster_feature(train, test, eps=0.1, min_samples=7):
    """
    Generate cluster-based features using DBSCAN and KNeighborsClassifier.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        eps (float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.1.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Default is 7.

    Returns:
        ndarray: A NumPy array of cluster labels for the test data.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(train)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # Find the core samples used by DBSCAN
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Remove noise points from the original dataset
    X_core = X[core_samples_mask]
    y_core = dbscan.labels_[core_samples_mask]

    # Train a KNeighborsClassifier on the core samples
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_core, y_core)

    X_new = scaler.transform(test)
    new_labels = knn.predict(X_new)

    return new_labels

def add_ghi(df):
    """
    This function takes a DataFrame with UTC timestamps, pressure, latitude, and longitude, 
    and calculates ghi (Global Horizontal Irradiance) - this is the total amount of shortwave radiation 
    received from above by a surface horizontal to the ground. This value includes both 
    Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI).
    Also the function adds columns for the solar position (apparent zenith, azimuth), 
    relative and absolute airmass, and Linke turbidity at each timestamp and location.
    
    The function uses the solarposition, atmosphere and clearsky modules 
    from the pvlib library to perform the calculations.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with 'utc_timestamp' as index (UTC), 'p' (pressure, Pa), and 'lat' (decimal degrees) and 'lon' (decimal degrees) columns. 
        The 'utc_timestamp' should be in UTC, and 'lat' and 'lon' should be in decimal degrees.
        
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with additional columns for 'apparent_zenith', 'azimuth', 'pressure', 
        'relative_airmass', 'absolute_airmass', 'linke_turbidity', and 'ghi'.
        
    """
    # Extracting latitude and longitude
    lat, lon = df['lat'], df['lon']
    
    # Creating location object
    loc = location.Location(lat, lon)

    # First, calculate the solar position for each timestamp and location
    solar_position = pvlib.solarposition.get_solarposition(df.index, df['lat'], df['lon'])
    df['apparent_zenith'] = solar_position['apparent_zenith']
    df['azimuth'] = solar_position['azimuth']
    
    # Calculate the relative and absolute air mass using pressure data
    relative_airmass = pvlib.atmosphere.get_relative_airmass(df['apparent_zenith'])
    df['relative_airmass'] = relative_airmass
    absolute_airmass = pvlib.atmosphere.get_absolute_airmass(relative_airmass, df['p'])
    df['absolute_airmass'] = absolute_airmass
    
    # Next step calculate the Linke turbidity (the amount of aerosols and other particles in the atmosphere that scatter and absorb sunlight)
    df['linke_turbidity'] = pvlib.clearsky.lookup_linke_turbidity(df.index, df['lat'].mean(), df['lon'].mean())
    
    # Finally, calculate the clear sky GHI
    clearsky = pvlib.clearsky.ineichen(df['apparent_zenith'], df['absolute_airmass'], df['linke_turbidity'])
    df['ghi'] = clearsky['ghi']
    
    return df

def prepare_data(production_data_file_name, weather_data_file_name, features, target, horizon, 
                 rolling_features, quantile_fitering_columns, lower_quantile, upper_quantile):
    """
    Read and prepare data for analysis.

    Args:
        production_data_file_name (str): The name of the production data file.
        weather_data_file_name (str): The name of the weather data file.
        wind_features (list): A list of wind-related feature names.
        wind_target (str): The name of the wind target variable.
        solar_features (list): A list of solar-related feature names.
        solar_target (str): The name of the solar target variable.

    Returns:
        tuple: A tuple containing the prepared wind_train, wind_test, solar_train, and solar_test DataFrames.
    """
    #read production data
    production_wind_solar = pd.read_csv(f'data/{production_data_file_name}', #.strip('\'')
                            usecols=(lambda s: s.startswith('utc') | s.startswith('DE')),
                            parse_dates=[0], index_col=0)
    production_wind_solar = production_wind_solar.loc[production_wind_solar.index.year == 2016, :][target]

    #read weather data
    weather = pd.read_csv(f'data/{weather_data_file_name}',
                        parse_dates=[0], index_col=0)
    weather_by_day = weather.groupby(weather.index).mean()

    #join production and weather data
    df = pd.merge(production_wind_solar, weather_by_day, how='left', left_index=True, right_index=True)
    print('Merged dataset:')
    print(df.head(2))
    print()

    print('Phik_matrix:')
    print(df.phik_matrix())
    print()

    #calculate ghi - Global Horizontal Irradiance
    df = add_ghi(df)

    #wind speed decomposition - need to add wind direction to a dataset to use this feature 
    # df = compute_wind_components(df, 'wind_speed', 'wind_direction')

    #add some time-dependent features 
    df = create_time_features(df)
    df = create_lag_features(df, [target], horizon)
    df = compute_rolling_stats(df, rolling_features, 5, horizon)

    df['cluster'] = 0
    df = df[features + [target]].copy()
    df = df.dropna()

    #apply quantile fitering
    df = quantile_filter(df, columns=quantile_fitering_columns, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    df_train, df_test = train_test_split(df, train_size=0.75,random_state=42)

    #add cluster-feature
    if 'cluster' in features:
        train_labels = get_cluster_feature(df_train, df_train)
        test_labels = get_cluster_feature(df_train, df_test)
        df_train.loc[:,'cluster'] = train_labels
        df_test.loc[:,'cluster'] = test_labels

    return df_train, df_test