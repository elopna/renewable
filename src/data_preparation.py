import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import phik
import os
import warnings
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

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
    wind_features = config['data']['wind_features']
    wind_features = [item.strip() for item in wind_features.split(',')]
    solar_features = config['data']['solar_features']
    solar_features = [item.strip() for item in solar_features.split(',')]
    wind_target = config['data']['wind_target']
    solar_target = config['data']['solar_target']
    return weather_data_file_name, production_data_file_name, num_samples, wind_features, solar_features, wind_target, solar_target

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

def prepare_data(production_data_file_name, weather_data_file_name, wind_features, wind_target, solar_features, solar_target):
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
    production_wind_solar = production_wind_solar.loc[production_wind_solar.index.year == 2016, :][['DE_wind_generation_actual', 'DE_solar_generation_actual']]

    #read weather data
    weather = pd.read_csv(f'data/{weather_data_file_name}',
                        parse_dates=[0], index_col=0)
    weather_by_day = weather.groupby(weather.index).mean()

    #join production and weather data
    df = pd.merge(production_wind_solar, weather_by_day, how='left', left_index=True, right_index=True)
    print('Merged dataset:')
    print(df.head(2))
    print()

    df = df.dropna()
    for col in df.columns:
        q1 = df[col].quantile(0.001)
        q2 = df[col].quantile(0.999)
        df = df[df[col].between(q1,q2)]

    #add some time-dependent features 
    df = create_time_features(df)

    print('Phik_matrix:')
    print(df.phik_matrix())
    print()

    df['cluster'] = 0
    df_wind = df[wind_features + [wind_target]].copy()
    df_solar = df[solar_features + [solar_target]].copy()
    wind_train, wind_test = train_test_split(df_wind, train_size=0.75,random_state=42)
    solar_train, solar_test = train_test_split(df_solar, train_size=0.75,random_state=42)

    #add cluster-feature
    train_labels = get_cluster_feature(wind_train, wind_train)
    test_labels = get_cluster_feature(wind_train, wind_test)
    wind_train.loc[:,'cluster'] = train_labels
    wind_test.loc[:,'cluster'] = test_labels
    train_labels = get_cluster_feature(solar_train, solar_train, eps=0.2, min_samples=7)
    test_labels = get_cluster_feature(solar_train, solar_test, eps=0.2, min_samples=7)
    solar_train.loc[:,'cluster'] = train_labels
    solar_test.loc[:,'cluster'] = test_labels

    return wind_train, wind_test, solar_train, solar_test