from src.data_preparation import prepare_data, parse_config
from src.model_training import train_model
from src.evaluation import calculate_metrics
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the configuration values
weather_data_file_name, production_data_file_name, num_samples, horizon, rolling_features, \
    quantile_fitering_columns, lower_quantile, upper_quantile, feature_sets, tune_hyperparams = parse_config(config)

# Fit and evaluate different models
for set in feature_sets:
    # Prepare data
    df_train, df_test = prepare_data(production_data_file_name, weather_data_file_name, feature_sets[set]['numeric_features']+feature_sets[set]['categorical_features'], 
                                     feature_sets[set]['target'], horizon, rolling_features, quantile_fitering_columns, lower_quantile, upper_quantile)
    feature_sets[set]['train_data'] = df_train
    feature_sets[set]['test_data'] = df_test

    # Train a model
    model, wind_train_error = train_model(feature_sets[set]['train_data'], feature_sets[set]['numeric_features'], feature_sets[set]['categorical_features'], \
                                          feature_sets[set]['target'], tune_hyperparams=tune_hyperparams)
    
    feature_sets[set]['model'] = model

metrics_df = calculate_metrics(feature_sets)
print('Models comparison:')
print(metrics_df)