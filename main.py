from src.data_preparation import prepare_data, parse_config
from src.model_training import train_model
import configparser

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the configuration values
weather_data_file_name, production_data_file_name, num_samples, wind_features, solar_features, wind_target, solar_target = parse_config(config)

# Prepare data
wind_train, wind_test, solar_train, solar_test = prepare_data(production_data_file_name, weather_data_file_name, wind_features, wind_target, solar_features, solar_target)

# Train wind model
wind_model = train_model(wind_train, wind_test, wind_features, wind_target)

# Train solar model
solar_model = train_model(solar_train, solar_test, solar_features, solar_target, learning_rate=0.04)