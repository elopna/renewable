# Renewable Energy Forecasting

This project aims to predict power production of renewable energy plants, specifically solar and wind energy plants. It leverages historical data, weather forecasts, and various other relevant features to predict energy output. The main tools utilized for modeling are the CatBoost library for the development of models and Optuna for hyperparameter optimization.

## Setting Up the Project

Follow the steps below to set up the project on your local machine:

1. **Create a virtual environment and activate it:**
    ```bash
    python -m venv renew_venv
    source renew_venv/bin/activate  # For Unix or MacOS
    renew_venv\Scripts\activate     # For Windows
    ```

2. **Install the necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Acquisition

The datasets `time_series_60min_singleindex.csv` and `weather_data_GER_2016.csv` are required for this project. They contain production data from solar and wind stations and weather data, respectively.

You can download these datasets from [Open Power System Data](https://data.open-power-system-data.org/).

Alternatively, you can use the `generate_data.ipynb` notebook provided in this repository to simulate data and avoid downloading. This feature is intended for quick testing and avoiding the need for data downloading, not as a replacement for real-world data during actual model training and testing.

## Configuration

Configuration of the project is performed via the `config.ini` file. Here, you can set the data paths, features for the model, and hyperparameters for the CatBoost model.

## Usage

After configuring your settings in `config.ini`, you can run the main script:

**Use the `main.py` script:** You can fit the model, and evaluate it directly via this script.

## Model

The CatBoost model is trained using a combination of numerical and categorical features specified in the `config.ini` file. The model's hyperparameters can be either set manually or optimized automatically using Optuna.

## Model Comparison

The performance of models trained with different feature sets can be compared by adding multiple feature set configurations in the `config.ini` file. The comparison results are printed in a table showing the mean absolute error (MAE) and coefficient of determination (R^2) for each model.

## Project Structure

The project is organized into several scripts and notebooks:

- `main.py`: Main script to run the entire pipeline.
- `generate_data.ipynb`: Notebook to simulate data.
- `src/data_preparation.py`: Contains functions for data preprocessing and feature engineering functions.
- `src/model_training.py`: Contains functions for model training and evaluation functions.
- `src/evaluation.py`: Contains functions for calculating metrics for a set of models.
- `src/eda.ipynb`: The notebook for exploratory data analysis, feature engineering, model training, and evaluation.

## Conclusions

The results of this project provide insights into the importance of various features and the effectiveness of the selected approach in predicting energy production for solar and wind energy plants. This project can serve as a starting point for further development and exploration of renewable energy forecasting techniques.
