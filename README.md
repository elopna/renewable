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

Alternatively, you can use the `generate_data.ipynb` notebook provided in this repository to simulate data and avoid downloading.

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




Suggestions for Future Improvements
Based on my study of recent research literature, here are some suggestions for future improvements to this project:



Wind Energy Forecasting:

Magnitude of Wind Velocity: Domain knowledge suggests that the magnitude of wind velocity, computed from horizontal and vertical components, is vital for accurate wind power forecasts. This variable should be computed and included as a feature in the model.

Turbines in Service: Including features related to wind farm operations, such as the fraction of turbines in service, could enhance forecast performance.

Artificial Neural Networks (ANNs): As per research [2], ANNs have been effective for short-term wind speed and wind power forecasting. A mixed input features-based cascade-connected artificial neural network (MIF-CANN) approach could be considered to train input features from multiple neighboring stations.

Error Correction Model: An error correction model could be created for the wind speed forecast data, as it's the variable with the most significant influence on wind power generation.



Solar Energy Forecasting:

Error Correction Model: Similar to wind energy forecasting, an error correction model for the irradiance forecast data could significantly improve the forecast accuracy for solar power generation.

PVWatts Model: The PVWatts model developed by NREL could be employed to predict the power output of the PV systems. This model uses the plane of array irradiance and the cell temperature to compute the DC output [3].



Common Ideas both for Solar and Wind Energy Forecasting:

Neighboring Meteorological Data: Including additional meteorological variables from neighboring stations could provide valuable surrounding information to the forecasting model, improving its performance.

Lost Power: The difference between the theoretical wind/solar power and the power generated by the wind turbine/solar plant at a given time point, could be included as a feature. This could be calculated as the loss on previous forecasting steps.

Ensemble Stacking: As per [4], using an ensemble stacking technique could improve model performance. This approach could utilize XGboost, Random Forest, and MLR models as base learners, and a linear Lasso model as a meta learner.



References
Hui Wei, Wen-sheng Wang, Xiao-xuan Kao, A novel approach to ultra-short-term wind power prediction based on feature engineering and informer, Energy Reports, Volume 9, 2023, pp. 1236-1250, ISSN 2352-4847, doi:10.1016/j.egyr.2022.12.062

Chen Q and Folly KA (2021) Short-Term Wind Power Forecasting Using Mixed Input Feature-Based Cascade-connected Artificial Neural Networks. Front. Energy Res. 9:634639. doi: 10.3389/fenrg.2021.634639

A Study on the Wind Power Forecasting Model Using Transfer Learning Approach by JeongRim Oh, JongJin Park, ChangSoo Ok, ChungHun Ha, Hong-Bae Jun, Electronics 2022, 11(24), 4125; doi:10.3390/electronics11244125

Ibtihal Ait Abdelmoula, Said Elhamaoui, Omaima Elalani, Abdellatif Ghennioui, Mohamed El Aroussi (2022) A photovoltaic power prediction approach enhanced by feature engineering and stacked machine learning model. Energy Reports, Volume 8, Supplement 9, pp.1288-1300.



I hope these suggestions could be useful for future improvements of the Renewable Energy Forecasting Project.