# renewable

create virtual env, activate and install essential libraries (bash):
python -m venv renew_venv

renew_venv\Scripts\activate

pip install -r requirements.txt

# Renewable Energy Forecasting

This project demonstrates the use of machine learning techniques to forecast power production for solar and wind energy plants. It uses historical data, weather forecasts, and other relevant features to predict the energy output. The project employs the CatBoost library for modeling and Optuna for hyperparameter optimization.

## Getting Started

Follow these steps to set up the project on your local machine:

### Get data and Running the Project

#### Data

You can download datasets time_series_60min_singleindex.csv (data about production of solar and wind stations) and weather_data_GER_2016.csv (weather) from https://data.open-power-system-data.org/
(also you can use generate_data.ipynb to simulate data and avoid downloading)

After cloning the repository you can:

 - use main.py to set params, fit model and evaluate it

 OR

 - open the `renewable_energy_forecasting.ipynb` notebook in Jupyter Notebook or your preferred notebook editor (e.g., JupyterLab, Visual Studio Code) and execute the cells in the notebook to perform data generation, feature engineering, model training, and evaluation.


## Conclusions

By analyzing the results, we can draw conclusions about the importance of various features and the overall effectiveness of the approach in predicting energy production for solar and wind energy plants. This project can serve as a starting point for further development and exploration of renewable energy forecasting techniques.
