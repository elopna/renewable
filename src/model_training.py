import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import optuna

def objective(trial, train_pool, test_pool, test, target):
    """
    Optuna objective function for hyperparameter optimization of CatBoost model.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object with suggested hyperparameters.
        train_pool (catboost.Pool): A CatBoost training dataset.
        test_pool (catboost.Pool): A CatBoost test dataset.
        test (pd.DataFrame): A test dataset with features and target.
        target (str): The name of the target variable.

    Returns:
        float: Mean absolute error on the test dataset.
    """
    # Suggest hyperparameters using Optuna's API
    iterations = trial.suggest_int("iterations", 100, 2000)
    depth = trial.suggest_int("depth", 4, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True)

    # Train the CatBoost model with suggested hyperparameters
    model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
                            loss_function='MAE', random_seed=42, verbose=False)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    # Calculate the RMSE on the test set
    test_preds = model.predict(test_pool)
    test_err = np.mean(np.abs(test[target] - test_preds))

    return test_err

def train_model(train, test, features, target, learning_rate=0.1, loss_function='MAE', iterations=1000, depth=6, optuna_n_trials=50):
    """
    Train a CatBoost model with default and optimized hyperparameters using Optuna.

    Args:
        train (pd.DataFrame): A training dataset with features and target.
        test (pd.DataFrame): A test dataset with features and target.
        features (list): A list of feature names.
        target (str): The name of the target variable.
        learning_rate (float, optional): The learning rate for the CatBoost model. Defaults to 0.1.
        loss_function (str, optional): The loss function for the CatBoost model. Defaults to 'MAE'.
        iterations (int, optional): The number of iterations for the CatBoost model. Defaults to 1000.
        depth (int, optional): The depth of trees in the CatBoost model. Defaults to 6.
        optuna_n_trials (int, optional): The number of trials for Optuna optimization. Defaults to 50.

    Returns:
        catboost.CatBoostRegressor: The trained CatBoost model with optimized hyperparameters.
    """
    # Create CatBoost datasets
    train_pool = Pool(train[features], train[target])
    test_pool = Pool(test[features], test[target])

    # Fit the CatBoost model
    model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function=loss_function, random_seed=42, verbose=100)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    # Get feature importances
    feature_importances = model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})

    # Plot feature importances
    plt.figure(figsize=(6, 3))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values('importance', ascending=False))
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=50)
    study.optimize(lambda trial: objective(trial, train_pool, test_pool, test, target), n_trials=optuna_n_trials)
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params
    best_model = CatBoostRegressor(**best_params, loss_function='MAE', random_seed=42, verbose=100)
    best_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    # Get feature importances
    feature_importances = best_model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})

    # Plot feature importances
    plt.figure(figsize=(6, 3))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values('importance', ascending=False))
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return best_model

