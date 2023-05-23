import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, train_pool, test_pool, test_y):
    """
    Optuna objective function for hyperparameter optimization of CatBoost model.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object with suggested hyperparameters.
        train_pool (catboost.Pool): A CatBoost training dataset.
        test_pool (catboost.Pool): A CatBoost test dataset.
        test_y (pd.DataFrame): A test target.

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
    test_err = np.mean(np.abs(test_y - test_preds))

    return test_err

def train_model(train, numeric_features, categorical_features, target, learning_rate=0.1, loss_function='MAE', iterations=1000, depth=6, optuna_n_trials=50, tune_hyperparams=True, n_splits=5):
    """
    Train a CatBoost model with the option of using Optuna for hyperparameter tuning. K-fold cross-validation is performed, feature importance is computed and plotted.

    Parameters
    ----------
    train : pd.DataFrame
        A DataFrame containing the training data with features and target.
    numeric_features : list
        A list of numerical feature names.
    categorical_features : list
        A list of categorical feature names.
    target : str
        The name of the target variable.
    learning_rate : float, optional
        The learning rate for the CatBoost model. Defaults to 0.1.
    loss_function : str, optional
        The loss function for the CatBoost model. Defaults to 'MAE'.
    iterations : int, optional
        The number of iterations for the CatBoost model. Defaults to 1000.
    depth : int, optional
        The depth of trees in the CatBoost model. Defaults to 6.
    optuna_n_trials : int, optional
        The number of trials for Optuna optimization. Defaults to 50.
    tune_hyperparams : bool, optional
        If True, hyperparameters are tuned using Optuna. Defaults to True.
    n_splits : int, optional
        Number of folds for K-fold cross-validation. Defaults to 5.

    Returns
    -------
    tuple
        A tuple containing the trained CatBoost model and the overall mean absolute error (MAE) of the out-of-fold predictions.

    Notes
    -----
    The function fits the CatBoost model, tunes hyperparameters if specified, generates out-of-fold predictions, 
    computes the overall MAE, and plots feature importances.
    """
    # create a kfold object
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create CatBoost datasets
    features = numeric_features + categorical_features
    X = train[features]
    y = train[target]

    # placeholder for the out of fold predictions
    oof_preds = np.zeros(X.shape[0])

    for train_index, val_index in kfold.split(X):
        # split data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # create catboost pool
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)

        # Fit the CatBoost model
        model = CatBoostRegressor(iterations=iterations, 
                                  depth=depth, 
                                  learning_rate=learning_rate, 
                                  loss_function=loss_function,
                                  random_seed=42, 
                                  verbose=False)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        # if hyperparameter tuning is enabled, use optuna
        if tune_hyperparams:
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, train_pool, val_pool, X_val, y_val, target), n_trials=optuna_n_trials, show_progress_bar=False)
            print("Best hyperparameters:", study.best_params)
            best_params = study.best_params
            model = CatBoostRegressor(**best_params, loss_function='MAE', random_seed=42, verbose=False)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        # generate and save the predictions
        oof_preds[val_index] = model.predict(val_pool)

    # calculate the overall performance
    oof_err = np.mean(np.abs(y - oof_preds))
    print(f'Overall MAE: {oof_err}')
    print()

    # Get feature importances
    feature_importances = model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})

    # Plot feature importances
    plt.figure(figsize=(6, 3))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values('importance', ascending=False))
    plt.title(f'Feature Importances for {target}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    
    return model, oof_err