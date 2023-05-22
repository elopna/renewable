from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import numpy as np

def calculate_metrics(sets):
    """
    Calculate metrics for a set of models.

    Args:
        sets (dict): A dict with data and trained models.

    Returns:
        pd.DataFrame: A DataFrame with metrics for each model.
    """

    metrics = []

    for set in sets:
        model = sets[set]['model']
        test_X = sets[set]['test_data'][sets[set]['numeric_features']+sets[set]['categorical_features']]
        test_y = sets[set]['test_data'][sets[set]['target']]

        predictions = model.predict(test_X)
        mae = mean_absolute_error(test_y, predictions)
        mse = mean_squared_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        metrics.append({
            'Model': type(model).__name__,
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
        })

    # Convert list of dicts to DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df
