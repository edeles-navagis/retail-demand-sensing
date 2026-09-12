import optuna
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from .features import CATEGORICAL_COLS, prepare_pipeline_data

def train_multi_lgb_with_optuna_production(X_train, y_train, categorical_cols, model_name="model", n_trials=12):
    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 64, 256),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 8.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 8.0),
            "cat_smooth": trial.suggest_float("cat_smooth", 10.0, 80.0),
            "cat_l2": trial.suggest_float("cat_l2", 0.0, 40.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "random_state": 42,
            "n_jobs": -1,
            "force_row_wise": True,
            "verbose": -1
        }

        tscv = TimeSeriesSplit(n_splits=3)
        wapes = []

        with mlflow.start_run(nested=True):
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                base_model = lgb.LGBMRegressor(**params)
                model = MultiOutputRegressor(base_model)
                
                # Fit the multi-output model
                model.fit(X_t, y_t, categorical_feature=categorical_cols)

                preds = model.predict(X_v)
                preds = np.maximum(preds, 0)
                
                y_v_array = y_v.values
                y_sum = np.sum(y_v_array)
                if y_sum == 0:
                    wape = 0.0
                else:
                    wape = np.sum(np.abs(y_v_array - preds)) / y_sum
                
                wapes.append(wape)
                
                mlflow.log_metric(f"fold_{fold}_wape", wape)
                trial.report(wape, fold)

                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()

            avg_wape = np.mean(wapes)
            
            mlflow.log_metric("avg_wape", avg_wape)
            return avg_wape

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    print("Best parameters:", study.best_params)

    best_params = study.best_params
    
    best_params.update({
        "objective": "regression_l1",
        "metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
        "force_row_wise": True,
        "verbose": -1
    })

    print("\nFitting final production model with best hyperparameters on full training set...")
    base_final_model = lgb.LGBMRegressor(**best_params)
    final_model = MultiOutputRegressor(base_final_model)
    final_model.fit(X_train, y_train, categorical_feature=categorical_cols)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(best_params)
        preds = final_model.predict(X_train)
        preds = np.maximum(preds, 0)
        
        y_train_array = y_train.values
        train_mae = mean_absolute_error(y_train_array, preds)
        train_wape = 0.0 if np.sum(y_train_array) == 0 else np.sum(np.abs(y_train_array - preds)) / np.sum(y_train_array)

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_wape", train_wape)
        
        trusted_types = [
            "collections.OrderedDict", 
            "lightgbm.basic.Booster", 
            "lightgbm.sklearn.LGBMRegressor",
            "numpy.dtype",
            "numpy.ndarray"
        ]
        mlflow.sklearn.log_model(final_model, "model", skops_trusted_types=trusted_types)

    return final_model

def run_training_pipeline(data_path="input/store_sales_multi_store.csv"):
    X_train, X_test, y_train_multi, y_test_multi, encoder = prepare_pipeline_data(data_path)
    
    print("\nTraining Multi-Output (1h, 2h, 3h) Horizon Pipeline...")
    model_multi = train_multi_lgb_with_optuna_production(
        X_train=X_train, 
        y_train=y_train_multi, 
        categorical_cols=CATEGORICAL_COLS, 
        model_name="demand_multi", 
        n_trials=10
    )

    print("\nEvaluating Production Model on Holdout Test Set...")
    preds_multi = model_multi.predict(X_test)
    preds_multi = np.maximum(preds_multi, 0)
    
    y_test_array = y_test_multi.values
    mae_multi = mean_absolute_error(y_test_array, preds_multi)

    y_test_sum = np.sum(y_test_array)
    wape_multi = 0.0 if y_test_sum == 0 else np.sum(np.abs(y_test_array - preds_multi)) / y_test_sum
    print(f"Multi-Output Test Set Results -> Overall MAE: {mae_multi:.4f} | Overall WAPE: {wape_multi:.4f} (Accuracy: {(1-wape_multi)*100:.2f}%)")

    joblib.dump(model_multi, "output/demand_model_multi.pkl")
    joblib.dump(encoder, "output/categorical_encoder.pkl")
    print("\nProduction assets for Multi-Output Horizon locked down successfully.")

if __name__ == "__main__":
    run_training_pipeline()
