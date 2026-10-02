import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from prepare_xgb_data import prepare_data


def rmsle(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            np.log1p(np.maximum(y_pred, 0))
        )
    )


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

X_train, y_train, X_val, y_val = prepare_data()


print("=" * 60)
print("FOODFLOW XGBOOST EXPERIMENT")
print("=" * 60)

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# ---------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    verbose=False
)

print("Training complete.")


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

print("\nGenerating validation predictions...")

predictions = model.predict(X_val)

predictions = np.maximum(predictions, 0)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

rmsle = rmsle(
    y_val,
    predictions
)


print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")