from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / 'data' / 'processed'

x_train_rf = pd.read_csv(processed_dir / 'x_train_rf.csv')
x_val_rf = pd.read_csv(processed_dir / 'x_val_rf.csv')
y_train = np.load(processed_dir / 'y_train.npy')
y_val = np.load(processed_dir / 'y_val.npy')

rf_baseline = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=392)
rf_baseline.fit(x_train_rf, y_train)

y_val_pred_rf = rf_baseline.predict(x_val_rf)

rf_mae = mean_absolute_error(y_val, y_val_pred_rf)
rf_rmse = root_mean_squared_error(y_val, y_val_pred_rf)
rf_r2 = r2_score(y_val, y_val_pred_rf)
print('Random Forest baseline validation results')
print(f'MAE: {rf_mae:.2f} cycles')
print(f'RMSE: {rf_rmse:.2f} cycles')
print(f'R²: {rf_r2:.3f}')

outputs_dir = project_dir / 'outputs'
evals_dir = outputs_dir / 'evals'
predictions_dir = outputs_dir / 'predictions'
figures_dir = outputs_dir / 'figures'

rf_baseline_predictions = pd.DataFrame({'actual_rul': y_val, 'predicted_rul': y_val_pred_rf, 'residual': y_val_pred_rf - y_val})
rf_baseline_predictions.to_csv(predictions_dir / 'rf_baseline_predictions.csv', index=False)
rf_baseline_eval = pd.DataFrame({'model': ['Random Forest baseline'], 'mae': [rf_mae], 'rmse': [rf_rmse], 'r_squared': [rf_r2]})
rf_baseline_eval.to_csv(evals_dir / 'rf_baseline_eval.csv', index=False)

plt.figure()
plt.scatter(y_val, y_val_pred_rf, s=0.5)
min_rul = min(y_val.min(), y_val_pred_rf.min())
max_rul = max(y_val.max(), y_val_pred_rf.max())
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Random Forest Baseline Model: Validation Prediction')
plt.tight_layout()
plt.savefig(figures_dir / 'rf_baseline_actual_vs_predicted.png')
plt.close()


# ALTERNATE VERSIONS (ultimately not selected to replace baseline)

# Random Forest V2: reduced leafs (min_samples_leaf)
rf_v2 = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=392)
rf_v2.fit(x_train_rf, y_train)
y_val_pred_rf_v2 = rf_v2.predict(x_val_rf)

rf_v2_mae = mean_absolute_error(y_val, y_val_pred_rf_v2)
rf_v2_rmse = root_mean_squared_error(y_val, y_val_pred_rf_v2)
rf_v2_r2 = r2_score(y_val, y_val_pred_rf_v2)

print('Random Forest V2 validation results')
print(f'MAE: {rf_v2_mae:.2f} cycles')
print(f'RMSE: {rf_v2_rmse:.2f} cycles')
print(f'R²: {rf_v2_r2:.3f}')

# Random Forest V3: reduced trees
rf_v3 = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=392)
rf_v3.fit(x_train_rf, y_train)
y_val_pred_rf_v3 = rf_v3.predict(x_val_rf)

rf_v3_mae = mean_absolute_error(y_val, y_val_pred_rf_v3)
rf_v3_rmse = root_mean_squared_error(y_val, y_val_pred_rf_v3)
rf_v3_r2 = r2_score(y_val, y_val_pred_rf_v3)

print('Random Forest V3 validation results')
print(f'MAE: {rf_v3_mae:.2f} cycles')
print(f'RMSE: {rf_v3_rmse:.2f} cycles')
print(f'R²: {rf_v3_r2:.3f}')

# Random Forest V4: introducing max_depth (avoid memorising noise) and max_features (avoid overfitting)
rf_v4 = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features='sqrt', max_depth=20, random_state=392)
rf_v4.fit(x_train_rf, y_train)
y_val_pred_rf_v4 = rf_v4.predict(x_val_rf)

rf_v4_mae = mean_absolute_error(y_val, y_val_pred_rf_v4)
rf_v4_rmse = root_mean_squared_error(y_val, y_val_pred_rf_v4)
rf_v4_r2 = r2_score(y_val, y_val_pred_rf_v4)

print('Random Forest V4 validation results')
print(f'MAE: {rf_v4_mae:.2f} cycles')
print(f'RMSE: {rf_v4_rmse:.2f} cycles')
print(f'R²: {rf_v4_r2:.3f}')
