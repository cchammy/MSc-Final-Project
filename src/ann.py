from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / 'data' / 'processed'

tf.keras.utils.set_random_seed(392)

x_train_ann = pd.read_csv(processed_dir / 'x_train_ann.csv')
x_val_ann = pd.read_csv(processed_dir / 'x_val_ann.csv')
y_train = np.load(processed_dir / 'y_train.npy')
y_val = np.load(processed_dir / 'y_val.npy')

ann_baseline = tf.keras.Sequential(name='ann_baseline')
ann_baseline.add(tf.keras.layers.Input(shape=(x_train_ann.shape[1],), name='input_layer'))
ann_baseline.add(tf.keras.layers.Dense(32, activation='relu', name='hidden_layer'))
ann_baseline.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
ann_baseline.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_baseline = ann_baseline.fit(x_train_ann, y_train, validation_data=(x_val_ann, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

y_val_pred_ann = ann_baseline.predict(x_val_ann, verbose=0).ravel()

ann_mae = mean_absolute_error(y_val, y_val_pred_ann)
ann_rmse = root_mean_squared_error(y_val, y_val_pred_ann)
ann_r2 = r2_score(y_val, y_val_pred_ann)
print('ANN baseline validation results')
print(f'MAE: {ann_mae:.2f} cycles')
print(f'RMSE: {ann_rmse:.2f} cycles')
print(f'R²: {ann_r2:.3f}')

outputs_dir = project_dir / 'outputs'
evals_dir = outputs_dir / 'evals'
predictions_dir = outputs_dir / 'predictions'
figures_dir = outputs_dir / 'figures'

ann_baseline_predictions = pd.DataFrame({'actual_rul': y_val, 'predicted_rul': y_val_pred_ann, 'residual': y_val_pred_ann - y_val})
ann_baseline_predictions.to_csv(predictions_dir / 'ann_baseline_predictions.csv', index=False)
ann_baseline_eval = pd.DataFrame({'model': ['ANN Baseline'], 'mae': [ann_mae], 'rmse': [ann_rmse], 'r_squared': [ann_r2]})
ann_baseline_eval.to_csv(evals_dir / 'ann_baseline_eval.csv', index=False)

plt.figure()
plt.plot(history_baseline.history['loss'], label='Training Loss')
plt.plot(history_baseline.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('ANN Baseline Model: Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / 'ann_baseline_learning_curve.png')
plt.close()

plt.figure()
plt.scatter(y_val, y_val_pred_ann, s=0.5)
min_rul = min(y_val.min(), y_val_pred_ann.min())
max_rul = max(y_val.max(), y_val_pred_ann.max())
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Artificial Neural Network Baseline Model: Validation Prediction')
plt.tight_layout()
plt.savefig(figures_dir / 'ann_baseline_actual_vs_predicted.png')
plt.close()

# ALTERNATE VERSION (ultimately not selected)

# ANN V2, increase neurons from 32 to 64
tf.keras.utils.set_random_seed(392)
ann_v2 = tf.keras.Sequential(name='ann_v2')
ann_v2.add(tf.keras.layers.Input(shape=(x_train_ann.shape[1],), name='input_layer'))
ann_v2.add(tf.keras.layers.Dense(64, activation='relu', name='hidden_layer'))
ann_v2.add(tf.keras.layers.Dense(1, activation='linear',name='output_layer'))
ann_v2.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping_v2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_v2 = ann_v2.fit(x_train_ann, y_train, validation_data=(x_val_ann, y_val), epochs=100, batch_size=32, callbacks=[early_stopping_v2], verbose=1)
y_val_pred_ann_v2 = ann_v2.predict(x_val_ann, verbose=0).ravel()

ann_v2_mae = mean_absolute_error(y_val, y_val_pred_ann_v2)
ann_v2_rmse = root_mean_squared_error(y_val, y_val_pred_ann_v2)
ann_v2_r2 = r2_score(y_val, y_val_pred_ann_v2)

print('ANN V2 validation results')
print(f'MAE: {ann_v2_mae:.2f} cycles')
print(f'RMSE: {ann_v2_rmse:.2f} cycles')
print(f'R²: {ann_v2_r2:.3f}')
