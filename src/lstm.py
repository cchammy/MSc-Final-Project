from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / 'data' / 'processed'

tf.keras.utils.set_random_seed(392)

x_train_lstm = np.load(processed_dir / 'x_train_lstm.npy')
x_val_lstm = np.load(processed_dir / 'x_val_lstm.npy')
y_train_lstm = np.load(processed_dir / 'y_train_lstm.npy')
y_val_lstm = np.load(processed_dir / 'y_val_lstm.npy')

lstm_baseline = tf.keras.Sequential(name='lstm_baseline')
lstm_baseline.add(tf.keras.layers.Input(shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), name='input_layer'))
lstm_baseline.add(tf.keras.layers.LSTM(64, activation='tanh', name='lstm_layer'))
lstm_baseline.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
lstm_baseline.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_baseline_lstm = lstm_baseline.fit(x_train_lstm, y_train_lstm, validation_data=(x_val_lstm, y_val_lstm), epochs=100, batch_size=32, callbacks=[early_stopping])

y_val_pred_lstm = lstm_baseline.predict(x_val_lstm).ravel()

lstm_mae = mean_absolute_error(y_val_lstm, y_val_pred_lstm)
lstm_rmse = root_mean_squared_error(y_val_lstm, y_val_pred_lstm)
lstm_r2 = r2_score(y_val_lstm, y_val_pred_lstm)
print('LSTM baseline validation results')
print(f'MAE: {lstm_mae:.2f} cycles')
print(f'RMSE: {lstm_rmse:.2f} cycles')
print(f'R²: {lstm_r2:.3f}')

outputs_dir = project_dir / 'outputs'
evals_dir = outputs_dir / 'evals'
predictions_dir = outputs_dir / 'predictions'
figures_dir = outputs_dir / 'figures'

lstm_baseline_predictions = pd.DataFrame({'actual_rul': y_val_lstm, 'predicted_rul': y_val_pred_lstm, 'residual': y_val_pred_lstm - y_val_lstm})
lstm_baseline_predictions.to_csv(predictions_dir / 'lstm_baseline_predictions.csv', index=False)
lstm_baseline_eval = pd.DataFrame({'model': ['LSTM Baseline'], 'mae': [lstm_mae], 'rmse': [lstm_rmse], 'r_squared': [lstm_r2]})
lstm_baseline_eval.to_csv(evals_dir / 'lstm_baseline_eval.csv', index=False)

plt.figure()
plt.plot(history_baseline_lstm.history['loss'], label='Training Loss')
plt.plot(history_baseline_lstm.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('LSTM Baseline Model: Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / 'lstm_baseline_learning_curve.png')
plt.close()

plt.figure()
plt.scatter(y_val_lstm, y_val_pred_lstm, s=0.5)
min_rul = min(y_val_lstm.min(), y_val_pred_lstm.min())
max_rul = max(y_val_lstm.max(), y_val_pred_lstm.max())
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Long Short Term Memory Network Baseline Model: Validation Prediction')
plt.tight_layout()
plt.savefig(figures_dir / 'lstm_baseline_actual_vs_predicted.png')
plt.close()

# ALTERNATE VERSION
# LSTM v2, same as baseline but with 128 neurons
tf.keras.utils.set_random_seed(392)
lstm_v2 = tf.keras.Sequential(name='lstm_v2')
lstm_v2.add(tf.keras.layers.Input(shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), name='input_layer'))
lstm_v2.add(tf.keras.layers.LSTM(128, activation='tanh', name='lstm_layer'))
lstm_v2.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
lstm_v2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
early_stopping_v2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_lstm_v2 = lstm_v2.fit(x_train_lstm, y_train_lstm, validation_data=(x_val_lstm, y_val_lstm), epochs=100, batch_size=32, callbacks=[early_stopping_v2])

y_val_pred_lstm_v2 = lstm_v2.predict(x_val_lstm).ravel()

lstm_v2_mae = mean_absolute_error(y_val_lstm, y_val_pred_lstm_v2)
lstm_v2_rmse = root_mean_squared_error(y_val_lstm, y_val_pred_lstm_v2)
lstm_v2_r2 = r2_score(y_val_lstm, y_val_pred_lstm_v2)
print('LSTM V2 validation results')
print(f'MAE: {lstm_v2_mae:.2f} cycles')
print(f'RMSE: {lstm_v2_rmse:.2f} cycles')
print(f'R²: {lstm_v2_r2:.3f}')

# NOTE: alternate performed better than baseline so save results and plots as above
lstm_v2_predictions = pd.DataFrame({'actual_rul': y_val_lstm, 'predicted_rul': y_val_pred_lstm_v2, 'residual': y_val_pred_lstm_v2 - y_val_lstm})
lstm_v2_predictions.to_csv(predictions_dir / 'lstm_v2_predictions.csv', index=False)

lstm_v2_eval = pd.DataFrame({'model': ['LSTM V2'], 'mae': [lstm_v2_mae], 'rmse': [lstm_v2_rmse], 'r_squared': [lstm_v2_r2]})
lstm_v2_eval.to_csv(evals_dir / 'lstm_v2_eval.csv', index=False)

plt.figure()
plt.plot(history_lstm_v2.history['loss'], label='Training Loss')
plt.plot(history_lstm_v2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('LSTM V2 Model: Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / 'lstm_v2_learning_curve.png')
plt.close()

plt.figure()
plt.scatter(y_val_lstm, y_val_pred_lstm_v2, s=0.5)
min_rul = min(y_val_lstm.min(), y_val_pred_lstm_v2.min())
max_rul = max(y_val_lstm.max(), y_val_pred_lstm_v2.max())
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Long Short Term Memory Network V2 Model: Validation Prediction')
plt.tight_layout()
plt.savefig(figures_dir / 'lstm_v2_actual_vs_predicted.png')
plt.close()
