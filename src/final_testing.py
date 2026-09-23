from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import tensorflow as tf

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / 'data' / 'processed'
raw_dir = project_dir / 'data' / 'raw'

# align test data format with formal NASA RUL prediction format (100 values)
# RN and ANN
test_metadata = pd.read_csv(processed_dir / 'test_metadata.csv')
NASA_test_rul = pd.read_csv(raw_dir / 'RUL_FD001.txt', sep=r'\s+', header=None).iloc[:, 0].to_numpy()
final_test_rows = test_metadata.groupby('unit_number').tail(1) # googled code to collapse messy loop i was using before
final_test_positions = final_test_rows.index
# LSTM
test_sequence_units = np.load(processed_dir / 'test_sequence_units.npy')
test_sequence_end_cycles = np.load(processed_dir / 'test_sequence_end_cycles.npy')
lstm_test_metadata = pd.DataFrame({'unit_number': test_sequence_units, 'end_cycle': test_sequence_end_cycles})
final_test_rows_lstm = lstm_test_metadata.groupby('unit_number').tail(1)
final_test_positions_lstm = final_test_rows_lstm.index

# Final RF
x_train_rf = pd.read_csv(processed_dir / 'x_train_rf.csv')
y_train = np.load(processed_dir / 'y_train.npy')
x_test_rf = pd.read_csv(processed_dir / 'x_test_rf.csv')
x_test_rf_100 = x_test_rf.iloc[final_test_positions]

rf_final = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=392)
rf_final.fit(x_train_rf, y_train)
rf_test_predictions = rf_final.predict(x_test_rf_100)

# Final ANN
# note even though early stopping wasn't triggered in development, i've opted to keep it in here incase useful for the test model. hence val data inclusion.
x_train_ann = pd.read_csv(processed_dir / 'x_train_ann.csv')
y_train = np.load(processed_dir / 'y_train.npy')
x_val_ann = pd.read_csv(processed_dir / 'x_val_ann.csv')
y_val = np.load(processed_dir / 'y_val.npy')
x_test_ann = pd.read_csv(processed_dir / 'x_test_ann.csv')
x_test_ann_100 = x_test_ann.iloc[final_test_positions]

tf.keras.utils.set_random_seed(392)
ann_final = tf.keras.Sequential(name='ann_final')
ann_final.add(tf.keras.layers.Input(shape=(x_train_ann.shape[1],), name='input_layer'))
ann_final.add(tf.keras.layers.Dense(32, activation='relu', name='hidden_layer'))
ann_final.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
ann_final.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_ann_final = ann_final.fit(x_train_ann, y_train, validation_data=(x_val_ann, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])

ann_test_predictions = ann_final.predict(x_test_ann_100).ravel()

# Final LSTM
x_train_lstm = np.load(processed_dir / 'x_train_lstm.npy')
y_train_lstm = np.load(processed_dir / 'y_train_lstm.npy')
x_val_lstm = np.load(processed_dir / 'x_val_lstm.npy')
y_val_lstm = np.load(processed_dir / 'y_val_lstm.npy')
x_test_lstm = np.load(processed_dir / 'x_test_lstm.npy')
x_test_lstm_100 = x_test_lstm[final_test_positions_lstm]

tf.keras.utils.set_random_seed(392)
lstm_final = tf.keras.Sequential(name='lstm_final')
lstm_final.add(tf.keras.layers.Input(shape=(x_train_lstm.shape[1], x_train_lstm.shape[2]), name='input_layer'))
lstm_final.add(tf.keras.layers.LSTM(128, activation='tanh', name='lstm_layer'))
lstm_final.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
lstm_final.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
early_stopping_lstm = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_lstm_final = lstm_final.fit(x_train_lstm, y_train_lstm, validation_data=(x_val_lstm, y_val_lstm), epochs=100, batch_size=32, callbacks=[early_stopping_lstm])

lstm_test_predictions = lstm_final.predict(x_test_lstm_100).ravel()


# Evaluation
# PHM08 s-score function, this is copied code; please see appendix and project_log for google prompting
def calculate_s_score(actual_rul, predicted_rul):
    errors = predicted_rul - actual_rul
    scores = np.where(errors < 0, np.exp(-errors/13)-1, np.exp(errors/10)-1)
    return np.sum(scores)

def evaluate_model(actual_rul, predicted_rul):
    return{'mae': mean_absolute_error(actual_rul, predicted_rul),
           'rmse': root_mean_squared_error(actual_rul, predicted_rul),
           'r_squared': r2_score(actual_rul, predicted_rul),
           's_score': calculate_s_score(actual_rul, predicted_rul)}

evals_dir = project_dir / 'outputs' / 'evals'
predictions_dir = project_dir / 'outputs' / 'predictions'

rf_test_results = evaluate_model(NASA_test_rul, rf_test_predictions)
ann_test_results = evaluate_model(NASA_test_rul, ann_test_predictions)
lstm_test_results = evaluate_model(NASA_test_rul, lstm_test_predictions)
test_results_table = pd.DataFrame([{'model': 'Random Forest', **rf_test_results}, {'model': 'ANN', **ann_test_results}, {'model': 'LSTM', **lstm_test_results}])
print(test_results_table.round(2))
test_results_table.to_csv(evals_dir / 'final_test_eval.csv', index=False)

final_predictions = pd.DataFrame({
    'unit_number': final_test_rows['unit_number'].to_numpy(),
    'actual_rul': NASA_test_rul,
    'rf_prediction': rf_test_predictions,
    'ann_prediction': ann_test_predictions,
    'lstm_prediction': lstm_test_predictions,
    'rf_error': rf_test_predictions - NASA_test_rul,
    'ann_error': ann_test_predictions - NASA_test_rul,
    'lstm_error': lstm_test_predictions - NASA_test_rul
})
final_predictions.to_csv(predictions_dir / 'final_test_predictions.csv', index=False)
