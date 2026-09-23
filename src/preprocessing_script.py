from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

project_dir = Path(__file__).resolve().parent.parent # googled how to solve problem with file reading without this line included
raw_dir = project_dir / 'data' / 'raw'

train_file = raw_dir / 'train_FD001.txt'
test_file = raw_dir / 'test_FD001.txt'
rul_file = raw_dir / 'RUL_FD001.txt'  # only used to check data shape then left untouched, RAW file will be used in final test eval

id_names = ['unit_number', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = id_names + setting_names + sensor_names

train_df = pd.read_csv(train_file, sep=r'\s+', header=None)
test_df = pd.read_csv(test_file, sep=r'\s+', header=None)
rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None)

train_df.columns = col_names
test_df.columns = col_names

# check: data is as expected and structure is read correctly. NO DETAILS from test or RUL sets
print('Training data shape:', train_df.shape)  # should be high number of rows, 26 columns
print('Test data shape:', test_df.shape)  # should be high number of rows (but less than train), 26 columns
print('RUL data shape:', rul_df.shape)  # should be 100 rows, 1 column

print('Training data preview:', train_df.head())
print('Number of engines in training data:', train_df['unit_number'].nunique())
print('Missing values in training data:', train_df.isna().sum().sum())
print('Duplicates in training data:', train_df.duplicated().sum())

# calculating & labelling RUL values (max cycle - current cycle) for training data, see project_log for code detail
train_df['rul'] = (
    train_df.groupby('unit_number')['cycle'].transform('max')
    - train_df['cycle']
)

rul_by_engine = train_df.groupby('unit_number')['rul']
rul_summary = rul_by_engine.agg(['min', 'max', 'count'])
print(rul_summary.head(10))

# split training set into training and validation sets (80/20) based on engine no.
unit_numbers = train_df['unit_number'].unique()
training_units, validation_units = train_test_split(unit_numbers, test_size=0.20, random_state=392)

train_split_df = train_df[train_df['unit_number'].isin(training_units)].copy()
val_split_df = train_df[train_df['unit_number'].isin(validation_units)].copy()

# feature selection on train_split_df [our ACTUAL training set], identfying redundant features and dropping from train val and test sets
all_cols = [col for col in train_split_df.columns if col not in ['unit_number', 'cycle', 'rul']]
constant_feats = [col for col in all_cols if train_split_df[col].nunique() == 1]
print('Constant features to exclude:', constant_feats)

train_split_df = train_split_df.drop(columns=constant_feats)
val_split_df = val_split_df.drop(columns=constant_feats)
test_df = test_df.drop(columns=constant_feats)

feature_cols = [col for col in train_split_df.columns if col not in ['unit_number', 'cycle', 'rul']]
print('Final features:', feature_cols)
print('Count:', len(feature_cols))

y_train = train_split_df['rul'].copy()
y_val = val_split_df['rul'].copy()

x_train_rf = train_split_df[feature_cols].copy()
x_val_rf = val_split_df[feature_cols].copy()
x_test_rf = test_df[feature_cols].copy()

scaler = MinMaxScaler()
# fit scaler to the training features ONLY
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_rf), columns=feature_cols, index=x_train_rf.index)

x_val_scaled = pd.DataFrame(scaler.transform(x_val_rf), columns=feature_cols, index=x_val_rf.index)
x_test_scaled = pd.DataFrame(scaler.transform(x_test_rf), columns=feature_cols, index=x_test_rf.index)
# check: values fall between 0-1
print('Scaled training min:', x_train_scaled.min().min())
print('Scaled training max:', x_train_scaled.max().max())

x_train_ann = x_train_scaled.copy()
x_val_ann = x_val_scaled.copy()
x_test_ann = x_test_scaled.copy()

# Creating sequenced datasets for the LSTM
# Please see project log and report appendix for AI prompts used to develop this code block (lines 87 to 129). comments have been added throughout to demonstrate understanding

# Creating fixed length sequences of 30 cycles, grouped by engine (no overlap between engines)
# Target is RUL at the final cycle in that sequence.
# First step create sequencing function
def create_lstm_sequences(data_df, scaled_features, sequence_length=30):
    sequences = []
    targets = []
    sequence_units = []  # engine ID for each sequence
    sequence_end_cycles = []  # final cycle in that sequence

    # test data has no row-level RUL labels
    has_targets = 'rul' in data_df.columns

    # One engine at a time to maintain order and ensure no sequence mixes cycles from different engines
    for unit_number, engine_data in data_df.groupby('unit_number'):
        engine_data = engine_data.sort_values('cycle')

        # select scaled features
        engine_features = scaled_features.loc[engine_data.index].to_numpy()

        # starts at row 29 as this is the first point with 30 cycles available
        for end_index in range(sequence_length - 1, len(engine_data)):
            start_index = end_index - sequence_length + 1

            # store cycles
            sequences.append(engine_features[start_index:end_index + 1])

            # target is RUL at final cycle
            if has_targets:
                targets.append(engine_data['rul'].iloc[end_index])

            # below used to identify units later
            sequence_units.append(unit_number)
            sequence_end_cycles.append(engine_data['cycle'].iloc[end_index])

    return (
        np.array(sequences),
        np.array(targets),
        np.array(sequence_units),
        np.array(sequence_end_cycles)
    )

x_train_lstm, y_train_lstm, train_sequence_units, train_sequence_end_cycles = (create_lstm_sequences(train_split_df, x_train_scaled))
x_val_lstm, y_val_lstm, val_sequence_units, val_sequence_end_cycles = (create_lstm_sequences(val_split_df, x_val_scaled))
x_test_lstm, _, test_sequence_units, test_sequence_end_cycles = (create_lstm_sequences(test_df, x_test_scaled))

# check: shapes [want 30 cycles and 17 features, and matched number of sequences with corresponding target]
print('LSTM training input:', x_train_lstm.shape)
print('LSTM training targets:', y_train_lstm.shape)
print('LSTM validation input:', x_val_lstm.shape)
print('LSTM validation targets:', y_val_lstm.shape)

# saving all processed data & metadata
processed_dir = project_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)  # added for reproducibility

x_train_rf.to_csv(processed_dir / 'x_train_rf.csv', index=False)
x_val_rf.to_csv(processed_dir / 'x_val_rf.csv', index=False)
x_test_rf.to_csv(processed_dir / 'x_test_rf.csv', index=False)

x_train_ann.to_csv(processed_dir / 'x_train_ann.csv', index=False)
x_val_ann.to_csv(processed_dir / 'x_val_ann.csv', index=False)
x_test_ann.to_csv(processed_dir / 'x_test_ann.csv', index=False)

np.save(processed_dir / 'y_train.npy', y_train.to_numpy())
np.save(processed_dir / 'y_val.npy', y_val.to_numpy())

np.save(processed_dir / 'x_train_lstm.npy', x_train_lstm)
np.save(processed_dir / 'y_train_lstm.npy', y_train_lstm)
np.save(processed_dir / 'x_val_lstm.npy', x_val_lstm)
np.save(processed_dir / 'y_val_lstm.npy', y_val_lstm)
np.save(processed_dir / 'x_test_lstm.npy', x_test_lstm)

test_df[['unit_number', 'cycle']].to_csv(processed_dir / 'test_metadata.csv', index=False)
np.save(processed_dir / 'test_sequence_units.npy', test_sequence_units)
np.save(processed_dir / 'test_sequence_end_cycles.npy', test_sequence_end_cycles)

# check: print summaries of all data to confirm shapes are correct
print('Dataset summary')
print('RF train/validation/test:', x_train_rf.shape, x_val_rf.shape, x_test_rf.shape)
print('ANN train/validation/test:', x_train_ann.shape, x_val_ann.shape, x_test_ann.shape)
print('Static targets:', y_train.shape, y_val.shape)
print('LSTM train/validation/test:', x_train_lstm.shape, x_val_lstm.shape, x_test_lstm.shape)
print('LSTM targets:', y_train_lstm.shape, y_val_lstm.shape)
