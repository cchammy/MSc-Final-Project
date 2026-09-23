# Project log
------------------------------
## Set-up
# Using NASA C-MAPSS 
# Full files can be downloaded at: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip 
# Using FD001, one Condition, one Fault Mode
# NASA FD001 test set [stops before failure] and FD001 RUL file reserved for final testing
# Training data [which goes until failure] to be split by engine ID into test/validation sets
# versions used in set up: pandas: 3.0.5, NumPy: 2.4.6, Scikit-learn: 1.9.0, TensorFlow: 2.21.0, Matplotlib: 3.11.1

-----------------------
## Data Preprocessing

## Data loading and initial checks
# loaded train_FD001.txt, test_FD001.txt and RUL_FD001.txt
# Training data shape: (20631, 26)
# Test data shape: (13096, 26)
# RUL data shape: (100, 1)
# In training data: 100 unique engines, no duplicate rows, no missing values

## RUL Labelling & checks
# attempted below loop, replaced with code in script after google prompt 'can you show me a more efficient way to achieve this'
# 
# train_df['rul'] = 0
# for unit_number, engine_data in train_df.groupby('unit_number'):
#     final_cycle = engine_data['cycle'].max()
#     train_df.loc[engine_data.index, 'rul'] = final_cycle - engine_data['cycle']
# confirmed code runs as intended with print of first 10 engine's RUL: 192, 287, 179, 189, 269, 188, 259, 150, 201, 222
# aligns with RUL values found in https://github.com/biswajitsahoo1111/rul_codes_open/blob/master/notebooks/cmapss_notebooks/CMAPSS_data_description_and_preprocessing.ipynb
# currently linear, future option to explore piecewise for further accuracy

## Split train_df into Training set and Validation set
# note, unlike in the proposal, have opted to do this prior to feature engineering/sensor selection to ensure this is carried out ONLY on the training set and there is no bleed
# NOT splitting by row, but by engine due to sequencing requirements & need avoid splitting engines accross train/val. so first need to group by engine.
# using sklearn train_test_split of 80/20, random seed of 392

## Feature selection using train_split_df
# identify constant (aka) useless features
# options to use standard deviation, min == max, or variance threshold
# went with just the one unique value as disqualifying factor
# quick print to have a look: constants include 1 setting and 6 sensors
# once columns identified in train_split_df, also dropped from val_split_df and test_df
# note, not exploring more advanced feature engineering (rolling stats, or correlation filtering) as i'm using same data across three different models and don't want to introduce bias

## Create static RUL targets for RF and ANN
# one for training and one for validation

## Create unscaled static data sets for Random Forest
# one for training, one for validation, one for testing

## Scaling features for ANN and LSTM
# use MinMaxScaler, fit ONLY on training data
# then applied to scaled versions of validation and test sets
# quick print of min and max values in scaled training set to ensure they fit between 0 and 1

## Build sequencing function and apply to data for LSTM model.
# After significant unsuccessful experinmentation, this was done using referencing from several github project, and prompts with Google Gemini and ChatGPT. Full detail in report appendix.
# Fixed sequence length was 30 cycles, using only scaled versions of data
# Grouped by engine_id

## Save all data sets to processed folder
# static RF input, static scaled ANN input
# static RUL target
# LSTM input sequences (train, val test) and targets (train, val)
# Shared code block with ChatGPT 'does this look okay before i run it', it suggested the three lines of metadata code verbatim

## Final print of shapes for all saved data. copied below:
# Dataset summary
# RF train/validation/test: (16497, 17) (4134, 17) (13096, 17)
# ANN train/validation/test: (16497, 17) (4134, 17) (13096, 17)
# Static targets: (16497,) (4134,)
# LSTM train/validation/test: (14177, 30, 17) (3554, 30, 17) (10196, 30, 17)
# LSTM targets: (14177,) (3554,)

---------------------------------
## Random Forest Development
# random forest, 500 trees, stopping threshold of 5 as Li et al (though not clear what hyperparameter they used, i opted for min_samples_leaf)
# train on train data, evaluate on val data, results as follows:
# MAE: 30.42 cycles
# RMSE: 42.11 cycles
# R²: 0.632
# created scatetr plot and saved too. looked like the baseline RF is underestimating engines with high RUL, gonna try lower min_sample_leaf
# NOTE: Li et al used piecewise RUL
# tried lower min_samples_leaf, fewer trees, and additional hyperparameters... generally slightly worsened predictions. Even got google to do a mega test loop with varying hyperparamaters and saw no meaningful change so ommitted from code and report.
# super quick look at feature importance revealed sensor 11 to be most imporant, then 4 and 6 but others notably low... not insightful enough to retain as part of the project

# alternative versions
# Random Forest V2 validation results
# MAE: 30.61 cycles
# RMSE: 42.26 cycles
# R²: 0.630
# Random Forest V3 validation results
# MAE: 30.48 cycles
# RMSE: 42.14 cycles
# R²: 0.632
# Random Forest V4 validation results
# MAE: 30.30 cycles
# RMSE: 41.71 cycles
# R²: 0.639
------------------------------------
## ANN Development
# baseline model of 32 neurons, 1 relu hidden layer, 1 linear output layer, early stopping
# baseline results:
# MAE: 34.33 cycles
# RMSE: 43.25 cycles
# R²: 0.612
# v2, 64 neurons all else same
# ANN V2 validation results
# MAE: 34.44 cycles
# RMSE: 43.38 cycles
# R²: 0.610
# compare validaiton and learning curves..... 
# didn't show overfitting (training loss continuing down while validation loss rises) so opted not to introduce drop out 
# also note, early stopping didn't come into play, and results plateaud after a small number of epochs! 

-------------------------------------
## LSTM Development
# baseline model, 1 lstm layer, 64 neurons, tanh activation, followed by single linear Dense output. adam optimiser, mse loss tracked. 100 epochs, batch size 32. same early stopping as ANN.
# training stopped at 30 epochs, early stopping kicked in!
# LSTM baseline validation results
# MAE: 18.92 cycles
# RMSE: 29.11 cycles
# R²: 0.782
# wayyy better than RF and ANN. no clear overfitting in validation prediction plot so no dropout introduced.
# actual vs predicted plot showed good alignment at lower RUL values, but performed poorly at higher RUL values which is in line with the widespread practice of piecewise rul and/or capping RUL at 125

# one alternative version at 128 neurons instead of 64. all else the same
# ran for 52 epochs
# LSTM V2 validation results
# MAE: 18.19 cycles
# RMSE: 27.98 cycles
# R²: 0.799
# THIS MODEL SELECTED! save plots and results alongside the baseline.

-------------------------------------
## Final Testing
# test data processing: load formal NASA RUL and algin with metadata 
# RF and ANN: last recorded cycle per engine
# LSTM: last 30 cycle sequence per engine
# this is so predictions align with the 100 RUL values NASA provides

# Random Forest final test results
# MAE: 25.14 cycles
# RMSE: 34.38 cycles
# R²: 0.315
# MAE and RMSE are improvements!
# suspect r2 dropped due to narrower RUL range with only 100 targets. similar absolute error can provide much lower r2

# ANN final test results
# MAE: 27.99 cycles
# RMSE: 34.54 cycles
# R²: 0.31

# LSTM final test results
# MAE: 16.68 cycles
# RMSE: 24.83 cycles
# R²: 0.643


# S-score. Googled 'how to calculate s-score phm08' and the google AI response provided following:
def calculate_phm08_score(y_true, y_pred):
    # Calculate the error vector
    errors = y_pred - y_true
    
    # Apply the asymmetric penalties
    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1,
        np.exp(errors / 10.0) - 1   
    )
    
    # Return the total sum of all scores
    return np.sum(scores)

# final test results!!
#            model    mae   rmse  r_squared   s_score
# 0  Random Forest  25.14  34.38       0.32  44277.06
# 1            ANN  27.99  34.54       0.31  21007.47
# 2           LSTM  16.68  24.83       0.64  19804.19
