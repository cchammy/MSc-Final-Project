# MSc-Final-Project
 Time Will Tell: benchmarking LSTM networks against traditional methods for remaining useful life prediction

This repository contains all source code and associated files for the MSc Data Science final project. This README details requirements and instructions for running the code.

**Contents**
- data/raw: NASA C-MAPSS FD001 source files and readme.txt
- data/processed: processed model inputs and metadata
- outputs/evals: saved evaluation results for baseline and selected models
- outputs/figures: saves plots
- outputs/predictions: saved predictions for baseline and selected models
- src: scripts for preprocessing, model development, final testing and results
 
**Requirements**

This project was developed using Python 3.11 in PyCharmEdu 2022.2.2 and requires:
- pandas 3.0.5
- NumPy 2.4.6
- scikit-learn 1.9.0
- TensorFlow 2.21.0
- Matplotlib 3.11.1

**Running the project**

Please run the scripts in the following order:
1. src/check_setup.py
   This script prints package versions, ensure they align with the above.
3. src/preprocessing_script.py
   This script creates the model-ready inputs and metadata in data/processed
4. src/random_forest.py
   This script was used to develop the RF models for validation and produces validation plots, results and predictions.
6. src/ann.py
   This script was used to develop the ANN models for validation and produces saved validation plots, results and predictions.
8. src/lstm.py
   This script was used to develop the LSTM models for validation and produces saved validation plots, results and predictions.
10. src/final_testing.py
    This script rebuilds the selected models and produces saved final test results and predictions.
12. src/final_plots.py
    This script produces saved comparison plots for the final test predictions, MAE and RMSE.

