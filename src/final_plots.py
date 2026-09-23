from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent.parent
evals_dir = project_dir / 'outputs' / 'evals'
predictions_dir = project_dir / 'outputs' / 'predictions'
figures_dir = project_dir / 'outputs' / 'figures'

final_test_eval_table = pd.read_csv(evals_dir / 'final_test_eval.csv')
final_test_predictions = pd.read_csv(predictions_dir / 'final_test_predictions.csv')

# MAE comparison
plt.figure()
plt.bar(final_test_eval_table['model'], final_test_eval_table['mae'])
plt.xlabel('Model')
plt.ylabel('MAE (cycles)')
plt.title('Final Test MAE Comparison')
plt.tight_layout()
plt.savefig(figures_dir / 'final_test_MAE_comparison.png')
plt.close()

# RMSE comparison
plt.figure()
plt.bar(final_test_eval_table['model'], final_test_eval_table['rmse'])
plt.xlabel('Model')
plt.ylabel('RMSE (cycles)')
plt.title('Final Test RMSE Comparison')
plt.tight_layout()
plt.savefig(figures_dir / 'final_test_rmse_comparison.png')
plt.close()

# scatter comparison limits
min_rul = min(
    final_test_predictions['actual_rul'].min(),
    final_test_predictions['rf_prediction'].min(),
    final_test_predictions['ann_prediction'].min(),
    final_test_predictions['lstm_prediction'].min())
max_rul = max(
    final_test_predictions['actual_rul'].max(),
    final_test_predictions['rf_prediction'].max(),
    final_test_predictions['ann_prediction'].max(),
    final_test_predictions['lstm_prediction'].max())

# final RF scatter
plt.figure()
plt.scatter(final_test_predictions['actual_rul'], final_test_predictions['rf_prediction'])
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlim(min_rul, max_rul)
plt.ylim(min_rul, max_rul)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Random Forest: Final Test Predictions')
plt.tight_layout()
plt.savefig(figures_dir / 'rf_final_test_actual_vs_predicted.png', dpi=200)
plt.close()

# final ANN scatter
plt.figure()
plt.scatter(final_test_predictions['actual_rul'], final_test_predictions['ann_prediction'])
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlim(min_rul, max_rul)
plt.ylim(min_rul, max_rul)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('ANN: Final Test Predictions')
plt.tight_layout()
plt.savefig(figures_dir / 'ann_final_test_actual_vs_predicted.png', dpi=200)
plt.close()

# final LSTM scatter
plt.figure()
plt.scatter(final_test_predictions['actual_rul'], final_test_predictions['lstm_prediction'])
plt.plot([min_rul, max_rul], [min_rul, max_rul], linestyle='--')
plt.xlim(min_rul, max_rul)
plt.ylim(min_rul, max_rul)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('LSTM: Final Test Predictions')
plt.tight_layout()
plt.savefig(figures_dir / 'lstm_final_test_actual_vs_predicted.png')
plt.close()
