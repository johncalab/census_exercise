# Income prediction model

### Training and evaluation.
* To train and evaluate model type `python income_train_eval.py`.
* This assumes `census-income.data` and `census-income.columns` to be in same directory as `income_train_eval.py`.
* Running this script will generate `income_model.sav` and `income.log`.

### Prediction
* To predict on new data, type `python income_predict.py`.
* Prompt will ask for path to data, path to model, and a threshold.
* Running this script will generate `income_predictions.np`.

# Segmentation

### Traning
* To train model run `python segmentation_train.py`.
* This assumes `census-income.data` and `census-income.columns` to be in the same directory as `segmentation_train.py`.
* Running this script will generate `segmentation_model.sav`.

### Prediction
* To predict on new data, type `python segmentation_predict.py`.
* Prompt will ask for path to data, and path to model.
* Running this script will generate `segmentation_predictions.py`.