from utils import load_for_training
print("Enter path to data (if blank, 'census-income.data' will be used):")
path_to_data = input()
if len(path_to_data) == 0:
    path_to_data = 'census-income.data'

df,var_cont,var_disc = load_for_training(path_to_data=path_to_data,make_dummies=True)
print("Data loaded.")

print("Enter path to model (if blank, 'segmentation_model.sav' will be used):")
path_to_model = input()
if len(path_to_model) == 0:
    path_to_model = 'segmentation_model.sav'

import joblib
model = joblib.load(path_to_model)

print("Model loaded. Making predictions.")

prediction = model.predict(df.drop(columns=['instance weight']))

print("Predictions made. Saving predictions.")
import numpy as np
np.save('segmentation_predictions.np',prediction)
print("Predictions saved as 'segmentation_predictions.np'.")