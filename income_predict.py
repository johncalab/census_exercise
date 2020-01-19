"""
Use trained model to predict income.
"""
print("Enter path to data (if blank, 'census-income.data' will be used:")
path_to_data = input()
if len(path_to_data) == 0:
    path_to_data = 'census-income.data'

print("Enter path to model (if blank, 'income_model.sav' will be used:")
path_to_model = input()
if len(path_to_model) == 0:
    path_to_model = 'income_model.sav'

print("Enter threshold for predictions (if blank, 0.5 will be used):")
t = input()
if len(t) == 0:
    t = 0.5
else:
    t = float(t)

print("Loading model.")
import joblib
model = joblib.load(path_to_model)
print("Model loaded. Loading data.")

from utils import load_data_from_two
df = load_data_from_two(path_to_data,'census-income.columns')
print("Data loaded. Preprocessing now.")

from utils import clean_data
x,_,_ = clean_data(df)
print("Data cleaned. Ready to predict.")

# I forgot to the pipe?
y_hat = model.predict_proba(x)[:,1]
y_pred = (y_hat > t)

import numpy as np
np.save('income_predictions.np', y_pred)
print("Predicions made, and saved as 'income_predictions.np'.")

