print("Loading data.")
from utils import load_for_training
df,var_cont,var_disc = load_for_training(make_dummies=True)
print("Data loaded.")

w = df['instance weight']
df = df.drop(columns=['instance weight'])

from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)

print("Training model.")
model.fit(df,sample_weight=w)

print("Done. Saving model.")

import joblib
model = joblib.dump(model,'segmentation_model.sav')

print("Model saved as 'segmentation_model.sav'.")