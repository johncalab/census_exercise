s = 'census-income.data'
from utils import load_for_training
print("Loading data.")
df,var_cont,var_disc = load_for_training(path_to_data=s)
# df is loaded and cleaned for training

from utils import ttsplit
train,test = ttsplit(df,df['label_encoded'])
print("Train/test split done.")

from utils import xyw
x_train,y_train,w_train = xyw(train)
x_test,y_test,w_test = xyw(test)

print("Preparing model.")
# instantiate model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

col_pipe = ColumnTransformer(
        transformers=[
            ('nothing', FunctionTransformer(), sorted(list(var_cont))),
            ('one_hot', OneHotEncoder(handle_unknown='ignore'), sorted(list(var_disc)))
        ])

model = Pipeline(steps=[ ('col_transf', col_pipe), ('rf', RandomForestClassifier(n_estimators=20)) ])
print("Model instantiated.")

# train model and save
print("Training model.")

model.fit(x_train,y_train,rf__sample_weight=w_train)
print("Model has been trained. Now saving.")
from sklearn.externals import joblib
joblib.dump(model,'income_model.sav')
print("Model was saved as 'income_model.sav'.")
# to load model use
# loaded_model = joblib.load(filename)

from utils import evaluate_preds

print("Evaluating over training set.")
y_train_hat = model.predict_proba(x_train)[:,1]
train_eval = evaluate_preds(y_train_hat,y_train,w_train)

print('\n')
s="During training, the model achieved the following scores:\n"
s += str(train_eval)
with open('income.log', 'a') as f:
    f.write(s)
print(s)

print("Evaluating over test set.")
# evaluate model
y_test_hat = model.predict_proba(x_test)[:,1]
test_eval = evaluate_preds(y_test_hat,y_test,w_test)

s = "\nDuring testing, the model achieved the following scores:\n"
s += str(test_eval)
with open('income.log', 'a') as f:
    f.write(s)
print(s)

