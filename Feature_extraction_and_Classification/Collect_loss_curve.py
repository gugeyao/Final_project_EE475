import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss
import sys
import os
import argparse
import matplotlib.pyplot as plt
import json
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from ML_methods import Basic_classification_methods
    
parser = argparse.ArgumentParser(description="Collect loss curves for multiple models")
parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")), help="Directory containing label/data files")
parser.add_argument("--labels-file", default="labels.txt", help="Labels file name inside data-dir")
parser.add_argument("--train-size", type=int, default=700)
parser.add_argument("--test-size", type=int, default=300)
args = parser.parse_args([] if hasattr(sys, 'ps1') else None)

test_size = args.test_size
train_size = args.train_size
n_estimators = [0,0,100,100]
divider = 4
directory_path = args.data_dir
filename = args.labels_file
method = Basic_classification_methods(directory_path, filename)
method.load_data(divider)
method.output_data = method.data[:,-4:]
x_train, x_test, y_train, y_test = method.preprocess_data_with_dominant_class(method.input_data,method.output_data,train_size,test_size)


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1, warm_start=True)
losses = []

for i in range(1, 101):  # Train for 100 iterations
    model.fit(x_train, y_train)
    if (i-1)%10 == 0:
        probas = model.predict_proba(x_train)
        losses.append(log_loss(y_train, probas))


# Example data
X, y = x_train, y_train
model = RandomForestClassifier(warm_start=True)
random_losses = []

for n_trees in range(1, 101, 10):  # Incrementally increase number of trees
    model.set_params(n_estimators=n_trees)
    model.fit(X, y)
    probas = model.predict_proba(X)
    random_losses.append(log_loss(y, probas))



# Initialize XGBoost model
xgb_model = XGBClassifier(verbosity=0)

# Train the model and track Gini impurity manually
xgb_losses = []
for n_estimators in range(1, 101, 10):  # Iterate over boosting rounds
    xgb_model.set_params(n_estimators=n_estimators)
    xgb_model.fit(x_train, y_train, verbose=False)
    pred_proba = xgb_model.predict_proba(x_train)
    xgb_losses.append(log_loss(y_train, pred_proba))

dictionary = {"LR": losses, "RF":random_losses,"XGB": xgb_losses}
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
json.dump(dictionary, open(os.path.join(directory_path, "loss_track.json"), "w"), cls=NumpyEncoder, indent=4)