import numpy as np
import sys
import os
import argparse
import json

from ML_methods import Basic_classification_methods
    

parser = argparse.ArgumentParser(description="Compare ML methods on Feature set 2")
parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")), help="Directory containing label/data files")
parser.add_argument("--labels-file", default="labels_with_PCA.txt", help="Labels file name inside data-dir")
parser.add_argument("--train-size", type=int, default=700)
parser.add_argument("--test-size", type=int, default=300)
args = parser.parse_args([] if hasattr(sys, 'ps1') else None)

test_size = args.test_size
n_estimators = [0,0,20,20]
divider = 8
directory_path = args.data_dir
filename = args.labels_file
method = Basic_classification_methods(directory_path, filename)
method.load_data(divider)
method.output_data = method.output_data[:,:4]
method.input_data = method.data[:,-6:-3]
classification_methods = ['LR','DT','RF','XGB']
train_size = 700
acc_train_macro_array = np.zeros((len(classification_methods)))
acc_test_macro_array = np.zeros((len(classification_methods)))
acc_train_weighted_array = np.zeros((len(classification_methods)))
acc_test_weighted_array = np.zeros((len(classification_methods)))
confusion_train_array = []
confusion_test_array = []
for i,cl in enumerate(classification_methods):
    #for j,train_size in enumerate(train_size_array):
    acc_train_macro, acc_test_macro, acc_train_weighted, acc_test_weighted, confusion_train,confusion_test= method.perform_classification(cl,train_size,test_size,n_estimators = n_estimators[i],max_depth = 4)
    acc_train_macro_array[i] = acc_train_macro
    acc_test_macro_array[i] = acc_test_macro
    acc_train_weighted_array[i] = acc_train_weighted
    acc_test_weighted_array[i] = acc_test_weighted
    confusion_train_array.append(confusion_train)
    confusion_test_array.append(confusion_test)
dictionary = {"ML method": classification_methods,"training size": train_size, "validation size": test_size,\
    "training macro acc": acc_train_macro_array, "validation macro acc": acc_test_macro_array, "training weighted acc": acc_train_weighted_array, "validation weighted acc": acc_test_weighted_array,
    "confusion train": confusion_train_array,"confusion validation": confusion_test_array}
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
json.dump(dictionary, open(os.path.join(directory_path, "ML_performance_PCA.json"), "w"), cls=NumpyEncoder, indent=4)