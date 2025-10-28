import numpy as np
import time
### preprocess data ###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

### machine learning method ###
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

### test data ###
from sklearn.metrics import log_loss,classification_report, confusion_matrix,precision_score

class Basic_classification_methods(object):    
    def __init__(self,data_path,file_name):
        self.number_of_samples = 1000
        self.data_path = data_path
        self.file_name = file_name
    def load_data(self,divider):
        self.data = np.loadtxt(open(self.data_path + "/"+self.file_name))
        self.input_data = self.data[:, :divider]
        self.output_data = self.data[:, divider:]
    
    def standardize_feature(self,input_data):
        self.scaler = StandardScaler()
        input_data_std = self.scaler.fit_transform(input_data)
        return input_data_std
    
    def random_selection_data(self,x,y,train_size,test_size):
        try: 
            rng = np.random.RandomState(42)
            indices = np.arange(y.shape[0])
            rng.shuffle(indices)
            x_shuffle = x[indices]
            y_shuffle = y[indices]
            x_test = x_shuffle[:test_size]
            y_test = y_shuffle[:test_size]
            x_rest = x_shuffle[test_size:]
            y_rest = y_shuffle[test_size:]
            indices_rest = np.arange(y_rest.shape[0])
            np.random.shuffle(indices_rest)
            x_train = x_rest[indices_rest][:train_size]
            y_train = y_rest[indices_rest][:train_size]
            
        except:
            print("Number of samples is greater than the size of data!")
        return x_train,x_test, y_train,y_test
    
    def dominant_class(self,y):
        y_dominant = np.argmax(y, axis=1)
        return y_dominant
    
    def compute_sample_weight(self,y):
        sample_weights = compute_sample_weight('balanced', y)
        return sample_weights
    
    # happened after loading data and split input and output
    def preprocess_data(self,input_data,output_data,train_size,test_size):
        # standardize the input
        input_data = self.standardize_feature(input_data)
        # take out fixed test set
        # choose from the rest randomly as the training set
        X_train, X_test, y_train, y_test = self.random_selection_data(input_data,output_data,train_size,test_size)
        return X_train, X_test, y_train, y_test
    
    def preprocess_data_with_dominant_class(self,input_data,output_data,train_size,test_size):
        output_data = self.dominant_class(output_data)
        X_train, X_test, y_train, y_test = self.preprocess_data(input_data,output_data,train_size,test_size)
        return X_train, X_test, y_train, y_test
    
    ####### evalulate performance ######
    def compute_log_loss(self,y,y_pred):
        loss = log_loss(y, y_pred)
        return loss
    
    def compute_accuracy(self,y,y_pred,weighted_method):
        acc = precision_score(y, y_pred, average= weighted_method)
        return acc
    
    #### perform classification ####
    def perform_logistic_regression(self,input_data,output_data,train_size,test_size):
        start = time.time()
        x_train, x_test, y_train, y_test = self.preprocess_data_with_dominant_class(input_data,output_data,train_size,test_size)
        #sample_weights = self.compute_sample_weight(y_train)
        # Train Logistic Regression
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(x_train, y_train)
        # Predict class probabilities
        y_train_pred = model.predict_proba(x_train)
        y_train_pred = np.argmax(y_train_pred,axis = 1)
        y_test_pred = model.predict_proba(x_test)
        y_test_pred = np.argmax(y_test_pred,axis = 1)
        end = time.time()
        return y_train, y_train_pred, y_test, y_test_pred, model,end-start
    
    def perform_decision_tree(self,input_data,output_data,train_size,test_size,max_depth = 100):
        start = time.time()
        x_train, x_test, y_train, y_test = self.preprocess_data_with_dominant_class(input_data,output_data,train_size,test_size)
        # Train a Decision Tree Classifier
        tree_model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        tree_model.fit(x_train, y_train)

        # Predict on the test set
        y_train_pred = tree_model.predict(x_train)
        y_test_pred = tree_model.predict(x_test)
        #acc_train = self.compute_accuracy(y_train,y_train_pred,weighted_method)
        #acc_test = self.compute_accuracy(y_test,y_test_pred,weighted_method)
        
        end = time.time()
        return y_train, y_train_pred, y_test, y_test_pred, tree_model,end-start
    
    def perform_random_forest(self,input_data,output_data,train_size,test_size,n_estimators = 100,max_depth = 100):
        start = time.time()
        x_train, x_test, y_train, y_test = self.preprocess_data_with_dominant_class(input_data,output_data,train_size,test_size)
        rf_model = RandomForestClassifier(n_estimators=n_estimators,max_depth = max_depth)  # Maximum depth of a tree)
        rf_model.fit(x_train, y_train)

        # Predict on the test set
        y_train_pred = rf_model.predict(x_train)
        y_test_pred = rf_model.predict(x_test)
        #acc_train = self.compute_accuracy(y_train,y_train_pred,weighted_method)
        #acc_test = self.compute_accuracy(y_test,y_test_pred,weighted_method)
        
        end = time.time()
        return y_train, y_train_pred, y_test, y_test_pred, rf_model,end-start
    
    def perform_xgboost(self,input_data,output_data,train_size,test_size,n_estimators = 4,max_depth = 100):
        start = time.time()
        x_train, x_test, y_train, y_test = self.preprocess_data_with_dominant_class(input_data,output_data,train_size,test_size)
        xgb_model = XGBClassifier(
            n_estimators=n_estimators,  # Number of boosting rounds
            learning_rate=0.1,  # Step size shrinkage
            max_depth = max_depth,  # Maximum depth of a tree
            eval_metric='mlogloss'  # Multiclass log loss for evaluation
        )
        sample_weights = self.compute_sample_weight(y_train)
        xgb_model.fit(x_train, y_train)
        # Predict on the test set
        y_train_pred = xgb_model.predict(x_train)
        y_test_pred = xgb_model.predict(x_test)
        #acc_train = self.compute_accuracy(y_train,y_train_pred,weighted_method)
        #acc_test = self.compute_accuracy(y_test,y_test_pred,weighted_method)
        
        end = time.time()
        return y_train, y_train_pred, y_test, y_test_pred, xgb_model,end-start
        
        
    def perform_classification(self,method,train_size,test_size,n_estimators = 4,weighted_method = 'macro',max_depth = 4):
        if method == "LR":
            y_train, y_train_pred, y_test, y_test_pred, model, simulation_time =  self.perform_logistic_regression(self.input_data,self.output_data,train_size,test_size)
        elif method == "DT":
            y_train, y_train_pred, y_test, y_test_pred, model, simulation_time =  self.perform_decision_tree(self.input_data,self.output_data,train_size,test_size,max_depth)
        elif method == "RF":
            y_train, y_train_pred, y_test, y_test_pred, model, simulation_time =  self.perform_random_forest(self.input_data,self.output_data,train_size,test_size,n_estimators,max_depth)
        elif method == "XGB":
            y_train, y_train_pred, y_test, y_test_pred, model, simulation_time =  self.perform_xgboost(self.input_data,self.output_data,train_size,test_size,n_estimators,max_depth)
        
        acc_train_macro = self.compute_accuracy(y_train,y_train_pred,'macro')
        acc_test_macro = self.compute_accuracy(y_test,y_test_pred,'macro')
        acc_train_weighted = self.compute_accuracy(y_train,y_train_pred,'weighted')
        acc_test_weighted = self.compute_accuracy(y_test,y_test_pred,'weighted')
        confusion_train = confusion_matrix(y_train, y_train_pred)
        confusion_test = confusion_matrix(y_test, y_test_pred)
        return acc_train_macro, acc_test_macro, acc_train_weighted, acc_test_weighted,confusion_train,confusion_test
    
