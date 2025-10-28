import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class post_process_methods(object):
    def __init__(self,data_path,num_trials):
        self.data_path = data_path
        self.num_trials = num_trials
    def collect_simulation_paras(self):
        self.folder_names = [
            item for item in os.listdir(self.data_path) 
            if os.path.isdir(os.path.join(self.data_path, item))
        ]
        self.folder_count = len(self.folder_names)
        print(f"Number of folders: {self.folder_count}")
                
        self.speeds, self.turning_rates, self.zoos, self.zors = (
            [float(folder.split("_")[i]) for folder in self.folder_names]
            for i in (1, 3, 5, 7)
        )
    
    def compute_mean_angular_momentum_directionality(self,num_trials,sub_path):
        time = np.arange(0,4000,100)
        time_trunc = 3000
        idx = np.where(time >= time_trunc)[0]
        P_array = np.zeros(num_trials)
        m_array = np.zeros(num_trials)
        for i in np.arange(0,num_trials,1):
            filename = "P_"+str(i+1)+".txt"
            P = np.loadtxt(open(sub_path+"/"+filename))
            P_t_mean = np.mean(P[idx])
            filename = "m_"+str(i+1)+".txt"
            m = np.loadtxt(open(sub_path+"/"+filename))
            m_t_mean = np.mean(m[idx])
            
            P_array[i] = P_t_mean
            m_array[i] = m_t_mean
        P_mean = np.mean(P_array)
        #P_err = np.std(P_array)/np.sqrt(num_trials)
        m_mean = np.mean(m_array)
        #m_err = np.std(m_array)/np.sqrt(num_trials)
        
        return P_mean,m_mean
        
    def label_phase(self,sub_path,trialID):
        label = 4
        time = np.arange(0,4000,100)
        time_trunc = 3000
        idx = np.where(time >= time_trunc)[0]
        filename = "P_"+str(trialID)+".txt"
        P = np.loadtxt(open(sub_path+"/"+filename))
        P_t_mean = np.mean(P[idx])
        filename = "m_"+str(trialID)+".txt"
        m = np.loadtxt(open(sub_path+"/"+filename))
        m_t_mean = np.mean(m[idx])
        if m_t_mean != 0 and P_t_mean != 0:
            if m_t_mean > 0.5:
                label = 2
            elif P_t_mean > 0.5:
                label = 3
            else:
                label = 1
        return label
    
    def count_labels(self,num_trials,sub_path):
        labels = []
        for i in np.arange(0,num_trials,1):
            label = self.label_phase(sub_path,i+1)
            labels.append(label)
        labels = np.array(labels)
        hist, _ = np.histogram(labels,bins=np.arange(0.5,5.5,1),density = True)
        return hist

    def save_individual_P_m(self,time_trunc =200):
        for i,folder_name in enumerate(self.folder_names):
            subpath = self.data_path +"/"+ folder_name
            P_mean,m_mean = self.compute_mean_angular_momentum_directionality(self.num_trials,subpath)
            hist = self.count_labels(self.num_trials,subpath)
            P_slope, P_var, m_slope, m_var = self.compute_gradient_momentum_directionality_early_stage(self.num_trials,subpath,time_trunc = time_trunc)
            #data_array = [self.speeds[i], self.turning_rates[i], self.zoos[i], self.zors[i],P_mean,P_err,m_mean,m_err]
            label_array = [self.speeds[i], self.turning_rates[i], self.zoos[i], self.zors[i],P_slope, P_var, m_slope, m_var,hist[0],hist[1],hist[2],hist[3]]
            #np.savetxt(subpath+"/mean_angular_momentum_directionality.txt",np.array(data_array))
            np.savetxt(subpath+"/label_hist.txt",np.array(label_array))
            if i%100 == 0:
                print(i)
         
#### compute the gradient of the data
    def compute_gradient_momentum_directionality_early_stage(self,num_trials,sub_path,time_trunc = 500):
        time = np.arange(0,4000,100)
        idx = np.where(time <= time_trunc)[0]
        time_sub = time[idx].reshape(-1,1)
        P_slope_array = np.zeros(num_trials)
        P_var_array = np.zeros(num_trials)
        m_slope_array = np.zeros(num_trials)
        m_var_array = np.zeros(num_trials)
        for i in np.arange(0,num_trials,1):
            filename = "P_"+str(i+1)+".txt"
            P = np.loadtxt(open(sub_path+"/"+filename))
            P_sub = P[idx]
            filename = "m_"+str(i+1)+".txt"
            m = np.loadtxt(open(sub_path+"/"+filename))
            m_sub = m[idx]
            
            ######## fit to linear regression
            P_model = LinearRegression().fit(time_sub, P_sub)
            P_slope = P_model.coef_[0]
            P_sub_predict = P_model.predict(time_sub)
            if (P_sub == 0).any():
                P_var = np.inf
            else:
                P_var = np.mean(np.sqrt((P_sub-P_sub_predict)**2)/P_sub)
                
            m_model = LinearRegression().fit(time_sub, m_sub)
            m_slope = m_model.coef_[0]
            m_sub_predict = m_model.predict(time_sub)
            if (m_sub == 0).any():
                m_var = np.inf
            else:
                m_var = np.mean(np.sqrt((m_sub-m_sub_predict)**2)/m_sub)
                
            
            P_slope_array[i] = P_slope
            P_var_array[i] = P_var
            m_slope_array[i] = m_slope
            m_var_array[i] = m_var
            
        P_slope = np.mean(P_slope_array)
        P_var = np.mean(P_var_array)
        
        m_slope = np.mean(m_slope_array)
        m_var = np.mean(m_var_array)
            
        return P_slope, P_var, m_slope, m_var
    
    
    def collect_all_P_m_data(self):    
        label_data = []
        for i,folder_name in enumerate(self.folder_names):
            subpath = self.data_path +"/"+ folder_name
            label = np.loadtxt(open(subpath+"/label_hist.txt"))
            label_data.append(label)
        np.savetxt(self.data_path+"/labels.txt",np.round(np.array(label_data),3),fmt=('%.3f', '%.3f', '%.3f', '%.3f','%.6f', '%.6f', '%.6f', '%.6f', '%.3f', '%.3f', '%.3f', '%.3f'))
    
    def collect_high_dim_data(self,traj_name):
        path = self.data_path
        folder_names = self.folder_names
        collected_data = []  # List to store data from all files
        for folder_name in folder_names:
            file = f"{path}/{folder_name}/{traj_name}"
            try:
                # Attempt to load the file
                data = np.loadtxt(open(file))
                flattened_data = data.flatten()  # Converts to 1D array
                collected_data.append(flattened_data)  # Add to the list
            except Exception as e:
                # Handle file loading errors
                print(f"Error loading file {file}: {e}")
        
        if collected_data:
            return np.array(collected_data)  
            print("No valid data collected.")
            return None
    
    def PCA_analysis(self,collected_data, n_components = 3):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(collected_data)
        pca = PCA(n_components)  # Reduce to 2 components
        principal_components = pca.fit_transform(scaled_data)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        print("Explained variance ratio:", explained_variance)
        return principal_components,explained_variance