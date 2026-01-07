import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Imbalanced Learning Metrics
from imblearn.metrics import geometric_mean_score, sensitivity_score

# Custom Functions
from libraries.functions import load_config, setup_logger, get_class_from_string
from libraries.data_loading import load_datasets
from libraries.ecoc import ECOC
from libraries.imbalance_degree import imbalance_degree
from libraries.functions import compute_imbalance_ratio

from libraries.functions import apply_ecoc_binarization

import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Configuration
config = load_config('config_test.yaml')

# Setup Logger
logger = setup_logger(config)

# Common Parameters
data_path = config["paths"]["data_folder"]
Test_size = config["simulation"]["test_size"]
N_max = config["simulation"]["N_MAX"]
n_simus = config["simulation"]["n_simus"]
ECOC_enc = config["simulation"]["ecoc_enc"]
flg_swp = config["simulation"]["flg_swp"]
maj_min = config["simulation"]["maj_min"]
min_maj = config["simulation"]["min_maj"]
output_path = config["paths"]["output_folder"]

# Model Configuration (Assuming LSEnsemble is always model 2)
model_list = config["models"]
model_list = [model_list[2]]

# Load Datasets
dataset_special_cases = {}
datasets = load_datasets(data_path, N_max, dataset_special_cases)

# Process Datasets
for dataset_name, (X, y, C0) in datasets.items():
    
    nclass = np.unique(y).shape[0]

    if nclass > 2:
        if maj_min:
            sorted_classes = pd.Series(y).value_counts().index.to_list()
            class_dict = {sorted_classes[k]: nclass - k for k in range(nclass)}
        elif min_maj:
            sorted_classes = pd.Series(y).value_counts().index.to_list()
            class_dict = {sorted_classes[k]: k + 1 for k in range(nclass)}
        else:
            class_dict = {k + 1: k + 1 for k in range(nclass)}
        y = np.vectorize(class_dict.get)(y)
    
        class_labels = np.array(sorted(class_dict.keys()))
    
        M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
        M = (M_ecoc._code_matrix > 0).astype(int)
    
        if True:
            M[M == 1] = -C0
            M[M == 0] = C0
    
        M_ecoc._code_matrix = M
        IB_degree = imbalance_degree(y, "EU")
        
    elif nclass == 2:
        # Ensure -1 is the label for the majority class
        if np.sum(y == -1) < np.sum(y == 1):
            y = -y  # Swap -1 and +1

        class_labels = np.unique(y)
        M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
        M = (M_ecoc._code_matrix > 0).astype(int)
        M[M == 1] = -C0
        M[M == 0] = C0
        M_ecoc._code_matrix = M
        num_dichotomies = M_ecoc._code_matrix.shape[1]
        IB_degree = compute_imbalance_ratio(y)
        # Convert binary labels -1 and +1 to 1 and 2
        label_map_binary = {-1: 1, 1: 2}
        y_converted = np.array([label_map_binary[label] for label in y])
        y = y_converted

    num_dichotomies = M_ecoc._code_matrix.shape[1]

    logger.info('Multiclass Results')
    logger.info('---------------------------------------------')
    logger.info(f'Dataset: {dataset_name}. Imbalance Degree = {IB_degree:.2f}')
    logger.info(f'ECOC encoding: {M_ecoc.encoding}. Flag swap: {flg_swp}')
    logger.info(f'Classes: {nclass}. Dichotomies: {num_dichotomies}')
    logger.info('---------------------------------------------')

    for model_item in model_list:
        model_metrics = {
            "avg_acc": [],
            "avg_bal_acc": [],
            "avg_kappa": [],
            "avg_geom_mean": [],
            "avg_sensitivity": [],
            "avg_f1": [],
            "avg_mcc": [],
        }
    
        # Load the trained model and its configuration using pickle
        model_name = model_item["name"]
        logger.info(f'Model: {model_name}')
        
        model_class = get_class_from_string(model_item['class'])
        
        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
            logger.info(f'    Switching: {int(SW_optimization)}, QC: {int(QC_optimization)}, Cost: {int(RI_C_optimization)}, Population: {int(RI_P_optimization)}')

    
            filename_o = f"{dataset_name}_{ECOC_enc}_{model_name}_SW_{int(SW_optimization)}_QC_{int(QC_optimization)}_RIC_{int(RI_C_optimization)}_RIP_{int(RI_P_optimization)}_train.pkl"
        else:
            filename_o = f"{dataset_name}_{ECOC_enc}_{model_name}_train.pkl"
    
        file_path = os.path.join(output_path, filename_o)
    
        try:
            with open(file_path, 'rb') as f:
                result_data = pickle.load(f)
                best_model_config = result_data["best_config"]
        except FileNotFoundError:
            logger.error(f"File {filename_o} not found.")
            continue
    
        acc_simulations = []
        bal_acc_simulations = []
        kappa_simulations = []
        geom_mean_simulations = []
        sensitivity_simulations = []
        f1_simulations = []
        mcc_simulations = []
    
        n_simus = max(n_simus, 20) # Assuming 100 test simulations
        for k_simu in range(n_simus):  
            logger.info(f'    Simulation: {k_simu} out of {n_simus}')
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01 * Test_size, random_state=42 + k_simu)
            M_tst = X_test.shape[0]
    
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_n = scaler.transform(X_train)
            X_test_n = scaler.transform(X_test)
    
            Ye_train, Ye_test, flag_swap_test, idx_train_ecoc, idx_test_ecoc = apply_ecoc_binarization(
                M, y_train, y_test, apply_flag_swap=flg_swp
            )
    
            Ye_pred = np.zeros((M_tst, num_dichotomies))
            for j_dic in range(num_dichotomies):
                x_train = X_train_n[idx_train_ecoc[j_dic], :]
                ye_train = Ye_train[j_dic]
                
                cw_train = np.ones(len(ye_train)) # Default sample weights
                
                x_test = X_test_n[idx_test_ecoc[j_dic], :]
                ye_test = Ye_test[j_dic]
    
                unique_labels = np.unique(ye_train)
                if len(unique_labels) < 2:
                    ye_pred = ye_test # np.zeros_like(ye_test)
                    if flag_swap_test[j_dic]:
                        ye_pred = ye_pred * -1
                    Ye_pred[:, j_dic] = ye_pred
                else:
                    cv_config = best_model_config[j_dic][0] # Get the best config only
                    model = model_class(**cv_config)
                        # Train the model
                    if model_name in ["MLPClassifier","kNN", "MultiRandBal"]:
                        model.fit(x_train, ye_train)
                    else:
                        model.fit(x_train, ye_train, sample_weight=cw_train)
                    ye_pred = model.predict(x_test)
                    if flag_swap_test[j_dic]:
                        ye_pred *= -1
                    Ye_pred[:, j_dic] = ye_pred
    
            if nclass > 2:
                y_pred_MC_ab = np.array([M_ecoc._get_closest_class(Ye_pred[row, :]) for row in range(M_tst)])
            elif nclass == 2:
                y_pred_MC_ab = (3+np.array([M_ecoc._get_closest_class(Ye_pred[row, :]) for row in range(M_tst)]))/2

    
            mat_confusion_ab = confusion_matrix(y_test, y_pred_MC_ab, labels=class_labels)
            cohen_kappa_ab = cohen_kappa_score(y_test, y_pred_MC_ab)
            acc_ab = accuracy_score(y_test, y_pred_MC_ab)
            bal_acc_ab = balanced_accuracy_score(y_test, y_pred_MC_ab)
            geom_mean_ab = geometric_mean_score(y_test, y_pred_MC_ab, average='weighted')
            sensitivity_ab = sensitivity_score(y_test, y_pred_MC_ab, average='weighted')
            if nclass > 2:
                f1_score_ab = f1_score(y_test, y_pred_MC_ab, average='weighted')
            elif nclass == 2:
                f1_score_ab = f1_score(y_test-1, y_pred_MC_ab-1)
            # Store metrics for the current simulation
            acc_simulations.append(acc_ab)
            bal_acc_simulations.append(bal_acc_ab)
            kappa_simulations.append(cohen_kappa_ab)
            geom_mean_simulations.append(geom_mean_ab)
            sensitivity_simulations.append(sensitivity_ab)
            f1_simulations.append(f1_score_ab)
            
        # Calculate the average and standard deviation for the model
        model_metrics = {
            "avg_acc": np.mean(acc_simulations),
            "avg_bal_acc": np.mean(bal_acc_simulations),
            "avg_kappa": np.mean(kappa_simulations),
            "avg_geom_mean": np.mean(geom_mean_simulations),
            "avg_sensitivity": np.mean(sensitivity_simulations),
            "avg_f1_score": np.mean(f1_simulations),
            "std_acc": np.std(acc_simulations),
            "std_bal_acc": np.std(bal_acc_simulations),
            "std_kappa": np.std(kappa_simulations),
            "std_geom_mean": np.std(geom_mean_simulations),
            "std_sensitivity": np.std(sensitivity_simulations),
            "std_f1_score": np.std(f1_simulations),
        }
    
        # Add mc_metrics to result_data
        result_data["multiclass_metrics"] = model_metrics  
        
        # Print results with 5 decimals
        logger.info('')
        logger.info(f"Accuracy: {model_metrics['avg_acc']:.5f} \u00B1 {model_metrics['std_acc']:.5f}")
        logger.info(f"Balanced Accuracy: {model_metrics['avg_bal_acc']:.5f} \u00B1 {model_metrics['std_bal_acc']:.5f}")
        logger.info(f"Cohen's Kappa: {model_metrics['avg_kappa']:.5f} \u00B1 {model_metrics['std_kappa']:.5f}")
        logger.info(f"Geometric Mean Score (weighted): {model_metrics['avg_geom_mean']:.5f} \u00B1 {model_metrics['std_geom_mean']:.5f}")
        logger.info(f"Sensitivity Score (weighted): {model_metrics['avg_sensitivity']:.5f} \u00B1 {model_metrics['std_sensitivity']:.5f}")
        logger.info(f"F1 Score (weighted): {model_metrics['avg_f1_score']:.5f} \u00B1 {model_metrics['std_f1_score']:.5f}")

    
        # Save the combined configuration and metrics using pickle
        file_path = os.path.join(output_path, filename_o)
        
        with open(file_path, 'wb') as f:
            pickle.dump(result_data, f)
            
        logger.info(f"Multiclass metrics and configuration saved to {filename_o}")
        logger.info('-------------------------------------------------------')
        logger.info('')
