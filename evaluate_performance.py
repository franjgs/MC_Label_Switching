"""
Evaluate Multiclass Performance (Program 2)
-------------------------------------------
This script evaluates the multiclass performance of the LSEnsemble (Label Switching Ensemble)
using the optimal configurations found in Program 1.

Key features:
- Uses the specific '_ab' reconstruction logic (Multiclass with Switching).
- Measures the execution time (training + inference) for the final optimized model.
- Generates 20+ Monte Carlo simulations for statistical robustness.
"""

import os
import pickle
import time
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
# Suppress ConvergenceWarnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Configuration
config = load_config('config_test.yaml')

# Setup Logger
logger = setup_logger(config)

# Common Parameters

data_path = config["paths"]["data_folder"]
force_results = config["simulation"]["force_results"]
Test_size = config["simulation"]["test_size"]
N_max = config["simulation"]["N_MAX"]
n_simus = config["simulation"]["n_simus"]
ECOC_enc = config["simulation"]["ecoc_enc"]
flg_swp = config["simulation"]["flg_swp"]
maj_min = config["simulation"]["maj_min"]
min_maj = config["simulation"]["min_maj"]
output_path = config["paths"]["output_folder"]

# Model Configuration - Full list of available models
model_list = config["models"]

# COMMENTS FOR SAFE SELECTION:
# - LogisticRegression: index 0 – Logistic regression with L2 regularization
# - RandomForestClassifier: index 1 – Random forest with class_weight='balanced'
# - LSEnsemble: index 2 – Asymmetric Label Switched Ensemble (ALSE) – OUR MAIN METHOD
# - MLPClassifier: index 3 – Standard multilayer perceptron from scikit-learn
# - LGBMClassifier: index 4 – LightGBM with is_unbalance=True
# - kNN: index 5 – K-Nearest Neighbors
# - C4.5: index 6 – Decision tree with entropy criterion (C4.5 approximation)
# - SVM: index 7 – Support Vector Machine with RBF kernel
# - MultiRandBal: index 8 – Ensemble with SMOTE oversampling + random undersampling

# Examples of selection (uncomment the desired line):

# 1. Run ALL models (default configuration)
# model_list = config["models"]

# 2. Run only our main method (LSEnsemble / ALSE)
# model_list = [config["models"][2]] # Only ALSE

# 3. Comparison between ALSE and classical baselines (example: RF + LightGBM + SVM + MLP)
# model_list = [config["models"][i] for i in [1, 4, 7, 3]]

# 4. Run only baselines without ALSE (for ablation or clean comparison)
# model_list = [config["models"][i] for i in [0, 1, 3, 4, 5, 6, 7, 8]]
# del model_list[2]  # Example of exclusion if using the full list

# Apply the desired selection here (uncomment only one option)
# model_list = config["models"]  # All models
# model_list = [config["models"][i] for i in [1, 4, 7, 3]]
model_list = [config["models"][2]] # Only ALSE


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
        if np.sum(y == -1) < np.sum(y == 1):
            y = -y  # Ensure -1 is majority

        class_labels = np.unique(y)
        M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
        M = (M_ecoc._code_matrix > 0).astype(int)
        M[M == 1] = -C0
        M[M == 0] = C0
        M_ecoc._code_matrix = M
        IB_degree = compute_imbalance_ratio(y)
        
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
        model_name = model_item["name"]
        logger.info(f'Model: {model_name}')
        
        model_class = get_class_from_string(model_item['class'])
        
        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
            logger.info(f'    Switching: {int(SW_optimization)}, QC: {int(QC_optimization)}, Cost: {int(RI_C_optimization)}, Population: {int(RI_P_optimization)}')

            filename_i = f"{dataset_name}_{ECOC_enc}_{model_name}_SW_{int(SW_optimization)}_QC_{int(QC_optimization)}_RIC_{int(RI_C_optimization)}_RIP_{int(RI_P_optimization)}_train.pkl"
        else:
            filename_i = f"{dataset_name}_{ECOC_enc}_{model_name}_train.pkl"
    
        filename_o = filename_i.replace('_train.pkl', '_test.pkl')
        file_path_i = os.path.join(output_path, filename_i)
        file_path_o = os.path.join(output_path, filename_o)

        try:
            with open(file_path_i, 'rb') as f:
                result_data = pickle.load(f)
                best_model_config = result_data["best_config"]
        except FileNotFoundError:
            logger.error(f"File {filename_i} not found.")
            continue
    
        updates = {
            "dataset": dataset_name,
            "nclass": nclass,
            "class_labels": class_labels.tolist(),
            "ECOC_enc": ECOC_enc,
            "num_dichotomies": num_dichotomies,
            'flag_swap': None,
            "model_name": model_name,
            "n_simus": n_simus,
            "Test_size": Test_size
        }
        
        # Solo actualizamos si no existe o está vacío
        for key, value in updates.items():
            if key not in result_data or result_data[key] is None:
                result_data[key] = value

                                
        # Check if execution time was already computed
        # We consider it "computed" if the key exists AND value is positive (>0)
        execution_time = result_data.get('execution_time_seconds', 0)
        if not force_results:
            if isinstance(execution_time, (int, float)) and execution_time > 0:
                logger.info(f"Skipping model '{model_name}': execution time already computed "
                            f"({execution_time:.2f} seconds)")
                # Optional: still save if you want to ensure latest format, but usually skip all
                continue
        
        # If we reach here → either no key, or value <=0 → proceed with simulations
        logger.info(f"Computing execution time and metrics for model '{model_name}'")
            
        # Metrics and time lists
        CM_simulations = []
        acc_simulations = []
        bal_acc_simulations = []
        kappa_simulations = []
        geom_mean_simulations = []
        sensitivity_simulations = []
        f1_simulations = []
        runtime_simulations = [] # New list for time tracking

        n_simus = max(n_simus, 20) 
        for k_simu in range(n_simus):  
            # Start timer for this simulation (Train + Test)
            start_time = time.time()

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.01 * Test_size, random_state=42 + k_simu)
            M_tst = X_test.shape[0]
    
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_n = scaler.transform(X_train)
            X_test_n = scaler.transform(X_test)
    
            Ye_train, Ye_test, flag_swap_test, idx_train_ecoc, idx_test_ecoc = apply_ecoc_binarization(
                M, y_train, y_test, apply_flag_swap=flg_swp, flag_swap=result_data["flag_swap"]
            )
    
            Ye_pred = np.zeros((M_tst, num_dichotomies))
            for j_dic in range(num_dichotomies):
                x_train = X_train_n[idx_train_ecoc[j_dic], :]
                ye_train = Ye_train[j_dic]
                cw_train = np.ones(len(ye_train)) 
                
                x_test = X_test_n[idx_test_ecoc[j_dic], :]
                ye_test = Ye_test[j_dic]
    
                unique_labels = np.unique(ye_train)
                if len(unique_labels) < 2:
                    ye_pred = ye_test 
                    if flag_swap_test[j_dic]:
                        ye_pred = ye_pred * -1
                    Ye_pred[:, j_dic] = ye_pred
                else:
                    cv_config = best_model_config[j_dic][0] 
                    model = model_class(**cv_config)
                    
                    # Train the model
                    if model_name in ["MLPClassifier", "kNN", "MultiRandBal"]:
                        model.fit(x_train, ye_train)
                        # Evaluate the model
                        ye_pred = model.predict(x_test)
                    elif model_name == "LGBMClassifier":
                        # Create feature names for consistency (f0, f1, ...)
                        feature_names = [f'f{i}' for i in range(x_train.shape[1])]
                        
                        # Convert training and validation/test sets to pandas DataFrame
                        x_train_df = pd.DataFrame(x_train, columns=feature_names)
                        x_test_df = pd.DataFrame(x_test, columns=feature_names)
                        
                        # Train the model using DataFrame (eliminates the warning)
                        model.fit(x_train_df, ye_train, sample_weight=cw_train)
                        
                        # Predict using DataFrame
                        y_pred = model.predict(x_test_df)
                    else:
                        model.fit(x_train, ye_train, sample_weight=cw_train)
                        # Evaluate the model
                        ye_pred = model.predict(x_test)
                        
                    if flag_swap_test[j_dic]:
                        ye_pred *= -1
                    Ye_pred[:, j_dic] = ye_pred
    
            # Multiclass Reconstruction (Switching-aware)
            if nclass > 2:
                y_pred_MC_ab = np.array([M_ecoc._get_closest_class(Ye_pred[row, :]) for row in range(M_tst)])
            elif nclass == 2:
                y_pred_MC_ab = (3+np.array([M_ecoc._get_closest_class(Ye_pred[row, :]) for row in range(M_tst)]))/2

            # End timer
            end_time = time.time()
            runtime_simulations.append((end_time - start_time))
    
            # Metrics Calculation
            CM_ab = confusion_matrix(y_test, y_pred_MC_ab, labels=class_labels)
            cohen_kappa_ab = cohen_kappa_score(y_test, y_pred_MC_ab)
            acc_ab = accuracy_score(y_test, y_pred_MC_ab)
            bal_acc_ab = balanced_accuracy_score(y_test, y_pred_MC_ab)
            geom_mean_ab = geometric_mean_score(y_test, y_pred_MC_ab, average='weighted')
            sensitivity_ab = sensitivity_score(y_test, y_pred_MC_ab, average='weighted')
            
            if nclass > 2:
                f1_score_ab = f1_score(y_test, y_pred_MC_ab, average='weighted')
            elif nclass == 2:
                f1_score_ab = f1_score(y_test-1, y_pred_MC_ab-1)

            CM_simulations.append(CM_ab)
            acc_simulations.append(acc_ab)
            bal_acc_simulations.append(bal_acc_ab)
            kappa_simulations.append(cohen_kappa_ab)
            geom_mean_simulations.append(geom_mean_ab)
            sensitivity_simulations.append(sensitivity_ab)
            f1_simulations.append(f1_score_ab)
            
            if (k_simu + 1) % 5 == 0:
                logger.info(f" -> Simulation {k_simu + 1}/{n_simus} completed")
            
        # Convert list of confusion matrices to a 3D array: (n_simus, n_classes, n_classes)
        CM_array = np.array(CM_simulations)  # Shape: (n_simus, n_classes, n_classes)

        # Compute mean and std of the confusion matrix across simulations
        mean_CM = np.mean(CM_array, axis=0)
        std_CM = np.std(CM_array, axis=0)

        # Optional: Round for cleaner storage (e.g., to integers if counts are whole numbers)
        mean_CM = np.round(mean_CM).astype(int)
        std_CM = np.round(std_CM, 2)  # Keep 2 decimals for std
        
        # Statistical aggregation
        model_metrics = {
            "avg_CM": mean_CM.tolist(), "std_CM": std_CM.tolist(),
            "avg_acc": np.mean(acc_simulations), "std_acc": np.std(acc_simulations),
            "avg_bal_acc": np.mean(bal_acc_simulations), "std_bal_acc": np.std(bal_acc_simulations),
            "avg_kappa": np.mean(kappa_simulations), "std_kappa": np.std(kappa_simulations),
            "avg_geom_mean": np.mean(geom_mean_simulations), "std_geom_mean": np.std(geom_mean_simulations),
            "avg_sensitivity": np.mean(sensitivity_simulations), "std_sensitivity": np.std(sensitivity_simulations),
            "avg_f1_score": np.mean(f1_simulations), "std_f1_score": np.std(f1_simulations),
        }

        result_data["multiclass_metrics"] = model_metrics  
        # Always update runtime (it's new information)
        result_data["execution_time_seconds"] = np.mean(runtime_simulations)        # total time 
        # Normalize by number of dichotomies (makes it comparable across datasets)
        result_data["execution_time_per_dichotomy"] = result_data["execution_time_seconds"] / num_dichotomies   # normalized (more useful)
    
        new_metrics = result_data["multiclass_metrics"] 
        logger.info(f"Balanced Accuracy: {new_metrics['avg_bal_acc']:.5f} ± {new_metrics['std_bal_acc']:.5f}")
        logger.info(f"Geo Mean: {new_metrics['avg_geom_mean']:.5f} ± {new_metrics['std_geom_mean']:.5f}")
        logger.info(f"Avg Runtime: {result_data['execution_time_seconds'] :.4f} s")
        logger.info(f"Avg Runtime per dichotomy: {result_data['execution_time_per_dichotomy']:.4f} s")

        
        with open(file_path_o, 'wb') as f:
            pickle.dump(result_data, f)
            
        logger.info(f"Results saved to {filename_o}\n")