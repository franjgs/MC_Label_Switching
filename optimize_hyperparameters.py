import os
import pickle

import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

# Create a mapping dictionary
METRIC_FUNCTIONS = {
    "balanced_accuracy_score": balanced_accuracy_score,
    "f1_score": f1_score,
    "matthews_corrcoef": matthews_corrcoef,
}

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Custom Libraries
from libraries.ecoc import ECOC
from libraries.data_loading import load_datasets
from libraries.functions import load_config, setup_logger, get_class_from_string
from libraries.functions import generate_model_configurations, apply_ecoc_binarization

from libraries.functions import compute_imbalance_ratio
from libraries.imbalance_degree import imbalance_degree

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DEFAULT_CONFIG_PATH = "config/config_smoke.yaml"

# Transition note:
# config_smoke.yaml is currently used as the validated structured
# LSEnsemble configuration while config_train.yaml is migrated.

parser = argparse.ArgumentParser(
    description="Optimize hyperparameters for binary-decomposition multiclass experiments."
)
parser.add_argument(
    "--config",
    default=DEFAULT_CONFIG_PATH,
    help="Path to the YAML configuration file."
)
args = parser.parse_args()

# Load Configuration
config = load_config(args.config)

# -----------------------------------------------------------------------------
# INITIALIZE LOGGER
# -----------------------------------------------------------------------------
# Setup Logger
logger = setup_logger(config)

# Common Parameters
verbose = config["simulation"]["verbose"]
data_path = config["paths"]["data_folder"]
output_path = config["paths"]["output_folder"]
if not os.path.exists(output_path):
    print(" New folder created: {}".format(output_path))
    os.makedirs(output_path)
    
n_simus = config["simulation"]["n_simus"]
Test_size = config["simulation"]["test_size"]
N_max = config["simulation"]["N_MAX"]
num_folds = config["simulation"]["num_folds"]
ECOC_enc = config["simulation"]["ecoc_enc"]
flg_swp = config["simulation"]["flg_swp"]
maj_min = config["simulation"]["maj_min"]
min_maj = config["simulation"]["min_maj"]
metric_function_name = config["simulation"]["f_sel"]

# Retrieve the actual function from the mapping
if metric_function_name not in METRIC_FUNCTIONS:
    raise ValueError(f"Invalid metric function '{metric_function_name}'. Choose from {list(METRIC_FUNCTIONS.keys())}.")
f_sel = METRIC_FUNCTIONS[metric_function_name]
model_selection = config["simulation"]["model_selection"]

# Model Configuration - Full list of available models
model_list = config["models"]

# COMMENTS FOR SAFE SELECTION:
# Select models by name so the script does not depend on YAML ordering.

# Examples of selection (uncomment the desired line):

# 1. Run ALL models (default configuration)
# model_list = config["models"]

# 2. Run only our main method (LSEnsemble / ALSE)
# model_list = [m for m in config["models"] if m["name"] == "LSEnsemble"]

# 3. Comparison between ALSE and classical baselines (example: RF + LightGBM + SVM + MLP)
# model_list = [m for m in config["models"] if m["name"] in {"RandomForestClassifier", "LGBMClassifier", "SVM", "MLPClassifier"}]

# 4. Run only baselines without ALSE (for ablation or clean comparison)
# model_list = [m for m in config["models"] if "LSEnsemble" not in m["name"]]

# Apply the desired selection here (uncomment only one option)
# model_list = config["models"]  # All models
model_list = [m for m in config["models"] if m["name"] == "LSEnsemble"]

# Generate Model Configurations
CV_config = generate_model_configurations(model_list)

# Load Datasets
dataset_special_cases = {}
datasets = load_datasets(data_path, N_max, dataset_special_cases)

Partial_saving = False

# Process Datasets
for dataset_name, (X, y, C0) in datasets.items():
    n_samples, n_attributes = X.shape
    nclass = np.unique(y).shape[0]
    logger.info('')
    logger.info('---------------------------------------------------------------------')
    logger.info(f'Dataset: {dataset_name}.')
    logger.info(f'Number of classes: {nclass}. Number of samples: {n_samples}. Dimensionality: {n_attributes}')
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
        IB_degree = imbalance_degree(y, "EU")
        
        class_labels = np.array(sorted(class_dict.keys()))
        M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
        M = (M_ecoc._code_matrix > 0).astype(int)
        if True:
            M[M == 1] = -C0
            M[M == 0] = C0
        M_ecoc._code_matrix = M
        num_dichotomies = M_ecoc._code_matrix.shape[1]
    elif nclass == 2:
        # Ensure -1 is the label for the majority class
        if np.sum(y == -1) < np.sum(y == 1):
            y = -y  # Swap -1 and +1
        IB_degree = compute_imbalance_ratio(y)
        
        class_labels = np.unique(y)
        M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
        M = (M_ecoc._code_matrix > 0).astype(int)
        M[M == 1] = -C0
        M[M == 0] = C0
        M_ecoc._code_matrix = M
        num_dichotomies = M_ecoc._code_matrix.shape[1]
        
        # Convert binary labels -1 and +1 to 1 and 2
        label_map_binary = {-1: 1, 1: 2}
        y = np.array([label_map_binary[label] for label in y])
    else:
        raise ValueError(f"Number of classes ({nclass}) must be greater than or equal to 2.")
        
    logger.info(f'ECOC encoding: {M_ecoc.encoding}. Flag swap: {flg_swp}')
    logger.info(f'Number of Dichotomies: {num_dichotomies}. Imbalance Degree = {IB_degree:.2f}')
    logger.info('f_sel: '+f_sel.__name__)
    
    CV_config_full = {}
    best_model_avg = {}
    metric_conf = {}
    CM_accumulated = {}
    best_metric_avg = dict()
    best_metric_peak = dict()
    best_model_peak = dict()
    filename_o = dict()
    partial_filename_o = dict()
    for model_item in model_list:
        model_name = model_item["name"]
        logger.info(f'Model: {model_name}')

        if 'LSEnsemble' in model_name:
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']

            logger.info(f'    Switching: {int(SW_optimization)}, QC: {int(QC_optimization)}, Cost: {int(RI_C_optimization)}, Population: {int(RI_P_optimization)}')
            filename_o[model_name] = f"{dataset_name}_{ECOC_enc}_{model_name}_SW_{int(SW_optimization)}_QC_{int(QC_optimization)}_RIC_{int(RI_C_optimization)}_RIP_{int(RI_P_optimization)}_train.pkl"
            if Partial_saving:
                partial_filename_o[model_name] = f"{dataset_name}_{ECOC_enc}_{model_name}_SW_{int(SW_optimization)}_QC_{int(QC_optimization)}_RIC_{int(RI_C_optimization)}_RIP_{int(RI_P_optimization)}_partial_train.pkl"
        else:
            if Partial_saving:
                partial_filename_o[model_name] = f"{dataset_name}_{ECOC_enc}_{model_name}_partial_train.pkl"
            filename_o[model_name] = f"{dataset_name}_{ECOC_enc}_{model_name}_train.pkl"

        n_conf_test = len(CV_config[model_name])

        CV_config_full[model_name] = [[] for j_dic in range(num_dichotomies)]
        CM_accumulated[model_name] = [np.zeros((n_conf_test, 2, 2)) for j_dic in range(num_dichotomies)]  # Accumulated confusion matrix
        metric_conf[model_name] = np.zeros((n_conf_test, num_dichotomies, num_folds, n_simus))
        best_metric_avg[model_name] = np.zeros(num_dichotomies)
        best_model_avg[model_name] = [dict() for j_dic in range(num_dichotomies)]
        best_metric_peak[model_name] = np.zeros(num_dichotomies)
        best_model_peak[model_name] = [dict() for j_dic in range(num_dichotomies)]
 
    logger.info('-----------------------------------------------------')
            
    for k_simu in range(n_simus):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01*Test_size, random_state=42+k_simu)
        M_tr = X_train.shape[0]
        M_tst = X_test.shape[0]
        input_size = X_train.shape[1]
    
        scaler = StandardScaler()  # Normalización de características
        scaler.fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_test_n = scaler.transform(X_test)
        
        # Inner n-Fold for validation
        # Inner CV / hold-out logic (simplified and consistent)
        if num_folds > 1:
            # Stratified K-Fold: divides X_train into multiple folds
            inner_cv = StratifiedKFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=42 + k_simu
            )
        else:
            # For num_folds == 1: use a dummy splitter that returns full train + empty val
            class FullTrainSplitter:
                def __init__(self, n_samples):
                    self.n_samples = n_samples
                
                def split(self, X, y=None, groups=None):
                    yield np.arange(self.n_samples, dtype=np.int64), np.array([], dtype=np.int64)
            
            inner_cv = FullTrainSplitter(n_samples=X_train_n.shape[0])
        
        # Bucle interno uniforme (sin if num_folds)
        for nFold, (train_index, val_index) in enumerate(inner_cv.split(X_train_n, y_train), 1):
            if verbose:
                logger.info(f"Simulation {k_simu+1}. Inner Fold {nFold}:")
            
            # Always use train_index for training
            X_train_cv = X_train_n[train_index]
            y_train_cv = y_train[train_index]
            
            # If there is a validation set (only when num_folds > 1)
            if len(val_index) > 0:
                X_test_cv = X_train_n[val_index]
                y_test_cv = y_train[val_index]
                # Use validation for metric/BIC/etc.
                # e.g., val_metric = f_sel(y_val_cv, model.predict(X_val_cv))
            else:
                # num_folds == 1 → no validation set → train with full X_train
                X_test_cv = X_test_n
                y_test_cv = y_test

            # Apply ECOC binarization
            Ye_train, Ye_test, flag_swap, idx_train_ecoc, idx_test_ecoc = apply_ecoc_binarization(M, y_train_cv, y_test_cv, apply_flag_swap = flg_swp)
                
            QP_tr = np.zeros((num_dichotomies))
            for j_dic in range(num_dichotomies):
                ye_train, ye_test = Ye_train[j_dic], Ye_test[j_dic]
                
                # Check if the dichotomy contains both classes
                unique_labels = np.unique(np.concatenate([ye_train, ye_test]))
                unique_labels_train = unique_labels[np.isin(unique_labels, ye_train)]
                unique_labels_test = unique_labels[np.isin(unique_labels, ye_test)]
                
                if len(unique_labels_train) < 2:
                    for model_item in model_list:
                        model_name = model_item["name"]
                        CV_config_full[model_name][j_dic] = CV_config[model_name]
                        
                        k_conf = 0 # Default configuration
                        ye_pred = ye_test
                        metric = 1.0 if np.array_equal(ye_test, ye_pred) else 0.0  # Fallback metric                    
                        metric_conf[model_name][k_conf, j_dic, nFold-1, k_simu] = metric
                        CM = confusion_matrix(ye_test, ye_pred, labels=np.array([-1,1]))
                        CM_accumulated[model_name][j_dic][k_conf] += CM
                        best_metric_peak[model_name][j_dic] = metric
                        best_model_peak[model_name][j_dic] = [CV_config[model_name][k_conf], metric, CM]
                       
                    continue

                QP_tr[j_dic] = compute_imbalance_ratio(ye_train)
                                                        
                x_train = X_train_cv[idx_train_ecoc[j_dic], :]
                x_test = X_test_cv[idx_test_ecoc[j_dic], :]
    
                cw_train = np.ones(len(ye_train)) # Default sample weights
                
                for model_item in model_list:
                    model_name = model_item["name"]
                    model_class = get_class_from_string(model_item["class"])
                    if 'LSEnsemble' in model_name:
                        CV_config[model_name] = generate_model_configurations(
                            [model_item],
                            x_train=x_train,
                            y_train_lab=ye_train,
                            logger=logger
                        )[model_name]
                        
                    CV_config_full[model_name][j_dic] = CV_config[model_name]
                
                    k_conf = 0
                    for cv_config in CV_config[model_name]:

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
                            ye_pred = model.predict(x_test_df)
                        else:
                            model.fit(x_train, ye_train, sample_weight=cw_train)
                            # Evaluate the model
                            ye_pred = model.predict(x_test)
                            
                        if len(unique_labels_test) == 1:
                            if len(np.unique(ye_pred)) == 2:
                                CM = confusion_matrix(ye_test, ye_pred, labels=unique_labels)
                                CM_accumulated[model_name][j_dic][k_conf] += CM
                                metric = f_sel(ye_pred, ye_test)  # Swap order to avoid warning
                            else:
                                CM = confusion_matrix(ye_test, ye_pred, labels=np.array([-1,1]))
                                CM_accumulated[model_name][j_dic][k_conf] += CM
                                metric = 1.0 if np.array_equal(ye_test, ye_pred) else 0  # Fallback metric
                        else:
                            CM = confusion_matrix(ye_test, ye_pred, labels=unique_labels)
                            CM_accumulated[model_name][j_dic][k_conf] += CM
                            metric = f_sel(ye_test, ye_pred)

                        metric_conf[model_name][k_conf, j_dic, nFold-1, k_simu] = metric
    
                        if metric > best_metric_peak[model_name][j_dic]:
                            best_metric_peak[model_name][j_dic] = metric
                            best_model_peak[model_name][j_dic] = [cv_config, metric, CM]
                            if verbose: #  and model_selection == "peak":
                                logger.info(f'  Model: {model_name}. Dichotomy: {j_dic+1}')
                                logger.info(f'      Best configuration (peak): {cv_config}')
                                logger.info(f'      Best metric (peak): {best_metric_peak[model_name][j_dic]:.5f}')
                                logger.info('')
                        k_conf += 1
        
        if Partial_saving:
            # After processing ALL dichotomies for this simulation k_simu
            # 1. Compute partial best configurations (avg and peak) up to this simulation
            partial_best_model_avg = {}
            partial_best_metric_avg = dict()
            partial_best_metric_peak = dict()
            partial_best_model_peak = dict()
            
            for model_item in model_list:
                model_name = model_item["name"]
                partial_best_metric_avg[model_name] = np.zeros(num_dichotomies)
                partial_best_model_avg[model_name] = [dict() for j_dic in range(num_dichotomies)]
                partial_best_metric_peak[model_name] = np.zeros(num_dichotomies)
                partial_best_model_peak[model_name] = [dict() for j_dic in range(num_dichotomies)]
                
                for j_dic in range(num_dichotomies):
                    k_conf = 0
                    for cv_config in CV_config_full[model_name][j_dic]:
                        # Average performance up to this simulation
                        avg_metric_conf = np.mean(metric_conf[model_name][k_conf, j_dic, :, :k_simu+1])
                        CM_avg = CM_accumulated[model_name][j_dic][k_conf] / ((num_folds * (k_simu+1)))
    
                        if model_selection == "avg":
                            if avg_metric_conf > partial_best_metric_avg[model_name][j_dic]:
                                partial_best_metric_avg[model_name][j_dic] = avg_metric_conf
                                partial_best_model_avg[model_name][j_dic] = [cv_config, avg_metric_conf, CM_avg]
    
                        # Peak: update if this simulation has a better fold
                        peak_metric = np.max(metric_conf[model_name][k_conf, j_dic, :, :k_simu+1])
                        if peak_metric > partial_best_metric_peak[model_name][j_dic]:
                            partial_best_metric_peak[model_name][j_dic] = peak_metric
                            partial_best_model_peak[model_name][j_dic] = [cv_config, peak_metric, CM_avg]  # or take the best fold CM
    
                        k_conf += 1
    
            partial_best_model = dict()
            partial_best_metric = dict()
            # Best model selection logic
            for model_item in model_list:
                model_name = model_item["name"]
                partial_best_model[model_name] = [dict() for j_dic in range(num_dichotomies)] #initialize
                partial_best_metric[model_name] = [0 for j_dic in range(num_dichotomies)] #initialize
            
                for j_dic in range(num_dichotomies):
                    if model_selection == "peak":
                        # Select based on peak performance (partial_best_metric_peak)
                        partial_best_model[model_name][j_dic] = partial_best_model_peak[model_name][j_dic]  # cv_config, metric, CM
                        partial_best_metric[model_name][j_dic] = partial_best_metric_peak[model_name][j_dic]
                    elif model_selection == "avg":
                        # Select based on average performance (partial_best_metric_avg)
                        partial_best_model[model_name][j_dic] = partial_best_model_avg[model_name][j_dic] # cv_config, metric, CM
                        partial_best_metric[model_name][j_dic] = partial_best_metric_avg[model_name][j_dic]
                    if verbose:
                        logger.info(f'Model: {model_name}. Dichotomy: {j_dic+1}')
                        logger.info(f'Partial Best configuration ({model_selection}): {partial_best_model[model_name][j_dic][0]}')  # cv_config
                        logger.info(f'Partial Best metric ({model_selection}): {partial_best_model[model_name][j_dic][1]:.5f}')  # metric
                        logger.info(f'Partial Best CM ({model_selection}): {partial_best_model[model_name][j_dic][2]}')  # CM
                        logger.info('')
                
                # 3. Save partial results after each simulation
                partial_result_data = {
                    "dataset": dataset_name,
                    'nclass': nclass,
                    "class_labels": class_labels.tolist(),
                    "ECOC_enc": ECOC_enc,
                    "num_dichotomies": num_dichotomies,
                    'flag_swap': flag_swap,
                    "model_name": model_name,
                    'n_simus_so_far': k_simu + 1,  # important to know how far it got
                    'num_folds': num_folds,
                    "Test_size": Test_size,
                    'f_sel': metric_function_name,
                   
                    "best_config": partial_best_model[model_name],
                    "binary_metrics": {
                        "best_metric_peak": partial_best_metric_peak[model_name],
                        "best_metric_avg": partial_best_metric_avg[model_name],
                        "best_metric": partial_best_metric[model_name],
                    },
                }                
                # Generate filename based on dataset and model
                partial_file_path = os.path.join(output_path, partial_filename_o[model_name])
        
                with open(partial_file_path, 'wb') as f:
                    pickle.dump(partial_result_data, f)
        
                logger.info(f"Partial results saved after simulation {k_simu+1}/{n_simus}: {partial_filename_o}")
    
    # Compute final best_metric_avg (average performance)
    for model_item in model_list:
        model_name = model_item["name"]
        for j_dic in range(num_dichotomies):
            k_conf = 0
            for cv_config in CV_config_full[model_name][j_dic]:
                # Compute the averaged metric for this configuration after all simulations and folds
                avg_metric_conf = np.mean(metric_conf[model_name][k_conf, j_dic, :, :])
                # Normalize by the total number of test samples to get the mean confusion matrix
                CM_avg = CM_accumulated[model_name][j_dic][k_conf] / (num_folds * n_simus)
    
                # Update best metric and model for the current configuration
                if avg_metric_conf >= best_metric_avg[model_name][j_dic]:
                    best_metric_avg[model_name][j_dic] = avg_metric_conf
                    best_model_avg[model_name][j_dic] = [cv_config, avg_metric_conf, CM_avg]
    
                k_conf += 1
    
    best_model = dict()
    best_metric = dict()
    # Final Best model selection logic
    for model_item in model_list:
        model_name = model_item["name"]
        best_model[model_name] = [dict() for j_dic in range(num_dichotomies)] #initialize
        best_metric[model_name] = [0 for j_dic in range(num_dichotomies)] #initialize
    
        for j_dic in range(num_dichotomies):
            if model_selection == "peak":
                # Select based on peak performance (best_metric_peak)
                best_model[model_name][j_dic] = best_model_peak[model_name][j_dic]  # cv_config, metric, CM
                best_metric[model_name][j_dic] = best_metric_peak[model_name][j_dic]
            elif model_selection == "avg":
                # Select based on average performance (best_metric_avg)
                best_model[model_name][j_dic] = best_model_avg[model_name][j_dic] # cv_config, metric, CM
                best_metric[model_name][j_dic] = best_metric_avg[model_name][j_dic]
            if verbose:
                logger.info(f'Model: {model_name}. Dichotomy: {j_dic+1}')
                logger.info(f'Best configuration ({model_selection}): {best_model[model_name][j_dic][0]}')  # cv_config
                logger.info(f'Best metric ({model_selection}): {best_model[model_name][j_dic][1]:.5f}')  # metric
                logger.info(f'Best CM ({model_selection}): {best_model[model_name][j_dic][2]}')  # CM
                logger.info('')
    
        # Prepare data to save
        
        result_data = {
            "dataset": dataset_name,
            'nclass': nclass,
            "class_labels": class_labels.tolist(),
            "ECOC_enc": ECOC_enc,
            "num_dichotomies": num_dichotomies,
            'flag_swap': flag_swap,
            "model_name": model_name,
            'n_simus': n_simus,
            'num_folds': num_folds,
            "Test_size": Test_size,
            'f_sel': metric_function_name,
            
            "best_config": best_model[model_name],
            "binary_metrics": {
                "best_metric_peak": best_metric_peak[model_name],
                "best_metric_avg": best_metric_avg[model_name],
                "best_metric": best_metric[model_name],
            },
        }
    
        # Generate filename based on dataset and model
        file_path = os.path.join(output_path, filename_o[model_name])
    
        # Save the combined configuration and metrics using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(result_data, f)
    
        logger.info(f"Binary Metrics and configuration saved to {filename_o}")
        logger.info('')
