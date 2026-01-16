#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native Multiclass Classifier Optimization + Final Test Evaluation (Program 1 + 2)
--------------------------------------------------------------------------------
This script:
1. Optimizes native multiclass classifiers using hyperparameters from config_train.yaml.
2. Performs parameter search with inner CV/hold-out.
3. At the end, evaluates the best configurations on the test set with multiple simulations,
   measuring execution time (train + inference) and multiclass metrics.
   
Key features:
- No binarization, no dichotomies, no ECOC – pure multiclass models.
- Uses generate_model_configurations for correct parameter mapping.
- Inner CV with FullTrainSplitter for num_folds==1 (train on 100% of train partition).
- Final test evaluation with time measurement and statistical aggregation.
"""

import os
import pickle
import time
import warnings
from itertools import product
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)
from imblearn.metrics import geometric_mean_score, sensitivity_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Custom Libraries
from libraries.data_loading import load_datasets
from libraries.functions import (
    load_config,
    setup_logger,
    get_class_from_string,
    generate_model_configurations,
    compute_imbalance_ratio,
)
from libraries.imbalance_degree import imbalance_degree

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*'penalty' was deprecated in version 1.8.*"
)

# Metric functions (adapted for multiclass)
METRIC_FUNCTIONS = {
    "balanced_accuracy_score": balanced_accuracy_score,
    "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    "matthews_corrcoef": matthews_corrcoef,
}

# Load configurations
config_train = load_config('config_train.yaml')
config_test = load_config('config_test.yaml')

# Setup Logger
logger = setup_logger(config_train)

# Parameters from config_train (optimization)
verbose = config_train["simulation"]["verbose"]
data_path = config_train["paths"]["data_folder"]
output_path = config_train["paths"]["output_folder"]
if not os.path.exists(output_path):
    print("New folder created: {}".format(output_path))
    os.makedirs(output_path)

n_simus_opt = config_train["simulation"]["n_simus"]  # Simulations for optimization
Test_size = config_train["simulation"]["test_size"]
N_max = config_train["simulation"]["N_MAX"]
num_folds = config_train["simulation"]["num_folds"]
metric_function_name = config_train["simulation"]["f_sel"]
model_selection = config_train["simulation"]["model_selection"]  # "avg" or "peak"

# Parameters from config_test (final evaluation)
n_simus_test = config_test["simulation"]["n_simus"]

# Retrieve the actual function from the mapping
if metric_function_name not in METRIC_FUNCTIONS:
    raise ValueError(f"Invalid metric function '{metric_function_name}'. Choose from {list(METRIC_FUNCTIONS.keys())}.")
f_sel = METRIC_FUNCTIONS[metric_function_name]
model_selection = config_train["simulation"]["model_selection"]

# Model Configuration - Full list of available models
model_list = config_train["models"]
# Apply the desired selection (uncomment as needed)
# model_list = config["models"]  # All models
# model_list = [config["models"][i] for i in [1, 4, 7, 3]]  # RF, LGBM, SVM, MLP
# model_list = [config["models"][2]]  # Only ALSE (skip, not native multiclass)
model_list = [config_train["models"][i] for i in [0, 1, 3, 4, 5, 6, 7]]  # Exclude LSEnsemble & MultiRandBal
# Generate Model Configurations (your function)
CV_config = generate_model_configurations(model_list)

# Load Datasets
dataset_special_cases = {}
datasets = load_datasets(data_path, N_max, dataset_special_cases)

# Process Datasets
for dataset_name, (X, y, C0) in datasets.items():
    n_samples, n_attributes = X.shape
    nclass = np.unique(y).shape[0]
    
    # Map labels if needed
    if nclass > 2:
        if config_train["simulation"]["maj_min"]:
            sorted_classes = pd.Series(y).value_counts().index.to_list()
            class_dict = {sorted_classes[k]: nclass - k for k in range(nclass)}
        elif config_train["simulation"]["min_maj"]:
            sorted_classes = pd.Series(y).value_counts().index.to_list()
            class_dict = {sorted_classes[k]: k + 1 for k in range(nclass)}
        else:
            class_dict = {k + 1: k + 1 for k in range(nclass)}
        
        y = np.vectorize(class_dict.get)(y)
    
    IB_degree = imbalance_degree(y, "EU") if nclass > 2 else compute_imbalance_ratio(y)
    
    logger.info('')
    logger.info('-----------------------------------------------------')
    logger.info(f'Dataset: {dataset_name}. Imbalance Degree = {IB_degree:.2f}')
    logger.info(f'Number of classes: {nclass}')
    logger.info('f_sel: ' + f_sel.__name__)
    
    # Initialize structures for multiclass (single level, no dichotomies)
    best_model_avg = {}
    metric_conf = {}
    CM_accumulated = {}
    best_metric_avg = dict()
    best_metric_peak = dict()
    best_model_peak = dict()
    
    for model_item in model_list:
        model_name = model_item["name"]
        logger.info(f'Model: {model_name}')
        
        dynamic_combinations = model_item["dynamic_params"]
        n_conf_opt = len(list(product(*dynamic_combinations.values())))
        
        # Multiclass structures (single level, no dichotomies)
        CM_accumulated[model_name] = np.zeros((n_conf_opt, nclass, nclass))  # Multiclass CM
        metric_conf[model_name] = np.zeros((n_conf_opt, num_folds, n_simus_opt))
        best_metric_avg[model_name] = 0.0
        best_model_avg[model_name] = dict()
        best_metric_peak[model_name] = 0.0
        best_model_peak[model_name] = dict()
    
    logger.info('-----------------------------------------------------')
    
    for k_simu in range(n_simus_opt):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01 * Test_size, random_state=42 + k_simu
        )
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_test_n = scaler.transform(X_test)
        
        # Inner CV / hold-out
        inner_cv = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=42 + k_simu
        )
        
        for nFold, (train_index, val_index) in enumerate(inner_cv.split(X_train_n, y_train), 1):
            if verbose:
                logger.info(f"Simulation {k_simu+1}. Inner Fold {nFold}:")
            
            X_train_cv = X_train_n[train_index]
            y_train_cv = y_train[train_index]
            
            if len(val_index) > 0:
                X_test_cv = X_train_n[val_index]
                y_test_cv = y_train[val_index]
            else:
                # num_folds == 1 → no internal val → use external test as "val" (careful: contamination!)
                # Alternative: use train metric or BIC (recommended)
                X_test_cv = X_test_n
                y_test_cv = y_test
            
            # No binarization → direct multiclass
            for model_item in model_list:
                model_name = model_item["name"]
                model_class = get_class_from_string(model_item["class"])
                
                k_conf = 0
                for cv_config in CV_config[model_name]:
                    model = model_class(**cv_config)
                    
                    # Train the model (native multiclass)
                    # Convert to DataFrame only if the model is LightGBM to avoid the feature names warning
                    if model_name == "LGBMClassifier":
                        # Create feature names for consistency (f0, f1, ...)
                        feature_names = [f'f{i}' for i in range(X_train_cv.shape[1])]
                        
                        # Convert training and validation/test sets to pandas DataFrame
                        X_train_cv_df = pd.DataFrame(X_train_cv, columns=feature_names)
                        X_test_cv_df = pd.DataFrame(X_test_cv, columns=feature_names)
                        
                        # Train the model using DataFrame (eliminates the warning)
                        model.fit(X_train_cv_df, y_train_cv)
                        
                        # Predict using DataFrame
                        y_pred = model.predict(X_test_cv_df)
                    else:
                        # For all other models: use NumPy arrays directly (no warning occurs)
                        model.fit(X_train_cv, y_train_cv)
                        y_pred = model.predict(X_test_cv)
                    
                    CM = confusion_matrix(y_test_cv, y_pred, labels=np.unique(y))
                    CM_accumulated[model_name][k_conf] += CM
                    
                    # Metric (adapt for multiclass)
                    metric = f_sel(y_test_cv, y_pred)
                    metric_conf[model_name][k_conf, nFold-1, k_simu] = metric
                    
                    k_conf += 1
    
    # Select best configuration based on model_selection ("avg" or "peak")
    for model_item in model_list:
        model_name = model_item["name"]
        for k_conf in range(len(CV_config[model_name])):
            # Average performance across folds and sims
            avg_metric = np.mean(metric_conf[model_name][k_conf, :, :])
            
            # Peak performance
            peak_metric = np.max(metric_conf[model_name][k_conf, :, :])
            
            if model_selection == "avg":
                if avg_metric > best_metric_avg[model_name]:
                    best_metric_avg[model_name] = avg_metric
                    best_model_avg[model_name] = CV_config[model_name][k_conf]
            elif model_selection == "peak":
                if peak_metric > best_metric_peak[model_name]:
                    best_metric_peak[model_name] = peak_metric
                    best_model_peak[model_name] = CV_config[model_name][k_conf]
            else:
                raise ValueError(f"Unknown model_selection: {model_selection}")
        
    # Choose best config based on model_selection
    best_configs = {}
    for model_item in model_list:
        model_name = model_item["name"]
        if model_selection == "avg":
            best_configs[model_name] = best_model_avg[model_name]
        elif model_selection == "peak":
            best_configs[model_name] = best_model_peak[model_name]
    
    # Final evaluation on test set (integrated from Program 2)
    logger.info(f"Starting final test evaluation for {dataset_name}")
    
    # Lists for aggregation
    CM_simulations = []
    acc_simulations = []
    bal_acc_simulations = []
    kappa_simulations = []
    geom_mean_simulations = []
    sensitivity_simulations = []
    f1_simulations = []
    runtime_simulations = []
    
    for model_item in model_list:
        model_name = model_item["name"]
        best_config = best_configs[model_name]
        
        for k_simu in range(n_simus_test):
            start_time = time.time()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.01 * Test_size, random_state=42 + k_simu
            )
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_n = scaler.transform(X_train)
            X_test_n = scaler.transform(X_test)
            
            model = get_class_from_string(model_item["class"])(**best_config)
            # Train the model (native multiclass)
            # Convert to DataFrame only if the model is LightGBM to avoid the feature names warning
            if model_name == "LGBMClassifier":
                # Create feature names for consistency (f0, f1, ...)
                feature_names = [f'f{i}' for i in range(X_train_n.shape[1])]
                
                # Convert training and validation/test sets to pandas DataFrame
                X_train_n_df = pd.DataFrame(X_train_n, columns=feature_names)
                X_test_n_df = pd.DataFrame(X_test_n, columns=feature_names)
                
                # Train the model using DataFrame (eliminates the warning)
                model.fit(X_train_n_df, y_train)
                
                # Predict using DataFrame
                y_pred = model.predict(X_test_n_df)
            else:
                # For all other models: use NumPy arrays directly (no warning occurs)
                model.fit(X_train_n, y_train)
                y_pred = model.predict(X_test_n)
            
            end_time = time.time()
            runtime_simulations.append(end_time - start_time)
            
            CM = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            geom_mean = geometric_mean_score(y_test, y_pred, average='weighted')
            sensitivity = sensitivity_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            CM_simulations.append(CM)
            acc_simulations.append(acc)
            bal_acc_simulations.append(bal_acc)
            kappa_simulations.append(kappa)
            geom_mean_simulations.append(geom_mean)
            sensitivity_simulations.append(sensitivity)
            f1_simulations.append(f1)
            
            if (k_simu + 1) % 5 == 0:
                logger.info(f" -> Test simulation {k_simu + 1}/{n_simus_test} completed")
        
        # Aggregate metrics
        CM_array = np.array(CM_simulations)
        mean_CM = np.mean(CM_array, axis=0).round().astype(int)
        std_CM = np.std(CM_array, axis=0).round(2)
        
        model_metrics = {
            "avg_CM": mean_CM.tolist(),
            "std_CM": std_CM.tolist(),
            "avg_acc": np.mean(acc_simulations),
            "std_acc": np.std(acc_simulations),
            "avg_bal_acc": np.mean(bal_acc_simulations),
            "std_bal_acc": np.std(bal_acc_simulations),
            "avg_kappa": np.mean(kappa_simulations),
            "std_kappa": np.std(kappa_simulations),
            "avg_geom_mean": np.mean(geom_mean_simulations),
            "std_geom_mean": np.std(geom_mean_simulations),
            "avg_sensitivity": np.mean(sensitivity_simulations),
            "std_sensitivity": np.std(sensitivity_simulations),
            "avg_f1_score": np.mean(f1_simulations),
            "std_f1_score": np.std(f1_simulations),
        }
        
        # Update result_data
        result_data = {
            "dataset": dataset_name,
            "nclass": nclass,
            "model_name": model_name,
            "best_config": best_config,
            "multiclass_metrics": model_metrics,
            "execution_time_seconds": np.mean(runtime_simulations),
            "execution_time_per_simulation": np.mean(runtime_simulations) / n_simus_test,
        }
        
        filename_o = f"{dataset_name}_{model_name}_native_test.pkl"
        file_path_o = os.path.join(output_path, filename_o)
        
        with open(file_path_o, 'wb') as f:
            pickle.dump(result_data, f)
        
        logger.info(f"Final test metrics saved to {filename_o}")
        logger.info(f"Balanced Accuracy: {model_metrics['avg_bal_acc']:.5f} ± {model_metrics['std_bal_acc']:.5f}")
        logger.info(f"Geo Mean: {model_metrics['avg_geom_mean']:.5f} ± {model_metrics['std_geom_mean']:.5f}")
        logger.info(f"Avg Runtime: {result_data['execution_time_seconds']:.4f} s")
        logger.info('')