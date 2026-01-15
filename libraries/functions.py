#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:51:23 2025

@author: fran
"""
import numpy as np
import yaml
import logging
from itertools import product
import importlib

import random

from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def compute_imbalance_ratio(targets):
    """
    Computes the imbalance ratio (majority class count / minority class count)
    for binary targets, handling cases with values 0 and 1, or False and True.

    Args:
        targets (np.ndarray): Array of binary target values. Can contain:
            - -1 and 1
            - 0 and 1
            - False and True

    Returns:
        float: The imbalance ratio. Returns 1 if classes are balanced or if
               only one class is present. Returns np.inf if only the minority
               class is present.
    """
    if targets.dtype == bool:
        negative_label = False
        positive_label = True
    else:
        negative_label = 0
        positive_label = 1
        if -1 in targets:
            negative_label = -1
            positive_label = 1

    n_negative = np.sum(targets == negative_label)
    n_positive = np.sum(targets == positive_label)

    if n_positive == 0 and n_negative == 0:
        return 1.0  # No data
    elif n_positive == 0:
        return np.inf  # Only negative class (majority)
    elif n_negative == 0:
        return 1000 # Only positive class (minority)

    if n_negative >= n_positive:
        imbalance_ratio = n_negative / n_positive
    else:
        imbalance_ratio = n_positive / n_negative

    return imbalance_ratio

def get_class_from_string(class_path):
    """Converts a string class path to a class object."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_config(filepath="config.yaml"):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(config):
    """Sets up the logger based on configuration, avoiding duplicate handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config["logging"]["level"])
    formatter = logging.Formatter(config["logging"]["format"], datefmt=config["logging"]["datefmt"])

    # Check for existing handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Matplotlib Logger
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    return logger

def generate_model_configurations(model_list):
    """Generates full model configurations from the model list, including LSEnsemble optimization logic."""
    CV_config = {}

    for model_item in model_list:
        model_name = model_item["name"]
        param_grid = model_item["params"]
        CV_config[model_name] = []

        # LSEnsemble optimization logic
        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
            
            if not SW_optimization:
                model_item["dynamic_params"]["LS_alpha"] = [0]
                model_item["dynamic_params"]["LS_beta"] = [0]
            else:
                model_item['params']['SW_mode'] = "normal"
                Alpha = model_item['dynamic_params']['LS_alpha']
                Beta = model_item['dynamic_params']['LS_alpha']
                Alpha_intervals = None  # Initialize Alpha_intervals
                Beta_intervals = None  # Initialize Beta_intervals
                if isinstance(Alpha, list) and len(Alpha) > 0:
                    if Alpha[0] == "auto":
                        if len(Alpha) > 1 and isinstance(Alpha[1], int) and Alpha[1] > 0:
                            model_item['params']['SW_mode'] = "auto"
                            Alpha_intervals = Alpha[1]
                            model_item['dynamic_params']['LS_alpha']= list(range(1, Alpha_intervals + 1))
                        else:
                            print("Error in ALSE configuration: When Alpha[0] is 'auto', the next element must be a positive integer.")
                            
            if not QC_optimization:
                model_item["dynamic_params"]["LS_Q_C"] = [1]
                
            if not RI_C_optimization:
                model_item["dynamic_params"]["LS_Q_RB_C"] = [1]
            else:
                model_item['params']['Q_RB_C_mode'] = "normal"
                Q_RB_C = model_item['dynamic_params']['LS_Q_RB_C']
                Q_RB_C_intervals = None  # Initialize Q_RB_C_intervals
                
                if isinstance(Q_RB_C, list) and len(Q_RB_C) > 0:
                    if Q_RB_C[0] == "auto":
                        if len(Q_RB_C) > 1 and isinstance(Q_RB_C[1], int) and Q_RB_C[1] > 0:
                            model_item['params']['Q_RB_C_mode'] = "auto"
                            Q_RB_C_intervals = Q_RB_C[1]
                            model_item['dynamic_params']['LS_Q_RB_C']= list(range(1, Q_RB_C_intervals + 1))
                        else:
                            print("Error in ALSE configuration: When Q_RB_C[0] is 'auto', the next element must be a positive integer.")
                    elif Q_RB_C[0] == "full":
                        model_item['params']['Q_RB_C_mode'] = "full"
                        Q_RB_C_intervals = 1
                    else:
                        Q_RB_C_intervals = len(Q_RB_C)
            if not RI_P_optimization:
                model_item["dynamic_params"]["LS_Q_RB_S"] = [1]
            else:
                model_item['params']['Q_RB_S_mode'] = "normal"
                Q_RB_S = model_item['dynamic_params']['LS_Q_RB_S']
                Q_RB_S_intervals = None  # Initialize Q_RB_S_intervals
                
                if isinstance(Q_RB_S, list) and len(Q_RB_S) > 0:
                    if Q_RB_S[0] == "auto":
                        if len(Q_RB_S) > 1 and isinstance(Q_RB_S[1], int) and Q_RB_S[1] > 0:
                            model_item['params']['Q_RB_S_mode'] = "auto"
                            Q_RB_S_intervals = Q_RB_S[1]
                            model_item['dynamic_params']['LS_Q_RB_S']= list(range(1, Q_RB_S_intervals + 1))
                        else:
                            print("Error in ALSE configuration: When Q_RB_S[0] is 'auto', the next element must be a positive integer.")
                    elif Q_RB_S[0] == "full":
                        model_item['params']['Q_RB_S_mode'] = "full"
                        Q_RB_S_intervals = 1
                    else:
                        Q_RB_S_intervals = len(Q_RB_S)

        dynamic_params = model_item["dynamic_params"]
        keys = list(dynamic_params.keys())
        values = list(dynamic_params.values())

        for combination in product(*values):
            updated_config = param_grid.copy()
            param_dict = dict(zip(keys, combination))

            if model_name == "MLPBayesBinW":
                updated_config.update({
                    "layers_size": (param_dict['s_NnBase'],),
                    "drop_out": [param_dict['s_pDOent'], param_dict['s_pDOocu']],
                    "activations": [param_dict['s_tActBase'], param_dict['s_tActSalida']],
                    "n_epoch": param_dict['s_nEpoch'],
                    "n_batch": param_dict['s_nBatch'],
                })
            elif model_name == "LGBMClassifier":
                updated_config.update({
                    "num_leaves": param_dict['LGBM_num_leaves'],
                    "learning_rate": param_dict['LGBM_learning_rate'],
                    "n_estimators": param_dict['LGBM_n_estimators'],
                })
            elif model_name == "LSEnsemble":
                updated_config.update({
                    "alpha": param_dict['LS_alpha'],
                    "beta": param_dict['LS_beta'],
                    "QC": param_dict['LS_Q_C'],
                    "Q_RB_S": param_dict['LS_Q_RB_S'],
                    "Q_RB_C": param_dict['LS_Q_RB_C'],
                    "num_experts": param_dict['LS_num_experts'],
                    "hidden_size": param_dict['LS_hidden_size'],
                    "drop_out": param_dict['LS_drop_out'],
                    "n_batch": param_dict['LS_n_batch'],
                    "n_epoch": param_dict['LS_n_epoch'],
                    "mode": param_dict['LS_mode'],
                })
            elif model_name == "LogisticRegression":
                updated_config.update({
                    "C": param_dict['LR_C'],
                    "penalty": param_dict['LR_penalty'],
                })
            elif model_name == "RandomForestClassifier":
                updated_config.update({
                    "n_estimators": param_dict['RF_n_estimators'],
                    "max_depth": param_dict['RF_max_depth'],
                    "min_samples_split": param_dict['RF_min_samples_split'],
                    "min_samples_leaf": param_dict['RF_min_samples_leaf'],
                })
            elif model_name == "MLPClassifier":
                updated_config.update({
                    "hidden_layer_sizes": param_dict['MLP_hidden_layer_sizes'],
                    "activation": param_dict['MLP_activation'],
                    "solver": param_dict['MLP_solver'],
                    "alpha": param_dict['MLP_alpha'],
                })
            elif model_name == "kNN":
                updated_config.update({
                    "n_neighbors": param_dict['kNN_n_neighbors'],
                    "metric": param_dict['kNN_metric'],
                })
            elif model_name == "C4.5":
                updated_config.update({
                    "max_depth": param_dict['C45_max_depth'],
                    "min_samples_split": param_dict['C45_min_samples_split'],
                    "min_samples_leaf": param_dict['C45_min_samples_leaf'],
                })
            elif model_name == "SVM":
                updated_config.update({
                    "C": param_dict['SVM_C'],
                    "gamma": param_dict['SVM_gamma'],
                })
            elif model_name == "MultiRandBal":
                updated_config.update({
                    "n_estimators": param_dict['RB_n_estimators'],
                    "base_estimator": param_dict['RB_base_estimator'],
                })
                
            CV_config[model_name].append(updated_config)

    return CV_config

def initialize_model_data(model_list, dynamic_combinations, num_dichotomies, num_folds, n_simus, logger):
    """Initializes data structures for models."""
    CM_accumulated = {}
    metric_conf = {}
    best_metric_conf = {}
    best_model_conf = {}
    best_metric_overall = {}
    best_model_overall = {}

    for model_item in model_list:
        model_name = model_item["name"]
        dynamic_combinations = model_item["dynamic_params"]
        num_configurations = len(dynamic_combinations)
        CM_accumulated[model_name] = [np.zeros((num_configurations, 2, 2)) for _ in range(num_dichotomies)]
        metric_conf[model_name] = np.zeros((num_configurations, num_dichotomies, num_folds, n_simus))
        best_metric_conf[model_name] = np.zeros(num_dichotomies)
        best_model_conf[model_name] = [dict() for _ in range(num_dichotomies)]
        best_metric_overall[model_name] = np.zeros(num_dichotomies)
        best_model_overall[model_name] = [dict() for _ in range(num_dichotomies)]

    return CM_accumulated, metric_conf, best_metric_conf, best_model_conf, best_metric_overall, best_model_overall


def apply_ecoc_binarization(M, y_train, y_test, apply_flag_swap=True, flag_swap=None, eps=1e-5, verbose=False):
    """
    Converts multiclass labels into multiple binary labels using the ECOC matrix (-1,1 format).

    Parameters:
    - M: ECOC matrix (shape: [n_classes, n_dichotomies]), defining class partitions.
    - y_train: Multiclass training labels (NumPy array) with values in [1, n_classes].
    - y_test: Multiclass test labels (NumPy array) with values in [1, n_classes].
    - apply_flag_swap: If True, applies label swapping for imbalanced dichotomies.
    - flag_swap: Optional NumPy array indicating swap status per dichotomy (computed if None).
    - eps: Threshold to identify highly imbalanced classes.
    - verbose: If True, prints additional processing details.

    Returns:
    - Y_train_ecoc: List of NumPy arrays, each containing binary training labels per dichotomy.
    - Y_test_ecoc: List of NumPy arrays, each containing binary test labels per dichotomy.
    - flag_swap: NumPy array indicating whether swapping was applied per dichotomy.
    - idx_train_ecoc: List of NumPy arrays containing training indices per dichotomy.
    - idx_test_ecoc: List of NumPy arrays containing test indices per dichotomy.
    """
    num_dichotomies = M.shape[1]
    Y_train_ecoc = [[] for _ in range(num_dichotomies)]
    Y_test_ecoc = [[] for _ in range(num_dichotomies)]
    idx_train_ecoc = [[] for _ in range(num_dichotomies)]
    idx_test_ecoc = [[] for _ in range(num_dichotomies)]
    QP_tr = np.zeros(num_dichotomies)

    if flag_swap is None:
        flag_swap = np.zeros(num_dichotomies)

    for j_dic in range(num_dichotomies):
        dicotomia = M[:, j_dic]

        # Prepare train labels and indices for the dichotomy
        y_train_ecoc = []
        idx_train_ecoc[j_dic] = []
        for i, clase in enumerate(y_train):
            if dicotomia[int(clase) - 1] != 0:  # Check for valid dichotomy label
                y_train_ecoc.append(dicotomia[int(clase) - 1])
                idx_train_ecoc[j_dic].append(i)


        y_train_ecoc = np.array(y_train_ecoc)
        N0_tr = np.sum(y_train_ecoc == -1)
        N1_tr = np.sum(y_train_ecoc == 1)

        # Apply label swapping if necessary
        if apply_flag_swap and flag_swap[j_dic] == 0 and N0_tr < 0.95 * N1_tr:
            y_train_ecoc *= -1
            flag_swap[j_dic] = 1
            N0_tr, N1_tr = N1_tr, N0_tr
        elif apply_flag_swap and flag_swap[j_dic] == 1:
            y_train_ecoc *= -1
            N0_tr, N1_tr = N1_tr, N0_tr

        P0_tr = N0_tr / (N0_tr + N1_tr)
        P1_tr = N1_tr / (N0_tr + N1_tr)
        QP_tr[j_dic] = 1000 if P1_tr < eps else P0_tr / P1_tr

        # Prepare test labels and indices for the dichotomy
        y_test_ecoc = []
        idx_test_ecoc[j_dic] = []
        for i, clase in enumerate(y_test):
            if dicotomia[int(clase) - 1] != 0:  # Check for valid dichotomy label
                y_test_ecoc.append(dicotomia[int(clase) - 1])
                idx_test_ecoc[j_dic].append(i)

        y_test_ecoc = np.array(y_test_ecoc)
        if flag_swap[j_dic]:
            y_test_ecoc *= -1

        # Store the results for the current dichotomy
        Y_train_ecoc[j_dic] = y_train_ecoc
        Y_test_ecoc[j_dic] = y_test_ecoc

        if verbose:
            print(f'Dichotomy {j_dic + 1}: {dicotomia}, N0_tr: {N0_tr}, N1_tr: {N1_tr}, IR = {QP_tr[j_dic]:.2f}')

    return Y_train_ecoc, Y_test_ecoc, flag_swap, idx_train_ecoc, idx_test_ecoc


def calc_MAE_AMAE_CM(CM):
    nClases = CM.shape[0]
    AMAEu = np.zeros(nClases)
    MAEu = np.zeros(nClases)
    for kclase in range(nClases):
        for kotra in range(nClases):
            MAEu[kclase] += CM[kclase,kotra]*np.abs(kclase-kotra)#/y_test.shape[0]
            AMAEu[kclase] += CM[kclase,kotra]*np.abs(kclase-kotra)/np.sum(CM, axis=1)[kclase]

    MAE = np.sum(MAEu)/np.sum(CM)
    AMAE = np.mean(AMAEu)
    
    return MAE, AMAE

def calc_MAE_AMAE(y,ye):
    #labels, ocurrences = np.unique(y,return_counts=True)
    labels = np.unique(y)
    
    MAE = np.mean(np.abs(y-ye))
    MAEu = np.zeros((labels.shape[0]))
    for kclase in range(labels.shape[0]):
        vc = np.nonzero(y==labels[kclase])
        MAEu[kclase] = np.mean(np.abs(y[vc]-ye[vc]))
        
    AMAE = np.mean(MAEu)    
    
    return MAE, AMAE

def compute_metrics(y,ye,metrics):
    
    metricas=np.zeros(len(metrics))
    for k in range(len(metrics)):
        name_metric = metrics[k]
        if name_metric.lower() == 'mae':
            #MAE, AMAE = calc_MAE_AMAE_CM(confusion_matrix(y, ye))
            MAE, AMAE = calc_MAE_AMAE(y, ye)
            metricas[k] = MAE
            
        elif name_metric.lower() == 'amae':
            #MAE, AMAE = calc_MAE_AMAE(confusion_matrix(y, ye))
            MAE, AMAE = calc_MAE_AMAE(y, ye)
            metricas[k] = AMAE
            
        elif name_metric.lower() == 'accuracy':                
            metricas[k] = accuracy_score(y, ye)
        
        elif name_metric.lower() == 'balanced_accuracy':                
            metricas[k] = balanced_accuracy_score(y, ye)
            
        elif name_metric.lower() == 'f1':                
            metricas[k] = f1_score(y, ye, average='macro')
            
        elif name_metric.lower() == 'cohen_kappa':                
            metricas[k] = cohen_kappa_score(y, ye) 
            
        elif name_metric.lower() == 'cohen_kappa_linear':                
            metricas[k] = cohen_kappa_score(y, ye, weights='linear') 
            
        elif name_metric.lower() == 'cohen_kappa_quadratic':                
            metricas[k] = cohen_kappa_score(y, ye, weights='quadratic')
            
        elif name_metric.lower() == 'matthews':                
            metricas[k] = matthews_corrcoef(y, ye) 
            
    return metricas

def generate_batches(nBatch, param, mode='random', seed=42):
    if seed is not None:
        np.random.seed(seed)

    if mode == 'random':
        # param: number of samples in the training set
        if nBatch == param:
            list_samples_batch = []
            list_samples_batch.append(list(range(param)))
            num_batches = 1

        else:

            indices = list(range(param))
            random.shuffle(indices)

            l = len(indices)
            #for ndx in range(0, l, nBatch):
            #    yield indices[ndx:min(ndx + nBatch, l)]

            num_batches = int(np.ceil(param/nBatch))
            list_samples_batch = []
            for ndx in range(0, l, nBatch):
                list_aux = indices[ndx:min(ndx + nBatch, l)]
                list_aux.sort()
                list_samples_batch.append(list_aux)


    elif mode == 'class_equitative':
        # All classes have the same number of samples in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]

            batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(samples_max/batch_samples_class).astype(int)

            list_samples_class = []
            for k in range(num_classes):
                mod = int((batch_samples_class*num_batches) // samples_class[k])
                rem = int((batch_samples_class*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class*kbatch:batch_samples_class*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)

    elif mode == 'representative':
        # All classes have at least 1 sample in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]
            num_samples = param.shape[0]

            #batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(num_samples/nBatch).astype(int)
            batch_samples_class = []
            list_samples_class = []
            for k in range(num_classes):
                batch_samples_class.append(np.ceil(nBatch*samples_class[k]/num_samples).astype(int))

                mod = int((batch_samples_class[k]*num_batches) // samples_class[k])
                rem = int((batch_samples_class[k]*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class[kclass]*kbatch:batch_samples_class[kclass]*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)


    for kbatch in range(num_batches):
        yield list_samples_batch[kbatch]


def estimate_alpha(ir: float, cap: bool = True) -> float:
    """
    Estimates a reasonable starting value for Alpha based on Imbalance Ratio (IR).
    
    The function uses a simple logarithmic relationship fitted to your data.
    For very high IR, it caps the estimate to avoid unrealistic values.
    
    Args:
        ir (float): Real Imbalance Ratio of the dichotomy (>=1.0)
        cap (bool): Whether to cap the maximum Alpha at 0.45 (default: True)
    
    Returns:
        float: Estimated Alpha value (between 0.0 and 0.45)
    """
    if ir <= 1.0:
        return 0.08  # Minimum reasonable value for nearly balanced cases
    
    alpha_est = 0.12 * np.log10(ir) + 0.08
    
    if cap:
        alpha_est = min(alpha_est, 0.45)
    
    return round(alpha_est, 3)