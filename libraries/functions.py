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
import copy
import importlib
import random

from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def compute_imbalance_ratio(targets):
    """
    Computes the imbalance ratio (Majority Count / Minority Count).
    Supports asymmetric targets: -(1-2*alpha) and (1-2*beta).
    
    Returns:
        float: (n_majority / n_minority). 
               Returns 1000.0 if only the majority class is present.
               Returns 1.0 if perfectly balanced or empty.
    """
    unique_values, counts = np.unique(targets, return_counts=True)
    n_unique = len(unique_values)

    inf = np.inf
    if n_unique == 0:
        return 1.0
    
    if n_unique == 1:
        # If only one class exists, we assume it's the majority 
        # because the minority is, by definition, rare or missing.
        return inf
    elif n_unique == 2:
        # Sort classes by count: counts[0] is the smaller count if we sort
        sorted_indices = np.argsort(counts)
        n_minority = counts[sorted_indices[0]]
        n_majority = counts[sorted_indices[1]]
    
        # Map which value belongs to which class for internal logic if needed
        # minority_val = unique_values[sorted_indices[0]]
        # majority_val = unique_values[sorted_indices[1]]
    
        if n_minority == 0:
            return inf
        
        imbalance_ratio = n_majority / n_minority
        return float(imbalance_ratio)
    else:
        return None

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

def _get_resolver_logger(logger=None):
    """
    Return the logger used by configuration resolver utilities.

    Parameters
    ----------
    logger : logging.Logger or None, optional
        External logger. If None, the module logger is used.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logger if logger is not None else logging.getLogger(__name__)


def _expand_lse_rebalance_values(values, param_name, qp_tr):
    """
    Expand LSE rebalance YAML specifications into explicit numeric search values.

    Supported conventions
    ---------------------
    Explicit list:
        [1, 2, 4, 8]
    Auto range:
        ['auto', n] -> np.linspace(1, qp_tr, n)
    Full imbalance:
        ['full'] -> [qp_tr]

    Parameters
    ----------
    values : list
        Raw YAML value list for one rebalance hyperparameter.
    param_name : str
        Parameter name used in error messages.
    qp_tr : float
        Imbalance ratio of the current training split.

    Returns
    -------
    tuple
        Tuple with:
        - resolved list of values
        - mode string in {'normal', 'auto', 'full'}
    """
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(
            f"Invalid {param_name} configuration. Expected a non-empty list."
        )

    if values[0] == "auto":
        if len(values) > 1 and isinstance(values[1], int) and values[1] > 0:
            n_items = values[1]
            return np.linspace(1, qp_tr, n_items).tolist(), "auto"

        raise ValueError(
            f"Invalid {param_name} configuration. "
            "When using ['auto', n], n must be a positive integer."
        )

    if values[0] == "full":
        return [qp_tr], "full"

    return values, "normal"


def _expand_lse_alpha_values(values, qp_tr):
    """
    Expand the special LS_alpha YAML specification into explicit search values.

    Supported conventions
    ---------------------
    Explicit list:
        [0, 0.05, 0.1, 0.15]
    Auto alpha grid:
        ['auto', n] -> local grid around estimate_alpha(qp_tr)

    The automatic alpha grid is built as:

        np.linspace(
            max(0.0, alpha_start - 0.2),
            min(0.45, alpha_start + 0.2),
            n
        )

    where alpha_start = estimate_alpha(qp_tr).

    Parameters
    ----------
    values : list
        Raw YAML value list for LS_alpha.
    qp_tr : float
        Imbalance ratio of the current training split.

    Returns
    -------
    tuple
        Tuple with:
        - resolved list of alpha values
        - mode string in {'normal', 'auto'}
    """
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(
            "Invalid LS_alpha configuration. Expected a non-empty list."
        )

    if values[0] == "auto":
        if len(values) > 1 and isinstance(values[1], int) and values[1] > 0:
            n_items = values[1]
            alpha_start = estimate_alpha(qp_tr)
            resolved = np.round(
                np.linspace(
                    max(0.0, alpha_start - 0.2),
                    min(0.45, alpha_start + 0.2),
                    n_items
                ),
                3
            ).tolist()
            return resolved, "auto"

        raise ValueError(
            "Invalid LS_alpha configuration. "
            "When using ['auto', n], n must be a positive integer."
        )

    if values[0] == "full":
        raise ValueError(
            "Invalid LS_alpha configuration. The 'full' mode is not defined for LS_alpha."
        )

    return values, "normal"

def _filter_lse_dynamic_params_by_base_learner(dynamic_params, base_learner, optim=None):
    """
    Filter the LSE dynamic search space according to the selected base learner
    and optimization method.

    The goal is to avoid generating redundant Cartesian-product combinations
    with hyperparameters that are ignored by the underlying expert type.

    This helper is mainly used for the legacy flat LSE YAML format. When the
    structured format with ``base_learner_configs`` is used, the resolver should
    keep only global LSE parameters and let ``LSEnsemble`` resolve the selected
    base-learner block internally.

    Parameters
    ----------
    dynamic_params : dict
        Resolved dynamic LSE parameter grid.
    base_learner : str
        Selected expert family.
    optim : str or None, optional
        Optimizer name from the fixed model parameters.

    Returns
    -------
    dict
        Filtered dynamic parameter dictionary.
    """
    always_keep = {
        "alpha",
        "beta",
        "Q_C",
        "Q_RB_S",
        "Q_RB_C",
        "num_experts",
    }

    shallow_or_deep_mlp_params = {
        "hidden_size",
        "drop_out",
        "n_batch",
        "n_epoch",
    }

    mlp_like = {
        "FAMLP",
        "AMLP",
        "DAMLP",
        "Parzen",
    }

    # ``mode`` controls minibatch generation. It is meaningful for minibatch
    # optimizers and for warm_start_lbfgs, because the warm-start phase uses
    # minibatches. Pure LBFGS ignores it.
    mode_applicable = False

    if base_learner in ("FAMLP", "Parzen"):
        mode_applicable = True
    elif base_learner in ("AMLP", "DAMLP", "LogReg") and optim != "lbfgs":
        mode_applicable = True

    if base_learner in mlp_like:
        allowed = always_keep | shallow_or_deep_mlp_params

    elif base_learner == "LogReg":
        allowed = always_keep | {
            "n_epoch",
            "n_batch",
        }

    elif base_learner == "CalibratedBooster":
        allowed = always_keep

    else:
        # Conservative fallback for legacy/custom neural learners.
        allowed = always_keep | shallow_or_deep_mlp_params

    if mode_applicable:
        allowed = allowed | {"mode"}

    filtered = {}
    for key, value in dynamic_params.items():
        if key in allowed:
            filtered[key] = value

    return filtered

def _prune_lse_params_by_base_learner(config, base_learner, optim=None):
    """
    Remove LSE constructor parameters that do not apply to the selected
    base learner family.

    This helper is mainly used for the legacy flat LSE YAML format. In the
    structured format with ``base_learner_configs``, base-learner-specific
    values remain inside the selected learner block and are resolved by
    ``LSEnsemble`` itself.

    Parameters
    ----------
    config : dict
        Constructor-ready parameter dictionary.
    base_learner : str
        Selected expert family.
    optim : str or None, optional
        Optimizer name when relevant.

    Returns
    -------
    dict
        Pruned configuration dictionary.
    """
    pruned = dict(config)

    if base_learner == "CalibratedBooster":
        for key in (
            "activation_fn",
            "optim",
            "loss_fn",
            "drop_out",
            "n_batch",
            "n_epoch",
            "mode",
            "hidden_size",
            "hidden_layer_sizes",
            "output_act",
            "learning_rate",
            "weight_decay",
            "debug",
            "early_stopping",
            "es_patience",
            "es_tol",
            "lbfgs_max_iter",
            "lbfgs_max_eval",
            "warm_start",
            "warm_start_optim",
            "warm_start_n_epoch",
            "warm_start_n_batch",
            "warm_start_mode",
            "warm_start_learning_rate",
            "warm_start_weight_decay",
            "warm_start_debug",
        ):
            pruned.pop(key, None)

    elif base_learner == "LogReg":
        for key in (
            "activation_fn",
            "loss_fn",
            "drop_out",
            "hidden_size",
            "hidden_layer_sizes",
            "learning_rate",
            "weight_decay",
            "lbfgs_max_iter",
            "lbfgs_max_eval",
            "warm_start",
            "warm_start_optim",
            "warm_start_n_epoch",
            "warm_start_n_batch",
            "warm_start_mode",
            "warm_start_learning_rate",
            "warm_start_weight_decay",
            "warm_start_debug",
        ):
            pruned.pop(key, None)

        if optim == "lbfgs":
            pruned.pop("mode", None)
            pruned.pop("n_batch", None)

    elif base_learner == "Parzen":
        for key in (
            "optim",
            "loss_fn",
            "learning_rate",
            "weight_decay",
            "debug",
            "early_stopping",
            "es_patience",
            "es_tol",
            "lbfgs_max_iter",
            "lbfgs_max_eval",
            "warm_start",
            "warm_start_optim",
            "warm_start_n_epoch",
            "warm_start_n_batch",
            "warm_start_mode",
            "warm_start_learning_rate",
            "warm_start_weight_decay",
            "warm_start_debug",
            "hidden_layer_sizes",
        ):
            pruned.pop(key, None)

    elif base_learner in ("AMLP", "DAMLP", "FAMLP"):
        # Pure LBFGS is full-batch, so minibatch-specific parameters are ignored.
        # For warm_start_lbfgs, mode and n_batch are kept because they apply to
        # the warm-start minibatch phase.
        if optim == "lbfgs":
            pruned.pop("mode", None)
            pruned.pop("n_batch", None)

    return pruned
def _resolve_lse_model_configurations(model_item, x_train=None, y_train_lab=None, logger=None):
    """
    Resolve one LSEnsemble YAML block into constructor-ready configurations.

    This function interprets the declarative LSE YAML syntax after the current
    training split is available, because some hyperparameter specifications
    depend on the imbalance ratio of y_train_lab.

    Two YAML styles are supported:

    1. Legacy flat LSE configuration:
       Base-learner-specific parameters such as hidden_size, n_batch, n_epoch,
       mode, optim, etc. are declared directly in params/dynamic_params.

    2. Structured base-learner configuration:
       Base-learner-specific parameters are declared under:

           params:
             base_learner_configs:
               AMLP:
                 hidden_size: ...
                 n_batch: ...
                 n_epoch: ...

       In this mode, this resolver only expands global LSE parameters
       such as alpha, beta, Q_C, Q_RB_S, Q_RB_C and num_experts. The active
       base-learner block is interpreted later by LSEnsemble itself.

    Parameters
    ----------
    model_item : dict
        One model block from the YAML configuration.
    x_train : np.ndarray or None, optional
        Training matrix. If provided, its number of columns is used as
        ``input_size``.
    y_train_lab : np.ndarray or None, optional
        Training labels. If provided, the imbalance ratio is computed from
        this array to resolve imbalance-dependent search modes.
    logger : logging.Logger or None, optional
        Logger instance.

    Returns
    -------
    list
        List of constructor-ready dictionaries for LSEnsemble.
    """
    logger = _get_resolver_logger(logger)

    item = copy.deepcopy(model_item)

    params = dict(item.get("params", {}))
    dynamic_params = dict(item.get("dynamic_params", {}))
    lse_optimization = dict(item.get("LSE_optimization", {}))
    base_learner_params = dict(item.get("base_learner_params", {}))

    base_learner = params.get("base_learner", "FAMLP")
    optim = params.get("optim", None)

    # New hierarchical YAML format. When this is present, base-learner-specific
    # parameters must not be injected as top-level LSEnsemble constructor
    # arguments. They are resolved inside LSEnsemble from the selected block.
    uses_structured_base_configs = "base_learner_configs" in params

    if x_train is not None:
        params["input_size"] = x_train.shape[1]

    qp_tr = 1.0
    if y_train_lab is not None:
        qp_value = compute_imbalance_ratio(y_train_lab)
        if qp_value is not None and np.isfinite(qp_value):
            qp_tr = float(qp_value)

    sw_optimization = lse_optimization.get("SW", False)
    qc_optimization = lse_optimization.get("QC", False)
    ri_c_optimization = lse_optimization.get("RI_C", False)
    ri_p_optimization = lse_optimization.get("RI_P", False)

    sw_mode = "disabled"
    q_rb_c_mode = "disabled"
    q_rb_s_mode = "disabled"

    # ------------------------------------------------------------------
    # Switching search space.
    # YAML names: alpha, beta
    # Constructor names: alpha, beta
    # ------------------------------------------------------------------
    if not sw_optimization:
        dynamic_params["alpha"] = [0]
        dynamic_params["beta"] = [0]
    else:
        alpha_values = dynamic_params.get("alpha", [])
        if isinstance(alpha_values, list) and len(alpha_values) > 0:
            dynamic_params["alpha"], sw_mode = _expand_lse_alpha_values(
                alpha_values,
                qp_tr
            )
        else:
            dynamic_params["alpha"] = [0]

        if "beta" not in dynamic_params:
            dynamic_params["beta"] = [0]

    # ------------------------------------------------------------------
    # QC search space.
    # YAML name: Q_C
    # Constructor name: QC
    # ------------------------------------------------------------------
    if not qc_optimization:
        dynamic_params["Q_C"] = [1]
    elif "Q_C" not in dynamic_params:
        dynamic_params["Q_C"] = [1]

    # ------------------------------------------------------------------
    # Cost rebalance search space.
    # YAML/constructor names:
    #   Q_RB_C -> Q_RB_C
    # ------------------------------------------------------------------
    if not ri_c_optimization:
        dynamic_params["Q_RB_C"] = [1]
    else:
        q_rb_c_values = dynamic_params.get("Q_RB_C", [])
        if isinstance(q_rb_c_values, list) and len(q_rb_c_values) > 0:
            dynamic_params["Q_RB_C"], q_rb_c_mode = _expand_lse_rebalance_values(
                q_rb_c_values,
                "Q_RB_C",
                qp_tr
            )
        else:
            dynamic_params["Q_RB_C"] = [1]

    # ------------------------------------------------------------------
    # Sample/population rebalance search space.
    # YAML/constructor names:
    #   Q_RB_S -> Q_RB_S
    # ------------------------------------------------------------------
    if not ri_p_optimization:
        dynamic_params["Q_RB_S"] = [1]
    else:
        q_rb_s_values = dynamic_params.get("Q_RB_S", [])
        if isinstance(q_rb_s_values, list) and len(q_rb_s_values) > 0:
            dynamic_params["Q_RB_S"], q_rb_s_mode = _expand_lse_rebalance_values(
                q_rb_s_values,
                "Q_RB_S",
                qp_tr
            )
        else:
            dynamic_params["Q_RB_S"] = [1]

    # ------------------------------------------------------------------
    # Defaults for missing dynamic parameters.
    #
    # In the structured YAML format, only global LSE parameters are filled here.
    # Base-learner-specific defaults must not be injected at top level, because
    # they would override or conflict with params.base_learner_configs.
    #
    # Legacy flat YAML remains supported by adding the old expert-specific
    # defaults only when base_learner_configs is absent.
    # ------------------------------------------------------------------
    dynamic_defaults = {
        "alpha": [0],
        "beta": [0],
        "Q_C": [1],
        "Q_RB_S": [1],
        "Q_RB_C": [1],
        "num_experts": [1],
    }

    if not uses_structured_base_configs:
        dynamic_defaults.update({
            "hidden_size": [64],
            "drop_out": [0.0],
            "n_batch": [128],
            "n_epoch": [100],
            "mode": ["random"],
        })

    for key, default_value in dynamic_defaults.items():
        if key not in dynamic_params:
            dynamic_params[key] = default_value

    # ------------------------------------------------------------------
    # Filter dynamic search space.
    #
    # With structured base_learner_configs, dynamic_params is intentionally
    # restricted to global LSE parameters. Expert-specific parameters are taken
    # from the selected base_learner_configs block inside LSEnsemble.
    #
    # With legacy flat YAML, keep the historical filtering logic.
    # ------------------------------------------------------------------
    if uses_structured_base_configs:
        allowed_dynamic_params = {
            "alpha",
            "beta",
            "Q_C",
            "Q_RB_S",
            "Q_RB_C",
            "num_experts",
        }

        dynamic_params = {
            key: value
            for key, value in dynamic_params.items()
            if key in allowed_dynamic_params
        }
    else:
        dynamic_params = _filter_lse_dynamic_params_by_base_learner(
            dynamic_params=dynamic_params,
            base_learner=base_learner,
            optim=optim
        )

    keys = list(dynamic_params.keys())
    values = list(dynamic_params.values())

    resolved_configs = []

    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        updated_config = dict(params)

        # --------------------------------------------------------------
        # Common LSE parameters.
        # Note:
        # - YAML uses Q_C
        # - LSEnsemble constructor expects QC
        # --------------------------------------------------------------
        updated_config.update({
            "alpha": param_dict["alpha"],
            "beta": param_dict["beta"],
            "QC": param_dict["Q_C"],
            "Q_RB_S": param_dict["Q_RB_S"],
            "Q_RB_C": param_dict["Q_RB_C"],
            "num_experts": param_dict["num_experts"],
        })

        # --------------------------------------------------------------
        # Base learner specific parameters.
        #
        # In the structured format, do not copy any expert-specific values
        # to the top-level constructor config. The complete
        # base_learner_configs block is already present in updated_config,
        # and LSEnsemble will select the active sub-block internally.
        #
        # In the legacy flat format, preserve the previous behavior.
        # --------------------------------------------------------------
        if not uses_structured_base_configs:
            if base_learner in ("FAMLP", "AMLP"):
                updated_config.update({
                    "hidden_size": param_dict["hidden_size"],
                    "drop_out": param_dict["drop_out"],
                    "n_batch": param_dict["n_batch"],
                    "n_epoch": param_dict["n_epoch"],
                })

                if "activation_fn" in params:
                    updated_config["activation_fn"] = params["activation_fn"]
                if "optim" in params:
                    updated_config["optim"] = params["optim"]
                if "loss_fn" in params:
                    updated_config["loss_fn"] = params["loss_fn"]

                if "mode" in param_dict and optim != "lbfgs":
                    updated_config["mode"] = param_dict["mode"]

            elif base_learner == "Parzen":
                updated_config.update({
                    "hidden_size": param_dict["hidden_size"],
                    "drop_out": param_dict["drop_out"],
                    "n_batch": param_dict["n_batch"],
                    "n_epoch": param_dict["n_epoch"],
                })

                if "activation_fn" in params:
                    updated_config["activation_fn"] = params["activation_fn"]

                if "mode" in param_dict:
                    updated_config["mode"] = param_dict["mode"]

            elif base_learner == "LogReg":
                updated_config.update({
                    "n_epoch": param_dict["n_epoch"],
                })

                if "optim" in params:
                    updated_config["optim"] = params["optim"]

                if "mode" in param_dict and optim != "lbfgs":
                    updated_config["mode"] = param_dict["mode"]

            elif base_learner == "CalibratedBooster":
                if base_learner_params:
                    updated_config["base_learner_params"] = dict(base_learner_params)

                if "calibration_method" in params:
                    updated_config["calibration_method"] = params["calibration_method"]

            updated_config = _prune_lse_params_by_base_learner(
                config=updated_config,
                base_learner=base_learner,
                optim=optim
            )

        else:
            # Structured format:
            # Keep backward-compatible base_learner_params only if explicitly
            # provided. These flat params can still be used as final overrides
            # inside LSEnsemble._resolve_base_learner_params().
            if base_learner_params:
                updated_config["base_learner_params"] = dict(base_learner_params)

        updated_config = {
            key: value
            for key, value in updated_config.items()
            if value is not None
        }

        resolved_configs.append(updated_config)

    logger.debug(
        "Resolved %d LSE configurations for model '%s' with base_learner='%s' "
        "(structured_base_configs=%s, SW_mode=%s, Q_RB_C_mode=%s, "
        "Q_RB_S_mode=%s, qp_tr=%.4f).",
        len(resolved_configs),
        item.get("name", "unknown"),
        base_learner,
        uses_structured_base_configs,
        sw_mode,
        q_rb_c_mode,
        q_rb_s_mode,
        qp_tr
    )

    return resolved_configs

def _resolve_standard_model_configurations(model_item):
    """
    Resolve one non-LSE model block into constructor-ready configurations.

    The resolver keeps all fixed parameters declared in ``params`` and combines
    them with every combination declared in ``dynamic_params``. For current YAML
    models, parameter names are used exactly as provided in the configuration.

    Legacy compatibility is preserved only for a few old model families that may
    still appear in historical experiments.

    Parameters
    ----------
    model_item : dict
        One model block from the YAML configuration.

    Returns
    -------
    list
        List of constructor-ready dictionaries.
    """
    item = copy.deepcopy(model_item)

    model_name = item["name"]
    fixed_params = dict(item.get("params", {}))
    dynamic_params = dict(item.get("dynamic_params", {}))

    if not dynamic_params:
        return [
            {
                key: value
                for key, value in fixed_params.items()
                if value is not None
            }
        ]

    keys = list(dynamic_params.keys())
    values = list(dynamic_params.values())

    resolved_configs = []

    for combination in product(*values):
        param_dict = dict(zip(keys, combination))

        # Start from all fixed parameters and add all dynamic parameters
        # exactly as they are declared in YAML.
        updated_config = dict(fixed_params)
        updated_config.update(param_dict)

        # --------------------------------------------------------------
        # Legacy compatibility blocks for older experiment names only.
        # These do not affect the current YAML models.
        # --------------------------------------------------------------
        if model_name == "kNN":
            if "kNN_n_neighbors" in param_dict:
                updated_config["n_neighbors"] = param_dict["kNN_n_neighbors"]
            if "kNN_metric" in param_dict:
                updated_config["metric"] = param_dict["kNN_metric"]

        elif model_name == "C4.5":
            if "C45_max_depth" in param_dict:
                updated_config["max_depth"] = param_dict["C45_max_depth"]
            if "C45_min_samples_split" in param_dict:
                updated_config["min_samples_split"] = param_dict["C45_min_samples_split"]
            if "C45_min_samples_leaf" in param_dict:
                updated_config["min_samples_leaf"] = param_dict["C45_min_samples_leaf"]

        elif model_name == "SVM":
            if "SVM_C" in param_dict:
                updated_config["C"] = param_dict["SVM_C"]
            if "SVM_gamma" in param_dict:
                updated_config["gamma"] = param_dict["SVM_gamma"]

        elif model_name == "MultiRandBal":
            if "RB_n_estimators" in param_dict:
                updated_config["n_estimators"] = param_dict["RB_n_estimators"]
            if "RB_base_estimator" in param_dict:
                updated_config["base_estimator"] = param_dict["RB_base_estimator"]

        updated_config = {
            key: value
            for key, value in updated_config.items()
            if value is not None
        }

        resolved_configs.append(updated_config)

    return resolved_configs


def resolve_model_configurations(model_block, x_train=None, y_train_lab=None, logger=None):
    """
    Resolve one model block from YAML into constructor-ready configurations.

    Parameters
    ----------
    model_block : dict
        One model block from the YAML configuration.
    x_train : np.ndarray or None, optional
        Training matrix. Used by LSE resolution when needed.
    y_train_lab : np.ndarray or None, optional
        Training labels. Used by LSE resolution when needed.
    logger : logging.Logger or None, optional
        Logger instance.

    Returns
    -------
    list
        List of resolved configurations.
    """
    model_name = model_block["name"]

    if "LSEnsemble" in model_name:
        return _resolve_lse_model_configurations(
            model_item=model_block,
            x_train=x_train,
            y_train_lab=y_train_lab,
            logger=logger
        )

    return _resolve_standard_model_configurations(model_block)


def generate_model_configurations(model_list, x_train=None, y_train_lab=None, logger=None):
    """
    Generate resolved configurations for all models in the provided list.

    This function preserves the historical public API used in the project,
    but it now delegates the real resolution logic to
    ``resolve_model_configurations``.

    Parameters
    ----------
    model_list : list
        List of model blocks from YAML.
    x_train : np.ndarray or None, optional
        Training matrix used by LSE resolution.
    y_train_lab : np.ndarray or None, optional
        Training labels used by LSE resolution.
    logger : logging.Logger or None, optional
        Logger instance.

    Returns
    -------
    dict
        Dictionary indexed by model name. Each value is a list of resolved
        constructor-ready configurations.
    """
    cv_config = {}

    for model_item in model_list:
        model_name = model_item["name"]
        cv_config[model_name] = resolve_model_configurations(
            model_block=model_item,
            x_train=x_train,
            y_train_lab=y_train_lab,
            logger=logger
        )

    return cv_config

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
    
    alpha_est = 0.2 * np.log10(ir) + 0.1
    
    if cap:
        alpha_est = min(alpha_est, 0.45)
    
    return round(alpha_est, 3)