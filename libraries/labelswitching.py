import numpy as np
import torch
import torch.nn as nn

from libraries.functions import compute_imbalance_ratio
from libraries.ML_models import LogisticRegressionTorch, AsymmetricMLP, DeepAsymmetricMLP
from libraries.ML_models import CalibratedBooster
from libraries.ML_models import MLPBayesSwitch


from imblearn.over_sampling import SMOTE

from libraries.optimizers import LOSS_FUNCTIONS

from libraries.optimizers import (weighted_mse_loss,
#                                  weighted_kl_loss,
#                                  weighted_bce_loss,
#                                  weighted_bce_logit_loss,
#                                  f1_loss,
)


TORCH_DTYPE = torch.float32
NUMPY_DTYPE = np.float32
eps=np.finfo(float).eps

def label_switching(y, alphasw=0.0, betasw=0.0, w=None, weight_mode="original", seed=None):
    """
    Perform label switching on binary labels and optionally rebuild sample
    weights from class weights.

    Parameters
    ----------
    y : np.ndarray
        Original labels with values -1 and +1.
    w : array-like of shape (2,) or None, optional
        Class weights in the form [negative_class_weight, positive_class_weight].
        If provided, the returned sample-weight vector is rebuilt from either
        the original labels or the switched labels, depending on
        ``weight_mode``.
    alphasw : float, optional
        Switching rate from negative class (-1) to positive class (+1).
    betasw : float, optional
        Switching rate from positive class (+1) to negative class (-1).
    weight_mode : {"original", "switched"}, optional
        Strategy used to rebuild class-based sample weights.

        - "original": weights are assigned from the original labels.
        - "switched": weights are assigned from the switched labels.
    seed : int or None, optional
        Random seed used to generate a reproducible switching realization.

    Returns
    -------
    ysw : np.ndarray
        Labels after switching.
    wsw : np.ndarray or None
        Sample weights rebuilt from class weights if class weights were
        provided. Otherwise, None.

    Notes
    -----
    This function is only valid for class-weight vectors of length 2.
    It must not be used with external per-sample weights.
    """
    y = np.asarray(y)
    ysw = np.copy(y)

    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.shape != (2,):
            raise ValueError(
                "label_switching expects class weights of length 2: "
                "[negative_class_weight, positive_class_weight]."
            )

    if weight_mode not in {"original", "switched"}:
        raise ValueError("weight_mode must be either 'original' or 'switched'.")

    if seed is not None:
        np.random.seed(seed)

    # Positive (+1) -> negative (-1)
    idx_pos = np.where(y == +1)[0]
    n_pos_sw = min(int(round(len(idx_pos) * betasw)), len(idx_pos))
    if n_pos_sw > 0:
        idx_pos_sw = np.random.choice(idx_pos, n_pos_sw, replace=False)
        ysw[idx_pos_sw] = -1

    # Negative (-1) -> positive (+1)
    idx_neg = np.where(y == -1)[0]
    n_neg_sw = min(int(round(len(idx_neg) * alphasw)), len(idx_neg))
    if n_neg_sw > 0:
        idx_neg_sw = np.random.choice(idx_neg, n_neg_sw, replace=False)
        ysw[idx_neg_sw] = +1

    if w is None:
        return ysw, None

    y_for_weights = y if weight_mode == "original" else ysw
    wsw = np.where(y_for_weights == -1, w[0], w[1]).astype(float, copy=False)

    return ysw, wsw

def compute_weights(targets_train, RI_C=1.0, Q_P=1.0, mode="normal", device=None):
    """
    Computes weights for training samples based on the desired cost Rebalance intensity (RI_C)
    and the imbalance ratio (Q_P).

    Args:
        targets_train (np.ndarray): Array of training targets (-1 or 1).
        RI_C (float): Desired cost Rebalance intensity. Defaults to 1 (no explicit rebalancing).
            - If float (>= 1):
                - If mode is "proportional_minority": Weight of the positive class (assumed minority if Q_P > 1) is RI_C (capped at Q_P).
                - If mode is "proportional_majority_reverse": Weight of the negative class is 1/RI_C (capped at 1/Q_P).
        Q_P (float): Imbalance ratio (number of majority class samples / number of minority class samples). Defaults to 1.
        mode (str): Mode of rebalancing. Defaults to 'proportional_minority'.
            - 'normal': Weights the minority class (positive if Q_P > 1) by RI_C.
            - 'reverse': Weights the majority class (negative if Q_P > 1) by 1/RI_C.

        device : torch.device or None
            Target device for the output tensor.

    Returns
    -------
    torch.Tensor
        Sample weights using the project-wide torch dtype.
    """
    if torch.is_tensor(targets_train):
        targets_np = targets_train.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets_train)

    weights = np.ones_like(targets_np, dtype=NUMPY_DTYPE)

    if RI_C >= 1:
        if mode == "normal":
            weights[targets_np > 0] = min(RI_C, Q_P)
        elif mode == "reverse":
            weights[targets_np <= 0] = 1.0 / min(RI_C, Q_P)

    return torch.as_tensor(weights, dtype=TORCH_DTYPE, device=device)

def identity(x):
    """
    Identity function: returns input unchanged.
    Used as fallback activation when no transformation is needed.
    """
    return x

# Mapping for activation functions
ACTIVATION_FUNCTIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    None: identity,
}

class LSEnsemble(nn.Module):
    def __init__(
        self,
        class_cost=None,
        hidden_size=None,
        num_experts=1,
        alpha=0,
        beta=0,
        QC=None,
        Q_RB_C=1,
        Q_RB_S=1,
        n_epoch=1,
        n_batch=1,
        mode="random",
        input_size=None,
        drop_out=0,
        base_learner="AMLP",
        activation_fn="tanh",
        output_act=1,
        optim="lbfgs",
        loss_fn="MSE",
        base_learner_params=None,
        base_learner_configs=None,
        calibration_method=None,
        ls_weight_mode="original",
        dtype=TORCH_DTYPE
    ):
        """
        Initialize the Label Switching Ensemble (LSEnsemble) model.

        This class implements an ensemble of experts with asymmetric label
        switching controlled by alpha and beta. It supports neural experts
        such as AMLP, DAMLP, Parzen and LogReg, as well as sklearn-like models
        wrapped through CalibratedBooster.

        Parameters
        ----------
        class_cost : array-like of shape (2,) or None, optional
            Class-dependent decision costs in the form
            [negative_class_cost, positive_class_cost].

            These costs are used only to derive the Bayes decision ratio QC
            when QC is None. They do not directly replace external per-sample
            weights nor the internal rebalancing factors.

            If None, [1.0, 1.0] is used.

        hidden_size : int or None, optional
            Number of neurons in the hidden layer for shallow MLP-based
            experts. This may be provided directly or through
            base_learner_configs.

        num_experts : int, optional
            Number of individual experts in the ensemble.

        alpha : float, optional
            Switching factor from majority to minority class.

        beta : float, optional
            Switching factor from minority to majority class.

        QC : float or None, optional
            Bayes decision cost ratio used in the final decision threshold.

            If provided, this value is used directly.

            If None, it is derived from class_cost as:

                QC = (C10 - C00) / (C01 - C11)

            with:
            - C10 = negative_class_cost  (false positive cost)
            - C00 = 0.0                  (true negative cost)
            - C01 = positive_class_cost  (false negative cost)
            - C11 = 0.0                  (true positive cost)

        Q_RB_C : float, optional
            Classification-oriented rebalancing cost factor.

        Q_RB_S : float, optional
            Rebalancing factor for neutral population, SMOTE-like.

        n_epoch : int, optional
            Number of training epochs. For warm_start_lbfgs, this controls
            the LBFGS refinement phase unless overridden in base_learner_configs.

        n_batch : int, optional
            Batch size during training. For warm_start_lbfgs, this controls
            the minibatch warm-start phase unless overridden in
            base_learner_configs.

        mode : str, optional
            Sampling mode used to create batches.

        input_size : int or None, optional
            Input feature dimensionality. If provided, experts are initialized
            immediately.

        drop_out : float, optional
            Dropout probability for hidden layers in MLP-based experts.

        base_learner : str, optional
            Type of base learner for each expert.

            Supported options include:
            - AMLP
            - DAMLP
            - Parzen
            - LogReg
            - CalibratedBooster

        activation_fn : str, optional
            Activation function for hidden layers. This may be overridden by
            the selected block in base_learner_configs.

        output_act : int, optional
            Output activation/compression variant. This may be overridden by
            the selected block in base_learner_configs.

        optim : str, optional
            Optimizer for neural base learners. This may be overridden by the
            selected block in base_learner_configs.

        loss_fn : str, optional
            Loss function identifier for training neural experts.

        base_learner_params : dict or None, optional
            Flat backward-compatible dictionary with additional keyword
            arguments passed to the selected base learner constructor.

            If both base_learner_params and base_learner_configs are provided,
            base_learner_params overrides the structured configuration.

        base_learner_configs : dict or None, optional
            Structured configuration dictionary with one block per base learner.

            Example:
                {
                    "AMLP": {...},
                    "CalibratedBooster": {
                        "model_type": "lgbm",
                        "model_configs": {"lgbm": {...}}
                    }
                }

            Only the block matching base_learner is used.

        calibration_method : str or None, optional
            Calibration strategy when using CalibratedBooster or other experts
            with explicit calibration support.

        ls_weight_mode : {"original", "switched"}, optional
            Strategy used to rebuild class-based weights after label switching.

        dtype : torch.dtype, optional
            Torch dtype used internally by the ensemble and its Torch experts.

        Notes
        -----
        - base_learner_configs is interpreted inside this class so that the
          generic training pipeline does not need model-specific logic.
        - class_cost affects the decision threshold only through QC when QC is
          None.
        - External per-sample weights should still be passed through fit().
        """
        super(LSEnsemble, self).__init__()

        self.dtype = dtype

        self.ls_weight_mode = ls_weight_mode
        if self.ls_weight_mode not in {"original", "switched"}:
            raise ValueError("ls_weight_mode must be either 'original' or 'switched'.")

        if class_cost is None:
            class_cost = [1.0, 1.0]

        class_cost = np.asarray(class_cost, dtype=float).ravel()
        if class_cost.shape != (2,):
            raise ValueError(
                "class_cost must contain exactly two elements: "
                "[negative_class_cost, positive_class_cost]."
            )
        if np.any(~np.isfinite(class_cost)) or np.any(class_cost <= 0):
            raise ValueError("class_cost values must be finite and strictly positive.")

        self.class_cost = class_cost.tolist()

        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta

        self.Q_RB_C = Q_RB_C
        self.Q_RB_S = Q_RB_S

        # Compute the theoretical Bayes cost ratio.
        C10 = float(self.class_cost[0])  # False positive cost
        C00 = 0.0
        C01 = float(self.class_cost[1])  # False negative cost
        C11 = 0.0
        
        bayes_cost_ratio = (C10 - C00) / (C01 - C11)
        
        if not np.isfinite(bayes_cost_ratio) or bayes_cost_ratio <= 0:
            raise ValueError("The Bayes cost ratio must be finite and strictly positive.")
        
        self.bayes_cost_ratio = bayes_cost_ratio
        
        if QC is None:
            self.QC_multiplier = 1.0
        else:
            self.QC_multiplier = float(QC)
        
        if not np.isfinite(self.QC_multiplier) or self.QC_multiplier <= 0:
            raise ValueError("QC must be finite and strictly positive.")
        
        self.QC = self.QC_multiplier * self.bayes_cost_ratio

        self.base_learner = base_learner

        # Structured base-learner configuration. This is expected to come from
        # params.base_learner_configs in the YAML, so src/training.py can remain
        # model-agnostic.
        self.base_learner_configs = (
            dict(base_learner_configs) if base_learner_configs is not None else {}
        )

        # Flat configuration kept for backward compatibility. If both sources
        # are provided, this one overrides the structured configuration.
        self.base_learner_params = (
            dict(base_learner_params) if base_learner_params is not None else {}
        )

        resolved_base_params = self._resolve_base_learner_params()

        # Base-learner-specific values override generic constructor defaults.
        hidden_size = resolved_base_params.pop("hidden_size", hidden_size)
        drop_out = resolved_base_params.pop("drop_out", drop_out)
        n_epoch = resolved_base_params.pop("n_epoch", n_epoch)
        n_batch = resolved_base_params.pop("n_batch", n_batch)
        mode = resolved_base_params.pop("mode", mode)
        optim = resolved_base_params.pop("optim", optim)
        activation_fn = resolved_base_params.pop("activation_fn", activation_fn)
        output_act = resolved_base_params.pop("output_act", output_act)

        # Remaining parameters are specific to the selected expert.
        # Examples:
        # - AMLP: warm_start_optim, warm_start_n_epoch, warm_start_learning_rate.
        # - CalibratedBooster: model_type, task, boosting_type, n_estimators, etc.
        self.base_learner_params = resolved_base_params

        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.mode = mode
        self.hidden_size = hidden_size
        self.drop_out = drop_out

        # Validate hidden_size only for expert families that actually need it.
        if self.base_learner in ("DAMLP", "AMLP", "Parzen") and self.hidden_size is None:
            raise ValueError(
                f"hidden_size must be provided when base_learner='{self.base_learner}'."
            )

        # Convert activation string to callable function.
        if activation_fn not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Invalid activation function '{activation_fn}'. "
                f"Available options: {list(ACTIVATION_FUNCTIONS.keys())}"
            )
        self.activation_fn = ACTIVATION_FUNCTIONS[activation_fn]

        self.optim = optim
        self.loss_fn_e = LOSS_FUNCTIONS.get(loss_fn, weighted_mse_loss)
        self.output_mode = output_act
        self.calibration_method = calibration_method

        # Initialize experts immediately if input_size is known.
        if input_size is not None:
            self.initialize_experts(input_size)

    def _resolve_base_learner_params(self):
        """
        Resolve the active base learner configuration.

        The class supports two configuration mechanisms:

        1. base_learner_configs:
           Structured YAML-style dictionary with one block per learner.
           Only the block matching self.base_learner is used.

        2. base_learner_params:
           Flat backward-compatible dictionary. If provided, it overrides the
           selected structured configuration.

        When self.base_learner is CalibratedBooster, this method also selects
        the internal model configuration indicated by model_type and merges it
        into the returned dictionary.

        Returns
        -------
        dict
            Resolved parameter dictionary for the selected base learner.

        Raises
        ------
        ValueError
            If the selected base learner has no configuration block, or if
            CalibratedBooster is missing a valid model_type/model_configs entry.
        """
        flat_params = dict(self.base_learner_params)

        if not self.base_learner_configs:
            return flat_params

        if self.base_learner not in self.base_learner_configs:
            raise ValueError(
                f"base_learner='{self.base_learner}' has no matching entry in "
                f"base_learner_configs. Available: {list(self.base_learner_configs.keys())}"
            )

        selected_params = dict(self.base_learner_configs[self.base_learner])

        if self.base_learner == "CalibratedBooster":
            model_type = selected_params.get("model_type", None)
            model_configs = selected_params.pop("model_configs", {})

            if model_type is None:
                raise ValueError(
                    "CalibratedBooster requires `model_type` in its configuration."
                )

            if model_type not in model_configs:
                raise ValueError(
                    f"CalibratedBooster model_type='{model_type}' has no matching "
                    f"model_configs entry. Available: {list(model_configs.keys())}"
                )

            internal_model_params = dict(model_configs[model_type])
            selected_params.update(internal_model_params)

        # Flat base_learner_params overrides structured config when both exist.
        selected_params.update(flat_params)

        return selected_params

    def initialize_experts(self, input_size):
        """
        Initialize the list of expert models based on the selected base_learner.

        This method creates num_experts instances of the chosen base learner,
        passing common LSEnsemble parameters such as alpha and beta, plus the
        learner-specific parameters resolved from base_learner_configs.

        Parameters
        ----------
        input_size : int
            Dimensionality of the input features.

        Notes
        -----
        - Experts are stored in self.experts as an nn.ModuleList.
        - For AMLP/DAMLP/LogReg, training-related parameters such as n_epoch,
          n_batch, mode and optim are passed to the expert.
        - For CalibratedBooster, internal model parameters such as model_type,
          task, n_estimators, scale_pos_weight, etc. are passed through
          self.base_learner_params.
        """
        if self.base_learner == "DAMLP":
            deep_params = dict(self.base_learner_params)

            hidden_layer_sizes = deep_params.pop(
                "hidden_layer_sizes",
                (self.hidden_size,)
            )

            deep_kwargs = {
                "input_dim": input_size,
                "hidden_layer_sizes": tuple(hidden_layer_sizes),
                "activation_fn": self.activation_fn,
                "alpha": self.alpha,
                "beta": self.beta,
                "batch_size": self.n_batch,
                "max_iter": self.n_epoch,
                "optim": self.optim,
                "output_act": self.output_mode,
                "dtype": self.dtype,
            }

            # Extra DAMLP-specific parameters, e.g. warm_start_optim.
            deep_kwargs.update(deep_params)

            self.experts = nn.ModuleList([
                DeepAsymmetricMLP(**deep_kwargs)
                for _ in range(self.num_experts)
            ])

        elif self.base_learner == "LogReg":
            logreg_kwargs = {
                "input_dim": input_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "n_epoch": self.n_epoch,
                "n_batch": self.n_batch,
                "optim": self.optim,
                "mode": self.mode,
                "loss_fn": self.loss_fn_e,
                "dtype": self.dtype,
                "output_act": self.output_mode,
            }
        
            logreg_kwargs.update(self.base_learner_params)
        
            self.experts = nn.ModuleList([
                LogisticRegressionTorch(**logreg_kwargs)
                for _ in range(self.num_experts)
            ])

        elif self.base_learner == "CalibratedBooster":
            booster_kwargs = {
                "input_dim": input_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "calibration_method": self.calibration_method,
            }

            # Extra CalibratedBooster parameters:
            # model_type, task, output_act, and selected internal model params.
            booster_kwargs.update(self.base_learner_params)

            self.experts = nn.ModuleList([
                CalibratedBooster(**booster_kwargs)
                for _ in range(self.num_experts)
            ])

        elif self.base_learner == "Parzen":
            parzen_kwargs = dict(self.base_learner_params)

            parzen_kwargs_default = {
                "n_epoch": self.n_epoch,
                "n_batch": self.n_batch,
                "type_batch": self.mode,
                "layers_size": [self.hidden_size],
                "drop_out": [0, self.drop_out],
                "activations": [self.activation_fn.__name__, "mod_tanh"],
                "class_prob": [0.5, 0.5],
                "alpha": self.alpha,
                "beta": self.beta,
            }

            parzen_kwargs_default.update(parzen_kwargs)

            self.experts = nn.ModuleList([
                MLPBayesSwitch(**parzen_kwargs_default)
                for _ in range(self.num_experts)
            ])

        else:
            # Default / fallback case: AsymmetricMLP.
            # This covers AMLP-style experts.
            expert_kwargs = {
                "input_size": input_size,
                "hidden_size": self.hidden_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "dropout_prob": self.drop_out,
                "activation_fn": self.activation_fn,
                "output_act": self.output_mode,
                "dtype": self.dtype,
                "n_epoch": self.n_epoch,
                "n_batch": self.n_batch,
                "optim": self.optim,
                "mode": self.mode,
                "loss_fn": self.loss_fn_e,
            }

            # Extra AMLP-specific parameters, e.g. warm_start_optim,
            # warm_start_n_epoch, warm_start_learning_rate, warm_start_debug.
            expert_kwargs.update(self.base_learner_params)

            self.experts = nn.ModuleList([
                AsymmetricMLP(**expert_kwargs)
                for _ in range(self.num_experts)
            ])
            
    def generate_experts_data(self, x, y, w=None, Q_RB_S=1, RB_each_expert=True):
        """
        Generate expert-specific datasets with optional SMOTE and label switching.
    
        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape (n_samples, n_features).
        y : torch.Tensor
            Binary label tensor of shape (n_samples,).
            Supported formats are {0, 1} and {-1, 1}.
        w : torch.Tensor or None, optional
            External sample-weight tensor of shape (n_samples,).
            This weight represents sample importance independent of class.
            If None, a unit-weight vector is used.
        Q_RB_S : float, optional
            Desired rebalancing ratio for SMOTE. Values greater than 1 activate
            synthetic oversampling of the minority class.
        RB_each_expert : bool, optional
            If True, generate an independent SMOTE dataset for each expert.
            If False, all experts share the same rebalanced dataset.
    
        Notes
        -----
        The stored expert weight vector `expert.w` represents the external
        per-sample weight and is preserved through label switching.
    
        Label switching modifies labels only.
    
        For synthetic minority samples generated by SMOTE, the external weight is
        set to the mean external weight of the original minority-class samples.
    
        The final training weight is not computed here. This function stores:
        - expert.X : expert features
        - expert.y : switched and compressed labels
        - expert.w : external sample weights preserved after SMOTE/LS
    
        Cost-sensitive class weights must be computed later during training and
        combined multiplicatively with expert.w.
        """
        x_np = x.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
        y_np = y.detach().cpu().numpy()
        w_np = (
            w.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
            if w is not None
            else np.ones_like(y_np, dtype=NUMPY_DTYPE)
        )
    
        # Binary label handling
        unique_labels = np.unique(y_np)
    
        if set(unique_labels) == {0, 1}:
            self.bin_format = 0
            y_np = np.where(y_np == 0, -1, 1)
        elif set(unique_labels) == {-1, 1}:
            self.bin_format = -1
        else:
            raise ValueError("Only binary labels {0,1} or {-1,1} are supported.")
    
        QP_tr = compute_imbalance_ratio(y_np)
        self.QP_tr = QP_tr
    
        apply_rebalancing = False
        sampling_strategy = None
    
        if Q_RB_S > 1:
            sampling_strategy = min(Q_RB_S / QP_tr, 1.0)
            self.Q_RB_S = QP_tr * sampling_strategy
            apply_rebalancing = True
    
        def apply_smote(x_np, y_np, w_np, random_state):
            """
            Apply SMOTE and extend the external sample-weight vector.
    
            Synthetic samples inherit the mean external weight of the original
            minority-class samples.
            """
            try:
                smote = SMOTE(
                    random_state=random_state,
                    sampling_strategy=sampling_strategy
                )
                X_RB, y_RB = smote.fit_resample(x_np, y_np)
                X_RB = np.asarray(X_RB, dtype=NUMPY_DTYPE)
    
                n_synthetic = len(X_RB) - len(x_np)
    
                if n_synthetic > 0:
                    classes, counts = np.unique(y_np, return_counts=True)
                    minority_class = classes[np.argmin(counts)]
    
                    minority_mask = (y_np == minority_class)
                    mean_minority_weight = np.asarray(
                        np.mean(w_np[minority_mask]),
                        dtype=NUMPY_DTYPE
                    )
    
                    synthetic_w = np.full(
                        n_synthetic,
                        mean_minority_weight,
                        dtype=NUMPY_DTYPE
                    )
    
                    w_RB = np.concatenate([w_np, synthetic_w]).astype(
                        NUMPY_DTYPE,
                        copy=False
                    )
                else:
                    w_RB = w_np
    
                return X_RB, y_RB, w_RB
    
            except ValueError:
                return x_np, y_np, w_np
    
        def process_labels_and_weights(X_RB, y_RB, w_RB, expert_idx=0):
            """
            Apply label switching to labels only and convert everything to tensors.
        
            Returns
            -------
            tuple
                (X_tensor, y_tensor, w_tensor)
            """
            if self.alpha == 0 and self.beta == 0:
                targets_sw = y_RB
                w_sw = np.ones_like(y_RB, dtype=NUMPY_DTYPE)
            else:
                targets_sw, w_sw = label_switching(
                    y_RB,
                    alphasw=self.alpha,
                    betasw=self.beta,
                    w=self.class_cost,
                    weight_mode=self.ls_weight_mode,
                    seed=42 + expert_idx
                )
                w_sw = np.asarray(w_sw, dtype=NUMPY_DTYPE)
        
            X_RB = np.asarray(X_RB, dtype=NUMPY_DTYPE)
            w_RB = np.asarray(w_RB, dtype=NUMPY_DTYPE) * w_sw
            targets_sw = np.asarray(targets_sw)
        
            y_RB_SW = np.where(
                targets_sw > 0,
                np.asarray(1 - 2 * self.beta, dtype=NUMPY_DTYPE),
                np.asarray(-(1 - 2 * self.alpha), dtype=NUMPY_DTYPE)
            ).astype(NUMPY_DTYPE, copy=False)
        
            X_RB_SW = torch.from_numpy(X_RB).to(x.device, dtype=self.dtype)
            y_RB_SW = torch.from_numpy(y_RB_SW).to(x.device, dtype=self.dtype)
            w_RB_SW = torch.from_numpy(
                w_RB.astype(NUMPY_DTYPE, copy=False)
            ).to(x.device, dtype=self.dtype)
        
            return X_RB_SW, y_RB_SW, w_RB_SW
    
        if apply_rebalancing:
            if RB_each_expert:
                # Generate a reproducible but different SMOTE dataset for each expert.
                for i, expert in enumerate(self.experts):
                    X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np, random_state=42 + i)
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB, expert_idx=i)
            else:
                # Generate a single reproducible SMOTE dataset shared by all experts.
                X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np, random_state=42)
                for i, expert in enumerate(self.experts):
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB, expert_idx=i)
        else:
            # No rebalancing; all experts use the original dataset.
            for i, expert in enumerate(self.experts):
                expert.X, expert.y, expert.w = process_labels_and_weights(x_np, y_np, w_np, expert_idx=i)

    def fit(self, x_train, y_train, sample_weight=None):
        """
        Fit the ensemble model using training data.
    
        Parameters:
        - x_train: Tensor or NumPy array of shape (n_samples, n_features) for training data.
        - y_train: Tensor or NumPy array of shape (n_samples,) for training labels.
        - sample_weight: Optional tensor or NumPy array of shape (n_samples,) for sample weights.
        """
        # Single conversion block: everything to self.dtype + correct device
        if len(list(self.parameters())) > 0:
            device = next(self.parameters()).device
        else:
            device = torch.device("cpu")

        if isinstance(x_train, np.ndarray):
            x_train_np = np.asarray(x_train, dtype=NUMPY_DTYPE).copy()
            x_train = torch.from_numpy(x_train_np).to(device=device, dtype=self.dtype)
        elif isinstance(x_train, torch.Tensor):
            x_train = x_train.to(device=device, dtype=self.dtype)
        else:
            raise TypeError("x_train must be torch.Tensor or np.ndarray")

        if isinstance(y_train, np.ndarray):
            y_train_np = np.asarray(y_train).copy()
            y_train = torch.from_numpy(y_train_np).to(device=device, dtype=self.dtype)
        elif isinstance(y_train, torch.Tensor):
            y_train = y_train.to(device=device, dtype=self.dtype)
        else:
            raise TypeError("y_train must be torch.Tensor or np.ndarray")

        # Sample weights: default to ones if None
        if sample_weight is None:
            sample_weight = torch.ones_like(y_train, dtype=self.dtype, device=device)
        elif isinstance(sample_weight, np.ndarray):
            sw_np = np.asarray(sample_weight, dtype=NUMPY_DTYPE).copy()
            sample_weight = torch.from_numpy(sw_np).to(device=device, dtype=self.dtype)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.to(device=device, dtype=self.dtype)
        else:
            raise TypeError("sample_weight must be torch.Tensor, np.ndarray or None")
        
        # Generate internal data for experts
        self.generate_experts_data(
            x_train,
            y_train,
            sample_weight,
            Q_RB_S=self.Q_RB_S,
            RB_each_expert=False
        )
        
        # Fit expert models using the already processed sample_weight
        self.fit_expert_model(
            sample_weight,
            epochs=self.n_epoch,
            batch_size=self.n_batch,
            optim=self.optim
        )
        
        return self
                
    def fit_expert_model(self, w_train, epochs=50, batch_size=256, optim='lbfgs'):
        """
        Train experts using their stored data with a specified optimization method.
    
        Handles different base learners (Parzen, AMLP, DAMLP, LogReg, etc.) and
        combines external sample weights with class-cost weights when supported.
    
        Parameters
        ----------
        w_train : torch.Tensor
            Reference tensor used for dtype/device alignment.
        epochs : int, optional
            Number of training epochs for gradient-based optimizers.
        batch_size : int, optional
            Batch size for gradient-based optimizers.
        optim : str, optional
            Optimization algorithm identifier.
    
        Returns
        -------
        self
            The updated ensemble.
        """
        # Initialize final weight matrix for all experts.
        weights = torch.ones(
            (self.experts[0].y.shape[0], self.num_experts),
            device=w_train.device,
            dtype=self.dtype
        )
    
        # Compute final training weights:
        # external sample weight × class-cost weight
        for i, expert in enumerate(self.experts):
            weights[:, i] = (
                compute_weights(
                    expert.y,
                    RI_C=self.Q_RB_C,
                    Q_P=self.QP_tr,
                    mode='normal'
                ).to(w_train.device, dtype=self.dtype)
                * expert.w.to(w_train.device, dtype=self.dtype)
            )
    
        if self.base_learner in ['AMLP', 'DAMLP', 'Parzen', 'CalibratedBooster', 'LogReg']:
            for i, expert in enumerate(self.experts):
                x_np = expert.X.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
                y_np = expert.y.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
                w_np = weights[:, i].detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
                expert.fit(x_np, y_np, w_np)
    
        elif self.base_learner in ['MLP']:
            # Example branch for models that do not support sample weights or costs.
            for i, expert in enumerate(self.experts):
                x_np = expert.X.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
                y_np = expert.y.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
                expert.fit(x_np, y_np)
    
        else:
            raise ValueError(f"Unsupported base learner: {self.base_learner}")
    
        return self
    
    # Get outputs from all experts
    def get_expert_outputs(self, x):
            """
            Get the outputs from all experts and concatenate them column-wise.
            """
            expert_outputs = self.experts[0](x)
            aux_outputs = torch.zeros_like(expert_outputs)
            for i in range(self.num_experts - 1):
                aux_outputs = self.experts[i + 1](x)
                expert_outputs = torch.cat((expert_outputs, aux_outputs), 1)
            return expert_outputs
    
    # Predict outputs of each expert
    def predict_expert_outputs(self, x):
            """
            Predict the outputs of each expert without tracking gradients.
            """
            with torch.no_grad():
                return self.get_expert_outputs(x)
    
    def forward(self, x):
        """
        This method performs the forward pass by calculating the predictions from the experts
        and returning the averaged prediction (o_pred) across experts.
        """
        if len(list(self.parameters())) > 0:
            device = next(self.parameters()).device
        else:
            device = torch.device("cpu")
    
        if isinstance(x, np.ndarray):
            x_np = np.asarray(x, dtype=NUMPY_DTYPE).copy()
            x_torch = torch.from_numpy(x_np).to(device=device, dtype=self.dtype)
        elif torch.is_tensor(x):
            x_torch = x.to(device=device, dtype=self.dtype)
        else:
            x_np = np.asarray(x, dtype=NUMPY_DTYPE).copy()
            x_torch = torch.from_numpy(x_np).to(device=device, dtype=self.dtype)
    
        expert_outputs = self.predict_expert_outputs(x_torch)
        o_pred = expert_outputs.mean(dim=1)
    
        return o_pred.detach().cpu().numpy().astype(NUMPY_DTYPE, copy=False)
        
        
    def predict(self, x):
            """
            Final prediction averaging over experts and applying threshold to obtain class labels.
            """
            QP_tr = self.QP_tr
            Q_tr = self.QC * self.QP_tr
            
            Q_RB_C = min(1.0, self.Q_RB_C)  # self.Q_RB_C #
            Q_RB_S = min(1.0, self.Q_RB_S)  # self.Q_RB_S #
            
            QR_tr = max(1.0, QP_tr / (Q_RB_C * Q_RB_S))
            Q_ratio = (Q_tr / (Q_tr + QR_tr))  # 1/2 #
                   
            # Get the averaged expert predictions (o_pred)
            o_pred = self.forward(x)
        
            # Apply thresholding to get the final class labels
            y = np.ones_like(o_pred)
            eta_th = (2.0 * (self.alpha + (1.0 - self.alpha - self.beta) * Q_ratio) - 1.0)
            y[o_pred < eta_th] = self.bin_format
            
            return y.astype(int)
        
    def predict_proba(self, x):
            """
            Returns the predicted probability (o_pred) without applying threshold logic.
            This method is equivalent to predict but returns o_pred instead of y.
            """
            # Get the averaged expert predictions (o_pred)
            o_pred = self.forward(x)
            o_pred = np.asarray(o_pred, dtype=NUMPY_DTYPE)
        
            # Map from [-1,1] -> [0,1]
            p1 = (o_pred + 1.0) / 2.0
            p1 = np.clip(p1, 0.0, 1.0)
        
            p0 = 1.0 - p1
            return np.column_stack([p0, p1]).astype(NUMPY_DTYPE, copy=False)