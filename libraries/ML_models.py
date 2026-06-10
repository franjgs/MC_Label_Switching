#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:58:28 2025

@author: fran
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F

import time
import logging


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    logging.getLogger("catboost").setLevel(logging.ERROR)
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None
    
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy import special

# --- Custom Libraries ---
from libraries.functions import generate_batches, compute_imbalance_ratio
from libraries.optimizers import LBFGSScipy, train_model, weighted_mse_loss

from libraries.optimizers import LOSS_FUNCTIONS

logger = logging.getLogger(__name__)

TORCH_DTYPE = torch.float32
NUMPY_DTYPE = np.float32
eps=np.finfo(float).eps

class TorchLSBase(nn.Module):
    """
    Common utilities for Torch-based learners used inside the Label Switching
    ensemble framework.

    This base class centralizes:
    - dtype and device handling,
    - NumPy to Torch conversion,
    - loss-function resolution,
    - optimizer construction,
    - common prediction helpers.

    Child classes are expected to implement `forward()`.
    """

    def __init__(self, alpha=0.0, beta=0.0, dtype=TORCH_DTYPE):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.model_dtype = dtype

    def _get_first_param(self):
        """
        Return the first trainable parameter, or None if the module has no parameters.
        """
        return next(self.parameters(), None)

    def _get_model_device(self):
        """
        Return the device used by the model parameters.
        """
        first_param = self._get_first_param()
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    def _get_model_dtype(self):
        """
        Return the dtype used by the model parameters.
        """
        first_param = self._get_first_param()
        if first_param is None:
            return self.model_dtype
        return first_param.dtype

    def _to_tensor(self, x, allow_none=False):
        """
        Convert an input object to a Torch tensor aligned with model dtype and device.
        """
        if x is None:
            if allow_none:
                return None
            raise TypeError("Input cannot be None.")

        device = self._get_model_device()
        model_dtype = self._get_model_dtype()

        if isinstance(x, np.ndarray):
            x_np = np.asarray(x).copy()
            return torch.from_numpy(x_np).to(device=device, dtype=model_dtype)

        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=model_dtype)

        raise TypeError("Input must be torch.Tensor or np.ndarray.")

    def _prepare_training_data(self, x_train, y_train, sample_weight=None):
        """
        Convert training inputs, targets, and optional sample weights to aligned tensors.
        """
        x_train = self._to_tensor(x_train)
        y_train = self._to_tensor(y_train)

        if sample_weight is None:
            sample_weight = torch.ones_like(
                y_train,
                dtype=self._get_model_dtype(),
                device=self._get_model_device()
            )
        else:
            sample_weight = self._to_tensor(sample_weight)

        return x_train, y_train, sample_weight

    def _resolve_loss_fn(self, loss_fn):
        """
        Resolve the loss function from either a callable or a registered name.
        """
        if callable(loss_fn):
            return loss_fn
        return LOSS_FUNCTIONS.get(loss_fn, weighted_mse_loss)

    def _build_lbfgs_optimizer(self, max_iter=150, max_eval=150):
        """
        Build the legacy SciPy-based L-BFGS optimizer.
        """
        return LBFGSScipy(
            self.parameters(),
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=1e-04,
            tolerance_change=10e6 * eps,
            history_size=10
        )

    def _build_standard_optimizer(self, optim, learning_rate=None, weight_decay=0.0):
        """
        Build a standard first-order optimizer following the legacy defaults
        used in the original codebase.
        """
        if optim == 'adam':
            lr = 0.001 if learning_rate is None else learning_rate
            return torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        if optim == 'adamw':
            lr = 0.001 if learning_rate is None else learning_rate
            return torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        if optim == 'rmsprop':
            lr = 0.003 if learning_rate is None else learning_rate
            return torch.optim.RMSprop(self.parameters(), lr=lr)

        if optim == 'sgd':
            lr = 0.1 if learning_rate is None else learning_rate
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        if optim == 'adagrad':
            lr = 0.01 if learning_rate is None else learning_rate
            return torch.optim.Adagrad(self.parameters(), lr=lr)

        if optim == 'adadelta':
            lr = 0.01 if learning_rate is None else learning_rate
            return torch.optim.Adadelta(self.parameters(), lr=lr)

        raise ValueError(f"Unsupported optimizer: {optim}")

    def predict(self, x):
        """
        Return the continuous compressed output.
        """
        self.eval()
        x = self._to_tensor(x)

        with torch.no_grad():
            outputs = self.forward(x)

        return outputs.detach().cpu().numpy().reshape(-1)

    def predict_proba(self, x):
        """
        Map the compressed output from [-1, 1] into [0, 1].
        """
        self.eval()
        x = self._to_tensor(x)

        with torch.no_grad():
            outputs = self.forward(x)
            prob_pos = torch.clamp((outputs + 1.0) / 2.0, 0.0, 1.0)
            prob_neg = 1.0 - prob_pos

        return torch.cat([prob_neg, prob_pos], dim=1).cpu().numpy()


class ActivationMixin:
    """
    Common utilities for callable hidden activation functions.
    """

    VALID_ACTIVATIONS = {torch.relu, torch.tanh, torch.sigmoid}

    def _validate_activation_fn(self, activation_fn):
        """
        Validate that the activation function is one of the supported callables.
        """
        if activation_fn not in self.VALID_ACTIVATIONS:
            valid_names = sorted(fn.__name__ for fn in self.VALID_ACTIVATIONS)
            raise ValueError(
                f"Unsupported activation function "
                f"'{getattr(activation_fn, '__name__', activation_fn)}'. "
                f"Supported activations: {valid_names}."
            )

    def _apply_hidden_activation(self, x):
        """
        Apply the configured callable hidden activation.
        """
        return self.activation_fn(x)


class AsymmetricOutputMixin:
    """
    Shared asymmetric output compression for LS-compatible Torch models.
    """

    def _apply_asymmetric_output(self, z, output_act=1):
        """
        Apply the asymmetric compressed output transformation.
        """
        if output_act == 1:
            return torch.where(
                z < 0,
                torch.tanh(z) * (1 - 2 * self.alpha),
                torch.tanh(z) * (1 - 2 * self.beta),
            )

        if output_act == 2:
            return torch.where(
                z < 0,
                torch.tanh(z / (1 - 2 * self.alpha)) * (1 - 2 * self.alpha),
                torch.tanh(z / (1 - 2 * self.beta)) * (1 - 2 * self.beta),
            )

        raise ValueError("Invalid output_act. Choose 1 or 2.")


class LegacyLinearInitMixin:
    """
    Initialization helpers that reproduce the legacy behavior of the original code.
    """

    def _init_linear_weights_xavier_keep_bias(self):
        """
        Apply Xavier uniform initialization to all linear weights while preserving
        the default PyTorch bias initialization.
        """
        def weight_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        self.apply(weight_init)

    def _init_linear_weights_xavier_zero_bias(self, layers):
        """
        Apply Xavier uniform initialization to linear weights and set all biases
        to zero. This reproduces the original deep MLP implementation.
        """
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class LogisticRegressionTorch(TorchLSBase, AsymmetricOutputMixin, LegacyLinearInitMixin):
    """
    Flexible linear expert for LSEnsemble.

    This class is the linear counterpart of AsymmetricMLP. It keeps the same
    training interface and optimization modes, while allowing both:

    1. Legacy LSE/ALSE behavior:
       - Linear layer followed by asymmetric compressed tanh output.
       - Typical loss: weighted_mse_loss.
       - Targets usually encoded as {-1, +1}, possibly after label switching.

    2. sklearn-like logistic regression behavior:
       - Linear logits during training.
       - BCEWithLogits loss.
       - Targets internally converted to {0, 1}.
       - Optional class_weight="balanced".
       - Optional L2 regularization controlled by C.
       - Optional asymmetric compressed output outside fit() for LSEnsemble.

    Optimization modes
    ------------------
    - optim="lbfgs":
      Full-batch LBFGS training using the project legacy LBFGSScipy wrapper.

    - optim="warm_start_lbfgs":
      Minibatch warm-start with a standard optimizer followed by full-batch
      LBFGS refinement.

    - optim in {"adam", "adamw", "rmsprop", "sgd", "adagrad", "adadelta"}:
      Minibatch training with the corresponding Torch optimizer.
    """

    def __init__(
        self,
        input_dim,
        alpha,
        beta,
        loss_fn=weighted_mse_loss,
        optim="lbfgs",
        max_iter=None,
        learning_rate=None,
        mode="random",
        batch_size=None,
        dtype=TORCH_DTYPE,
        output_act=1,
        n_epoch=50,
        n_batch=256,
        output_mode="asymmetric_tanh",
        train_output_mode=None,
        eval_output_mode=None,
        C=None,
        penalty="l2",
        class_weight=None,
        weight_decay=0.0,
        debug=False,
        early_stopping=True,
        es_patience=10,
        es_tol=1e-4,
        lbfgs_max_iter=150,
        lbfgs_max_eval=150,
        warm_start=None,
        warm_start_optim="rmsprop",
        warm_start_n_epoch=300,
        warm_start_learning_rate=0.003,
        warm_start_weight_decay=0.0,
        warm_start_debug=None,
        random_state=None
    ):
        """
        Initialize the flexible linear expert.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        alpha : float
            Label switching factor from majority to minority class.

        beta : float
            Label switching factor from minority to majority class.

        loss_fn : callable or str
            Loss function. Use "MSE" for legacy LSE behavior and
            "BCE_logit" for sklearn-like logistic regression.

        output_mode : str
            Default output mode. Used as fallback for train/eval modes.

        train_output_mode : str or None
            Output mode used during fit(). If None, it is inferred:
            - "logits" for BCE_logit.
            - output_mode otherwise.

        eval_output_mode : str or None
            Output mode used outside fit(). If None, output_mode is used.

        C : float or None
            Inverse L2 regularization strength, sklearn-style.

        class_weight : dict, "balanced" or None
            Optional class weighting. class_weight="balanced" follows the
            sklearn convention.
        """
        super().__init__(alpha=alpha, beta=beta, dtype=dtype)

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Backward compatibility with the previous constructor.
        if max_iter is not None:
            n_epoch = max_iter
        if batch_size is not None:
            n_batch = batch_size

        self.input_dim = input_dim
        self.output_act = output_act

        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.optim = optim
        self.mode = mode
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.debug = debug

        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_tol = es_tol

        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval

        self.output_mode = output_mode

        self.train_output_mode = train_output_mode
        self.eval_output_mode = eval_output_mode

        if self.train_output_mode is None:
            if self._is_bce_logit_loss(loss_fn):
                self.train_output_mode = "logits"
            else:
                self.train_output_mode = output_mode

        if self.eval_output_mode is None:
            self.eval_output_mode = output_mode

        self._fit_in_progress = False

        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight

        warm_start = {} if warm_start is None else dict(warm_start)

        self.warm_start_optim = warm_start.get("optim", warm_start_optim)
        self.warm_start_n_epoch = warm_start.get("n_epoch", warm_start_n_epoch)
        self.warm_start_learning_rate = warm_start.get(
            "learning_rate",
            warm_start_learning_rate
        )
        self.warm_start_weight_decay = warm_start.get(
            "weight_decay",
            warm_start_weight_decay
        )
        self.warm_start_debug = warm_start.get("debug", warm_start_debug)

        self.linear = nn.Linear(input_dim, 1)
        self._init_linear_weights_xavier_keep_bias()

        self.model_dtype = dtype
        self.to(self.model_dtype)

        self._validate_configuration()

    def _validate_configuration(self):
        """
        Validate configuration values.
        """
        valid_output_modes = {"asymmetric_tanh", "logits", "sigmoid"}

        for name, value in {
            "output_mode": self.output_mode,
            "train_output_mode": self.train_output_mode,
            "eval_output_mode": self.eval_output_mode,
        }.items():
            if value not in valid_output_modes:
                raise ValueError(
                    f"Invalid {name}='{value}'. "
                    f"Available values: {sorted(valid_output_modes)}"
                )

        if self.penalty not in {None, "none", "l2"}:
            raise ValueError(
                "Only penalty=None, penalty='none' and penalty='l2' are supported."
            )

        if self.C is not None and self.C <= 0:
            raise ValueError("C must be strictly positive when provided.")

        if self._is_bce_logit_loss(self.loss_fn) and self.train_output_mode != "logits":
            raise ValueError(
                "loss_fn='BCE_logit' requires train_output_mode='logits'. "
                "BCEWithLogits must receive raw logits during training."
            )

    def _is_bce_logit_loss(self, loss_fn):
        """
        Return True when the selected loss is BCE with logits.
        """
        if isinstance(loss_fn, str):
            return loss_fn.lower() in {
                "bce_logit",
                "bce_logits",
                "bcewithlogits",
                "bcewithlogitsloss"
            }

        return False

    def _resolve_logreg_loss_fn(self, loss_fn):
        """
        Resolve the effective loss function for this linear expert.

        BCE_logit is handled inside this class because it needs:
        - target conversion from {-1, +1} to {0, 1},
        - optional class_weight,
        - optional C-based L2 regularization.
        """
        if self._is_bce_logit_loss(loss_fn):
            return self._bce_logit_loss_with_options

        resolved = self._resolve_loss_fn(loss_fn)

        if self.C is not None and self.penalty not in {None, "none"}:
            return self._wrap_loss_with_l2(resolved)

        return resolved

    def _targets_to_zero_one(self, target):
        """
        Convert binary targets from {-1, +1} or {0, 1} to {0, 1}.
        """
        return torch.where(
            target > 0,
            torch.ones_like(target),
            torch.zeros_like(target)
        )

    def _class_weight_tensor(self, target_01):
        """
        Build a per-sample class-weight tensor.

        For class_weight="balanced", the sklearn convention is used:

            n_samples / (n_classes * n_samples_class)
        """
        if self.class_weight is None:
            return torch.ones_like(target_01)

        class_weight = self.class_weight

        if isinstance(class_weight, str):
            class_weight = class_weight.strip().lower()

        if class_weight == "balanced":
            target_flat = target_01.view(-1)

            n_samples = target_flat.numel()
            n_pos = torch.sum(target_flat == 1)
            n_neg = torch.sum(target_flat == 0)

            if n_pos == 0 or n_neg == 0:
                return torch.ones_like(target_01)

            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)

            weights = torch.where(
                target_flat == 1,
                w_pos,
                w_neg
            ).view(-1, 1)

            return weights.to(
                device=target_01.device,
                dtype=target_01.dtype
            )

        if isinstance(class_weight, dict):
            w0 = float(class_weight.get(0, class_weight.get(-1, 1.0)))
            w1 = float(class_weight.get(1, 1.0))

            return torch.where(
                target_01 > 0,
                torch.full_like(target_01, w1),
                torch.full_like(target_01, w0)
            )

        raise ValueError(f"Unsupported class_weight={self.class_weight}")

    def _l2_penalty(self, normalizer=None):
        """
        Compute sklearn-like L2 penalty controlled by C.

        The intercept is intentionally not regularized.

        Important
        ---------
        When the data loss is normalized by the sum of effective sample
        weights, the L2 term is normalized by the same quantity. This avoids
        over-regularizing when C is small.
        """
        zero = torch.zeros(
            (),
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype
        )

        if self.C is None:
            return zero

        if self.penalty in {None, "none"}:
            return zero

        if self.penalty != "l2":
            raise ValueError(f"Unsupported penalty={self.penalty}")

        if normalizer is None:
            normalizer = torch.tensor(
                1.0,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype
            )

        normalizer = torch.clamp(normalizer, min=1.0)

        return 0.5 * torch.sum(self.linear.weight ** 2) / (
            float(self.C) * normalizer
        )

    def _bce_logit_loss_with_options(self, inputs, target, weights=None):
        """
        Compute BCEWithLogits loss with optional class weights and L2.

        Effective weights follow sklearn's convention:

            effective_weight[i] = sample_weight[i] * class_weight[y_i]

        This method never modifies the input weights tensor.
        """
        target_01 = self._targets_to_zero_one(target)

        if weights is None:
            weights_ext = torch.ones_like(target_01)
        else:
            weights_ext = weights.to(
                device=target_01.device,
                dtype=target_01.dtype
            ).view(-1, 1).clone()

        class_weights = self._class_weight_tensor(target_01)
        weights_eff = weights_ext * class_weights

        normalizer = torch.clamp(weights_eff.sum(), min=1.0)

        data_loss = F.binary_cross_entropy_with_logits(
            inputs,
            target_01,
            weight=weights_eff,
            reduction="sum"
        )

        data_loss = data_loss / normalizer
        l2_loss = self._l2_penalty(normalizer=normalizer)

        return data_loss + l2_loss

    def _wrap_loss_with_l2(self, base_loss_fn):
        """
        Add optional C-based L2 regularization to a legacy/custom loss.

        For legacy losses, the normalizer is estimated from the provided
        sample weights.
        """
        def wrapped_loss(outputs, labels, weights=None):
            """
            Compute wrapped loss plus normalized L2 penalty.
            """
            data_loss = base_loss_fn(outputs, labels, weights)

            if hasattr(data_loss, "ndim") and data_loss.ndim > 0:
                data_loss = torch.mean(data_loss)

            if weights is None:
                normalizer = torch.tensor(
                    float(labels.numel()),
                    device=outputs.device,
                    dtype=outputs.dtype
                )
            else:
                weights_ext = weights.to(
                    device=outputs.device,
                    dtype=outputs.dtype
                ).view(-1, 1).clone()

                normalizer = torch.clamp(weights_ext.sum(), min=1.0)

            return data_loss + self._l2_penalty(normalizer=normalizer)

        return wrapped_loss

    def forward(self, x):
        """
        Run the forward pass through the linear layer.

        During fit(), train_output_mode is used. Outside fit(),
        eval_output_mode is used.

        Returns
        -------
        torch.Tensor
            Output according to the active output mode:
            - "logits": raw linear logits.
            - "sigmoid": sigmoid(logits).
            - "asymmetric_tanh": asymmetric compressed tanh output.
        """
        x = x.to(
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype
        )

        z = self.linear(x)

        if getattr(self, "_fit_in_progress", False):
            active_output_mode = self.train_output_mode
        else:
            active_output_mode = self.eval_output_mode

        if active_output_mode == "logits":
            return z

        if active_output_mode == "sigmoid":
            return torch.sigmoid(z)

        if active_output_mode == "asymmetric_tanh":
            return self._apply_asymmetric_output(
                z,
                output_act=self.output_act
            )

        raise ValueError(f"Unsupported output mode: {active_output_mode}")

    def _fit_with_standard_optimizer(
        self,
        x_train,
        y_train,
        sample_weight,
        loss_fn,
        optim,
        n_epoch,
        n_batch,
        mode,
        learning_rate,
        weight_decay,
        debug,
        loss_already_normalized=False
    ):
        """
        Train the model with a standard first-order optimizer.
        """
        optimizer = self._build_standard_optimizer(
            optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        train_model(
            x_train,
            y_train,
            self,
            loss_fn,
            optimizer,
            sample_weight,
            num_epochs=n_epoch,
            batch_size=n_batch,
            mode=mode,
            lbfgs=False,
            debug=debug,
            early_stopping=self.early_stopping,
            es_patience=self.es_patience,
            es_tol=self.es_tol,
            normalize_monitor_loss=not loss_already_normalized
        )

    def _fit_with_lbfgs(
        self,
        x_train,
        y_train,
        sample_weight,
        loss_fn,
        n_epoch,
        mode,
        debug,
        loss_already_normalized=False
    ):
        """
        Train the model with full-batch LBFGS.

        This uses TorchLSBase._build_lbfgs_optimizer, preserving the legacy
        LBFGSScipy configuration used by the project.
        """
        optimizer = self._build_lbfgs_optimizer(
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_eval
        )

        train_model(
            x_train,
            y_train,
            self,
            loss_fn,
            optimizer,
            sample_weight,
            num_epochs=n_epoch,
            batch_size=None,
            mode=mode,
            lbfgs=True,
            debug=debug,
            early_stopping=self.early_stopping,
            es_patience=self.es_patience,
            es_tol=self.es_tol,
            normalize_monitor_loss=not loss_already_normalized
        )

    def fit(
        self,
        x_train,
        y_train,
        sample_weight=None,
        n_epoch=None,
        n_batch=None,
        optim=None,
        mode=None,
        loss_fn=None,
        debug=None
    ):
        """
        Fit the flexible linear expert.

        If n_epoch, n_batch, optim, mode, loss_fn or debug are not provided,
        the values stored in the expert are used.
        """
        x_train, y_train, sample_weight = self._prepare_training_data(
            x_train,
            y_train,
            sample_weight
        )

        # Avoid accidental in-place contamination of sample weights during
        # repeated LBFGS closure calls or interactive debugging sessions.
        sample_weight = sample_weight.clone().detach()

        effective_n_epoch = self.n_epoch if n_epoch is None else n_epoch
        effective_n_batch = self.n_batch if n_batch is None else n_batch
        effective_optim = self.optim if optim is None else optim
        effective_mode = self.mode if mode is None else mode
        effective_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        effective_debug = self.debug if debug is None else debug

        loss_already_normalized = self._is_bce_logit_loss(effective_loss_fn)
        resolved_loss_fn = self._resolve_logreg_loss_fn(effective_loss_fn)

        if sample_weight.max() > 10:
            logger.warning(
                "Unexpected large external sample_weight before LogReg training | "
                "min=%.6f | max=%.6f | mean=%.6f. "
                "This may indicate that class weights were already applied upstream.",
                float(sample_weight.min().detach().cpu().item()),
                float(sample_weight.max().detach().cpu().item()),
                float(sample_weight.mean().detach().cpu().item())
            )

        self._fit_in_progress = True

        try:
            if effective_optim == "lbfgs":
                lbfgs_t0 = time.perf_counter()

                self._fit_with_lbfgs(
                    x_train=x_train,
                    y_train=y_train,
                    sample_weight=sample_weight,
                    loss_fn=resolved_loss_fn,
                    n_epoch=effective_n_epoch,
                    mode=effective_mode,
                    debug=effective_debug,
                    loss_already_normalized=loss_already_normalized
                )

                lbfgs_seconds = time.perf_counter() - lbfgs_t0
                logger.debug("LBFGS timing | lbfgs=%.2fs", lbfgs_seconds)

            elif effective_optim == "warm_start_lbfgs":
                warm_debug = (
                    effective_debug
                    if self.warm_start_debug is None
                    else self.warm_start_debug
                )

                warm_start_t0 = time.perf_counter()

                self._fit_with_standard_optimizer(
                    x_train=x_train,
                    y_train=y_train,
                    sample_weight=sample_weight,
                    loss_fn=resolved_loss_fn,
                    optim=self.warm_start_optim,
                    n_epoch=self.warm_start_n_epoch,
                    n_batch=effective_n_batch,
                    mode=effective_mode,
                    learning_rate=self.warm_start_learning_rate,
                    weight_decay=self.warm_start_weight_decay,
                    debug=warm_debug,
                    loss_already_normalized=loss_already_normalized
                )

                warm_start_seconds = time.perf_counter() - warm_start_t0

                lbfgs_t0 = time.perf_counter()

                self._fit_with_lbfgs(
                    x_train=x_train,
                    y_train=y_train,
                    sample_weight=sample_weight,
                    loss_fn=resolved_loss_fn,
                    n_epoch=effective_n_epoch,
                    mode=effective_mode,
                    debug=effective_debug,
                    loss_already_normalized=loss_already_normalized
                )

                lbfgs_seconds = time.perf_counter() - lbfgs_t0
                total_seconds = warm_start_seconds + lbfgs_seconds

                logger.debug(
                    "Warm-start LBFGS timing | warm_start=%.2fs | lbfgs=%.2fs | total=%.2fs",
                    warm_start_seconds,
                    lbfgs_seconds,
                    total_seconds
                )

            else:
                std_opt_t0 = time.perf_counter()

                self._fit_with_standard_optimizer(
                    x_train=x_train,
                    y_train=y_train,
                    sample_weight=sample_weight,
                    loss_fn=resolved_loss_fn,
                    optim=effective_optim,
                    n_epoch=effective_n_epoch,
                    n_batch=effective_n_batch,
                    mode=effective_mode,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    debug=effective_debug,
                    loss_already_normalized=loss_already_normalized
                )

                std_opt_seconds = time.perf_counter() - std_opt_t0
                logger.debug("Standard Optimizer timing | std_opt=%.2fs", std_opt_seconds)

        finally:
            self._fit_in_progress = False

        return self


class AsymmetricMLP(TorchLSBase, ActivationMixin, AsymmetricOutputMixin, LegacyLinearInitMixin):
    """
    Legacy shallow asymmetric MLP with exactly one hidden layer.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        alpha,
        beta,
        dropout_prob=0.0,
        activation_fn=torch.tanh,
        output_act=1,
        dtype=TORCH_DTYPE,
        n_epoch=50,
        n_batch=256,
        optim="lbfgs",
        mode="random",
        loss_fn=weighted_mse_loss,
        learning_rate=None,
        weight_decay=0.0,
        debug=False,
        early_stopping=True,
        es_patience=10,
        es_tol=1e-4,
        lbfgs_max_iter=150,
        lbfgs_max_eval=150,
        warm_start=None,
        warm_start_optim="rmsprop",
        warm_start_n_epoch=300,
        warm_start_learning_rate=0.003,
        warm_start_weight_decay=0.0,
        warm_start_debug=None
    ):
        """
        Initialize a shallow asymmetric MLP expert.

        Training parameters are stored in the expert so that LSEnsemble can
        configure all learners through a common interface.

        Optimization modes
        ------------------
        - optim="lbfgs":
          Full-batch LBFGS training. n_batch and mode are ignored.

        - optim in {"adam", "adamw", "rmsprop", "sgd"}:
          Minibatch training. n_batch and mode are used.

        - optim="warm_start_lbfgs":
          First minibatch training using n_batch and mode, then full-batch
          LBFGS refinement without reinitializing the weights.

        The optional warm_start dictionary can override the warm-start optimizer,
        number of epochs, learning rate, weight decay and debug flag.
        """
        super().__init__(alpha=alpha, beta=beta, dtype=dtype)

        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.output_act = output_act
        self.dropout_prob = dropout_prob

        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.optim = optim
        self.mode = mode
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.debug = debug

        self.early_stopping = early_stopping
        self.es_patience = es_patience
        self.es_tol = es_tol

        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval

        # Warm-start configuration.
        # Supports both the new nested YAML block `warm_start` and the older
        # flat keyword arguments for backward compatibility.
        warm_start = {} if warm_start is None else dict(warm_start)

        self.warm_start_optim = warm_start.get("optim", warm_start_optim)
        self.warm_start_n_epoch = warm_start.get("n_epoch", warm_start_n_epoch)
        self.warm_start_learning_rate = warm_start.get(
            "learning_rate",
            warm_start_learning_rate
        )
        self.warm_start_weight_decay = warm_start.get(
            "weight_decay",
            warm_start_weight_decay
        )
        self.warm_start_debug = warm_start.get("debug", warm_start_debug)

        self._validate_activation_fn(self.activation_fn)

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout_prob)

        self._init_linear_weights_xavier_keep_bias()

        self.model_dtype = dtype
        self.to(self.model_dtype)

    def forward(self, x):
        """
        Run the forward pass through one hidden layer and asymmetric output.
        """
        x = x.to(
            device=self.hidden0[0].weight.device,
            dtype=self.hidden0[0].weight.dtype
        )

        o = self.activation_fn(self.hidden0(x))
        o = self.dropout(o)
        z = self.out(o)

        return self._apply_asymmetric_output(z, output_act=self.output_act)

    def _fit_with_standard_optimizer(
        self,
        x_train,
        y_train,
        sample_weight,
        loss_fn,
        optim,
        n_epoch,
        n_batch,
        mode,
        learning_rate,
        weight_decay,
        debug
    ):
        """
        Train the model with a standard first-order optimizer.

        This helper is used both for direct minibatch training and for the
        warm-start phase before LBFGS.
        """
        optimizer = self._build_standard_optimizer(
            optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        train_model(
            x_train,
            y_train,
            self,
            loss_fn,
            optimizer,
            sample_weight,
            num_epochs=n_epoch,
            batch_size=n_batch,
            mode=mode,
            lbfgs=False,
            debug=debug,
            early_stopping=self.early_stopping,
            es_patience=self.es_patience,
            es_tol=self.es_tol
        )

    def _fit_with_lbfgs(
        self,
        x_train,
        y_train,
        sample_weight,
        loss_fn,
        n_epoch,
        mode,
        debug
    ):
        """
        Train the model with full-batch LBFGS.

        The current model weights are used as starting point, so this method can
        be called after a previous optimizer to perform warm-start fine-tuning.
        """
        optimizer = self._build_lbfgs_optimizer(
            max_iter=self.lbfgs_max_iter,
            max_eval=self.lbfgs_max_eval
        )

        train_model(
            x_train,
            y_train,
            self,
            loss_fn,
            optimizer,
            sample_weight,
            num_epochs=n_epoch,
            batch_size=None,
            mode=mode,
            lbfgs=True,
            debug=debug,
            early_stopping=self.early_stopping,
            es_patience=self.es_patience,
            es_tol=self.es_tol
        )

    def fit(
        self,
        x_train,
        y_train,
        sample_weight=None,
        n_epoch=None,
        n_batch=None,
        optim=None,
        mode=None,
        loss_fn=None,
        debug=None
    ):
        """
        Fit the shallow asymmetric MLP.

        If n_epoch, n_batch, optim, mode, loss_fn or debug are not provided
        explicitly, the values stored in the expert are used. This allows
        LSEnsemble to configure all experts through a common interface.

        Supported optimization modes
        ----------------------------
        - 'lbfgs':
          Full-batch LBFGS. n_batch is ignored.

        - 'warm_start_lbfgs':
          Minibatch warm-start followed by full-batch LBFGS refinement.
          The warm-start phase uses n_batch and mode. The LBFGS phase ignores
          n_batch.

        - any standard optimizer name:
          Minibatch training. n_batch and mode are used.
        """
        x_train, y_train, sample_weight = self._prepare_training_data(
            x_train,
            y_train,
            sample_weight
        )

        effective_n_epoch = self.n_epoch if n_epoch is None else n_epoch
        effective_n_batch = self.n_batch if n_batch is None else n_batch
        effective_optim = self.optim if optim is None else optim
        effective_mode = self.mode if mode is None else mode
        effective_loss_fn = self.loss_fn if loss_fn is None else loss_fn
        effective_debug = self.debug if debug is None else debug

        resolved_loss_fn = self._resolve_loss_fn(effective_loss_fn)

        if effective_optim == "lbfgs":
            lbfgs_t0 = time.perf_counter()
            self._fit_with_lbfgs(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sample_weight,
                loss_fn=resolved_loss_fn,
                n_epoch=effective_n_epoch,
                mode=effective_mode,
                debug=effective_debug
            )
            lbfgs_seconds = time.perf_counter() - lbfgs_t0
            logger.debug("LBFGS timing | lbfgs=%.2fs", lbfgs_seconds)

        elif effective_optim == "warm_start_lbfgs":
            # First phase: minibatch optimizer using n_batch and mode.
            # Second phase: full-batch LBFGS refinement without reinitialization.
            # LBFGS ignores n_batch by construction.
            warm_debug = (
                effective_debug
                if self.warm_start_debug is None
                else self.warm_start_debug
            )

            warm_start_t0 = time.perf_counter()

            self._fit_with_standard_optimizer(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sample_weight,
                loss_fn=resolved_loss_fn,
                optim=self.warm_start_optim,
                n_epoch=self.warm_start_n_epoch,
                n_batch=effective_n_batch,
                mode=effective_mode,
                learning_rate=self.warm_start_learning_rate,
                weight_decay=self.warm_start_weight_decay,
                debug=warm_debug
            )

            warm_start_seconds = time.perf_counter() - warm_start_t0

            lbfgs_t0 = time.perf_counter()

            self._fit_with_lbfgs(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sample_weight,
                loss_fn=resolved_loss_fn,
                n_epoch=effective_n_epoch,
                mode=effective_mode,
                debug=effective_debug
            )

            lbfgs_seconds = time.perf_counter() - lbfgs_t0
            total_seconds = warm_start_seconds + lbfgs_seconds

            logger.debug(
                "Warm-start LBFGS timing | warm_start=%.2fs | lbfgs=%.2fs | total=%.2fs",
                warm_start_seconds,
                lbfgs_seconds,
                total_seconds
            )

        else:
            std_opt_t0 = time.perf_counter()
            self._fit_with_standard_optimizer(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sample_weight,
                loss_fn=resolved_loss_fn,
                optim=effective_optim,
                n_epoch=effective_n_epoch,
                n_batch=effective_n_batch,
                mode=effective_mode,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                debug=effective_debug
            )
            std_opt_seconds = time.perf_counter() - std_opt_t0
            logger.debug( "Standard Optimizer timing | std_opt=%.2fs", std_opt_seconds)

        return self

class DeepAsymmetricMLP(TorchLSBase, ActivationMixin, AsymmetricOutputMixin, LegacyLinearInitMixin):
    """
    Deep asymmetric MLP with one or more hidden layers.

    This class reproduces the original deep MLP behavior while using callable
    activation functions.
    """

    def __init__(
        self,
        input_dim,
        hidden_layer_sizes=(100,),
        activation_fn=torch.relu,
        alpha=0.0,
        beta=0.0,
        alpha_tr=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=200,
        optim='adam',
        output_act=1,
        dtype=TORCH_DTYPE
    ):
        super().__init__(alpha=alpha, beta=beta, dtype=dtype)

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation_fn = activation_fn
        self.alpha_tr = alpha_tr
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.optim = optim
        self.output_act = output_act

        self._validate_activation_fn(self.activation_fn)

        layer_sizes = [input_dim] + list(self.hidden_layer_sizes) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]).to(dtype=dtype)
            for i in range(len(layer_sizes) - 1)
        ])

        self._init_linear_weights_xavier_zero_bias(self.layers)

    def forward(self, x):
        """
        Run the forward pass through multiple hidden layers and asymmetric output.
        """
        x = x.to(device=self.layers[0].weight.device, dtype=self.layers[0].weight.dtype)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self._apply_hidden_activation(x)

        z = self.layers[-1](x)
        return self._apply_asymmetric_output(z, output_act=self.output_act)

    def fit(self, x_train, y_train, sample_weight=None):
        """
        Fit the deep asymmetric MLP using the original optimization behavior.
        """
        x_train, y_train, sample_weight = self._prepare_training_data(
            x_train,
            y_train,
            sample_weight
        )

        if y_train.ndimension() == 2 and y_train.shape[1] == 1:
            y_train_1d = y_train.squeeze(1)
        else:
            y_train_1d = y_train

        if self.optim == 'adam':
            optimizer = self._build_standard_optimizer(
                'adam',
                learning_rate=self.learning_rate_init,
                weight_decay=self.alpha_tr
            )

            train_model(
                x_train,
                y_train_1d,
                self,
                weighted_mse_loss,
                optimizer,
                sample_weight,
                num_epochs=self.max_iter,
                batch_size=self.batch_size,
                mode='random',
                lbfgs=False,
                debug=False
            )
        elif self.optim == 'lbfgs':
            optimizer = self._build_lbfgs_optimizer(
                max_iter=self.max_iter,
                max_eval=self.max_iter
            )

            train_model(
                x_train,
                y_train_1d,
                self,
                weighted_mse_loss,
                optimizer,
                sample_weight,
                num_epochs=self.max_iter,
                batch_size=None,
                mode='random',
                lbfgs=True,
                debug=False
            )
        else:
            raise ValueError("Only 'adam' and 'lbfgs' are supported solvers.")

        return self
    
class CalibratedBooster(nn.Module):
    """
    Flexible wrapper for boosting/linear models with asymmetric output compression,
    tailored for Asymmetric Label Switched Ensemble (ALSE / LSEnsemble).

    Goal: approximate the compressed switched posterior o_S(X) = 2 * Pr_S(C1|X) - 1,
    bounded approximately in [-(1-2*alpha), +(1-2*beta)].

    Recommendation for ALSE:
    - Use task='regression' (default) to directly regress on the continuous compressed targets.
      This preserves full information about confidence/magnitude.
    - Classification mode is provided for experimentation, but it binarizes targets (loses magnitude),
      so it is less aligned with learning a continuous bounded posterior.

    Parameters
    ----------
    model_type : str, {'lgbm', 'xgb', 'cat', 'rf', 'logreg'}, default='lgbm'
        Base model family.
    task : str, {'regression', 'classification'}, default='regression'
        Preferred: 'regression' to directly fit compressed o_S(X) values.
    alpha : float, default=0.0
        Switching rate majority → minority (affects negative bound).
    beta : float, default=0.0
        Switching rate minority → majority (affects positive bound).
    output_act : int, default=1
        Compression function variant (1 or 2).
    calibration_method : str or None, default='isotonic'
        - 'isotonic' recommended for regression (non-parametric adjustment)
        - 'sigmoid' or 'isotonic' for classification
    cv : int, splitter object or None, default=None
        For classification calibration only. None = fit on full data.
    **model_params
        Passed directly to the underlying model.
    """

    def __init__(
        self,
        input_dim,
        model_type='lgbm',
        task='regression',                  # changed default to regression
        alpha=0.0,
        beta=0.0,
        output_act=1,
        calibration_method='isotonic',      # default to isotonic for better range fit
        cv=None,
        **model_params
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.task = task.lower()
        self.alpha = alpha
        self.beta = beta
        self.output_act = output_act
        self.calibration_method = calibration_method
        self.cv = cv
        self.model_params = model_params

        self.model = None
        self.calibrator = None
        self._is_fitted = False

        if self.task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        self._validate_model_type()

    def _validate_model_type(self):
        """Validate model_type availability."""
        available = ['lgbm', 'xgb', 'cat', 'rf', 'logreg']
        if self.model_type not in available:
            raise ValueError(f"model_type must be one of {available}")

        if self.model_type == 'lgbm' and LGBMRegressor is None:
            raise ImportError("LightGBM not installed.")
        if self.model_type == 'xgb' and XGBRegressor is None:
            raise ImportError("XGBoost not installed.")
        if self.model_type == 'cat' and CatBoostRegressor is None:
            raise ImportError("CatBoost not installed.")

    def _get_model_class(self):
        """
        Return the estimator class associated with the selected model type and task.
    
        For classification tasks, the returned class must expose a classifier-style
        API, typically including `fit()`, `predict()`, and ideally `predict_proba()`.
    
        For regression tasks, the returned class must expose a regressor-style API
        and be compatible with continuous compressed ALSE-like targets.
    
        Returns
        -------
        type
            Estimator class to instantiate later.
    
        Raises
        ------
        ValueError
            If the model type is unknown or if the task/model combination is not
            supported.
        """
        # Tree boosting models
        if self.model_type == 'lgbm':
            if self.task == 'classification':
                return LGBMClassifier
            if self.task == 'regression':
                return LGBMRegressor
    
        elif self.model_type == 'xgb':
            if self.task == 'classification':
                return XGBClassifier
            if self.task == 'regression':
                return XGBRegressor
    
        elif self.model_type == 'cat':
            if self.task == 'classification':
                return CatBoostClassifier
            if self.task == 'regression':
                return CatBoostRegressor
    
        # Random Forest
        elif self.model_type == 'rf':
            if self.task == 'classification':
                return RandomForestClassifier
            if self.task == 'regression':
                return RandomForestRegressor
    
        # Linear / custom logistic-style models
        elif self.model_type == 'logreg':
            if self.task == 'classification':
                return LogisticRegression
            if self.task == 'regression':
                return LogisticRegressionTorch
    
        raise ValueError(
            f"Unsupported combination: model_type={self.model_type}, task={self.task}"
        )

    def _get_default_params(self):
        """
        Return robust default hyperparameters tailored to each model and task.
        Optimized for speed and handling imbalanced datasets (ALSE-style).
        """
        if self.model_type in ['lgbm', 'xgb', 'cat']:
            # Base common parameters
            base = {
                'random_state': 42,
            }

            if self.model_type == 'lgbm':
                base.update({
                    'boosting_type': 'gbdt',
                    'n_jobs': 1,
                    # 'objective': 'binary' if self.task == 'classification' else 'regression',
                    # 'metric': 'binary_logloss' if self.task == 'classification' else 'rmse',
                    'verbosity': -1,
                    'force_col_wise': True,
                    # Tree Structure
                    'max_depth': -1,
                    'num_leaves': 50,
                    'subsample': 0.8,
                    'min_child_samples': 20,
                    # 'min_child_weight': 1e-3,
                    # Learning
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    # Regularization
                    # 'reg_alpha': 0.0,
                    # 'reg_lambda': 0.0,
                    # Imbalance handling
                    'is_unbalance': True if self.task == 'classification' else False,
                })

            elif self.model_type == 'xgb':
                base.update({
                    'verbosity': 0,
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'reg_lambda': 0.0,
                    'reg_alpha': 0.0,
                    'tree_method': 'hist',
                    'n_jobs': 1,
                })
                # Note: scale_pos_weight is injected dynamically in fit()

            elif self.model_type == 'cat':
                base.update({
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 1.0,
                    'grow_policy': 'Depthwise',
                    'logging_level': 'Silent',
                    'thread_count': 1,
                })
                # Add balance weights for CatBoost
                if self.task == 'classification':
                    base['auto_class_weights'] = 'Balanced'

            # --- Critical: Final Objective Assignment ---
            if self.task == 'regression':
                if self.model_type == 'xgb':
                    base['objective'] = 'reg:squarederror'
                elif self.model_type == 'cat':
                    base['loss_function'] = 'RMSE'
                else: # lgbm
                    base['objective'] = 'regression'
            else: # classification
                if self.model_type == 'xgb':
                    base['objective'] = 'binary:logistic'
                elif self.model_type == 'cat':
                    # Fix: 'binary' is not a valid CatBoost objective
                    base['loss_function'] = 'Logloss' 
                else: # lgbm
                    base['objective'] = 'binary'

            return base

        elif self.model_type == 'rf':
            params = {
                'n_estimators': 60,
                'max_depth': None,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'n_jobs': -1,
                'random_state': 42,
            }
    
            if self.task == 'classification':
                params['class_weight'] = 'balanced_subsample'
    
            return params
        elif self.model_type == 'logreg':
            if self.task == 'classification':
                return {
                    'max_iter': 50000,
                    'C': 0.000001,
                    'random_state': 42,
                    # 'class_weight': 'balanced',
                    'n_jobs': 1,
                    'tol': 0.001,
                    'solver': "saga",
                    'penalty': "l2"
                }
            else: # ALSE LogisticRegressionTorch
                return {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'optim': 'lbfgs',
                    'max_iter': 500,
                    'loss_fn': weighted_mse_loss # Custom loss for switching
                }

        return {}

    def fit(self, X, y, sample_weight=None):
        """
        Fit the base model and, optionally, an isotonic calibrator.
    
        In classification mode:
        - the base estimator is trained on binary labels,
        - the optional calibrator maps raw positive-class probabilities into the
          compressed ALSE-compatible output space.
    
        In regression mode:
        - the base estimator is trained directly on the compressed target,
        - the optional calibrator refines predictions while preserving the same
          theoretical output bounds.
        """
        # Input standardization
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    
        # Theoretical bounds in z-space
        z_lower = - (1.0 - 2.0 * self.alpha)  # negative
        z_upper = (1.0 - 2.0 * self.beta)     # positive
    
        # Imbalance ratio for binary labels.
        ir = compute_imbalance_ratio(y_np > 0)
        
        # effective_weights are used in the first fit and contain only external
        # sample weights. Class imbalance is handled by native model parameters
        # when available, e.g. XGBoost scale_pos_weight="auto" resolved to IR.
        #
        # calib_weights are used for calibration/regression and additionally weight
        # positives by IR. Do not use them in the first classifier fit to avoid
        # double-counting class imbalance.
        effective_weights = (
            np.asarray(sample_weight, dtype=NUMPY_DTYPE).copy()
            if sample_weight is not None
            else np.ones_like(y_np, dtype=NUMPY_DTYPE)
        )
        
        calib_weights = np.where(y_np > 0, ir, 1.0) * effective_weights
        
        # Prepare model.
        model_class = self._get_model_class()
        model_name = model_class.__name__
        
        params = {**self._get_default_params(), **self.model_params}
        
        # Resolve XGBoost imbalance parameter.
        if "XGB" in model_name:
            scale_pos_weight = params.get("scale_pos_weight", None)
        
            if scale_pos_weight == "auto":
                params["scale_pos_weight"] = ir
            elif scale_pos_weight is None and ir > 1:
                params["scale_pos_weight"] = ir
        
        # Custom torch model support.
        if model_name == "LogisticRegressionTorch":
            params["input_dim"] = X_np.shape[1]
        
        self.model = model_class(**params)
    
        # LGBM compatibility: convert to DataFrame if needed
        X_fit = X_np
        if "LGBM" in model_name:
            feature_names = [f"f{i}" for i in range(X_np.shape[1])]
            X_fit = pd.DataFrame(X_np, columns=feature_names)
    
        # ────────────────────────────────────────────────
        #               Task-specific training
        # ────────────────────────────────────────────────
        if self.task == "classification":
            # Binarize for training the classifier
            y_train = (y_np > 0).astype(int)
            self.model.fit(X_fit, y_train, sample_weight=effective_weights)
    
            if self.calibration_method == "isotonic":
                raw_probs = self.model.predict_proba(X_fit)[:, 1]
    
                # Convert compressed y back to probability-like space for isotonic target
                p_target = np.clip(0.5 * (y_np + 1.0), self.alpha, 1.0 - self.beta)
                   
                self.calibrator = IsotonicRegression(
                    y_min=self.alpha,
                    y_max=1.0 - self.beta,
                    out_of_bounds="clip"
                )
                self.calibrator.fit(raw_probs, p_target, sample_weight=calib_weights)
    
        else:  # regression
            # Safety clip (just in case of tiny numerical drift)
            y_train = np.clip(y_np, z_lower, z_upper)
            self.model.fit(X_fit, y_train, sample_weight=calib_weights)
    
            if self.calibration_method == "isotonic":
                raw_pred = self.model.predict(X_fit)
    
                self.calibrator = IsotonicRegression(
                    y_min=z_lower,
                    y_max=z_upper,
                    out_of_bounds="clip"
                )
                self.calibrator.fit(raw_pred, y_train, sample_weight=calib_weights)
    
        self._is_fitted = True
        return self
    
    
    def forward(self, X):
        """
        Run inference and return the compressed ALSE-compatible output.
    
        The returned value is always in compressed output space:
    
            z in [z_min, z_max]
    
        with:
            z_min = -(1 - 2 * alpha)
            z_max = +(1 - 2 * beta)
    
        In classification mode:
        - without calibration, probabilities are converted as z = 2p - 1;
        - with isotonic calibration, the calibrator is assumed to return z directly.
    
        In regression mode:
        - the base model already predicts z-like values;
        - the optional calibrator refines them in the same output space.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
    
        # Input preparation
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    
        if "LGBM" in self._get_model_class().__name__:
            feature_names = [f"f{i}" for i in range(X_np.shape[1])]
            X_np = pd.DataFrame(X_np, columns=feature_names)
    
        # Theoretical bounds in z-space (z = 2p - 1)
        z_lower = - (1.0 - 2.0 * self.alpha)  # negative
        z_upper = (1.0 - 2.0 * self.beta)     # positive
    
        # Get raw prediction in z-space
        if self.task == "classification":
            raw_probs = self.model.predict_proba(X_np)[:, 1]
    
            if self.calibrator is not None:
                z_np = 2.0 * self.calibrator.transform(raw_probs) - 1.0
            else:
                clipped_probs = np.clip(raw_probs, self.alpha, 1.0 - self.beta)
                z_np = 2.0 * clipped_probs - 1.0
        else:
            z_np = self.model.predict(X_np)
            if self.calibrator is not None:
                z_np = self.calibrator.transform(z_np)
    
        # Convert to torch once
        device = X.device if isinstance(X, torch.Tensor) else torch.device("cpu")
        
        z = torch.as_tensor(z_np, device=device, dtype=torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(1)
    
        # Optional asymmetric tanh compression
        if self.output_act == 1:
            z = torch.where(
                z < 0,
                z_lower * torch.tanh(z / z_lower),
                z_upper * torch.tanh(z / z_upper)
            )
    
        # Single final clamp (covers both cases)
        z = torch.clamp(z, min=z_lower, max=z_upper)
    
        return z
    
    
    def predict(self, X):
        """
        Return the compressed model output.
    
        This method mirrors forward() and therefore returns values in the
        theoretical compressed range [z_min, z_max].
        """
        return self.forward(X).detach().cpu().numpy()
    
    
    def predict_proba(self, X):
        """
        Return approximate class probabilities derived from the compressed output.
    
        The conversion is:
    
            p = 0.5 * (z + 1)
    
        followed by clipping to the valid asymmetric probability range:
    
            p in [alpha, 1 - beta]
        """
        z = self.forward(X).squeeze(1)
    
        probs_pos = 0.5 * (z + 1.0)
        probs_pos = torch.clamp(probs_pos, min=self.alpha, max=1.0 - self.beta)
        probs_neg = 1.0 - probs_pos
    
        return torch.stack([probs_neg, probs_pos], dim=1).cpu().numpy()

# ---------------------------------------------
# PyTorch implementation of a custom MLP Bayes
# ---------------------------------------------

#------------------------------------------------------------------------------
# Dictionary to map textual names of Parzen windows with numbers
#------------------------------------------------------------------------------
dict_parzen = {
    'gauss': 0,
    'uniform': 1,
    'linear': 2,
    'linear_inv': -2,
    'quadratic': 3,
    'quadratic_inv': -3,
    'cubic': 4, 
    'cubic_inv': -4, 
    'fourth': 5,
    'fourth_inv': -5,
    'triangle': 10,
    'abs': 11,
    'exponential': 13
}

#------------------------------------------------------------------------------
"""
% y = fdpGeneralN(x,tFDP);
%
% Función que calcula la función densidad de probabilidad especificada
% y parámetro deltaS
%
%  tFDP : parametros FDP base para el estimador de Parzen
%
%     tFDP(1) indica el tipo de FDP 
%      >  0 : FDP Gausiana (99% de probabilidad en centroS +/- deltaS)
%      >  1 : FDP Uniforme en centroS +/- deltaS  (norma L1 en +/-1)
%      >  2 : FDP Lineal Creciente en centroS +/- deltaS (norma L2 en +/-1)
%      > -2 : FDP Lineal Decreciente en centroS +/- deltaS
%      >  3 : FDP cuadrática Creciente en centroS +/- deltaS (norma L3 en +/-1)
%      > -3 : FDP cuadrática Decreciente en centroS +/- deltaS (complementp norma L3 en +/-1)
%      >  4 : FDP cúbica Creciente en centroS +/- deltaS (norma L4 en +/-1)
%      > -4 : FDP cúbica Decreciente en centroS +/- deltaS (complementp norma L4 en +/-1)
%      >  5 : FDP cuarta Creciente en centroS +/- deltaS (norma L5 en +/-1)
%      > -5 : FDP cuarta Decreciente en centroS +/- deltaS (complementp norma L5 en +/-1)
%
%      > 10 : FDP Triangular simétrica en centroS +/- deltaS
%      > 11 : FDP Valor Absoluto en centroS +/- deltaS
%      > 12 : FDP Integral de activación tanh 
%           (con 99% de probabilidad en centroS +/- deltaS)
%      > 13 : FDP exponencial decreciente, inicio en centroS, con factor de caída deltaS 
%
%      > 102, -102, 110, 111: generalizaciones de 2,-2,10,11 pero con
%           transiciones sinusoidales en lugar de lineales
%
%      > 202, -202, 210, 211: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales (x^2) en lugar de lineales
%
%      > 302, -302, 310, 311: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales inversas (1 - x^2) 
%
%
%     tFDP(2) indica el parámetro de ancho deltaS (soporte centroS +/- deltaS)
%     tFDP(3) indica el parámetro de media (centroS)
%
%--------------------------------------------------------------------------
%         Autor: Marcelino Lázaro
%      Creación: diciembre 2014
% Actualización: abril 2017
%--------------------------------------------------------------------------
"""
def fdpGeneralN(x,tipoFDP=[0,1,0]):
    tFDP=tipoFDP[0]
    if type(tFDP)==str:
        tFDP = dict_parzen[tFDP]
        
    deltaS=tipoFDP[1]
    centroS=tipoFDP[2]
        
    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758 # 99%
        y=np.exp(-np.power((x-centroS),2)/(2*np.power(float(deltaS)/adapta,2)))/(np.sqrt(2*3.1415926535)*float(deltaS)/adapta)
    elif tFDP == 1:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/(2.0*float(deltaS)), 0)
      
    elif tFDP == 2:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)
        
    elif tFDP == -2:
        y = np.where(np.abs(x-centroS) <= deltaS, (-x+centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)
        
    elif tFDP == 3:
        y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),2), 0)
        
    elif tFDP == -3:
        y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),2), 0)
                
    elif tFDP == 4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),3), 0)
        
    elif tFDP == -4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),3), 0)
        
    elif tFDP == 5:
        y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),4), 0)
        
    elif tFDP == -5:
        y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),4), 0)
        
    elif tFDP == 10:
        y = np.where(np.abs(x-centroS) <= deltaS, (1-np.abs((x-centroS)/float(deltaS)))/float(deltaS), 0)
        
    elif tFDP == 11:
        y = np.where(np.abs(x-centroS) <= deltaS, np.abs(x-centroS)/float(deltaS)/float(deltaS), 0)
        
    elif tFDP == 13:
        y = np.where(x >= centroS, deltaS*np.exp(-deltaS*(x-centroS)), 0)
        
    return y    

#------------------------------------------------------------------------------
"""
% y = intGeneralN(x,tFDP);
%
% Función que calcula la integral de la distribución especificada
% desde menos infinito a x
%
%  tFDP : tipo de FDP base para el estimador de Parzen
%
%  tFDP : parametros FDP base para el estimador de Parzen
%
%     tFDP(1) indica el tipo de FDP 
%      >  0 : FDP Gausiana (99% de probabilidad en centroS +/- deltaS)
%      >  1 : FDP Uniforme en centroS +/- deltaS  (norma L1 en +/-1)
%      >  2 : FDP Lineal Creciente en centroS +/- deltaS (norma L2 en +/-1)
%      > -2 : FDP Lineal Decreciente en centroS +/- deltaS
%      >  3 : FDP cuadrática Creciente en centroS +/- deltaS (norma L3 en +/-1)
%      > -3 : FDP cuadrática Decreciente en centroS +/- deltaS (complementp norma L3 en +/-1)
%      >  4 : FDP cúbica Creciente en centroS +/- deltaS (norma L4 en +/-1)
%      > -4 : FDP cúbica Decreciente en centroS +/- deltaS (complementp norma L4 en +/-1)
%      >  5 : FDP cuarta Creciente en centroS +/- deltaS (norma L5 en +/-1)
%      > -5 : FDP cuarta Decreciente en centroS +/- deltaS (complementp norma L5 en +/-1)
%
%      > 10 : FDP Triangular simétrica en centroS +/- deltaS
%      > 11 : FDP Valor Absoluto en centroS +/- deltaS
%      > 12 : FDP Integral de activación tanh 
%           (con 99% de probabilidad en centroS +/- deltaS)
%      > 13 : FDP exponencial decreciente, inicio en centroS, con factor de caída deltaS 
%
%      > 102, -102, 110, 111: generalizaciones de 2,-2,10,11 pero con
%           transiciones sinusoidales en lugar de lineales
%
%      > 202, -202, 210, 211: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales (x^2) en lugar de lineales
%
%      > 302, -302, 310, 311: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales inversas (1 - x^2) 
%
%
%     tFDP(2) indica el parámetro de ancho deltaS (soporte centroS +/- deltaS)
%     tFDP(3) indica el parámetro de media (centroS)
%
%--------------------------------------------------------------------------
%         Autor: Marcelino Lázaro
%      Creación: diciembre 2014
% Actualización: abril 2017
%--------------------------------------------------------------------------
%
"""
def intGeneralN(x,tipoFDP=[0,1,0]):
    tFDP=tipoFDP[0]
    if type(tFDP)==str:
        tFDP = dict_parzen[tFDP]
        
    deltaS=tipoFDP[1]
    centroS=tipoFDP[2]
    
    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758 # 99%
        
        y=special.erfc(0.7071*(centroS-x)/(deltaS/adapta))/2.0
        
    elif tFDP == 1:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+float(deltaS))/(2*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0
        
    elif tFDP == 2:
        y = np.where(np.abs(x-centroS) <= deltaS, np.power(x-centroS+float(deltaS),2)/(4.0*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0
            
    elif tFDP == -2:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-np.power(centroS+float(deltaS)-x,2)/(4.0*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     
        
    elif tFDP == 3:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.125*np.power(1+(x-centroS)/deltaS,3), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == -3:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.125*np.power(1-(x-centroS)/deltaS,3), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 4:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.0625*np.power(1+(x-centroS)/deltaS,4), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == -4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.0625*np.power(1-(x-centroS)/deltaS,4), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 5:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.03125*np.power(1+(x-centroS)/deltaS,5), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == -5:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.03125*np.power(1-(x-centroS)/deltaS,5), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 10:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/deltaS - 0.5-np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

        
    elif tFDP == 11:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.5+np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP ==13:
        y = np.where(x >= centroS, 1-np.exp(-deltaS*(x-centroS)), 0.0)
        #y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    return y



class MLPBayesSwitch(nn.Module):
    def __init__(self, layers_size=[100], 
                 activations=['relu','identity'],
                 parzen_window = 'gauss',
                 #parzen_params=[['gauss',1,0],['gauss',1,0]],      
                 class_prob=[0.5, 0.5],
                 cost_classes=[1,0.8],
                 update='momentum', 
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 1000,
                 n_batch = 250,                  
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start=False,
                 flag_Evo=False,
                 alpha = 0,
                 beta = 0
                 ):
        
        super().__init__()
        
        #self.name = 'MLPBayesD'
        self.layers_size = layers_size
        self.activations = activations
        self.parzen_window = parzen_window
        #self.parzen_params = parzen_params        
        self.class_prob = class_prob
        self.cost_classes = cost_classes 
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch                       
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_Evo = flag_Evo
        self.alpha = alpha
        self.beta = beta
        #
        self.parzen_params = [[parzen_window,1-2*alpha,0],[parzen_window,1-2*beta,0]]
        
    
    def _preprocess_data_LS(self, x, y):
        nClases = 2

        # Labels converted to -1, +1
        y_pre = np.where(y <= 0, -1, 1)
        
        """
        if 0 not in list(np.unique(y)):
            y = y - 1
        """        
        self.class_labels = list(np.unique(y_pre))                        
                        
        
        if len(self.class_prob) == 0:
            # Calculate class probabilities
            unique, counts = np.unique(y_pre, return_counts=True)
            self.class_prob = counts / y_pre.shape[0]
                
        if len(self.cost_classes) == 0:
            self.cost_classes = np.ones(nClases)
                                    
        # yout = np.where(y==0, -1,1) 
        yout = y                  
        xout = x.T                          
        
        return xout, yout
    
    
    def _init_coefs(self, dim_in, dim_out, type_init):
        # Use the initialization method recommended by
        # Glorot et al.
        if type_init == 'normal':
            sd = np.sqrt(2.0 / (dim_in + dim_out))
            coefs = np.float32(np.random.normal(0.0, sd, (dim_out,dim_in+1)))
    
        elif type_init == 'uniform':
            sd = np.sqrt(6.0 / (dim_in + dim_out))
            coefs = np.float32(np.random.uniform(-sd, sd, (dim_out,dim_in+1)))
    
        return coefs

    def _initialize(self, dim_in, dim_out, type_init='uniform'):
    
        self.coefs_ = []
        self.dW = []
        self.dWm = []
        self.num_hidden = len(self.layers_size)
        self.learning_rate = self.learning_rate_init
    
        nDims = [dim_in] + list(self.layers_size) + [dim_out]
        for k in range(self.num_hidden+1):
            self.coefs_.append(self._init_coefs(nDims[k], nDims[k+1], type_init))
            self.dW.append(self.coefs_[k]*0.0)
            self.dWm.append(self.coefs_[k]*0.0)
            
        
    def _cost(self, y, ye, sample_weight):       
        
        v0 = np.nonzero(y<=0) # ==-1)
        v1 = np.nonzero(y>0) # ==1)
        
        # coste = self.cost_classes[0] * self.class_prob[0] * np.mean(intGeneralN(ye[0,v0],self.parzen_params[0]))
        # coste += self.cost_classes[1] * self.class_prob[1] * np.mean(intGeneralN(-ye[0,v1],self.parzen_params[1]))
            
        coste = self.cost_classes[0] * self.class_prob[0] * np.matmul(intGeneralN(ye[0,v0],self.parzen_params[0]), sample_weight[v0])
        coste += self.cost_classes[1] * self.class_prob[1] * np.matmul(intGeneralN(-ye[0,v1],self.parzen_params[1]), sample_weight[v1])
        
        return coste
    
    def _derivatives_cost(self, y, ye, sample_weight):
        """
        Computes the derivative of the cost with respect to the network output.
    
        Parameters:
        y : np.ndarray
            True labels.
        ye : np.ndarray
            Network output.
        sample_weight : np.ndarray
            Weights assigned to each sample, used to adjust the importance of individual samples during cost calculation.
    
        Returns:
        d : np.ndarray
            Derivative of the cost with respect to the output layer.
        """
        nBatchSize = y.shape[0]
        d = np.zeros((1, nBatchSize))
    
        v0 = np.nonzero(y <= 0)
        v1 = np.nonzero(y > 0)
    
        # Handle cases where v0 or v1 might be empty
        if len(v0[0]) > 0:
            d[0, v0] = self.cost_classes[0] * self.class_prob[0] * np.multiply(fdpGeneralN(ye[0, v0], self.parzen_params[0]), sample_weight[v0]) / len(v0[0])
        if len(v1[0]) > 0:
            d[0, v1] = -self.cost_classes[1] * self.class_prob[1] * np.multiply(fdpGeneralN(-ye[0, v1], self.parzen_params[1]), sample_weight[v1]) / len(v1[0])
    
        # Propagate through the nonlinear function of the output layer
        if self.activations[-1] == 'tanh':
            d = np.multiply(d, 1 - np.square(ye))
    
        elif self.activations[-1] == 'logistic':
            d = np.multiply(d, ye - np.square(ye))
    
        elif self.activations[-1] == 'relu':
            d = np.where(ye <= 0, 0, d)
    
        elif self.activations[-1] == 'mod_tanh':
            d = np.where(
                ye < 0,
                np.multiply(d, (1 - 2 * self.alpha) * (1 - np.square(ye))),
                np.multiply(d, (1 - 2 * self.beta) * (1 - np.square(ye)))
            )
    
        return d

    def _backpropagation(self, x, Os, d):
        # x: input, Os: network outputs, d: derivative of loss function (with activation)

        nCapas = self.num_hidden + 1
        dim_in, nBatchSize = x.shape
        
        dW=[[] for k in range(nCapas)]
                        
        dW[nCapas-1]=np.matmul(d,np.transpose(np.concatenate((Os[-1],np.ones((1,nBatchSize))))))
                
        o=Os[-1]
        
        for ko in range(nCapas-2,-1,-1):
            wp=self.coefs_[ko+1]
            d=np.matmul(np.transpose(wp[:,0:-1]),d)
            
            if self.activations[ko]=='relu':
                d = np.where(o<=0,0,d)
                
            elif self.activations[ko]=='tanh':
                d=np.multiply(d,1-np.square(o))
                
            elif self.activations[ko]=='logistic':
                d=np.multiply(d,o-np.square(o))
            
            elif self.activations[ko] == 'mod_tanh':
                d = np.where(
                    o < 0,
                    np.multiply(d, (1 - 2 * self.alpha) * (1 - np.square(o))),
                    np.multiply(d, (1 - 2 * self.beta) * (1 - np.square(o)))
                )
                
            if ko == 0:
                o=x
            else:
                o=Os[ko-1]
             
            dW[ko]=np.matmul(d,np.transpose(np.concatenate((o,np.ones((1,nBatchSize))))))
            
        return dW 
    
    def _update_W(self, dW,mX,mW):
                        
        nCapas = self.num_hidden + 1        
        if self.update =='momentum':
            for ko in range(nCapas):
                self.dWm[ko]=self.dWm[ko]*self.momentum+self.learning_rate*np.multiply(dW[ko],mW[ko])
                self.coefs_[ko]=self.coefs_[ko]-np.multiply(self.dWm[ko],mW[ko])
            
        elif self.update =='gradient':
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.learning_rate*np.multiply(dW[ko],mW[ko])
        

    def predict(self, x):
        
        ys = self._forward_pass_fast(x.T)
        #ye = np.argmax(ys, axis=0)
        ye = np.where(ys[0,:]>0,1,0)
        
        return ye
                
    def _forward_pass_W(self, x, W):
        nCapas=len(W)
        No=nCapas-1
        
        Np=x.shape[1]
        
        Os=list()
        
        xcapa = x.copy()    
        for ko in range(nCapas):
    
            o=np.matmul(W[ko],np.concatenate((xcapa,np.ones((1,Np)))));
            
            if self.activations[ko] == 'tanh':
                o = np.tanh(o)
            elif self.activations[ko] == 'logistic':
                o = 1.0/(1+np.exp(-1*o))
            elif self.activations[ko] == 'relu':
                o[o<0]=0.0
            elif self.activations[ko] == 'softmax':            
                #o = np.exp(o)
                # Option to avoid numeric issues (same result)
                o = np.exp(o-np.max(o,axis=0))
                o = o/np.sum(o,axis=0)
            elif self.activations[ko] == 'mod_tanh':
                o = np.where(
                    o < 0,
                    (1 - 2 * self.alpha) * np.tanh(o),
                    (1 - 2 * self.beta) * np.tanh(o)
                )    
            Os.append(o)
            xcapa=o.copy()
        
        return (Os[-1],Os[0:No])  

    def _forward_pass_fast(self, x):
        nCapas=len(self.coefs_)
        
        Np=x.shape[1]        
        
        xcapa = x.copy()    
        for ko in range(nCapas):

            o=np.matmul(self.coefs_[ko],np.concatenate((xcapa,np.ones((1,Np)))));
            
            # Ensure o is a NumPy array if it is a float
            if isinstance(o, float):
                o = np.array([o])
            elif isinstance(o, list):
                o = np.array(o)
                
            # Ensure o is a NumPy array and has the correct dtype
            o = np.array(o, dtype=np.float64)

            
            if self.activations[ko] == 'tanh':
                o = np.tanh(o)
            elif self.activations[ko] == 'logistic':
                o = 1.0/(1+np.exp(-1*o))
            elif self.activations[ko] == 'relu':
                o[o<0]=0.0
            elif self.activations[ko] == 'softmax':            
                #o = np.exp(o)
                # Option to avoid numeric issues (same result)
                o = np.exp(o-np.max(o,axis=0))
                o = o/np.sum(o,axis=0)
            elif self.activations[ko] == 'mod_tanh':
                o = np.where(
                    o < 0,
                    (1 - 2 * self.alpha) * np.tanh(o),
                    (1 - 2 * self.beta) * np.tanh(o)
                )
                
            xcapa=o.copy()
        
        return o
    
    def soft_output(self, x):
        # Ensure x is a NumPy array
        if not isinstance(x, np.ndarray):
            x = np.array(x)
    
        ys = self._forward_pass_fast(x.T)        
        ye = ys[0,:]
        ye = np.clip(ye, 2*self.alpha-1, 1-2*self.beta)
        
        return ye
    
    def forward(self, x):
        # Convert input to numpy
        x_np = x.cpu().numpy()
        y_np = self.soft_output(x_np)
        # Convert output to tensor
        y_torch = torch.from_numpy(y_np).float().unsqueeze(1)
        
        return y_torch


    def _fit(self, x, y, sample_weight=None, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):
        if type(x_val) == float:
            #print('    Generating validation set...')
            
            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):            
                indices_CV.append(val)
                
            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values        
            ind_train = list(set(ind_train))
            
            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]
            
            #print('    Validation set generated')
            
                
        x, y = self._preprocess_data_LS(x,y)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data_LS(x_val,y_val)
            
        
        dim_in, num_pat = x.shape
        dim_out = 1
        
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )
        
        if first_pass:#(len(self.coefs_) == 0) or (self.warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)
            
        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nBatch = num_pat if nBatch is None or nBatch == 'auto' or int(nBatch) > num_pat or int(nBatch) == 0 else int(nBatch)
        
        nCapas = self.num_hidden + 1
        
        if (self.flag_Evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye)
                
                
            evoCosteEpoch[0]=coste
            
            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()
                

        #dW=[[] for k in range(nCapas)]        
        Wdo=[[] for k in range(nCapas)]    
        Wdon=[[] for k in range(nCapas)]   
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]
        
        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]
        
        if len(self.dWm) == 0:
            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))                        
                            
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))
         
        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y
        elif  self.type_batch == 'random':
            paramsBatch = len(y)
            
            
        for kEpoch in range(nEpochs):
            
            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):                                            
                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:                                    
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])
                        
                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()
                                    
                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[indBatch], ye, sample_weight[indBatch])
                #d = self.gradiente_Numerico(y[indBatch], ye)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------    
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[indBatch],ye)
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[indBatch],paramsBatch,coste)
                    
                    if salida:                            
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])
                                                                                
                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste    
                        print("Step size in the limit (zero) ...")
                                                
                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W
                        
                        return opt_epoch, evoCosteEpoch                                        
                    
                else:
                    self._update_W(dW,mX,mW)
                           
            #----------------------------------------------------------------------
            # Actualización del coste (global) y validación
            #----------------------------------------------------------------------
            if (self.flag_Evo) or (val_stop[0] > 0):
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])                                        
                
                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y,ye)
                
                
                evoCosteEpoch[kEpoch+1] = coste
                
                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()
                        
                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])
                        
                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W
                            
                            break
                                    
        # Training is over            
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])
            
        if (self.flag_Evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye,sample_weight)
            
            """
            # Convert predictions to class labels (-1 or 1)
            yepred = np.where(ye <= 0, -1, 1)[0]
            
            # Print cost after training
            print(f"Cost after training: {evoCosteEpoch:.4f}")
            
            # Print balanced accuracy score
            balanced_acc = balanced_accuracy_score(y, yepred)
            print(f"Balanced Accuracy = {balanced_acc:.4f}")
            """
            opt_epoch = nEpochs
            
        elif val_stop[0] == 0:
            opt_epoch = nEpochs
            
        self.loss_ = evoCosteEpoch
        self.opt_epoch = opt_epoch
        
                                        
        return self 
        
    def fit(self, X, y, sample_weight=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
            
        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        # Handle sample_weight
        if sample_weight is None:
            # Initialize weights to 1.0 for all samples if no weights are provided
            sample_weight = np.ones_like(y)
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, sample_weight, incremental=False)
    
    
