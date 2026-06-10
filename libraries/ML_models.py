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
import torch.optim as optim
import torch.nn.functional as F
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

# ---------------------------------------------
# PyTorch implementation of a custom MLP
# ---------------------------------------------
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', 
                 alpha=0.0, beta=0.0, alpha_tr=0.0001, batch_size='auto', 
                 learning_rate_init=0.001, max_iter=200, optim='adam'):
        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.alpha_tr = alpha_tr
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.optim = optim

        # Build layers
        layer_sizes = [input_dim] + list(hidden_layer_sizes) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])

        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Optimizer
        if optim == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha_tr)
        elif optim == 'lbfgs':
            self.optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate_init)
        else:
            raise ValueError("Only 'adam' and 'lbfgs' are supported solvers.")

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(x)
        
        # Output layer with custom transformation
        z = self.layers[-1](x)
        return torch.where(
            z < 0,
            torch.tanh(z) * (1 - 2 * self.alpha),
            torch.tanh(z) * (1 - 2 * self.beta),
        )

    def fit(self, X, y, sample_weight=None):
        # Handle sample_weight
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = torch.from_numpy(sample_weight).float()
        else:
            # Initialize weights to 1.0 for all samples if no weights are provided
            sample_weight = torch.ones_like(y, dtype=torch.float32)
            
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        batch_size = len(X) if self.batch_size == 'auto' else self.batch_size
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        criterion = nn.MSELoss()  # Change from BCELoss to MSELoss
    
        for epoch in range(self.max_iter):
            for batch_X, batch_y in dataloader:
                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs, batch_y)  # Compute MSE loss
                    loss.backward()
                    return loss
    
                if self.optim == 'adam':
                    loss = closure()
                    self.optimizer.step()
                elif self.optim == 'lbfgs':
                    self.optimizer.step(closure)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.forward(X)
        return (y_pred > 0.0).int().numpy()

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.forward(X)
        return torch.cat([1 - y_pred, y_pred], dim=1).numpy()
    

# ---------------------------------------------
# PyTorch implementation of a custom LogReg
# ---------------------------------------------
class LogisticRegressionTorch_old(nn.Module):
    """
    Custom Logistic Regression with asymmetric activation, 
    compatible with MLPClassifierTorch API.
    """
    def __init__(self, input_dim, alpha, beta, alpha_tr=0.0001, batch_size='auto', 
                 learning_rate_init=0.01, max_iter=100, solver='adam'):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha
        self.beta = beta
        self.alpha_tr = alpha_tr
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.solver = solver

        # Optimizer initialization
        if solver == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha_tr)
        elif solver == 'lbfgs':
            self.optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate_init)
        else:
            raise ValueError("Only 'adam' and 'lbfgs' are supported.")

    def forward(self, x):
        """Asymmetric tanh transformation."""
        z = self.linear(x)
        return torch.where(
            z < 0,
            torch.tanh(z) * (1 - 2 * self.alpha),
            torch.tanh(z) * (1 - 2 * self.beta)
        )

    def fit(self, X, y, sample_weight=None):
        """
        Unified fit method supporting mini-batches, sample weights, and L-BFGS.
        """
        # Convert inputs to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Handle sample weights (Weighted MSE)
        if sample_weight is not None:
            weights = torch.tensor(sample_weight, dtype=torch.float32).unsqueeze(1)
        else:
            weights = torch.ones_like(y_tensor)

        # DataLoader setup
        batch_sz = len(X_tensor) if self.batch_size == 'auto' else self.batch_size
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, weights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True)

        self.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y, batch_w in dataloader:
                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_X)
                    # Manual weighted MSE: (w * (out - target)^2).mean()
                    loss = (batch_w * (outputs - batch_y)**2).mean()
                    loss.backward()
                    return loss

                if self.solver == 'adam':
                    closure()
                    self.optimizer.step()
                elif self.solver == 'lbfgs':
                    self.optimizer.step(closure)
        
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict class labels for X. Required for sklearn compatibility.
        
        Args:
            X (ndarray): Input features.
        Returns:
            ndarray: Binary predictions (0 or 1).
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.forward(X_tensor)
            # Assuming threshold 0 for tanh-based output
            return (outputs > 0).float().numpy().flatten()

    def predict_proba(self, X):
        """
        Predict class probabilities for X. Required for CalibratedClassifierCV.
        
        Args:
            X (ndarray): Input features.
        Returns:
            ndarray: Array of shape (n_samples, 2) with class probabilities.
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.forward(X_tensor)
            
            # Map asymmetric output to a [0, 1] probability range
            # Note: Since outputs can be slightly outside [-1, 1] depending on alpha/beta,
            # we clamp the values to ensure valid probabilities.
            prob_pos = (outputs + 1) / 2
            prob_pos = torch.clamp(prob_pos, 0, 1).numpy().reshape(-1, 1)
            
            # Return [P(class 0), P(class 1)]
            return np.hstack([1 - prob_pos, prob_pos])
        
class LogisticRegressionTorch(nn.Module):
    """
    Custom Logistic Regression compatible with external training routines.
    The fit method acts as a wrapper for the ensemble's train_model function.
    """
    def __init__(self, input_dim, alpha, beta, loss_fn, optim='adam', max_iter=150, 
                 learning_rate=0.001, mode='random', batch_size='auto'):
        super(LogisticRegressionTorch, self).__init__()
        # Using double precision (float64) for better L-BFGS convergence
        self.linear = nn.Linear(input_dim, 1).double()
        self.alpha = alpha
        self.beta = beta
        self.loss_fn_e = weighted_mse_loss  # Default to MSE if not found
        
        # Training hyperparameters
        self.optim = optim
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.mode = mode
        self.batch_size = batch_size

    def forward(self, x):
        """Asymmetric tanh activation."""
        z = self.linear(x)
        return torch.where(
            z < 0,
            torch.tanh(z) * (1 - 2 * self.alpha),
            torch.tanh(z) * (1 - 2 * self.beta)
        )

    def fit(self, X, y, sample_weight=None):
        """
        Internal fit that configures the optimizer and calls train_model.
        """
        # 1. Prepare data (ensure they are Tensors and double precision)
        X_tensor = torch.from_numpy(X).double() if isinstance(X, np.ndarray) else X.double()
        y_tensor = torch.from_numpy(y).double() if isinstance(y, np.ndarray) else y.double()
        
        if sample_weight is None:
            sample_weight = torch.ones(len(X_tensor), dtype=torch.double)
        else:
            sample_weight = torch.from_numpy(sample_weight).double() if isinstance(sample_weight, np.ndarray) else sample_weight.double()

        # 2. Configure Optimizer based on self.solver
        if self.optim == 'lbfgs':
            # Note: eps should be defined or passed; using 1e-16 as standard for float64
            eps = 1e-16 
            optimizer = LBFGSScipy(
                self.parameters(),
                max_iter=150,
                max_eval=150,
                tolerance_grad=1e-04,
                tolerance_change=10e6 * eps,
                history_size=10
            )
            train_model(
                X_tensor,
                y_tensor,
                self,
                self.loss_fn_e,
                optimizer,
                sample_weight,
                num_epochs=1, # L-BFGS usually needs 1 epoch with full batch
                batch_size=None,
                mode=self.mode,
                lbfgs=True,
                debug=False
            )
        else:
            if self.optim == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            elif self.optim == 'adamw':
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
            else:
                raise ValueError(f"Unsupported optimizer: {self.solver}")

            # 3. Call your specialized training function
            # We assume weighted_mse_loss is available in the namespace
            train_model(
                X_tensor,
                y_tensor,
                self,
                self.loss_fn_e, # Your custom loss function
                optimizer,
                sample_weight,
                num_epochs=self.max_iter,
                batch_size=self.batch_size,
                mode=self.mode,
                lbfgs=(self.optim == 'lbfgs'),
                debug=False
            )
        return self

    def predict(self, X):
        """Binary prediction for sklearn compatibility."""
        self.eval()
        X_tensor = torch.from_numpy(X).double() if isinstance(X, np.ndarray) else X.double()
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            return outputs.cpu().numpy().flatten()

    def predict_proba(self, X):
        """Probability estimation for CalibratedClassifierCV."""
        self.eval()
        X_tensor = torch.from_numpy(X).double() if isinstance(X, np.ndarray) else X.double()
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            prob_pos = (outputs + 1) / 2
            prob_pos = torch.clamp(prob_pos, 0, 1).cpu().numpy().reshape(-1, 1)
            return np.hstack([1 - prob_pos, prob_pos])


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
        """Return model class based on task and model_type."""
        if self.model_type == 'lgbm':
            return LGBMRegressor if self.task == 'regression' else LGBMClassifier
        elif self.model_type == 'xgb':
            return XGBRegressor if self.task == 'regression' else XGBClassifier
        elif self.model_type == 'cat':
            return CatBoostRegressor if self.task == 'regression' else CatBoostClassifier
        elif self.model_type == 'rf':
            return RandomForestRegressor if self.task == 'regression' else RandomForestClassifier
        elif self.model_type == 'logreg':
            # return LinearRegression if self.task == 'regression' else LogisticRegression
            return LogisticRegressionTorch if self.task == 'regression' else LogisticRegression
        raise ValueError(f"Unsupported model_type: {self.model_type}")

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
                    'objective': 'binary' if self.task == 'classification' else 'regression',
                    'metric': 'binary_logloss' if self.task == 'classification' else 'rmse',
                    'verbosity': -1,
                    'force_col_wise': True,
                    # Tree Structure
                    'max_depth': -1,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'min_child_weight': 1e-3,
                    # Learning
                    'learning_rate': 0.1,
                    'n_estimators': 60,
                    # Regularization
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.0,
                    # Imbalance handling
                    # 'is_unbalance': True if self.task == 'classification' else False,
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
                    'n_jobs': -1,
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
                    'thread_count': -1,
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
            return {
                'n_estimators': 60,
                'max_depth': None,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'n_jobs': -1,
                'random_state': 42,
                # Dynamic balancing per tree (robust for imbalanced dichotomies)
                'class_weight': 'balanced_subsample' if self.task == 'classification' else None
            }

        elif self.model_type == 'logreg':
            if self.task == 'classification':
                return {
                    'solver': 'lbfgs',
                    'max_iter': 500,
                    'C': 1.0,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            else: # ALSE Regression
                return {
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'solver': 'lbfgs',
                    'max_iter': 200,
                    'loss_fn': weighted_mse_loss # Custom loss for switching
                }

        return {}

    def fit(self, X, y, sample_weight=None):
        """
        Fit model and optional calibrator with unified weight scaling.
        Estimates Pr_S(C1|X) to satisfy the ALSE soft output requirement.
        """
        # 1. Standardize inputs to NumPy and compute Imbalance Ratio (IR)
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        ir = compute_imbalance_ratio(y_np)
        
        model_class = self._get_model_class()
        model_name = model_class.__name__

        # 2. Parameter Preparation: Priority is given to self.model_params
        params = {**self._get_default_params(), **self.model_params}

        # 3. Dynamic Imbalance Injection: Only if not already handled by the user
        if model_name in ['LGBMClassifier', 'XGBClassifier']:
            # Check if user already provided an imbalance strategy
            is_handled = params.get('is_unbalance') or params.get('scale_pos_weight')
            if not is_handled and ir > 1:
                params['scale_pos_weight'] = ir
        
        # 4. Sample Weights setup
        effective_weights = np.array(sample_weight, dtype=float).copy() if sample_weight is not None else np.ones_like(y_np, dtype=float)

        # 5. Model Initialization
        if model_name == 'LogisticRegressionTorch':
            params['input_dim'] = X_np.shape[1]
            
        self.model = model_class(**params)

        # 6. Task Execution
        if self.task == 'classification':
            y_train = (y_np > 0).astype(int)
            
            if "LGBM" in model_name:
                feature_names = [f'f{i}' for i in range(X_np.shape[1])]
                X_np = pd.DataFrame(X_np, columns=feature_names)
                
            self.model.fit(X_np, y_train, sample_weight=effective_weights)
            
            # Calibration: Mapping scores to Pr_S in [alpha, 1-beta]
            if self.calibration_method == 'isotonic':
                raw_probs = self.model.predict_proba(X_np)[:, 1]
                y_pr_s = 0.5 * (y_np + 1)
                
                # Fit with strict theoretical boundaries
                self.calibrator = IsotonicRegression(
                    y_min=self.alpha, 
                    y_max=1 - self.beta, 
                    out_of_bounds='clip'
                )
                self.calibrator.fit(raw_probs, y_pr_s, sample_weight=effective_weights)
                
        else:  # Regression mode
            if "LGBM" in model_name:
                feature_names = [f'f{i}' for i in range(X_np.shape[1])]
                X_np = pd.DataFrame(X_np, columns=feature_names)

            self.model.fit(X_np, y_np, sample_weight=effective_weights)
            
            if self.calibration_method == 'isotonic':
                preds = self.model.predict(X_np)
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(preds, y_np, sample_weight=effective_weights)

        self._is_fitted = True
        return self

    def forward(self, x):
        """
        Perform forward pass and apply linear probability transformation.
        The output maps to z = 2*Pr_S - 1, respecting alpha/beta bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
    
        # 1. Input conversion
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        model_name = self._get_model_class().__name__
        
        if "LGBM" in model_name:
            feature_names = [f'f{i}' for i in range(x_np.shape[1])]
            x_np = pd.DataFrame(x_np, columns=feature_names)
    
        # 2. Prediction & Calibration
        if self.task == 'classification':
            probs = self.model.predict_proba(x_np)[:, 1]
            if self.calibrator:
                probs = self.calibrator.transform(probs)
            
            # Final safety clip before linear mapping
            probs = np.clip(probs, self.alpha, 1 - self.beta)
            z_np = 2 * probs - 1
        else:
            z_np = self.model.predict(x_np)
            if self.calibrator:
                z_np = self.calibrator.transform(z_np)
    
        # 3. Conversion to Torch
        z = torch.as_tensor(z_np, device=x.device, dtype=torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(1)
            
        # 4. Final Clipping (No tanh distortion if output_act=0)
        neg_bound = (1 - 2 * self.alpha)
        pos_bound = (1 - 2 * self.beta)
        
        if self.output_act == 0:
            return torch.clamp(z, -neg_bound, pos_bound)
        
        # Non-linear activation fallback (only if specifically requested)
        if self.output_act == 1:
            return torch.where(z < 0, neg_bound * torch.tanh(z), pos_bound * torch.tanh(z))
        
        return torch.clamp(z, -neg_bound, pos_bound)

    def predict(self, X):
        """Predict class label from sign of compressed output."""
        return self.forward(X).cpu().numpy()

    def predict_proba(self, X):
        """Approximate class probabilities from compressed output."""
        compressed = self.forward(X)
        probs = (compressed + 1) / 2
        probs = torch.clamp(probs, self.alpha, 1-self.beta)
        return torch.stack([1 - probs, probs], dim=1).cpu().numpy()


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
    
    
