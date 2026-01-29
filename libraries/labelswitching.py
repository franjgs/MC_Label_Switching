import numpy as np
import torch
import torch.nn as nn

from libraries.functions import compute_imbalance_ratio
from libraries.ML_models import MLPClassifierTorch, LogisticRegressionTorch
from libraries.ML_models import CalibratedBooster
from libraries.ML_models import MLPBayesSwitch

import random
from imblearn.over_sampling import SMOTE

from libraries.optimizers import LBFGSScipy, train_model
from libraries.optimizers import (weighted_mse_loss,
                                  weighted_kl_loss,
                                  weighted_bce_loss,
                                  weighted_bce_logit_loss,
                                  f1_loss)

eps=np.finfo(float).eps

def label_switching(y, w=None, alphasw=0.0, betasw=0.0):
    """
    Perform label switching with optional weight adjustment for sample-level weights.

    Parameters:
    - y: Original labels (numpy array with values -1 and +1).
    - w: Class weights (list or numpy array of size 2, [majority_weight, minority_weight]).
         If None, weights are initialized to [1.0, 1.0].
    - alphasw: Label switching rate from Majority to Minority class.
    - betasw: Label switching rate from Minority to Majority class.

    Returns:
    - ysw: Labels after switching.
    - wsw: Updated sample weights after switching.
    """
    # Initialize class weights if not provided
    if w is None:
        w = np.array([1.0, 1.0])  # [majority_class_weight, minority_class_weight]

    # Copy original labels
    ysw = np.copy(y)

    # Initialize sample weights based on the original labels
    wsw = np.where(y == -1, w[0], w[1])

    # Find indices of each class
    idx1 = np.where(y == +1)[0]  # Minority class
    l1 = len(idx1)
    bet_1 = int(round(l1 * betasw))  # Number of switches from Minority to Majority
    bet_1 = min(bet_1, l1)  # Ensure bet_1 <= l1
    if bet_1 > 0:
        idx1_sw = np.random.choice(idx1, bet_1, replace=False)
        # Perform label switching for the minority class
        ysw[idx1_sw] = -1
        wsw[idx1_sw] = w[0]  # Update weights to majority class weight

    idx0 = np.where(y == -1)[0]  # Majority class
    l0 = len(idx0)
    alph_0 = int(round(l0 * alphasw))  # Number of switches from Majority to Minority
    alph_0 = min(alph_0, l0)  # Ensure alph_0 <= l0
    if alph_0 > 0:
        idx0_sw = np.random.choice(idx0, alph_0, replace=False)
        # Perform label switching for the majority class
        ysw[idx0_sw] = +1
        wsw[idx0_sw] = w[1]  # Update weights to minority class weight

    return ysw, wsw


def compute_weights(targets_train, RI_C=1, Q_P=1, mode='normal'):
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

    Returns:
        torch.Tensor: Tensor of weights for each training sample.
    """
    weights = np.ones_like(targets_train, dtype=float)

    if RI_C >= 1:
        if mode == 'normal':
            weights[targets_train > 0] = min(RI_C, Q_P)
        elif mode == 'reverse':
            weights[targets_train <= 0] = 1 / min(RI_C, Q_P)

    return torch.from_numpy(weights)     # /np.sum(weights))


# Base class to handle common initialization logic
class BaseAsymmetricMLP(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, beta, activation_fn=torch.tanh):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.activation_fn = activation_fn

    def init_weights(self):
        """
        Applies Xavier uniform initialization to weights.
        """
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # Bias remains with PyTorch default random initialization
        self.apply(weight_init)

class AsymmetricMLP(BaseAsymmetricMLP):
    def __init__(self, input_size, hidden_size, alpha, beta, dropout_prob=0.0, activation_fn=torch.tanh, output_act=1):
        super().__init__(input_size, hidden_size, alpha, beta, activation_fn)
        
        # We use Sequential to exactly match the structure and initialization order of Program 1
        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        
        # We define 'out' immediately after 'hidden0' to match the PRNG sequence
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
        # Define dropout separately (it doesn't use random numbers during init, 
        # but we place it here to not interfere with the layer registration order)
        self.dropout = nn.Dropout(dropout_prob)

        # Apply weight initialization
        self.init_weights()

        # Match precision
        self.to(torch.float64)
        self.output_act = output_act

    def forward(self, x):
        x = x.to(torch.float64)
        
        # Matches Program 1: tanh(hidden0(x))
        o = self.activation_fn(self.hidden0(x))
        o = self.dropout(o)
        z = self.out(o)

        if self.output_act == 1:
            return torch.where(
                z < 0,
                torch.tanh(z) * (1 - 2 * self.alpha),
                torch.tanh(z) * (1 - 2 * self.beta),
            )
        elif self.output_act == 2:
            return torch.where(
                z < 0,
                torch.tanh(z / (1 - 2 * self.alpha)) * (1 - 2 * self.alpha),
                torch.tanh(z / (1 - 2 * self.beta)) * (1 - 2 * self.beta),
            )
        else:
            raise ValueError("Invalid output_mode. Choose 1 or 2.")

# Mapping for activation functions
ACTIVATION_FUNCTIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "none": lambda x: x  # Identity function if no activation is needed
}

# Mapping for loss functions
LOSS_FUNCTIONS = {
    "MSE": weighted_mse_loss,
    "KL": weighted_kl_loss,
    "BCE": weighted_bce_loss,
    "BCE_logit": weighted_bce_logit_loss,
    "F1": f1_loss,
}

class LSEnsemble(nn.Module):
    def __init__(self, hidden_size, num_experts, alpha=0, beta=0, QC=1, Q_RB_C=1, 
                 Q_RB_S=1, n_epoch=1, n_batch=1, mode='random', 
                 input_size=None, drop_out=0, base_learner='FAMLP', 
                 activation_fn="tanh", output_act=1, optim='lbfgs', loss_fn='MSE'):  
        """
        Parameters:
        - hidden_size (int): 
            Number of neurons in the hidden layer for each expert in the ensemble.
        - num_experts (int): 
            Number of experts (individual models) in the ensemble.
        - alpha (float, optional, default=0): 
            Switching factor: majority to minority.
        - beta (float, optional, default=0): 
            Switching factor: minority to majority.
        - Q_RB_C (float, optional, default=1): 
            Classification cost.
        - Q_RB_S (float, optional, default=1): 
            Rebalancing factor for neutral population rebalance (SMOTE).
        - n_epoch (int, optional, default=1): 
            Number of training epochs for the ensemble.
        - n_batch (int, optional, default=1): 
            Batch size used during training.
        - mode (str, optional, default='random'): 
            Mode of sampling or initialization for the ensemble. Options may include:
            'random', 'class_equitative', or others depending on the implementation.
        - input_size (int, optional, default=None): 
            Number of input features in the dataset. This must be set before initializing the model.
        - drop_out (float, optional, default=0): 
            Dropout probability for the hidden layer to prevent overfitting.
        - base_learner (str, optional, default='FAMLP'):
            Type of base learner used in the ensemble:
                - 'FAMLP': Full Asymmetric MLP.
                - 'LogReg': Customized Logistic Regression.
                - 'Parzen': MLP Bayes.
                - 'AMLP': Simple Asymmetric MLP.
        - activation_fn (str, optional, default="tanh"): 
            Activation function for the hidden layers of the model. Common options:
            'tanh', 'relu', or 'sigmoid'.
        - output_act (int, optional, default=1): 
            Type of activation function applied at the output layer. Options:
            - 0: Simple compressor
            - 1: Customized tanh-based activation function.
            - 2: Scaled tanh activation function.
        - optim (str, optional, default='lbfgs'): 
            Optimization method.
        - loss_fn (str, optional, default='MSE'): 
            Loss function used for training. Common options:
            - 'MSE': Mean Squared Error.
            - 'KL': Kullback-Leibler Divergence.
            - 'BCE': Binary Cross-Entropy for classification tasks.
            - 'F1': F1 Score
        """
        super(LSEnsemble, self).__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.drop_out = drop_out
      
        self.Q_RB_C = Q_RB_C
        self.Q_RB_S = Q_RB_S
        
        if QC == None:
            C10 = 1  # False Positive (FP)
            C00 = 0  # True Negative (TN)
            C01 = 1  # False Negative (FN)
            C11 = 0  # True Positive (TP)
            self.QC = float(C10 - C00) / float(C01 - C11)
        else:
            self.QC = QC 
        
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.mode = mode
        self.base_learner = base_learner
        self.hidden_size = hidden_size
        # Convert string activation function to actual function
        if activation_fn not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Invalid activation function '{activation_fn}'. Choose from {list(ACTIVATION_FUNCTIONS.keys())}.")
        self.activation_fn = ACTIVATION_FUNCTIONS[activation_fn]

        self.optim = optim
        # Map the provided string to the actual loss function
        self.loss_fn_e = LOSS_FUNCTIONS.get(loss_fn, weighted_mse_loss)  # Default to MSE if not found

        self.output_mode = output_act

        # Initialize the experts if input_size is provided during initialization
        if input_size is not None:
            self.initialize_experts(input_size)

    def initialize_experts(self, input_size):
        if self.base_learner == 'AMLP':  # Use MLPClassifierTorch if lbfgs is set to 'MLP'
            self.experts = nn.ModuleList([
                MLPClassifierTorch(
                    input_dim=input_size,
                    hidden_layer_sizes=(self.hidden_size,),
                    activation=self.activation_fn.__name__,
                    alpha=self.alpha,   # Pass alpha
                    beta=self.beta,     # Pass beta
                    batch_size=self.n_batch,
                    max_iter=self.n_epoch,
                    solver=self.optim, # 'lbfgs'#  if self.lbfgs else 'adam'
                ) for _ in range(self.num_experts)
            ])
        elif self.base_learner == 'LogReg':
            self.experts = nn.ModuleList([
                LogisticRegressionTorch(
                    input_dim=input_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    num_epochs=self.n_epoch
                ) for _ in range(self.num_experts)
            ])
        elif self.base_learner == 'CalibratedBooster':
            self.experts = nn.ModuleList([
                CalibratedBooster(
                    input_dim=input_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    model_type= 'lgbm', # 'cat', # 'rf', # 'logreg', # 'xgb', #  
                    task= 'classification', # 'regression', # 
                    calibration_method= None, #'isotonic', # 'sigmoid', # 
                    output_act=0 # self.output_mode
                ) for _ in range(self.num_experts)
            ])
        elif self.base_learner == 'Parzen':
            self.experts = nn.ModuleList([
                MLPBayesSwitch(
                    n_epoch = self.n_epoch,
                    n_batch = self.n_batch,
                    type_batch = self.mode,
                    layers_size=[self.hidden_size],
                    drop_out = [0, self.drop_out],
                    activations=[self.activation_fn.__name__, 'mod_tanh'],
                    class_prob = [0.5, 0.5],
                    alpha=self.alpha,
                    beta=self.beta
                ) for _ in range(self.num_experts)
            ])
        else:
            # Create experts with the actual input size
            self.experts = nn.ModuleList([
                AsymmetricMLP(
                    input_size,
                    self.hidden_size,
                    self.alpha,
                    self.beta,
                    self.drop_out,
                    self.activation_fn,
                    self.output_mode
                ) for _ in range(self.num_experts)
            ])
            
            
    def generate_experts_data(self, x, y, w=None, Q_RB_S=1, RB_each_expert=True):
        """
        Generates rebalanced and augmented datasets for experts using SMOTE, while adjusting weights and applying label switching.
    
        Parameters:
        -----------
        x : torch.Tensor
            Features tensor (N_samples, N_features).
        y : torch.Tensor
            Labels tensor (N_samples,).
        w : torch.Tensor, optional
            Sample weights tensor (N_samples,). Defaults to uniform weights if not provided.
        Q_RB_S : float, optional
            Desired ratio of the majority to minority class in the rebalanced dataset. Defaults to 1 (no rebalancing).
        RB_each_expert : bool, optional
            If True, applies SMOTE separately for each expert. If False, shares the same rebalanced dataset among all experts.
    
        Workflow:
        ---------
        1. Converts input tensors `x`, `y`, and `w` to numpy arrays for SMOTE compatibility.
        2. Handles binary labels (supports both 0/1 and -1/+1 formats) and computes class proportions.
        3. Applies SMOTE to rebalance the dataset, optionally extending sample weights for synthetic data.
        4. Adjusts labels and weights for each expert based on label switching parameters `alpha` and `beta`.
        5. Converts the final datasets back to PyTorch tensors and assigns them to each expert.
    
        Notes:
        ------
        - SMOTE generates synthetic samples for the minority class to achieve the desired class ratio.
        - If `RB_each_expert` is True, each expert gets a unique rebalanced dataset; otherwise, all share the same.
        - Label switching adjusts targets (`y`) and weights (`w`) based on predefined perturbation factors (`alpha` and `beta`).
    
        """
        x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
        w_np = w.cpu().numpy() if w is not None else np.ones_like(y_np, dtype=np.float32)
    
        # Binary label handling
        unique_labels = np.unique(y_np)
        
        if set(unique_labels) == {0, 1}:  # Case for binary {0, 1}
            self.bin_format = 0
            y_np = np.where(y_np == 0, -1, 1)
        elif set(unique_labels) == {-1, 1}:  # Case for binary {-1, 1}
            self.bin_format = -1
        
        QP_tr = compute_imbalance_ratio(y_np)
        
        self.QP_tr = QP_tr
    
        apply_rebalancing = False
        if Q_RB_S > 1:
            sampling_strategy = min(Q_RB_S / QP_tr, 1.0)
            self.Q_RB_S = QP_tr*sampling_strategy
            apply_rebalancing = True
    
        def apply_smote(x_np, y_np, w_np):
            """
            Applies SMOTE to rebalance the dataset and adjusts weights for synthetic samples.
            
            Returns:
            --------
            X_RB : numpy.ndarray
                Features of the rebalanced dataset.
            y_RB : numpy.ndarray
                Labels of the rebalanced dataset.
            w_RB : numpy.ndarray
                Weights of the rebalanced dataset.
            """
            try:
                smote = SMOTE(random_state=random.randint(1, 100), sampling_strategy=sampling_strategy)
                X_RB, y_RB = smote.fit_resample(x_np, y_np)
                n_synthetic = len(X_RB) - len(x_np)
                if n_synthetic > 0:
                    minority_class = min(y_np)
                    avg_weight = np.mean(w_np[y_np == minority_class])
                    synthetic_weights = np.full(n_synthetic, avg_weight)
                    w_RB = np.concatenate([w_np, synthetic_weights])
                else:
                    w_RB = w_np
                return X_RB, y_RB, w_RB
            except ValueError:
                return x_np, y_np, w_np
    
        def process_labels_and_weights(X_RB, y_RB, w_RB):
            """
            Processes rebalanced labels and weights, including optional label switching.
            
            Returns:
            --------
            X_RB : torch.Tensor
                Features tensor for the expert.
            y_RB_SW : torch.Tensor
                Adjusted labels tensor for the expert.
            w_RB_SW : torch.Tensor
                Adjusted weights tensor for the expert.
            """
            targets_sw, w_RB_SW = (y_RB, w_RB) if self.alpha == 0 and self.beta == 0 else label_switching(y_RB, w_RB, self.alpha, self.beta)
            targets_sw = torch.from_numpy(targets_sw).to(x.device)
            w_RB_SW = torch.from_numpy(w_RB_SW).to(x.device)
            y_RB_SW = torch.where(targets_sw > 0, (1 - 2 * self.beta), -(1 - 2 * self.alpha))
            return torch.from_numpy(X_RB).float().to(x.device), y_RB_SW, w_RB_SW
    
        if apply_rebalancing:
            if RB_each_expert:
                # Generate unique SMOTE data for each expert
                for expert in self.experts:
                    X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np)
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB)
            else:
                # Generate a single rebalanced dataset shared by all experts
                X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np)
                for expert in self.experts:
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB)
        else:
            # No rebalancing; all experts use the original dataset
            for expert in self.experts:
                expert.X, expert.y, expert.w = process_labels_and_weights(x_np, y_np, w_np)

    def fit(self, x_train, y_train, sample_weight=None):
        """
        Fit the ensemble model using training data.
    
        Parameters:
        - x_train: Tensor or NumPy array of shape (n_samples, n_features) for training data.
        - y_train: Tensor or NumPy array of shape (n_samples,) for training labels.
        - sample_weight: Optional tensor or NumPy array of shape (n_samples,) for sample weights.
        """
        # Ensure x_train and y_train are PyTorch tensors
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()  # Use float to handle potential binary labels
    
        # Handle sample_weight once at the beginning
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = torch.from_numpy(sample_weight).float()
        else:
            # Initialize weights to 1.0 for all samples if no weights are provided
            sample_weight = torch.ones_like(y_train, dtype=torch.float32)
            
        # Ensure the data is on the same device as the model
        if len(list(self.parameters())) > 0:
            device = next(self.parameters()).device
        else:
            device = torch.device("cpu")  # Fallback to CPU if no model parameters exist

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        sample_weight = sample_weight.to(device)
        
        # Generate internal data for experts
        self.generate_experts_data(x_train, y_train, sample_weight,
                                   Q_RB_S=self.Q_RB_S,
                                   RB_each_expert=False)
        
        # Fit expert models using the already processed sample_weight
        self.fit_expert_model(sample_weight, epochs=self.n_epoch, batch_size=self.n_batch, optim=self.optim)
        
        return self
                
    def fit_expert_model(self, w_train, epochs=50, batch_size=256, optim='lbfgs'):
        """
        Train experts using their stored data with a specified optimization method.
    
        Handles different base learners (Parzen, AMLP, LogReg) and optimizers (LBFGS, Adam, RMSprop, etc.).
        Sample weights are applied during training to adjust the importance of individual samples for each expert.
    
        Parameters:
        - w_train: Training weights (tensor) to scale expert-specific weights.
        - epochs: Number of training epochs (int) for gradient-based optimizers.
        - batch_size: Batch size for training with gradient-based optimizers (int).
        - optim: Optimization algorithm to use ('lbfgs', 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta').
                 For 'Parzen' base learner, this parameter is ignored as it uses a direct fit method.
    
        Returns:
        - self: The updated model after training.
        """
    
        # Initialize weights for each expert
        weights = torch.ones((self.experts[0].y.shape[0], self.num_experts)).to(w_train.device)
    
        # Compute weights for each expert
        for i, expert in enumerate(self.experts):
            weights[:, i] = compute_weights(expert.y, RI_C=self.Q_RB_C, Q_P=self.QP_tr, mode='normal') * expert.w  # Scale by expert's internal weights
    
        if self.base_learner in ['Parzen', "CalibratedBooster"]:
            for i, expert in enumerate(self.experts):
                x_np, y_np = expert.X.cpu().numpy(), expert.y.cpu().numpy()
                expert.fit(x_np, y_np, weights[:, i].cpu().numpy())  # Pass weights to Parzen fit
        elif self.base_learner in ['AMLP', 'LogReg']:
            for i, expert in enumerate(self.experts):
                x_np, y_np = expert.X.cpu().numpy(), expert.y.cpu().numpy()
                expert.fit(x_np, y_np) 
        elif self.base_learner == 'FAMLP':  # This condition is redundant if the previous one covers it
            if optim == 'lbfgs':
                for i, expert in enumerate(self.experts):
                    optim_LBFGS_scipy = LBFGSScipy(
                        expert.parameters(),
                        max_iter=150,
                        max_eval=150,
                        tolerance_grad=1e-04,
                        tolerance_change=10e6 * eps,  # `eps` is assumed to be defined outside
                        history_size=10
                    )
                    train_model(
                        expert.X,
                        expert.y,
                        expert,
                        self.loss_fn_e,
                        optim_LBFGS_scipy,
                        weights[:, i],
                        num_epochs=epochs,
                        batch_size=None,
                        mode=self.mode,
                        lbfgs=True,
                        debug=False
                    )
            else:
                for i, expert in enumerate(self.experts):
                    if optim == 'adam':
                        optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)
                    elif optim == 'adamw':
                        optimizer = torch.optim.AdamW(expert.parameters(), lr=0.01, weight_decay=1e-2)
                    elif optim == 'rmsprop':
                        optimizer = torch.optim.RMSprop(expert.parameters(), lr=0.001)
                    elif optim == 'sgd':
                        optimizer = torch.optim.SGD(expert.parameters(), lr=0.1, momentum=0.9)
                    elif optim == 'adagrad':
                        optimizer = torch.optim.Adagrad(expert.parameters(), lr=0.01)
                    elif optim == 'adadelta':
                        optimizer = torch.optim.Adadelta(expert.parameters(), lr=0.01)
                    else:
                        raise ValueError(f"Unsupported optimizer: {optim}")
    
                    train_model(
                        expert.X,
                        expert.y,
                        expert,
                        self.loss_fn_e,
                        optimizer,
                        weights[:, i],
                        num_epochs=epochs,
                        batch_size=batch_size,
                        mode=self.mode,
                        lbfgs=(optim == 'lbfgs'),
                        debug=False
                    )
        else:
            raise ValueError(f"Unsupported base learner: {self.base_learner}")
    
        return self
    
    # Get outputs from all experts
    def get_expert_outputs(self, x):
        expert_outputs = self.experts[0](x)
        aux_outputs = torch.zeros_like(expert_outputs)
        for i in range(self.num_experts-1):
            aux_outputs = self.experts[i+1](x)
            expert_outputs = torch.cat((expert_outputs, aux_outputs), 1)
        return expert_outputs

    # Predict outputs of each expert
    def predict_expert_outputs(self, x):
        with torch.no_grad():
            return self.get_expert_outputs(x) # .double())

    def forward(self, x):
        """
        This method performs the forward pass by calculating the predictions from the experts
        and returning the averaged prediction (o_pred) across experts.
        """
    
        # Convert input to tensor
        x_torch = torch.from_numpy(x).float()
    
        # Get expert outputs and compute their average
        expert_outputs = self.predict_expert_outputs(x_torch)
        o_pred = expert_outputs.mean(dim=1)  # Average over experts
        
        return o_pred.numpy()  # Forward method now gives output in the 0 to 1 range
    
    
    def predict(self, x):
        """
        Final prediction averaging over experts and applying threshold to obtain class labels.
        """
        QP_tr = self.QP_tr
        Q_tr = self.QC * self.QP_tr
        
        Q_RB_C = min(1, self.Q_RB_C) # self.Q_RB_C # 
        Q_RB_S = min(1, self.Q_RB_S) # self.Q_RB_S #
        
        QR_tr = max(1, QP_tr / (Q_RB_C * Q_RB_S))
        Q_ratio = (Q_tr / (Q_tr + QR_tr)) # 1/2 # 
               
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
    
        # Apply thresholding to get the final class labels
        y = np.ones_like(o_pred)
        eta_th = (2 * (self.alpha + (1 - self.alpha - self.beta) * Q_ratio) - 1)
        y[o_pred < eta_th] = self.bin_format
        
        return y.astype(int)
    
    def predict_proba(self, x):
        """
        Returns the predicted probability (o_pred) without applying threshold logic.
        This method is equivalent to predict but returns o_pred instead of y.
        """
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
        
        # Normalize to be between 0 and 1 (for proba outputs)
        o_pred_01 = (o_pred + 1) / 2
        
        return o_pred_01.numpy().astype(int)  # Convert back to numpy if necessary
