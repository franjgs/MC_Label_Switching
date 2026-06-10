import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer
from functools import reduce

from libraries.functions import generate_batches

eps=np.finfo(float).eps

# LBFGSScipy: taken from https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
class LBFGSScipy(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now CPU only
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._n_iter = 0
        self._last_loss = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """
        Perform a single outer optimization step using SciPy L-BFGS-B.
    
        Parameters
        ----------
        closure : callable
            Closure that reevaluates the model, computes the loss, and performs
            backward propagation. It must call zero_grad() internally.
    
        Returns
        -------
        torch.Tensor or None
            Final loss recorded during the SciPy optimization call.
        """
        assert len(self.param_groups) == 1
    
        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']
    
        first_param = self._params[0]
        param_dtype = first_param.dtype
        param_device = first_param.device
    
        def wrapped_closure(flat_params):
            """
            SciPy-compatible closure. The provided flat parameters are injected
            into the model before reevaluating the objective.
            """
            flat_params_torch = torch.from_numpy(flat_params).to(
                device=param_device,
                dtype=param_dtype
            )
            self._distribute_flat_params(flat_params_torch)
    
            loss = closure()
            self._last_loss = loss
    
            loss_value = float(loss.detach().cpu().item())
            flat_grad = self._gather_flat_grad().detach().cpu().numpy()
    
            return loss_value, flat_grad
    
        def callback(flat_params):
            self._n_iter += 1
    
        initial_params = self._gather_flat_params().detach().cpu().numpy()
    
        result = fmin_l_bfgs_b(
            wrapped_closure,
            initial_params,
            maxiter=max_iter,
            maxfun=max_eval,
            factr=tolerance_change / eps,
            pgtol=tolerance_grad,
            epsilon=1e-08,
            m=history_size,
            callback=callback
        )
    
        # Explicitly distribute the final parameters returned by SciPy.
        final_params = torch.from_numpy(result[0]).to(
            device=param_device,
            dtype=param_dtype
        )
        self._distribute_flat_params(final_params)
    
        return self._last_loss
        
        
def create_dataloader(X, y, weights, batch_size, mode='random', seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = TensorDataset(X, y, weights)

    # Generate indices using the original generate_batches (NumPy)
    batch_indices_orig = list(generate_batches(batch_size, y.numpy() if mode != "random" else len(y), mode, seed))
    batch_indices_orig = [sorted(batch) for batch in batch_indices_orig]  # Sort indices

    # Create a custom sampler using the generated indices
    from torch.utils.data import Sampler

    class BatchSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __iter__(self):
            for batch in self.indices:
                yield torch.tensor(batch)  # Yield PyTorch tensors

    batch_sampler = BatchSampler(batch_indices_orig)

    # Create the DataLoader using the custom sampler
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    return dataloader


def train_model(
    x,
    y,
    model,
    loss_fn,
    optimizer,
    weights,
    num_epochs=10,
    batch_size=None,
    mode='random',
    lbfgs=False,
    debug=False,
    early_stopping=False,
    es_patience=5,
    es_tol=1e-4,
    normalize_monitor_loss=True
):
    """
     Train a model using either gradient-based optimizers or LBFGS.
    
     Parameters
     ----------
     x : torch.Tensor
         Input features.
     y : torch.Tensor
         Target labels.
     model : torch.nn.Module
         PyTorch model to train.
     loss_fn : callable
         Loss function with signature loss_fn(outputs, labels, weights).
     optimizer : torch.optim.Optimizer
         Optimizer instance.
     weights : torch.Tensor
         Per-sample training weights.
     num_epochs : int, optional
         Maximum number of training epochs.
     batch_size : int or str or None, optional
         Batch size for the data loader. For LBFGS, full-batch mode is enforced.
     mode : str, optional
         Sampling mode for the data loader.
     lbfgs : bool, optional
         If True, use LBFGS optimization logic.
     debug : bool, optional
         If True, print debug information.
     early_stopping : bool, optional
         If True, stop training when the epoch loss stops improving.
     es_patience : int, optional
         Number of consecutive epochs allowed without meaningful improvement.
     es_tol : float, optional
         Minimum relative improvement required to reset patience.
         For example, es_tol=1e-4 means that the epoch loss must improve by
         more than 0.01% relative to the best loss seen so far.
    
     Returns
     -------
     torch.nn.Module
         Trained model.
     """
    
    if lbfgs:
        effective_batch_size = len(x)
    else:
        effective_batch_size = len(x) if batch_size is None or batch_size == 'auto' else int(batch_size)

    trainloader = create_dataloader(
        x,
        y,
        weights,
        batch_size=effective_batch_size,
        mode=mode
    )

    first_param = next(model.parameters(), None)
    if first_param is None:
        model_dtype = torch.float32
        model_device = torch.device("cpu")
    else:
        model_dtype = first_param.dtype
        model_device = first_param.device

    model.train()

    best_epoch_loss = np.inf
    no_improve_count = 0

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for batch_idx, (features, labels, batch_weights) in enumerate(trainloader):
            features = features.to(device=model_device, dtype=model_dtype)
            labels = labels.to(device=model_device, dtype=model_dtype).view(-1, 1)
            batch_weights = batch_weights.to(device=model_device, dtype=model_dtype).view(-1, 1)

            batch_n = int(features.shape[0])

            if lbfgs:
                def closure():
                    """Compute LBFGS loss and gradients for the current full batch."""
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = loss_fn(outputs, labels, batch_weights)
                    loss.backward()
                    return loss

                step_result = optimizer.step(closure)
                raw_loss = float(step_result.detach().cpu().item())

            else:
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels, batch_weights)
                loss.backward()
                optimizer.step()
                raw_loss = float(loss.detach().cpu().item())

            # Use raw_loss for optimization, but normalized loss for monitoring.
            if normalize_monitor_loss:
                monitored_batch_loss = raw_loss / max(batch_n, 1)
            else:
                monitored_batch_loss = raw_loss

            epoch_loss_sum += monitored_batch_loss * batch_n
            epoch_sample_count += batch_n

            """
            if debug and batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}, "
                    f"raw_loss={raw_loss:.5f}, "
                    f"monitored_loss={monitored_batch_loss:.8f}, "
                    f"batch_size={batch_n}"
                )
            """
        epoch_loss = epoch_loss_sum / max(epoch_sample_count, 1)

        if debug:
            print(
                f"Epoch {epoch + 1} completed | "
                f"normalized_epoch_loss={epoch_loss:.8f} | "
                f"samples={epoch_sample_count}"
            )

        if early_stopping:
            if np.isfinite(best_epoch_loss):
                rel_improvement = (best_epoch_loss - epoch_loss) / max(abs(best_epoch_loss), 1e-12)
            else:
                rel_improvement = np.inf

            if rel_improvement > es_tol:
                best_epoch_loss = epoch_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if debug:
                print(
                    f"Early stopping monitor | "
                    f"best_loss={best_epoch_loss:.8f} | "
                    f"current_loss={epoch_loss:.8f} | "
                    f"rel_improvement={rel_improvement:.6e} | "
                    f"no_improve_count={no_improve_count}/{es_patience}"
                )

            if no_improve_count >= es_patience:
                if debug:
                    print(
                        f"Stopping early at epoch {epoch + 1} after "
                        f"{es_patience} epochs without sufficient relative improvement."
                    )
                break

    return model


def weighted_mse_loss(inputs, target, weights=None):
    """
    Compute the weighted Mean Squared Error loss.
    
    Args:
        inputs (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        weights (torch.Tensor, optional): Sample weights. Defaults to 1 for all samples.
        
    Returns:
        torch.Tensor: Scalar loss value (0.5 * sum of weighted squared errors).
    """
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    if weights is None:
        weights = torch.ones_like(inputs)

    # Compute weighted MSE
    weighted_diff = weights * (inputs - target) ** 2
    # Calculate 0.5 * sum(w * (y_pred - y_true)^2)
    return 0.5 * torch.sum(weighted_diff)

def weighted_kl_loss(inputs, target, weights=None):
    """
    Compute a weighted loss similar to KL Divergence.

    Parameters:
        inputs (torch.Tensor or np.ndarray): Predicted logits or probabilities.
        target (torch.Tensor or np.ndarray): Target probabilities.
        weights (torch.Tensor or None): Weights for each sample. If None, uniform weights are used.

    Returns:
        torch.Tensor: The computed weighted KL divergence loss.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    
    # Ensure inputs and target are in the range [0, 1]
    inputs_01 = torch.clip(0.5 * (inputs + 1), 1e-5, 1 - 1e-5)  # Avoid log(0)
    target_01 = torch.clip(0.5 * (target + 1), 1e-5, 1 - 1e-5)  # Avoid log(0)
    
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs_01)

    # Compute the KL divergence term
    kl_div = target_01 * torch.log(target_01 / inputs_01) + (1 - target_01) * torch.log((1 - target_01) / (1 - inputs_01))
    
    # Apply weights and sum the loss
    weighted_kl = weights * kl_div
    return torch.sum(weighted_kl)
    
def weighted_bce_loss(inputs, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).double()  # Convert to double (float64)

    # Ensure inputs and targets are within [0, 1] for BCE loss
    inputs_01 = torch.clamp(0.5 * (inputs + 1), 0, 1).double()  # Ensure double type
    targets_01 = torch.clamp(0.5 * (target + 1), 0, 1).double()  # Ensure double type

    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs_01).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double
        
    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]
    
    # Compute the BCE loss
    loss_bce = nn.BCELoss(weight=weights)
    return loss_bce(inputs_01, targets_01)

def weighted_bce_logit_loss(inputs, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).double()  # Convert to double (float64)
        
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double

    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]
    
    # Compute the BCEWithLogits loss
    loss_bce_logit = nn.BCEWithLogitsLoss(weight=weights)
    return loss_bce_logit(inputs, target)

def f1_loss(predict, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    target = 0.5 * (target + 1)
    if isinstance(predict, np.ndarray):
        predict = torch.from_numpy(predict).double()  # Convert to double (float64)
    predict = torch.clamp(0.5 * (predict + 1), 0, 1)  # Ensure values are within [0, 1
    
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(target).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double

    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]]

    # Initialize BCEWithLogitsLoss once, passing weight
    bce_loss_fn = nn.BCEWithLogitsLoss(weight=weights)

    loss = 0
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += bce_loss_fn(predict[:, lack_cls], target[:, lack_cls])

    tp = predict * target
    tp = tp.sum(dim=0)
    
    fp = predict * (1 - target)
    fp = fp.sum(dim=0)
    
    fn = ((1 - predict) * target)
    fn = fn.sum(dim=0)
    
    tn = (1 - predict) * (1 - target)
    tn = tn.sum(dim=0)
    
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-8)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-8)
    cost_class1 = 1 - soft_f1_class1  # Reduce 1 - soft_f1_class1 to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # Reduce 1 - soft_f1_class0 to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # Take into account both class 1 and class 0
    macro_cost = cost.mean()  # Average on all labels
    
    return macro_cost + loss

# Mapping for loss functions
LOSS_FUNCTIONS = {
    "MSE": weighted_mse_loss,
    "KL": weighted_kl_loss,
    "BCE": weighted_bce_loss,
    "BCE_logit": weighted_bce_logit_loss,
    "F1": f1_loss,
}
