#!/usr/bin/env python3
"""SuperBigFLICA utilities with corrected held-out dictionary projection."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch import optim
from torch.utils.data import DataLoader, Dataset


def load_nidp_csv(path):
    """Load a numeric CSV as subjects by variables, preserving missing cells."""
    values = np.genfromtxt(path, delimiter=',', dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values[:, None]
    return values


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    print(f"Random seed set to {seed} for reproducibility.")


def _as_2d_targets(y):
    y = np.asarray(y)
    y = np.atleast_2d(y)
    if y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T
    return y


def _safe_pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(scipy.stats.pearsonr(x, y)[0])


class loss_SuperBigFLICA_regression(nn.Module):
    def __init__(self, ntask, nmod, nsub, device='cuda',
                 auto_weight=(1, 1, 1, 1),
                 lambdas=(0.25, 0.25, 0.25, 0.25)):
        super().__init__()
        self.ntask = int(ntask)
        self.nmod = int(nmod)
        self.nsub = int(nsub)
        self.auto_weight = list(auto_weight)

        lam = torch.as_tensor(np.asarray(lambdas).flatten(),
                              dtype=torch.float32, device=device)
        if lam.numel() != 4 or torch.sum(lam) <= 0:
            raise ValueError("lambdas must contain four values with positive sum")
        self.register_buffer('lambdas', lam / torch.sum(lam))

        self.sigma1 = nn.Parameter(torch.ones(1, self.nmod, device=device))
        self.sigma2 = nn.Parameter(torch.ones(1, self.nmod, device=device))
        self.sigma3 = nn.Parameter(torch.ones(1, self.ntask, device=device))
        self.sigma4 = nn.Parameter(torch.ones(1, self.ntask, device=device))
        self.sigma5 = nn.Parameter(torch.ones(1, self.ntask, device=device))

    def forward(self, recon_x, x_orig, sptial_loadings, y_pred,
                y_train, pred_weights, lat_train):
        del lat_train
        batch_prop = recon_x[0].shape[0] / float(self.nsub)
        sigma1, sigma2 = self.sigma1.square(), self.sigma2.square()
        sigma3, sigma4 = self.sigma3.square(), self.sigma4.square()
        sigma5 = self.sigma5.square()

        if self.auto_weight[0] == 0:
            self.sigma1.requires_grad_(False)
        if self.auto_weight[1] == 0:
            self.sigma2.requires_grad_(False)
        if self.auto_weight[2] == 0:
            self.sigma3.requires_grad_(False)
        if self.auto_weight[3] == 0:
            self.sigma4.requires_grad_(False)
            self.sigma5.requires_grad_(False)

        loss_recon = torch.zeros((), device=y_pred.device)
        for i in range(self.nmod):
            diff = recon_x[i] - x_orig[i]
            loss_recon += diff.square().mean() / sigma1[0, i].square() / 2
        loss_recon += torch.sum(torch.log(sigma1 + 1))

        loss_spatial = torch.zeros((), device=y_pred.device)
        for i in range(self.nmod):
            loss_spatial += (sptial_loadings[i].abs().mean() * batch_prop /
                             sigma2[0, i])
        loss_spatial += 2 * torch.sum(torch.log(sigma2 + 1))

        # Preserve [batch, target] shape. The old squeeze caused broadcasting.
        target = y_train
        if target.ndim == 1:
            target = target[:, None]
        if target.shape != y_pred.shape:
            if target.numel() != y_pred.numel():
                raise ValueError(f"target shape {target.shape} != prediction {y_pred.shape}")
            target = target.reshape_as(y_pred)
        target = torch.where(torch.isnan(target), y_pred.detach(), target)

        loss_mse = ((target - y_pred).square() / sigma3.square() / 2).mean()
        loss_mse += torch.sum(torch.log(sigma3 + 1))

        loss_pred = torch.mean(torch.abs(pred_weights) / sigma4 * batch_prop)
        loss_pred += 2 * torch.sum(torch.log(sigma4 + 1))
        loss_pred += torch.mean(pred_weights.square() * batch_prop /
                                sigma5.square() / 2)
        loss_pred += torch.sum(torch.log(sigma5 + 1))

        total = (self.lambdas[0] * loss_recon +
                 self.lambdas[1] * loss_spatial +
                 self.lambdas[2] * loss_mse +
                 self.lambdas[3] * loss_pred)
        return total, loss_recon, loss_spatial, loss_mse, loss_pred


class SupervisedFLICAmodel(nn.Module):
    def __init__(self, nfea, nlat, ntask, dropout=0.5, device='cpu',
                 init_spatial_loading=None, init_weight_pred=None,
                 init_bais_pred=None):
        super().__init__()
        self.ntask = int(ntask)
        self.dropout = float(dropout)
        self.nlat = int(nlat)
        self.nmod = len(nfea)

        self.spatial_loading = nn.ParameterList()
        if init_spatial_loading is None:
            for nf in nfea:
                self.spatial_loading.append(
                    nn.Parameter(torch.randn(int(nf), self.nlat, device=device)))
        else:
            for loading in init_spatial_loading:
                self.spatial_loading.append(nn.Parameter(
                    torch.as_tensor(loading, dtype=torch.float32, device=device)))

        self.mod_weight = nn.Parameter(torch.ones(self.nlat, self.nmod,
                                                   device=device))
        if init_weight_pred is None:
            self.weight_pred = nn.Parameter(torch.randn(self.nlat, self.ntask,
                                                        device=device))
            self.bias_pred = nn.Parameter(torch.randn(1, self.ntask,
                                                       device=device))
        else:
            self.weight_pred = nn.Parameter(torch.as_tensor(
                init_weight_pred, dtype=torch.float32, device=device))
            self.bias_pred = nn.Parameter(torch.as_tensor(
                init_bais_pred, dtype=torch.float32, device=device))
        self.batch_norm = nn.BatchNorm1d(self.nlat)

    def forward(self, x, device='cpu'):
        mod_weight = F.softmax(self.mod_weight, dim=1)
        latent = torch.zeros(x[0].shape[0], self.nlat, device=device,
                             dtype=torch.float32)
        for i in range(self.nmod):
            dat = F.dropout(x[i], self.dropout, training=self.training)
            latent += (dat.float().matmul(self.spatial_loading[i].float())
                       .matmul(torch.diag(mod_weight[:, i].float())))
        latent /= float(self.nmod)
        latent = self.batch_norm(latent)
        latent = F.dropout(latent, self.dropout, training=self.training)

        outputs = []
        for i in range(self.nmod):
            w = self.spatial_loading[i].matmul(torch.diag(mod_weight[:, i]))
            outputs.append(latent.matmul(w.T))
        pred = latent.matmul(self.weight_pred) + self.bias_pred
        return outputs, self.spatial_loading, latent, pred, self.weight_pred


def _nets_zscore_exact(x):
    """Column z-score used by the released SuperBigFLICA mMIGP code."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, ddof=0)
    std[std == 0] = 0.1
    return centered / std


def MultiModality_MIGP_faster_exact(x, k):
    """Exact small-N algebra of the released multimodal MIGP initializer.

    This follows ``MultiModality_MIGP_faster`` from Gong et al.'s released
    SuperBigFLICA code.  For N=192 it evaluates the full subject covariance,
    so no incremental approximation or subject ordering is involved.
    """
    arrays = [np.asarray(item, dtype=np.float64) for item in x]
    n_subjects = arrays[0].shape[0]
    if k <= 0 or k >= n_subjects:
        raise ValueError(f"mMIGP k must be in [1, {n_subjects - 1}], got {k}")
    covariance = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    for data in arrays:
        standardized = _nets_zscore_exact(data)
        covariance += standardized @ standardized.T / data.shape[1]
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance, subset_by_index=(n_subjects - k, n_subjects - 1))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    # The released code returns U * eigenvalue, then its initializer
    # column-normalizes this matrix before the modality back-projections.
    return eigenvectors * eigenvalues[np.newaxis, :], eigenvalues


def Initialize_SuperBigFLICA(x_train, y_train, nlat, train_ind=None):
    """Released multimodal-MIGP initialization, isolated for sensitivity."""
    del train_ind
    y_train = _as_2d_targets(y_train).astype(np.float64)
    mmigp_scaled, eigenvalues = MultiModality_MIGP_faster_exact(
        x_train, nlat)
    norms = np.sqrt(np.sum(mmigp_scaled ** 2, axis=0, keepdims=True))
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise RuntimeError("mMIGP produced zero or non-finite component norm")
    mmigp_unit = mmigp_scaled / norms
    spatial = [
        (np.asarray(data, dtype=np.float64).T @ (mmigp_unit / norms))
        .astype(np.float32)
        for data in x_train
    ]
    y_filled = y_train.copy()
    y_filled[~np.isfinite(y_filled)] = 0.0
    common = np.zeros((x_train[0].shape[0], nlat), dtype=np.float64)
    for data, loading in zip(x_train, spatial):
        common += np.asarray(data, dtype=np.float64) @ loading
    common_z = _nets_zscore_exact(common)
    design = np.column_stack((common_z, np.ones(common_z.shape[0])))
    coef = np.linalg.pinv(design) @ y_filled
    diagnostics = {
        "method": "released_multimodality_migp_faster_small_n_algebra",
        "n_subjects": int(x_train[0].shape[0]),
        "n_modalities": int(len(x_train)),
        "n_components": int(nlat),
        "eigenvalues": eigenvalues.tolist(),
        "covariance_effective_rank": int(np.linalg.matrix_rank(
            sum(
                _nets_zscore_exact(data) @ _nets_zscore_exact(data).T /
                data.shape[1]
                for data in x_train
            )
        )),
        "unit_basis_orthogonality_max_abs_error": float(np.max(np.abs(
            mmigp_unit.T @ mmigp_unit - np.eye(nlat)
        ))),
        "initial_common_abs_max": float(np.max(np.abs(common_z))),
        "initial_prediction_weight_abs_max": float(np.max(np.abs(coef[:-1]))),
    }
    return (spatial, coef[:-1].astype(np.float32),
            coef[-1:].astype(np.float32), mmigp_unit.astype(np.float32),
            diagnostics)

# -------------------------------------------------------------------------
# Dictionary learning and fixed-dictionary projection
# -------------------------------------------------------------------------

def dictionary_learning_pytorch(X, n_components=100, lambda_l1=0.1,
                                n_iter=8, lr=0.01, device='cpu',
                                random_seed=42, ridge=1e-4):
    """Consistent canonical sparse dictionary learning on training data.

    The coefficient objective is the per-subject canonical form
    ``0.5 ||x-Da||_2^2 + alpha ||a||_1``.  Unlike the historical mean loss,
    its sparse penalty is independent of feature count and split size.

    ``ridge`` is a dimensionless factor.  At each dictionary update the
    effective diagonal ridge is ``ridge * mean(diag(A.T @ A))`` so its
    strength remains stable when coefficient scale changes.
    """
    del lr
    set_random_seeds(random_seed)
    X = np.asarray(X, dtype=np.float32)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n_subjects, n_features = X.shape

    D = torch.randn(n_features, n_components, device=device)
    order = torch.randperm(n_subjects, device=device)
    seeded = min(n_subjects, n_components)
    D[:, :seeded] = Xt[order[:seeded]].T
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    A = torch.zeros(n_subjects, n_components, device=device)

    history = []
    total_reinitialized = 0
    for iteration in range(n_iter):
        A_np, solver = apply_dictionary(
            D.detach().cpu().numpy(), X, lambda_l1=lambda_l1,
            max_iter=1000, tol=5e-5, device=device,
            return_diagnostics=True)
        A = torch.as_tensor(A_np, dtype=torch.float32, device=device)

        sd = torch.std(A, dim=0, correction=0)
        positive = sd[sd > 0]
        reference = (torch.median(positive) if positive.numel()
                     else torch.zeros((), device=device))
        dead = sd < torch.maximum(
            torch.as_tensor(1e-8, device=device), reference * 1e-4)
        dead_indices = torch.nonzero(dead, as_tuple=False).flatten()
        for offset, atom_index in enumerate(dead_indices.tolist()):
            subject = offset % n_subjects
            residual = Xt[subject] - A[subject].matmul(D.T)
            norm = residual.norm()
            if norm < 1e-10:
                residual = torch.randn(n_features, device=device)
                norm = residual.norm()
            D[:, atom_index] = residual / norm
            A[:, atom_index] = 0.0
            A[subject, atom_index] = norm
        total_reinitialized += int(dead_indices.numel())

        ata = A.T.matmul(A)
        ata_mean_diagonal = torch.diagonal(ata).mean()
        if (not torch.isfinite(ata_mean_diagonal) or
                float(ata_mean_diagonal.item()) <= 0.0):
            raise RuntimeError(
                "Invalid A.T@A mean diagonal during dictionary update")
        effective_ridge = float(ridge) * float(ata_mean_diagonal.item())
        ridge_matrix = effective_ridge * torch.eye(
            n_components, device=device, dtype=ata.dtype)
        xta = Xt.T.matmul(A)
        try:
            D = torch.linalg.solve(ata + ridge_matrix, xta.T).T
        except RuntimeError:
            D = xta.matmul(torch.linalg.pinv(ata + ridge_matrix))
        D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)

        history.append({
            "outer_iteration": int(iteration + 1),
            "coefficient_solver": solver,
            "active_atoms_before_update": int(n_components - dead_indices.numel()),
            "reinitialized_atoms": int(dead_indices.numel()),
            "ata_mean_diagonal": float(ata_mean_diagonal.item()),
            "ridge_factor": float(ridge),
            "effective_ridge": float(effective_ridge),
        })
        print(f"Canonical dictionary iteration {iteration+1}/{n_iter}; "
              f"active={n_components-dead_indices.numel()}; "
              f"reinitialized={dead_indices.numel()}; "
              f"ata_mean_diag={float(ata_mean_diagonal.item()):.6g}; "
              f"effective_ridge={effective_ridge:.6g}")

    final_A, final_solver = apply_dictionary(
        D.detach().cpu().numpy(), X, lambda_l1=lambda_l1,
        max_iter=5000, tol=1e-5, device=device, return_diagnostics=True)
    return D.detach().cpu().numpy(), final_A, {
        "objective": "0.5*||x-Da||_2^2 + alpha*||a||_1 per subject",
        "alpha": float(lambda_l1),
        "dictionary_update_ridge_policy": (
            "factor * mean(diag(A.T @ A)) at every outer iteration"
        ),
        "dictionary_update_ridge_factor": float(ridge),
        "effective_ridge_min": float(min(
            item["effective_ridge"] for item in history)),
        "effective_ridge_max": float(max(
            item["effective_ridge"] for item in history)),
        "outer_iterations": int(n_iter),
        "total_reinitialized_atoms": int(total_reinitialized),
        "history": history,
        "final_training_solver": final_solver,
    }


def _soft_threshold(x, threshold):
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def apply_dictionary(D, X, lambda_l1=0.1, max_iter=5000,
                     tol=1e-5, device='cpu', return_diagnostics=False):
    """
    Estimate coefficients for a fixed dictionary with adaptive-restart FISTA.

    The same converged solver is used for train, validation, and test, fixing
    the systematic held-out coefficient shrinkage from the old 20-step SGD.
    """
    D = np.asarray(D, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    Dt = torch.as_tensor(D, dtype=torch.float32, device=device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)

    n_subjects, n_features = Xt.shape
    n_components = Dt.shape[1]
    if Dt.shape[0] != n_features:
        raise ValueError(f"D has {Dt.shape[0]} features; X has {n_features}")

    gram = Dt.T.matmul(Dt)
    x_times_d = Xt.matmul(Dt)
    largest_eigenvalue = torch.linalg.eigvalsh(gram).max()
    lipschitz = largest_eigenvalue.clamp_min(1e-12)
    step_size = 1.0 / lipschitz
    threshold = step_size * float(lambda_l1)

    A = torch.zeros(n_subjects, n_components, device=device)
    Y = A.clone()
    momentum = 1.0
    relative_change = np.inf
    converged_at = max_iter
    restart_count = 0

    for iteration in range(1, max_iter + 1):
        gradient = Y.matmul(gram) - x_times_d
        A_new = _soft_threshold(Y - step_size * gradient, threshold)
        delta = A_new - A
        relative_change = float((torch.linalg.norm(delta) /
                                 torch.linalg.norm(A_new).clamp_min(1e-12)).item())

        momentum_new = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        Y_new = A_new + ((momentum - 1.0) / momentum_new) * delta

        # O'Donoghue-Candes gradient restart.  This must use the current
        # extrapolated point Y.  Using Y_new here makes the dot product a
        # positive multiple of ||A_new - A||^2 and restarts every iteration,
        # accidentally reducing FISTA to unaccelerated ISTA.
        if torch.sum((Y - A_new) * (A_new - A)) > 0:
            momentum_new = 1.0
            Y_new = A_new.clone()
            restart_count += 1

        A, Y, momentum = A_new, Y_new, momentum_new
        if relative_change < tol:
            converged_at = iteration
            break

    residual = Xt - A.matmul(Dt.T)
    objective = (0.5 * residual.square().sum(dim=1).mean() +
                 float(lambda_l1) * A.abs().sum(dim=1).mean())
    print("Fixed-dictionary coding: "
          f"subjects={n_subjects}, features={n_features}, "
          f"components={n_components}, iterations={converged_at}, "
          f"relative_change={relative_change:.3e}, "
          f"objective={objective.item():.6g}")
    coefficients = A.detach().cpu().numpy()
    diagnostics = {
        "subjects": int(n_subjects),
        "features": int(n_features),
        "components": int(n_components),
        "iterations": int(converged_at),
        "max_iterations": int(max_iter),
        "relative_change": float(relative_change),
        "tolerance": float(tol),
        "converged": bool(relative_change < tol),
        "adaptive_restart_count": int(restart_count),
        "objective": float(objective.item()),
    }
    if return_diagnostics:
        return coefficients, diagnostics
    return coefficients


def process_modality(i, X_train, dicl_dim, X_val, X_test,
                     lambda_l1=0.1, projection_max_iter=5000,
                     projection_tol=1e-5, random_seed=42,
                     dictionary_update_ridge=1e-4):
    print(f"Starting dictionary learning for modality {i}")
    feature_mean = np.mean(X_train, axis=0, dtype=np.float64).astype(np.float32)
    X_train = np.asarray(X_train, dtype=np.float32) - feature_mean
    X_val = np.asarray(X_val, dtype=np.float32) - feature_mean
    X_test = np.asarray(X_test, dtype=np.float32) - feature_mean
    global_rms = float(np.sqrt(np.mean(X_train.astype(np.float64) ** 2)))
    if not np.isfinite(global_rms) or global_rms <= 0:
        raise RuntimeError(f"Invalid training-derived RMS for modality {i}")
    X_train = (X_train / global_rms).astype(np.float32)
    X_val = (X_val / global_rms).astype(np.float32)
    X_test = (X_test / global_rms).astype(np.float32)

    D, _, learning_diagnostics = dictionary_learning_pytorch(
        X_train, n_components=dicl_dim, lambda_l1=lambda_l1,
        n_iter=8, lr=0.01, device='cpu', random_seed=random_seed + i,
        ridge=dictionary_update_ridge)

    # Re-estimate all splits against the same final dictionary.
    A_train, train_solver = apply_dictionary(
        D, X_train, lambda_l1, projection_max_iter, projection_tol, 'cpu', True)
    A_val, val_solver = apply_dictionary(
        D, X_val, lambda_l1, projection_max_iter, projection_tol, 'cpu', True)
    A_test, test_solver = apply_dictionary(
        D, X_test, lambda_l1, projection_max_iter, projection_tol, 'cpu', True)

    train_sd = np.std(A_train, axis=0, ddof=0)
    train_sd[train_sd == 0] = np.nan
    val_ratio = np.nanmedian(np.std(A_val, axis=0, ddof=0) / train_sd)
    test_ratio = np.nanmedian(np.std(A_test, axis=0, ddof=0) / train_sd)
    print(f"Modality {i} projection scale: validation/train SD={val_ratio:.4f}; "
          f"test/train SD={test_ratio:.4f}")
    solver_diagnostics = {
        "modality_index": int(i),
        "canonical_alpha": float(lambda_l1),
        "equivalent_historical_mean_lambda": (
            float(lambda_l1) * float(dicl_dim) / float(X_train.shape[1])
        ),
        "dictionary_update_ridge_policy": (
            "factor * mean(diag(A.T @ A)) at every outer iteration"
        ),
        "dictionary_update_ridge_factor": float(dictionary_update_ridge),
        "dictionary_learning": learning_diagnostics,
        "train": train_solver,
        "validation": val_solver,
        "test": test_solver,
    }
    input_transform = {
        "feature_mean": feature_mean,
        "global_rms": float(global_rms),
    }
    return (i, D, A_train, A_val, A_test, solver_diagnostics,
            input_transform)


def parallel_dictionary_learning(Data_train, Data_validation, Data_test,
                                 dicl_dim, n_jobs=-1, lambda_l1=0.1,
                                 projection_max_iter=5000,
                                 projection_tol=1e-5, random_seed=42,
                                 dictionary_update_ridge=1e-4):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_modality)(
            i, Data_train[i], dicl_dim, Data_validation[i], Data_test[i],
            lambda_l1, projection_max_iter, projection_tol, random_seed,
            dictionary_update_ridge)
        for i in range(len(Data_train)))
    results = sorted(results, key=lambda item: item[0])

    dictionaries = []
    solver_diagnostics = []
    input_transforms = []
    for (i, D, A_train, A_val, A_test, diagnostics,
         input_transform) in results:
        Data_train[i] = A_train
        Data_validation[i] = A_val
        Data_test[i] = A_test
        dictionaries.append(D)
        solver_diagnostics.append(diagnostics)
        input_transforms.append(input_transform)
    return (dictionaries, Data_train, Data_validation, Data_test,
            solver_diagnostics, input_transforms)


def _coefficient_scale_guard(train, validation, test, inactive_sd_ratio=1e-3):
    """Standardize using training data and zero training-near-dead atoms."""
    train = np.asarray(train, dtype=np.float32)
    validation = np.asarray(validation, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mean = np.mean(train, axis=0, dtype=np.float64)
    sd = np.std(train, axis=0, ddof=0, dtype=np.float64)
    positive = sd[np.isfinite(sd) & (sd > 0)]
    if positive.size == 0:
        raise RuntimeError("All dictionary coefficient columns have zero variance")
    reference_sd = float(np.median(positive))
    sd_floor = max(float(np.finfo(np.float32).eps),
                   float(inactive_sd_ratio) * reference_sd)
    active = np.isfinite(sd) & (sd >= sd_floor)
    if not np.any(active):
        raise RuntimeError("Coefficient scale guard removed every dictionary atom")

    divisor = np.where(active, sd, 1.0)
    transformed = []
    for matrix in (train, validation, test):
        z = (matrix.astype(np.float64) - mean) / divisor
        z[:, ~active] = 0.0
        transformed.append(z.astype(np.float32))

    split_names = ("train", "validation", "test")
    split_stats = {}
    for name, raw, z in zip(split_names, (train, validation, test), transformed):
        finite_z = z[np.isfinite(z)]
        split_stats[name] = {
            "raw_abs_max": float(np.max(np.abs(raw))),
            "standardized_abs_max": float(np.max(np.abs(finite_z))),
            "standardized_abs_p99_9": float(np.percentile(np.abs(finite_z), 99.9)),
        }
    diagnostics = {
        "reference_train_sd_median_positive": reference_sd,
        "inactive_sd_ratio": float(inactive_sd_ratio),
        "train_sd_floor": sd_floor,
        "n_atoms": int(sd.size),
        "n_active_atoms": int(np.sum(active)),
        "n_inactive_atoms": int(np.sum(~active)),
        "train_sd_min": float(np.min(sd)),
        "train_sd_p01": float(np.percentile(sd, 1)),
        "train_sd_p05": float(np.percentile(sd, 5)),
        "train_sd_median": float(np.median(sd)),
        "train_sd_max": float(np.max(sd)),
        "split_stats": split_stats,
    }
    return transformed[0], transformed[1], transformed[2], active, diagnostics


class load_Multimodal_data(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.K = len(X)
        self.y = y
        self.transform = transform
        if self.K == 0:
            raise ValueError("At least one modality is required")
        n = len(self.X[0])
        for i, modality in enumerate(self.X):
            if len(modality) != n:
                raise ValueError(f"Modality {i} has {len(modality)} subjects; expected {n}")
        if y is not None and len(y) != n:
            raise ValueError(f"Targets have {len(y)} subjects; imaging has {n}")

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        image = [self.X[i][index, :] for i in range(self.K)]
        if self.y is None:
            return image
        return image, torch.as_tensor(self.y[index, :], dtype=torch.float32)


def SupervisedFLICA(x_train, y_train, nlat, x_test, y_test, Data_test,
                    output_dir, random_seed=666, train_ind=None,
                    init_method='random', dropout=0.25, device='cpu',
                    auto_weight=(1, 1, 1, 1),
                    lambdas=(0.25, 0.25, 0.25, 0.25), lr=0.001,
                    batch_size=512, maxiter=50, dicl_dim=100,
                    dictionary_alpha=0.1,
                    dictionary_update_ridge=1e-4,
                    inactive_sd_ratio=1e-3,
                    projection_max_iter=5000,
                    projection_tol=1e-5):
    output_path = Path(output_dir)
    y_train = _as_2d_targets(y_train).astype(np.float32)
    y_test = _as_2d_targets(y_test).astype(np.float32)
    x_train = [np.asarray(x, dtype=np.float32).copy() for x in x_train]
    x_test = [np.asarray(x, dtype=np.float32).copy() for x in x_test]
    Data_test = [np.asarray(x, dtype=np.float32).copy() for x in Data_test]

    set_random_seeds(random_seed)
    nmod, ntask = len(x_train), y_train.shape[1]
    device = torch.device(device)

    print("Applying Dictionary Learning")
    (dictionaries, x_train, x_test, Data_test,
     solver_diagnostics, input_transforms) = parallel_dictionary_learning(
        x_train, x_test, Data_test, dicl_dim, n_jobs=-1,
        lambda_l1=dictionary_alpha, projection_max_iter=projection_max_iter,
        projection_tol=projection_tol, random_seed=random_seed,
        dictionary_update_ridge=dictionary_update_ridge)
    Data_train = copy.deepcopy(x_train)
    Data_validation = copy.deepcopy(x_test)
    coefficient_dir = output_path / "coefficient_diagnostics"
    coefficient_dir.mkdir(exist_ok=False)
    dictionary_dir = output_path / "dictionary_artifacts"
    dictionary_dir.mkdir(exist_ok=False)
    for i, (dictionary, transform) in enumerate(zip(
            dictionaries, input_transforms)):
        np.save(dictionary_dir / f"modality_{i}_dictionary.npy", dictionary)
        np.savez_compressed(
            dictionary_dir / f"modality_{i}_input_transform.npz",
            feature_mean=transform["feature_mean"],
            global_rms=np.asarray([transform["global_rms"]], dtype=np.float64))
    with (coefficient_dir / "fista_solver_diagnostics.json").open("w") as stream:
        json.dump(solver_diagnostics, stream, indent=2)
        stream.write("\n")
    # Persist the computed coefficients before enforcing convergence so a
    # failed diagnostic remains fully auditable and need not be recomputed.
    for i in range(nmod):
        np.savez_compressed(
            coefficient_dir / f"modality_{i}_raw_coefficients.npz",
            train=x_train[i], validation=x_test[i], test=Data_test[i])
    unconverged = [
        (entry["modality_index"], split)
        for entry in solver_diagnostics
        for split in ("train", "validation", "test")
        if not entry[split]["converged"]
    ]
    if unconverged:
        raise RuntimeError(
            "Fixed-dictionary FISTA failed its convergence requirement for "
            f"{unconverged}; diagnostics were preserved in {coefficient_dir}"
        )
    print("Dictionary Learning Done.")

    nfea = [x.shape[1] for x in x_train]
    scale_diagnostics = []
    for i in range(nmod):
        (x_train[i], x_test[i], guarded_test, active,
         diagnostics) = _coefficient_scale_guard(
            x_train[i], x_test[i], Data_test[i], inactive_sd_ratio)
        raw_train = Data_train[i]
        coefficient_mean = np.mean(raw_train, axis=0, dtype=np.float64)
        coefficient_sd = np.std(raw_train, axis=0, ddof=0, dtype=np.float64)
        np.savez_compressed(
            dictionary_dir / f"modality_{i}_coefficient_normalization.npz",
            mean=coefficient_mean,
            sd=coefficient_sd,
            active=active)
        diagnostics["modality_index"] = int(i)
        diagnostics["active_atom_indices"] = np.flatnonzero(active).tolist()
        centered_active = (raw_train[:, active].astype(np.float64) -
                           coefficient_mean[active])
        diagnostics["active_coefficient_effective_rank"] = int(
            np.linalg.matrix_rank(centered_active))
        scale_diagnostics.append(diagnostics)
        if (diagnostics["n_active_atoms"] < 50 or
                diagnostics["active_coefficient_effective_rank"] < 50):
            with (coefficient_dir / "scale_guard_diagnostics.json").open("w") as stream:
                json.dump(scale_diagnostics, stream, indent=2)
                stream.write("\n")
            raise RuntimeError(
                f"Modality {i} has fewer than 50 active/effective "
                "coefficient directions")
        if max(
            diagnostics["split_stats"][split]["standardized_abs_max"]
            for split in ("validation", "test")
        ) > 100.0:
            with (coefficient_dir / "scale_guard_diagnostics.json").open("w") as stream:
                json.dump(scale_diagnostics, stream, indent=2)
                stream.write("\n")
            raise RuntimeError(
                f"Modality {i} still has an absolute standardized held-out "
                "coefficient above 100 after the training-only guard"
            )
        # Keep the unnormalized coefficients for reproducible extraction;
        # guarded_test is checked here and regenerated identically later.
        del guarded_test
    with (coefficient_dir / "scale_guard_diagnostics.json").open("w") as stream:
        json.dump(scale_diagnostics, stream, indent=2)
        stream.write("\n")

    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_stds[y_stds == 0] = 1.0
    y_train_z = (y_train - y_mean) / y_stds
    y_test_z = (y_test - y_mean) / y_stds
    print("Normalization done.")

    set_random_seeds(random_seed)
    train_loader = DataLoader(load_Multimodal_data(x_train, y_train_z),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(load_Multimodal_data(x_test, y_test_z),
                            batch_size=batch_size, shuffle=False)

    if init_method == 'random':
        model = SupervisedFLICAmodel(nfea, nlat, ntask, dropout, device).to(device)
    elif init_method == 'migp':
        (init_spatial, init_weight, init_bias, mmigp_basis,
         mmigp_diagnostics) = Initialize_SuperBigFLICA(
            x_train, y_train_z, nlat, train_ind)
        np.save(dictionary_dir / "mmigp_subject_basis.npy", mmigp_basis)
        with (coefficient_dir / "mmigp_initialization_diagnostics.json").open(
                "w") as stream:
            json.dump(mmigp_diagnostics, stream, indent=2)
            stream.write("\n")
        model = SupervisedFLICAmodel(
            nfea, nlat, ntask, dropout, device,
            init_spatial, init_weight, init_bias).to(device)
    else:
        raise ValueError(f"Unsupported init_method: {init_method}")

    loss_fun = loss_SuperBigFLICA_regression(
        ntask, nmod, x_train[0].shape[0], device, auto_weight, lambdas).to(device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_stds_t = torch.as_tensor(y_stds, dtype=torch.float32, device=device)

    optimizer1 = torch.optim.Adam(loss_fun.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    model_steps = max(1, (maxiter + 1) * len(train_loader))
    loss_steps = max(1, max(0, maxiter - 9) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_steps)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=loss_steps)

    loss_all_test = np.zeros((maxiter, 4), dtype=np.float64)
    best_corr, pred_best, best_model = -np.inf, None, None
    checkpoint_audit = []
    target_min = np.nanmin(y_train, axis=0)
    target_max = np.nanmax(y_train, axis=0)
    target_range = np.maximum(target_max - target_min, y_stds)
    allowed_min = target_min - target_range
    allowed_max = target_max + target_range
    mae_limit = float(np.max(np.maximum(target_range, 3.0 * y_stds)))

    for epoch in range(maxiter + 1):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)
        start = time.time()
        model.train()
        train_loss = 0.0
        train_abs = 0.0

        for data_batch, labels in train_loader:
            data_batch = [x.to(device) for x in data_batch]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if epoch >= 10:
                optimizer1.zero_grad(set_to_none=True)

            recon, spatial, lat, pred, weight = model(data_batch, device)
            loss, _, _, _, _ = loss_fun(
                recon, data_batch, spatial, pred, labels, weight, lat)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch >= 10:
                optimizer1.step()
                scheduler1.step()

            train_loss += float(loss.item())
            pred_orig = pred * y_stds_t + y_mean_t
            label_orig = labels * y_stds_t + y_mean_t
            train_abs += float(torch.abs(pred_orig - label_orig).sum().item())

        train_mae = train_abs / float(y_train_z.shape[0] * ntask)

        model.eval()
        l1 = l2 = l3 = l4 = val_loss = 0.0
        pred_batches = []
        with torch.no_grad():
            for data_batch, labels in val_loader:
                data_batch = [x.to(device) for x in data_batch]
                labels = labels.to(device)
                recon, spatial, lat, pred, weight = model(data_batch, device)
                loss, a, b, c, d = loss_fun(
                    recon, data_batch, spatial, pred, labels, weight, lat)
                val_loss += float(loss.item())
                l1 += float(a.item()); l2 += float(b.item())
                l3 += float(c.item()); l4 += float(d.item())
                pred_batches.append((pred * y_stds_t + y_mean_t).cpu().numpy())

        pred_val = np.vstack(pred_batches)
        y_val_orig = y_test_z * y_stds + y_mean
        corr_each = np.array([_safe_pearsonr(pred_val[:, j], y_val_orig[:, j])
                              for j in range(ntask)])
        corr_val = (np.nansum(corr_each[corr_each > 0.1]) if ntask > 1
                    else float(np.nanmean(corr_each)))
        finite = np.isfinite(pred_val) & np.isfinite(y_val_orig)
        val_mae = float(np.mean(np.abs(pred_val[finite] - y_val_orig[finite])))

        predictions_finite = bool(np.isfinite(pred_val).all())
        predictions_in_range = bool(
            predictions_finite and
            np.all(pred_val >= allowed_min[None, :]) and
            np.all(pred_val <= allowed_max[None, :]))
        mae_valid = bool(np.isfinite(val_mae) and val_mae <= mae_limit)
        scale_valid = predictions_finite and predictions_in_range and mae_valid
        selected = bool(
            epoch >= 1 and scale_valid and np.isfinite(corr_val) and
            corr_val > best_corr)
        if selected:
            best_corr = corr_val
            pred_best = pred_val.copy()
            best_model = copy.deepcopy(model)
        checkpoint_audit.append({
            "epoch": int(epoch),
            "validation_corr": float(corr_val),
            "validation_mae": float(val_mae),
            "prediction_min": float(np.min(pred_val)),
            "prediction_max": float(np.max(pred_val)),
            "allowed_prediction_min": allowed_min.tolist(),
            "allowed_prediction_max": allowed_max.tolist(),
            "mae_limit": mae_limit,
            "scale_valid": scale_valid,
            "selected": selected,
        })
        if epoch >= 1:
            loss_all_test[epoch - 1] = [l1, l2, l3, l4]

        print(f"Epoch {epoch} | Train Loss: "
              f"{train_loss/len(train_loader.dataset):.6f} | "
              f"Train MAE: {train_mae:.4f} | Validation Loss: "
              f"{val_loss/len(val_loader.dataset):.4f} | "
              f"Validation MAE: {val_mae:.4f} | "
              f"Validation Corr: {corr_val:.4f} | "
              f"Scale Valid: {scale_valid} | "
              f"Time: {time.time()-start:.2f}s")

    with (output_path / "checkpoint_selection_audit.json").open("w") as stream:
        json.dump(checkpoint_audit, stream, indent=2)
        stream.write("\n")
    if best_model is None:
        raise RuntimeError(
            "No checkpoint satisfied finite prediction, training-derived "
            "prediction-range, and validation-MAE scale gates"
        )
    return (pred_best, best_model.to('cpu'), loss_all_test, best_corr,
            model.to('cpu'), Data_train, Data_validation, Data_test,
            dictionaries)

# -------------------------------------------------------------------------
# Model extraction and spatial regression
# -------------------------------------------------------------------------

def get_model_param(x_train, x_test, y_train, y_test, best_model,
                    get_sp_load=1, inactive_sd_ratio=1e-3):
    y_train = _as_2d_targets(y_train)
    y_test = _as_2d_targets(y_test)
    nmod = len(x_train)

    best_model = best_model.to('cpu')
    best_model.eval()
    modality_weights = F.softmax(best_model.mod_weight.detach(), dim=1).cpu().numpy()
    prediction_weights = best_model.weight_pred.detach().cpu().numpy()

    print('Data Normalization...')
    train_norm, test_norm = [], []
    for i in range(nmod):
        tr, te, _, _, _ = _coefficient_scale_guard(
            x_train[i], x_test[i], x_test[i], inactive_sd_ratio)
        train_norm.append(torch.as_tensor(tr, dtype=torch.float32))
        test_norm.append(torch.as_tensor(te, dtype=torch.float32))

    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_stds[y_stds == 0] = 1.0
    print('Done.')

    with torch.no_grad():
        _, _, lat_train_t, pred_train_t, _ = best_model(train_norm, 'cpu')
        _, _, lat_test_t, pred_test_t, _ = best_model(test_norm, 'cpu')

    lat_train = lat_train_t.cpu().numpy()
    lat_test = lat_test_t.cpu().numpy()
    pred_train = pred_train_t.cpu().numpy() * y_stds + y_mean
    pred_test = pred_test_t.cpu().numpy() * y_stds + y_mean
    pred_test = _as_2d_targets(pred_test)

    if pred_test.shape != y_test.shape:
        raise ValueError(f"prediction shape {pred_test.shape} != target {y_test.shape}")
    best_performance = np.array([
        _safe_pearsonr(pred_test[:, i], y_test[:, i])
        for i in range(pred_test.shape[1])
    ])

    spatial_loadings = []
    if get_sp_load == 1:
        cov = np.ones((lat_train.shape[0], 1), dtype=np.float32)
        for i in range(nmod):
            spatial_loadings.append(
                sKPCR_regression(lat_train, train_norm[i].numpy(), cov))

    return (lat_train, lat_test, spatial_loadings, modality_weights,
            prediction_weights, pred_train, pred_test, best_performance)


def sKPCR_regression(X, Y, cov):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    contrast = np.vstack((np.eye(X.shape[1]),
                          np.zeros((cov.shape[1], X.shape[1]))))
    design = np.hstack((X, cov))
    df = design.shape[0] - np.linalg.matrix_rank(design)
    if df <= 0:
        raise ValueError("Non-positive regression degrees of freedom")
    ss = np.linalg.pinv(design.T @ design)
    beta = ss @ design.T @ Y
    residual = Y - design @ beta
    sigma = np.sqrt(np.sum(residual ** 2, axis=0) / float(df))[None, :]
    numerator = beta.T @ contrast
    contrast_var = np.diag(contrast.T @ ss @ contrast)[None, :]
    denominator = sigma.T @ np.sqrt(np.maximum(contrast_var,
                                                np.finfo(float).eps))
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = numerator / denominator
    return np.nan_to_num(tstat, nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def clean_modality_name(modality_name):
    modality_name = str(modality_name)
    for suffix in ('_test', '_train', '_validation'):
        modality_name = modality_name.replace(suffix, '')
    return modality_name.strip()


def sort_and_filter_by_modality_order(data_list, order):
    index = {clean_modality_name(m): i for i, m in enumerate(order)}
    filtered = [item for item in data_list
                if clean_modality_name(item['modality']) in index]
    return sorted(filtered,
                  key=lambda item: index[clean_modality_name(item['modality'])])


def rms(IN, dim, options):
    del options
    arr = np.asarray(IN)
    if dim == []:
        return np.sqrt(np.sum(np.square(arr)) / arr.size)
    return np.sqrt(np.sum(np.square(arr), axis=dim) / arr.shape[dim])


def SBF_load(opts, behavioral_train, behavioral_test,
             behavioral_validation):
    """Load NIfTI, MGH, text, dscalar, or dtseries SBF inputs."""
    behavioral_train = _as_2d_targets(behavioral_train)
    behavioral_test = _as_2d_targets(behavioral_test)
    behavioral_validation = _as_2d_targets(behavioral_validation)

    allowed = opts.get('modalities_order', [])
    root = opts['brain_data_main_folder']
    directories = [os.path.join(root, d) for d in sorted(os.listdir(root))
                   if os.path.isdir(os.path.join(root, d))]
    directories = [d for d in directories
                   if any(m in os.path.basename(d) for m in allowed)]
    if not directories:
        raise FileNotFoundError(f"No modality directories under {root}")

    os.makedirs(opts['output_dir'], exist_ok=True)
    with open(os.path.join(opts['output_dir'], 'order_of_loaded_data.txt'), 'w') as f:
        for d in directories:
            f.write(d + ' \n')

    suffixes = ('.dscalar.nii', '.dtseries.nii', '.nii.gz', '.mgh', '.txt')

    def infer_ext(path):
        if path.endswith('.dscalar.nii'):
            return '.dscalar.nii'
        if path.endswith('.dtseries.nii'):
            return '.dtseries.nii'
        if path.endswith('.nii.gz'):
            return '.gz'
        return os.path.splitext(path)[1]

    def choose_file(folder):
        files = sorted(os.listdir(folder))
        preferred = [('.nii.gz' if s == '.gz' else s)
                     for s in opts.get('modalities_order_filetypes', [])]
        preferred += list(suffixes)
        seen = set()
        for suffix in preferred:
            if suffix in seen:
                continue
            seen.add(suffix)
            matches = [os.path.join(folder, f) for f in files
                       if f.endswith(suffix)]
            if len(matches) > 1:
                # FreeSurfer modalities are represented by one LH/RH pair.
                # The upstream example documents this layout, but its generic
                # multiple-file guard rejected it before the MGH loader ran.
                if suffix == '.mgh' and len(matches) == 2:
                    by_name = {
                        os.path.basename(match)[:2]: match for match in matches
                    }
                    if set(by_name) == {'lh', 'rh'}:
                        return by_name['lh']
                raise RuntimeError(f"Multiple {suffix} files in {folder}: {matches}")
            if matches:
                return matches[0]
        return None

    paths = [choose_file(d) for d in directories]
    missing = [directories[i] for i, p in enumerate(paths) if p is None]
    if missing:
        raise FileNotFoundError("No supported file in:\n" + '\n'.join(missing))
    print('Files selected for loading:')
    for path in paths:
        print('  ' + path)

    data = [None] * len(paths)
    masks = [None] * len(paths)
    filetypes = [None] * len(paths)
    names = [None] * len(paths)
    shapes, affine, header = {}, {}, {}
    cifti_axes, cifti_nifti_headers, cifti_kinds = {}, {}, {}

    valid_train = np.asarray(~pd.isna(behavioral_train).any(axis=1)).ravel()
    valid_test = np.asarray(~pd.isna(behavioral_test).any(axis=1)).ravel()
    valid_val = np.asarray(~pd.isna(behavioral_validation).any(axis=1)).ravel()
    removed = {'train': int(np.sum(~valid_train)),
               'test': int(np.sum(~valid_test)),
               'validation': int(np.sum(~valid_val))}

    mni_mask = None
    if any(p.endswith('.nii.gz') for p in paths):
        mni_path = os.path.join(opts['fsl_path'], 'data', 'standard',
                                'MNI152_T1_2mm_brain_mask.nii.gz')
        mni_mask = nib.load(mni_path).get_fdata()
    mask_gz, mask_mgh = None, None

    for i, path in enumerate(paths):
        print('i = ', i)
        ext = infer_ext(path)
        filetypes[i], names[i] = ext, path
        modality_name = os.path.basename(os.path.dirname(path))
        base = clean_modality_name(modality_name)
        key = base + '_train'
        print('loading data from modality number', i + 1, '=', path)

        if ext == '.gz':
            img = nib.load(path)
            raw = np.asarray(img.get_fdata())
            if raw.ndim != 4:
                raise ValueError(f"Expected 4D NIfTI, got {raw.shape}")
            shape3 = raw.shape[:3]
            shapes[modality_name] = shapes[key] = shape3
            affine[modality_name] = deepcopy(img.affine)
            affine[key] = deepcopy(img.affine)
            header[modality_name] = img.header.copy()
            header[key] = img.header.copy()
            data[i] = np.reshape(raw, (np.prod(shape3), raw.shape[3]), order='F')
            mask_gz = mni_mask

        elif ext == '.txt':
            loaded = np.asarray(np.loadtxt(path))
            data[i] = loaded[:, None] if loaded.ndim == 1 else loaded

        elif ext == '.mgh':
            folder, infile = os.path.dirname(path), os.path.basename(path)
            left_path = os.path.join(folder, 'l' + infile[1:])
            right_path = os.path.join(folder, 'r' + infile[1:])
            left, right = nib.load(left_path), nib.load(right_path)
            affine[modality_name] = affine[key] = left.affine
            header[modality_name] = header[key] = left.header
            vol = np.concatenate((left.get_fdata(), right.get_fdata()), axis=0)
            data2d = np.reshape(vol, (vol.shape[0], vol.shape[3]), order='F')
            # Use FreeSurfer's authoritative fsaverage cortex labels, rather
            # than inferring the mask independently from each modality.
            if mask_mgh is None:
                vertices_per_hemi = left.shape[0]
                surface_mask = np.zeros(2 * vertices_per_hemi, dtype=bool)
                left_label = nib.freesurfer.read_label(os.path.join(
                    opts['path_to_fsavg'], 'label', 'lh.cortex.label'))
                right_label = nib.freesurfer.read_label(os.path.join(
                    opts['path_to_fsavg'], 'label', 'rh.cortex.label'))
                surface_mask[left_label] = True
                surface_mask[vertices_per_hemi + right_label] = True
                mask_mgh = surface_mask.astype(np.float32)
            else:
                surface_mask = mask_mgh.astype(bool)
            outside_max = float(np.max(np.abs(data2d[~surface_mask, :])))
            if outside_max > 1e-6:
                raise ValueError(
                    f"Non-cortex MGH values are not zero in {folder}; "
                    f"maximum absolute value={outside_max}"
                )
            masks[i] = surface_mask.astype(np.float32)
            data[i] = data2d[surface_mask, :]

        elif ext in ('.dscalar.nii', '.dtseries.nii'):
            img = nib.load(path)
            if not isinstance(img, nib.Cifti2Image):
                raise TypeError(f"Expected Cifti2Image for {path}")
            raw = np.asarray(img.get_fdata())
            if raw.ndim != 2:
                raise ValueError(f"Expected 2D CIFTI, got {raw.shape}")
            axes = [img.header.get_axis(a) for a in range(raw.ndim)]
            brain_index = next((a for a, axis in enumerate(axes)
                                if axis.__class__.__name__ == 'BrainModelAxis'), 1)
            if brain_index == 1:
                data[i], brain_axis = raw.T, axes[1]
            elif brain_index == 0:
                data[i], brain_axis = raw, axes[0]
            else:
                raise ValueError(f"Unexpected CIFTI brain axis {brain_index}")
            header[modality_name] = header[key] = img.header
            affine[modality_name] = affine[key] = None
            cifti_axes[modality_name] = cifti_axes[key] = brain_axis
            cifti_nifti_headers[modality_name] = img.nifti_header.copy()
            cifti_nifti_headers[key] = img.nifti_header.copy()
            cifti_kinds[modality_name] = cifti_kinds[key] = ext
            print(f"{ext} raw shape: {raw.shape}")
            print(f"{ext} oriented shape (brain x subjects/maps): {data[i].shape}")
        else:
            raise ValueError(f"Unsupported extension {ext}")

        lower_dir, lower_path = modality_name.lower(), path.lower()
        if '_train' in lower_dir or '_train_' in lower_path:
            mask, split = valid_train, 'train'
        elif '_test' in lower_dir or '_test_' in lower_path:
            mask, split = valid_test, 'test'
        elif '_validation' in lower_dir or '_validation_' in lower_path:
            mask, split = valid_val, 'validation'
        else:
            raise ValueError(f"Cannot identify split for {path}")
        if data[i].shape[1] != len(mask):
            raise ValueError(f"{split} subject mismatch for {path}: "
                             f"{data[i].shape[1]} columns vs {len(mask)} targets")
        print(f"{split.capitalize()} group: Removing {removed[split]} subjects "
              "with missing behavioral data.")
        data[i] = data[i][:, mask]

    behavioral_train = behavioral_train[valid_train]
    behavioral_test = behavioral_test[valid_test]
    behavioral_validation = behavioral_validation[valid_val]

    scales = [None] * len(data)
    for i in range(len(data)):
        matrix = np.asarray(data[i], dtype=np.float32)
        matrix -= np.mean(matrix, axis=1, keepdims=True)
        scale = float(rms(matrix, [], []))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scales[i] = scale
        data[i] = matrix / scale
    data = [matrix.T for matrix in data]

    modalities = [os.path.basename(os.path.dirname(path)) for path in names]
    output_masks = {}
    if mask_mgh is not None:
        output_masks['mask_for_mgh'] = mask_mgh
    if mask_gz is not None:
        output_masks['mask_for_gz'] = mask_gz

    info = {
        'data_directories': directories,
        'masks_for_SBF_inputs': masks,
        'scaling_data_transform': scales,
        'filetype': filetypes,
        'names': names,
        'affine': affine,
        'header': header,
        'shapes': shapes,
        'masks_for_SBF_outputs': output_masks,
        'cifti_brain_axes': cifti_axes,
        'cifti_nifti_headers': cifti_nifti_headers,
        'cifti_output_kinds': cifti_kinds,
        'behavioral_data': {
            'train': behavioral_train,
            'test': behavioral_test,
            'validation': behavioral_validation,
        },
    }
    return data, info, modalities

# -------------------------------------------------------------------------
# Output reconstruction
# -------------------------------------------------------------------------

def SBF_save_everything(output_dir, spatial_loadings, img_info, opts,
                        dictionaries):
    os.makedirs(output_dir, exist_ok=True)

    for k, z_map_dictionary in enumerate(spatial_loadings):
        ext = opts['modalities_order_filetypes'][k]
        modality = opts['modalities_order'][k]
        key = modality + '_train'
        z_map_features = np.asarray(np.dot(dictionaries[k], z_map_dictionary))

        if ext == '.gz':
            if key not in img_info['shapes']:
                raise ValueError(f"No shape information for {key}")
            shape3 = img_info['shapes'][key]
            reshaped = np.zeros((*shape3, z_map_features.shape[1]),
                                dtype=np.float32)
            for component in range(z_map_features.shape[1]):
                reshaped[..., component] = np.reshape(
                    z_map_features[:, component], shape3, order='F')
            image = nib.Nifti1Image(
                np.nan_to_num(reshaped),
                affine=img_info['affine'][key],
                header=img_info['header'][key].copy())
            nib.save(image, os.path.join(output_dir,
                                         f'SBFOut_{modality}.nii.gz'))

        elif ext == '.mgh':
            mask = img_info['masks_for_SBF_outputs']['mask_for_mgh']
            nonzero = np.flatnonzero(mask != 0)
            full = np.tile(np.asarray(mask)[:, None],
                           (1, z_map_features.shape[1]))
            full[nonzero] = z_map_features
            half = len(mask) // 2
            left = np.nan_to_num(full[:half, :, None, None]
                                 .transpose(0, 2, 3, 1)).astype(np.float32)
            right = np.nan_to_num(full[half:, :, None, None]
                                  .transpose(0, 2, 3, 1)).astype(np.float32)
            affine = img_info['affine'][key]
            header = img_info['header'][key]
            nib.save(nib.freesurfer.mghformat.MGHImage(
                left, affine=affine, header=header),
                os.path.join(output_dir, f'SBFOut_lh_{modality}.mgh'))
            nib.save(nib.freesurfer.mghformat.MGHImage(
                right, affine=affine, header=header),
                os.path.join(output_dir, f'SBFOut_rh_{modality}.mgh'))

        elif ext in ('.dscalar.nii', '.dtseries.nii'):
            if key not in img_info['cifti_brain_axes']:
                raise ValueError(f"No CIFTI brain axis for {key}")
            brain_axis = img_info['cifti_brain_axes'][key]
            nifti_header = img_info['cifti_nifti_headers'].get(key)
            cifti_data = np.nan_to_num(z_map_features.T).astype(np.float32)

            if ext == '.dscalar.nii':
                names = [f'SBF_component_{i+1:03d}'
                         for i in range(cifti_data.shape[0])]
                map_axis = nib.cifti2.cifti2_axes.ScalarAxis(names)
                outname = f'SBFOut_{modality}.dscalar.nii'
            else:
                map_axis = nib.cifti2.cifti2_axes.SeriesAxis(
                    start=0.0, step=1.0, size=cifti_data.shape[0],
                    unit='SECOND')
                outname = f'SBFOut_{modality}.dtseries.nii'

            new_header = nib.cifti2.cifti2_axes.to_header(
                (map_axis, brain_axis))
            image = nib.Cifti2Image(cifti_data, header=new_header,
                                    nifti_header=nifti_header)
            nib.save(image, os.path.join(output_dir, outname))

        elif ext == '.txt':
            np.savetxt(os.path.join(output_dir,
                                    f'SBFOut_{modality}.txt'),
                       z_map_features)
        else:
            raise ValueError(f"Unsupported output extension: {ext}")

    return spatial_loadings
