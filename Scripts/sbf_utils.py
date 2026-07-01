#!/usr/bin/env python3
"""SuperBigFLICA utilities with corrected held-out dictionary projection."""
from __future__ import annotations

import copy
import os
import random
import time
from copy import deepcopy

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


def Initialize_SuperBigFLICA(x_train, y_train, nlat, train_ind=None):
    del train_ind
    y_train = _as_2d_targets(y_train)
    spatial, latent_sets = [], []
    for x in x_train:
        x = np.asarray(x, dtype=np.float64)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        loading = vt[:min(nlat, vt.shape[0]), :].T
        if loading.shape[1] < nlat:
            loading = np.pad(loading, ((0, 0), (0, nlat-loading.shape[1])))
        spatial.append(loading.astype(np.float32))
        latent_sets.append(x @ loading)
    common = np.mean(latent_sets, axis=0)
    valid = np.isfinite(y_train).all(axis=1)
    design = np.column_stack((common[valid], np.ones(np.sum(valid))))
    coef = np.linalg.pinv(design) @ y_train[valid]
    return spatial, coef[:-1].astype(np.float32), coef[-1:].astype(np.float32), common

# -------------------------------------------------------------------------
# Dictionary learning and fixed-dictionary projection
# -------------------------------------------------------------------------

def dictionary_learning_pytorch(X, n_components=100, lambda_l1=0.1,
                                n_iter=20, lr=0.01, device='cpu',
                                random_seed=42):
    """Learn a dictionary from training data only."""
    set_random_seeds(random_seed)
    X = np.asarray(X, dtype=np.float32)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n_subjects, n_features = X.shape

    D = torch.randn(n_features, n_components, device=device)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    A = torch.zeros(n_subjects, n_components, device=device)

    for iteration in range(n_iter):
        A = A.detach().requires_grad_(True)
        optA = optim.SGD([A], lr=lr)
        for _ in range(5):
            optA.zero_grad(set_to_none=True)
            recon = A.matmul(D.T)
            lossA = (Xt - recon).square().mean() + lambda_l1 * A.abs().mean()
            lossA.backward()
            optA.step()
        A = A.detach()

        ata = A.T.matmul(A)
        ridge = 1e-6 * torch.eye(n_components, device=device, dtype=ata.dtype)
        xta = Xt.T.matmul(A)
        try:
            D = torch.linalg.solve(ata + ridge, xta.T).T
        except RuntimeError:
            D = xta.matmul(torch.linalg.pinv(ata + ridge))
        D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)

        if iteration == 0 or (iteration + 1) % 5 == 0:
            print(f"Dictionary iteration {iteration+1}/{n_iter}; "
                  f"coefficient loss={lossA.item():.6g}")

    return D.detach().cpu().numpy(), A.detach().cpu().numpy()


def _soft_threshold(x, threshold):
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def apply_dictionary(D, X, lambda_l1=0.1, max_iter=1000,
                     tol=1e-6, device='cpu'):
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
    lipschitz = (2.0 * largest_eigenvalue / float(n_features)).clamp_min(1e-12)
    step_size = 1.0 / lipschitz
    threshold = step_size * float(lambda_l1) / float(n_components)

    A = torch.zeros(n_subjects, n_components, device=device)
    Y = A.clone()
    momentum = 1.0
    relative_change = np.inf
    converged_at = max_iter

    for iteration in range(1, max_iter + 1):
        gradient = 2.0 / float(n_features) * (Y.matmul(gram) - x_times_d)
        A_new = _soft_threshold(Y - step_size * gradient, threshold)
        delta = A_new - A
        relative_change = float((torch.linalg.norm(delta) /
                                 torch.linalg.norm(A_new).clamp_min(1e-12)).item())

        momentum_new = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        Y_new = A_new + ((momentum - 1.0) / momentum_new) * delta

        # Adaptive restart to reduce FISTA oscillation.
        if torch.sum((Y_new - A_new) * (A_new - A)) > 0:
            momentum_new = 1.0
            Y_new = A_new.clone()

        A, Y, momentum = A_new, Y_new, momentum_new
        if relative_change < tol:
            converged_at = iteration
            break

    objective = ((Xt - A.matmul(Dt.T)).square().mean() +
                 float(lambda_l1) * A.abs().mean())
    print("Fixed-dictionary coding: "
          f"subjects={n_subjects}, features={n_features}, "
          f"components={n_components}, iterations={converged_at}, "
          f"relative_change={relative_change:.3e}, "
          f"objective={objective.item():.6g}")
    return A.detach().cpu().numpy()


def process_modality(i, X_train, dicl_dim, X_val, X_test,
                     lambda_l1=0.1, projection_max_iter=1000,
                     projection_tol=1e-6, random_seed=42):
    print(f"Starting dictionary learning for modality {i}")
    D, _ = dictionary_learning_pytorch(
        X_train, n_components=dicl_dim, lambda_l1=lambda_l1,
        n_iter=20, lr=0.01, device='cpu', random_seed=random_seed + i)

    # Re-estimate all splits against the same final dictionary.
    A_train = apply_dictionary(D, X_train, lambda_l1,
                               projection_max_iter, projection_tol, 'cpu')
    A_val = apply_dictionary(D, X_val, lambda_l1,
                             projection_max_iter, projection_tol, 'cpu')
    A_test = apply_dictionary(D, X_test, lambda_l1,
                              projection_max_iter, projection_tol, 'cpu')

    train_sd = np.std(A_train, axis=0, ddof=0)
    train_sd[train_sd == 0] = np.nan
    val_ratio = np.nanmedian(np.std(A_val, axis=0, ddof=0) / train_sd)
    test_ratio = np.nanmedian(np.std(A_test, axis=0, ddof=0) / train_sd)
    print(f"Modality {i} projection scale: validation/train SD={val_ratio:.4f}; "
          f"test/train SD={test_ratio:.4f}")
    return i, D, A_train, A_val, A_test


def parallel_dictionary_learning(Data_train, Data_validation, Data_test,
                                 dicl_dim, n_jobs=-1, lambda_l1=0.1,
                                 projection_max_iter=1000,
                                 projection_tol=1e-6, random_seed=42):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_modality)(
            i, Data_train[i], dicl_dim, Data_validation[i], Data_test[i],
            lambda_l1, projection_max_iter, projection_tol, random_seed)
        for i in range(len(Data_train)))
    results = sorted(results, key=lambda item: item[0])

    dictionaries = []
    for i, D, A_train, A_val, A_test in results:
        Data_train[i] = A_train
        Data_validation[i] = A_val
        Data_test[i] = A_test
        dictionaries.append(D)
    return dictionaries, Data_train, Data_validation, Data_test


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
                    batch_size=512, maxiter=50, dicl_dim=100):
    del output_dir
    y_train = _as_2d_targets(y_train).astype(np.float32)
    y_test = _as_2d_targets(y_test).astype(np.float32)
    x_train = [np.asarray(x, dtype=np.float32).copy() for x in x_train]
    x_test = [np.asarray(x, dtype=np.float32).copy() for x in x_test]
    Data_test = [np.asarray(x, dtype=np.float32).copy() for x in Data_test]

    set_random_seeds(random_seed)
    nmod, ntask = len(x_train), y_train.shape[1]
    device = torch.device(device)

    print("Applying Dictionary Learning")
    dictionaries, x_train, x_test, Data_test = parallel_dictionary_learning(
        x_train, x_test, Data_test, dicl_dim, n_jobs=-1,
        lambda_l1=0.1, projection_max_iter=1000,
        projection_tol=1e-6, random_seed=random_seed)
    Data_train = copy.deepcopy(x_train)
    Data_validation = copy.deepcopy(x_test)
    print("Dictionary Learning Done.")

    nfea = [x.shape[1] for x in x_train]
    for i in range(nmod):
        mean = x_train[i].mean(axis=0)
        sd = x_train[i].std(axis=0, ddof=0)
        sd[sd == 0] = 1e-3
        x_train[i] = (x_train[i] - mean) / sd
        x_test[i] = (x_test[i] - mean) / sd

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
    else:
        init_spatial, init_weight, init_bias, _ = Initialize_SuperBigFLICA(
            x_train, y_train_z, nlat, train_ind)
        model = SupervisedFLICAmodel(
            nfea, nlat, ntask, dropout, device,
            init_spatial, init_weight, init_bias).to(device)

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

        if epoch >= 1 and np.isfinite(corr_val) and corr_val > best_corr:
            best_corr = corr_val
            pred_best = pred_val.copy()
            best_model = copy.deepcopy(model)
        if epoch >= 1:
            loss_all_test[epoch - 1] = [l1, l2, l3, l4]

        print(f"Epoch {epoch} | Train Loss: "
              f"{train_loss/len(train_loader.dataset):.6f} | "
              f"Train MAE: {train_mae:.4f} | Validation Loss: "
              f"{val_loss/len(val_loader.dataset):.4f} | "
              f"Validation MAE: {val_mae:.4f} | "
              f"Validation Corr: {corr_val:.4f} | "
              f"Time: {time.time()-start:.2f}s")

    if best_model is None:
        raise RuntimeError("No valid best model was selected")
    return (pred_best, best_model.to('cpu'), loss_all_test, best_corr,
            model.to('cpu'), Data_train, Data_validation, Data_test,
            dictionaries)

# -------------------------------------------------------------------------
# Model extraction and spatial regression
# -------------------------------------------------------------------------

def get_model_param(x_train, x_test, y_train, y_test, best_model,
                    get_sp_load=1):
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
        tr = torch.as_tensor(x_train[i], dtype=torch.float32).clone()
        te = torch.as_tensor(x_test[i], dtype=torch.float32).clone()
        mean = tr.mean(dim=0)
        try:
            sd = tr.std(dim=0, correction=0)
        except TypeError:
            sd = tr.std(dim=0, unbiased=False)
        sd = torch.where(sd == 0, torch.full_like(sd, 1e-3), sd)
        train_norm.append((tr - mean) / sd)
        test_norm.append((te - mean) / sd)

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
            masks[i] = np.ones(data2d.shape[0], dtype=np.float32)
            mask_mgh = masks[i].copy()
            data[i] = data2d

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
