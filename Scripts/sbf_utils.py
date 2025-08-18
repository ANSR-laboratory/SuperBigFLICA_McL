import numpy as np
import scipy
from scipy import linalg
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import copy
import pandas as pd
from torch import optim
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from copy import deepcopy
import nibabel as nib

class loss_SuperBigFLICA_regression(nn.Module):
    def __init__(self, ntask, nmod, nsub, device = 'cuda', auto_weight = [1,1,1,1], lambdas = [0.25, 0.25, 0.25, 0.25]):
        super(loss_SuperBigFLICA_regression, self).__init__()
        
        self.ntask = ntask
        self.auto_weight = auto_weight
        self.nmod=nmod
        self.mse = nn.MSELoss()
        self.lambdas = torch.FloatTensor(np.array(lambdas).flatten()).to(device)     
        #weight for reconstruction loss
        self.sigma1 = nn.Parameter(torch.ones(1,nmod,device = device))
        #weight for spatial loadings
        self.sigma2 = nn.Parameter(torch.ones(1,nmod,device = device))
        #weight for mse loss of each task
        self.sigma3 = nn.Parameter(torch.ones(1,ntask,device = device))
        #weight for regression coefs of each task
        self.sigma4 = nn.Parameter(torch.ones(1,ntask,device = device))#l1
        self.sigma5 = nn.Parameter(torch.ones(1,ntask,device = device))#l2
        self.nsub = nsub
               
    def forward(self, recon_x, x_orig, sptial_loadings, y_pred, y_train, pred_weights, lat_train):
        

        batch_prop = recon_x[0].size()[0]/self.nsub # the ratio of the current batch size to the total dataset size. 
        
        #make these parameters always positive
        sigma1 =self.sigma1**2
        sigma2 =self.sigma2**2
        sigma3 =self.sigma3**2 
        sigma4 =self.sigma4**2
        sigma5 =self.sigma5**2
        
        if self.auto_weight[0] == 0:
            self.sigma1.requires_grad = False
        if self.auto_weight[1] == 0:
            self.sigma2.requires_grad = False
        if self.auto_weight[2] == 0:
            self.sigma3.requires_grad = False
        if self.auto_weight[3] == 0:
            self.sigma4.requires_grad = False
            self.sigma5.requires_grad = False

        loss_recon = 0
        for i in range(0,self.nmod):
            diff = recon_x[i] - x_orig[i]
            loss_test = (diff * diff).mean() / sigma1[0,i]**2 / 2
            # print(loss_test)
            loss_recon = loss_recon + (diff * diff).mean() / sigma1[0,i]**2 / 2
        loss_recon = loss_recon + torch.sum(torch.log(sigma1+1)) 

        loss_sptial_loadings = 0
        for i in range(0,self.nmod):         
            loss_sptial_loadings = loss_sptial_loadings + sptial_loadings[i].abs().mean() * batch_prop / sigma2[0,i]
        loss_sptial_loadings = loss_sptial_loadings + 2 * torch.sum(torch.log(sigma2+1)) 
            
        y_train = y_train.squeeze(1)  # LN added to remove unnecessary singleton dimension in y_train
        index_NaN = torch.isnan(y_train)
        y_train[index_NaN] = y_pred[index_NaN]
        
        diff2 = (y_train - y_pred)**2 / sigma3**2 / 2
        loss_mse = torch.mean(diff2) + torch.sum(torch.log(sigma3 +1))
        
        loss_pred_weights = torch.mean(torch.abs(pred_weights) / sigma4  * batch_prop) + 2 * torch.sum(torch.log(sigma4+1)) #l1
        loss_pred_weights = loss_pred_weights + torch.mean((pred_weights)**2  * batch_prop / sigma5 ** 2 / 2) + torch.sum(torch.log(sigma5+1)) #l2
        
        self.lambdas = self.lambdas / torch.sum(self.lambdas)
        
        l = self.lambdas[0] * loss_recon + self.lambdas[1] * loss_sptial_loadings + self.lambdas[2] * loss_mse + self.lambdas[3] * loss_pred_weights #+ loss_lat
        # print('total loss', l)
        # print('loss recon', self.lambdas[0] * loss_recon)
        # print('loss spatial loadings', self.lambdas[1] * loss_sptial_loadings)
        # print('loss mse', self.lambdas[2] * loss_mse)
        # print('loss pred weights', self.lambdas[3] * loss_pred_weights)

        return l, loss_recon, loss_sptial_loadings, loss_mse, loss_pred_weights

# import pandas as pd

# class load_Multimodal_data1(Dataset):

# class load_Multimodal_data(Dataset):


# import torch
# import numpy as np
# import copy
# import scipy
# import time
# from torch.utils.data import DataLoader

# This function assumes that the following functions/classes are defined elsewhere and imported:
# - set_random_seeds
# - load_Multimodal_data
# - SupervisedFLICAmodel
# - Initialize_SuperBigFLICA
# - loss_SuperBigFLICA_regression
# - parallel_dictionary_learning

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Random seed set to {seed} for reproducibility.")

class SupervisedFLICAmodel(nn.Module):
    def __init__(self,nfea, nlat, ntask, dropout = 0.5, device = 'cpu', init_spatial_loading = None, init_weight_pred=None, init_bais_pred=None):
        
        #nfea: list of length K, number of features in each modality
        #nlat: number of components to extract
        
        super(SupervisedFLICAmodel, self).__init__()
        
        nmod = len(nfea)
        self.ntask = ntask
        self.dropout = dropout
        self.nlat = nlat

        #initialize the spatial loading of each modality
        if init_spatial_loading is None:
            self.spatial_loading = nn.ParameterList([])
            for i in range(0,nmod):
                self.spatial_loading.append( nn.Parameter(torch.randn(nfea[i],nlat).to(device)) )
                #torch.nn.init.normal_(self.spatial_loading[i])
        else:
            self.spatial_loading = nn.ParameterList([])
            for i in range(0,nmod):
                self.spatial_loading.append( nn.Parameter(torch.FloatTensor(init_spatial_loading[i]).to(device)) )                

        #initialize the modalitity weights (initilize to 1)
        self.mod_weight = nn.Parameter(torch.ones(nlat,nmod).to(device))
        
        #initalize the prediction weights        
        if init_weight_pred is None:
            set_random_seeds()

            self.weight_pred = nn.Parameter(torch.randn(nlat,self.ntask).to(device))
            #torch.nn.init.normal_(self.weight_pred)
            self.bias_pred = nn.Parameter(torch.randn(1,self.ntask).to(device))
            #torch.nn.init.normal_(self.bias_pred)
            
        else:
            self.weight_pred = nn.Parameter(torch.FloatTensor(init_weight_pred).to(device))
            self.bias_pred = nn.Parameter(torch.FloatTensor(init_bais_pred).to(device))

        self.batch_norm = nn.BatchNorm1d(nlat)
        
        #other variables
        self.nlat = nlat    
        self.nmod = nmod   
        
    def forward(self, x, device='cpu'):
        #x is a list of length K [nsub * nfeature]
        # print("before softmax", self.mod_weight)
        mod_weight = F.softmax(self.mod_weight,dim=1)
        # print("after softmax", mod_weight)
        latents_common = torch.zeros(x[0].size()[0],self.nlat,device=device)
        for i in range(0,self.nmod): 
            dat = F.dropout(x[i], self.dropout, training=self.training)
           # latents = dat.matmul(self.spatial_loading[i]).matmul(torch.diag(mod_weight[:,i]))  LN changed to line 370 to make all floats so code can execute
            latents = dat.float().matmul(self.spatial_loading[i].float()).matmul(torch.diag(mod_weight[:, i].float()))

            latents = latents.squeeze(1)        # LN added to remove extra singleton dimension so latents can be added to latents_common
            latents_common = latents_common + latents

            
        latents_common = latents_common / self.nmod
        #print("Shape of latents_common before batch norm:", latents_common.shape)  # Debugging
        
        #batch norm
        self.batch_norm = self.batch_norm.to(device)
        latents_common = self.batch_norm(latents_common)
        #dropout
        latents_common = F.dropout(latents_common, self.dropout, training=self.training)
        
        output = []  ##the same size as x
        for i in range(0,self.nmod):
            w = (self.spatial_loading[i]).matmul(torch.diag(mod_weight[:,i]))
            output.append( (latents_common.matmul(w.t())) )
        
        #doing prediction         
        pred = latents_common.matmul(self.weight_pred)+self.bias_pred ##it is nsubj_train * 1            

        return output, self.spatial_loading, latents_common, pred, self.weight_pred



def SupervisedFLICA(x_train, y_train, nlat, x_test, y_test, Data_test, output_dir, random_seed=666,
                    train_ind=None, init_method='random', dropout=0.25, device='cpu',
                    auto_weight=[1,1,1,1], lambdas=[0.25, 0.25, 0.25, 0.25],
                    lr=0.001, batch_size=512, maxiter=50, dicl_dim=100):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nmod = len(x_train)
    ntask = y_train.shape[1]
    device = torch.device(device)

    print("Applying Dictionary Learning")
    dictionaries, x_train, x_test, Data_test = parallel_dictionary_learning(x_train, x_test, Data_test, dicl_dim)
    Data_train = copy.deepcopy(x_train)
    Data_validation = copy.deepcopy(x_test)
    print("Dictionary Learning Done.")

    nfea = [x.shape[1] for x in x_train]

    is_norm = True
    if is_norm:
        for i in range(len(x_train)):
            x_mean = x_train[i].mean(axis=0)
            x_stds = x_train[i].std(axis=0)
            epsilon = 1e-3
            x_stds[x_stds == 0] = epsilon
            x_train[i] = (x_train[i] - x_mean) / x_stds
            x_test[i] = (x_test[i] - x_mean) / x_stds

    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_train = (y_train - y_mean) / y_stds
    y_test = (y_test - y_mean) / y_stds

    print('Done...')
    set_random_seeds()

    train_dataset = load_Multimodal_data(X=x_train, y=y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = load_Multimodal_data(X=x_test, y=y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if init_method == 'random':
        model = SupervisedFLICAmodel(nfea=nfea, nlat=nlat, ntask=ntask, dropout=dropout,
                                     device=device, init_spatial_loading=None,
                                     init_weight_pred=None, init_bais_pred=None).to(device)
    else:
        print('Multimodal MIGP initialization...')
        init_spatial_loading, init_weight_pred, init_bais_pred, uu = Initialize_SuperBigFLICA(x_train, y_train, nlat, train_ind)
        print('Done...')
        model = SupervisedFLICAmodel(nfea=nfea, nlat=nlat, ntask=ntask, dropout=dropout,
                                     device=device, init_spatial_loading=init_spatial_loading,
                                     init_weight_pred=init_weight_pred, init_bais_pred=init_bais_pred).to(device)

    loss_fun_reg = loss_SuperBigFLICA_regression(ntask=ntask, nmod=nmod, device=device, nsub=x_train[0].shape[0],
                                                 auto_weight=auto_weight, lambdas=lambdas).to(device)

    y_mean1 = torch.FloatTensor(y_mean).to(device)
    y_stds1 = torch.FloatTensor(y_stds).to(device)

    optimizer1 = torch.optim.Adam(loss_fun_reg.parameters(), lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxiter)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=maxiter)

    epochs = maxiter
    loss_all_train = []
    loss_all_test = np.zeros((maxiter, 4))
    best_corr = -1

    for epoch in range(epochs + 1):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        tt = time.time()

        model.train()
        train_loss = 0
        train_MAE = 0

        for batch_idx, (data_batch, labels) in enumerate(train_loader):
            for i in range(nmod):
                data_batch[i] = data_batch[i].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if epoch >= 10:
                optimizer1.zero_grad()

            recon_train, spatial_loadings, lat_train, pred_all, weight_pred = model(x=data_batch, device=device)
            loss, _, _, _, _ = loss_fun_reg(recon_train, data_batch, spatial_loadings, pred_all, labels, weight_pred, lat_train)
            loss.backward()
            optimizer.step()
            if epoch >= 10:
                optimizer1.step()
            scheduler.step()
            scheduler1.step()

            train_loss += loss.item()
            pred_all1 = (pred_all * y_stds1) + y_mean1
            labels1 = (labels * y_stds1) + y_mean1
            train_MAE += torch.sum(torch.abs(pred_all1 - labels1)) / pred_all.size()[1]

        train_MAE /= y_train.shape[0]
        loss_all_train.append(train_loss)

        model.eval()
        test_l1 = test_l2 = test_l3 = test_l4 = test_loss = 0

        with torch.no_grad():
            for batch_idx, (data_batch, labels) in enumerate(test_loader):
                for i in range(nmod):
                    data_batch[i] = data_batch[i].to(device)
                labels = labels.to(device)

                recon_train, spatial_loadings, lat_test, pred_all, weight_pred = model(x=data_batch, device=device)
                loss, l1, l2, l3, l4 = loss_fun_reg(recon_train, data_batch, spatial_loadings, pred_all, labels, weight_pred, lat_train)

                test_loss += loss
                test_l1 += l1
                test_l2 += l2
                test_l3 += l3
                test_l4 += l4

                if batch_idx == 0:
                    pred_all1 = (pred_all * y_stds1) + y_mean1
                    pred_test = torch.Tensor.cpu(pred_all1).detach().numpy()
                    lat_test_all = torch.Tensor.cpu(lat_test).detach().numpy()
                else:
                    pred_all1 = (pred_all * y_stds1) + y_mean1
                    pred_test = np.vstack((pred_test, torch.Tensor.cpu(pred_all1).detach().numpy()))
                    lat_test_all = np.vstack((lat_test_all, torch.Tensor.cpu(lat_test).detach().numpy()))

            idx_nonnan = np.isnan(y_test) == 0
            corr_test1 = np.zeros((y_test.shape[1],))

            y_test1 = np.array(y_test) * np.array(y_stds1.cpu()) + np.array(y_mean1.cpu())

            for ij in range(y_test1.shape[1]):
                if np.sum(idx_nonnan[:, ij]) >= 10:
                    pred_valid = pred_test.astype(np.float64).flatten()
                    y_valid = y_test1.astype(np.float64).flatten()
                    if np.isnan(pred_valid).any() or np.isnan(y_valid).any():
                        valid_mask = ~np.isnan(pred_valid) & ~np.isnan(y_valid)
                        pred_valid = pred_valid[valid_mask]
                        y_valid = y_valid[valid_mask]
                    corr_test1[ij] = scipy.stats.pearsonr(pred_valid, y_valid)[0]

            corr_test = np.nansum(corr_test1[corr_test1 > 0.1]) if y_test.shape[1] > 1 else np.nanmean(corr_test1)
            test_MAE = np.mean(np.abs(pred_test[idx_nonnan].flatten() - y_test[idx_nonnan].flatten()))

            if epoch >= 1 and corr_test > best_corr:
                best_corr = corr_test
                pred_best = pred_test.copy()
                best_model = copy.deepcopy(model)

            loss_all_test[epoch - 1, 0] = test_l1.cpu().numpy()
            loss_all_test[epoch - 1, 1] = test_l2.cpu().numpy()
            loss_all_test[epoch - 1, 2] = test_l3.cpu().numpy()
            loss_all_test[epoch - 1, 3] = test_l4.cpu().numpy()

        print(f'Epoch {epoch} | Train Loss: {train_loss / len(train_loader.dataset):.6f} | '
              f'Test Loss: {test_loss / len(test_loader.dataset):.4f} | '
              f'Test MAE: {test_MAE:.4f} | '
              f'Test Corr (Sum r>0.1): {corr_test:.4f} | Time: {time.time() - tt:.2f}s')

    best_model.to('cpu')
    last_model = model.to('cpu')

    return pred_best, best_model, loss_all_test, best_corr, last_model, Data_train, Data_validation, Data_test, dictionaries

def dictionary_learning_pytorch(X, n_components=100, lambda_l1=0.1, n_iter=20, lr=0.01, device='cpu'):
    set_random_seeds()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    N, M = X.shape
    D = torch.randn(M, n_components, device=device)
    D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
    A = torch.zeros(N, n_components, device=device)
    for _ in range(n_iter):
        A.requires_grad_()
        optA = optim.SGD([A], lr=lr)
        for _ in range(5):
            optA.zero_grad()
            recon = A.mm(D.t())
            lossA = ((Xt - recon)**2).mean() + lambda_l1 * A.abs().mean()
            lossA.backward()
            optA.step()
        A = A.detach()
        ATA = A.t().mm(A) + 1e-8 * torch.eye(n_components, device=device)
        XTA = Xt.t().mm(A)
        D = XTA.mm(torch.inverse(ATA))
        D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
    return D.cpu().numpy(), A.cpu().numpy()

def apply_dictionary(D, X):
    A = torch.zeros(X.shape[0], D.shape[1], requires_grad=True)
    Dt = torch.tensor(D, dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    optA = optim.SGD([A], lr=0.01)
    for _ in range(20):
        optA.zero_grad()
        recon = A.mm(Dt.t())
        loss = ((Xt - recon)**2).mean() + 0.1 * A.abs().mean()
        loss.backward()
        optA.step()
    return A.detach().cpu().numpy()


def process_modality(i, X_train, dicl_dim, X_val, X_test):
    D, A_train = dictionary_learning_pytorch(X_train, n_components=dicl_dim)
    A_val = apply_dictionary(D, X_val)
    A_test = apply_dictionary(D, X_test)
    return i, D, A_train, A_val, A_test

def parallel_dictionary_learning(Data_train, Data_validation, Data_test, dicl_dim, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_modality)(i, Data_train[i], dicl_dim, Data_validation[i], Data_test[i])
        for i in range(len(Data_train))
    )
    dicts = []
    for i, D, A_tr, A_val, A_te in results:
        Data_train[i]      = A_tr
        Data_validation[i] = A_val
        Data_test[i]       = A_te
        dicts.append(D)
    return dicts, Data_train, Data_validation, Data_test


class load_Multimodal_data(Dataset):
        def __init__(self, X, y=None, transform=None):
            # A list of K data modalities, where each modality is a 2D array of size [N, M_k].
            #X is a list of data of size [N,M_k]
            self.X = X
            junk=self.X[0] # not used
            # Sets self.K as the number of modalities in the dataset (the length of X).
            self.K = len(X)
            #y is the label of size [N,Q]
            self.y = y
            
            self.transform = transform # passed as a parameter, but not used
            
        def __len__(self):
            # returns the total number of samples in the dataset.
            # assume all modalities have the same number of samples
            #return self.X[0].size()[0]
            return len(self.X[0])
        
        def __getitem__(self, index):
            # Initializes an empty list to store the multimodal data for a single sample.
            image = []
            # Iterates through each modality (self.X[i]) to collect the data for the specified index.
            for i in range(0,self.K):
                image.append(self.X[i][index,:])
                
            if self.y is not None:
                # Converts the label for the given index to a PyTorch FloatTensor.
                return image, torch.FloatTensor(self.y[index,:]).float()
            else:
                return image

def get_model_param(x_train, x_test, y_train, y_test, best_model, get_sp_load=1):
    nmod = len(x_train)
    best_model.eval()
    modality_weights = F.softmax(torch.Tensor.cpu(best_model.mod_weight), dim=1).detach().numpy()
    prediction_weights = torch.Tensor.cpu(best_model.weight_pred).detach().numpy()
    print('Data Normalization...')
    is_norm = True
    if is_norm is True:
        for i in range(0, len(x_train)):
            x_mean = x_train[i].mean(axis=0)
            x_stds = x_train[i].std(axis=0)
            epsilon = 0.001
            x_stds[x_stds == 0] = epsilon
            x_train[i] = (x_train[i] - x_mean) / x_stds
            x_test[i] = (x_test[i] - x_mean) / x_stds
    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_train = (y_train - y_mean) / y_stds
    print('Done...')
    _, _, lat_train, pred_train, _ = best_model(x=x_train, device='cpu')
    _, _, lat_test, pred_test, _ = best_model(x=x_test, device='cpu')
    lat_train = torch.Tensor.cpu(lat_train).detach().numpy()
    lat_test = torch.Tensor.cpu(lat_test).detach().numpy()
    pred_train = np.multiply(pred_train.detach().numpy(), y_stds) + y_mean
    pred_test = np.multiply(pred_test.detach().numpy(), y_stds) + y_mean

    pred_test = np.atleast_2d(pred_test)
    y_test = np.atleast_2d(y_test)


    # ensure plain ndarrays (not np.matrix) and consistent 2D shape
    pred = np.asarray(pred_test)
    yt   = np.asarray(y_test)
    if pred.ndim == 1: pred = pred[:, None]
    if yt.ndim   == 1: yt   = yt[:, None]
    assert pred.shape == yt.shape, f"{pred.shape=} {yt.shape=}"

    # per‑target Pearson r (no lumping)
    best_performance = np.array([scipy.stats.pearsonr(pred[:, i], yt[:, i])[0]
                                for i in range(pred.shape[1])])

    # best_performance = scipy.stats.pearsonr(pred_test, y_test)[0]
    spatial_loadings = []
    if get_sp_load == 1:
        for i in range(0, nmod):
            spatial_loadings.append(sKPCR_regression(lat_train, x_train[i].numpy(), np.ones((lat_train.shape[0], 1))))
    return (lat_train, lat_test, spatial_loadings, modality_weights, prediction_weights, pred_train, pred_test, best_performance)

def sKPCR_regression(X, Y, cov):
    contrast = np.transpose(np.hstack((np.eye(X.shape[1], X.shape[1]), np.zeros((X.shape[1], cov.shape[1])))))
    contrast = np.array(contrast, dtype='float32')
    design = np.hstack((X, cov))
    df = design.shape[0] - design.shape[1]
    ss = np.linalg.inv(np.dot(np.transpose(design), design))
    beta = np.dot(np.dot(ss, np.transpose(design)), Y)
    Res = Y - np.dot(design, beta)
    sigma = np.reshape(np.sqrt(np.divide(np.sum(np.square(Res), axis=0), df)), (1, beta.shape[1]))
    tmp1 = np.dot(beta.T, contrast)
    tmp2 = np.array(np.diag(np.dot(np.dot(contrast.T, ss), contrast)), ndmin=2)
    Tstat = np.divide(tmp1, np.dot(sigma.T, np.sqrt(tmp2)))
    return Tstat



def SBF_load(opts, behavioral_train, behavioral_test, behavioral_validation):
    allowed_modalities = opts.get('modalities_order', [])
    data_directories = [os.path.join(opts['brain_data_main_folder'], d) for d in os.listdir(opts['brain_data_main_folder']) if os.path.isdir(os.path.join(opts['brain_data_main_folder'], d))]
    data_directories = [d for d in data_directories if any((modality in os.path.basename(d) for modality in allowed_modalities))]
    f = open(opts['output_dir'] + '/order_of_loaded_data.txt', 'w')
    for i in range(len(data_directories)):
        f.write(data_directories[i] + ' \n')
    f.close()
    paths2data = [a for a in range(len(data_directories))]
    for folders in range(len(data_directories)):
        sub_files = os.listdir(data_directories[folders])
        for files in range(len(sub_files)):
            fileName, fileExtension = os.path.splitext(sub_files[files])
            if fileExtension == '.gz' or fileExtension == '.mgh' or fileExtension == '.txt':
                paths2data[folders] = os.path.join(data_directories[folders], sub_files[files])
    list_of_arrays = [np.array(a) for a in range(0, len(paths2data))]
    print(list_of_arrays)
    Data_Modality = deepcopy(list_of_arrays)
    mask = deepcopy(list_of_arrays)
    filetypes = deepcopy(list_of_arrays)
    names = deepcopy(list_of_arrays)
    fsl_path = opts['fsl_path']
    path2MNImask = fsl_path + 'data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    img_MNI_mask = nib.load(path2MNImask)
    MNI_mask = img_MNI_mask.get_fdata()
    shape_MNI = MNI_mask.shape
    shapes = {}
    affine = {}
    header = {}
    valid_train = ~pd.isna(behavioral_train).any(axis=1)
    n_removed_train = np.sum(~valid_train)
    valid_train = np.asarray(valid_train).ravel()
    valid_test = ~pd.isna(behavioral_test).any(axis=1)
    n_removed_test = np.sum(~valid_test)
    valid_test = np.asarray(valid_test).ravel()
    valid_validation = ~pd.isna(behavioral_validation).any(axis=1)
    n_removed_validation = np.sum(~valid_validation)
    valid_validation = np.asarray(valid_validation).ravel()
    for i in range(0, len(paths2data)):
        print('i = ', i)
        fileName, fileExtension = os.path.splitext(paths2data[i])
        print('loading data from modality number', i + 1, '=', fileName)
        filetypes[i] = fileExtension
        names[i] = fileName
        if fileExtension == '.gz':
            img = nib.load(paths2data[i])
            Data_Modality[i] = img.get_fdata()
            shape = Data_Modality[i].shape
            shape2 = Data_Modality[i].shape[:3]
            modality_name = os.path.basename(os.path.dirname(paths2data[i]))
            shapes[modality_name] = shape2
            affine[modality_name] = deepcopy(img.affine)
            header[modality_name] = img.header.copy()
            print('nii.gz masked raw shape: ', shape)
            Data_Modality[i] = np.reshape(Data_Modality[i], [shape[0] * shape[1] * shape[2], shape[3]], order='F')
            shape2d = Data_Modality[i].shape
            print('nii.gz 2d flattened shape: ', shape2d)
            mask_gz = MNI_mask
            del img
        elif fileExtension == '.txt':
            Data_Modality[i] = np.loadtxt(paths2data[i])
        elif fileExtension == '.mgh':
            fs_path = opts['fs_path']
            direct = os.path.dirname(paths2data[i])
            infile = os.path.basename(paths2data[i])
            new_name = list(infile)
            new_name[0] = 'r'
            right_side = direct + '/' + ''.join(new_name)
            new_name[0] = 'l'
            left_side = direct + '/' + ''.join(new_name)
            print(f'{left_side}')
            img1 = nib.load(left_side)
            img2 = nib.load(right_side)
            modality_name = os.path.basename(os.path.dirname(paths2data[i]))
            affine[modality_name] = img1.affine
            header[modality_name] = img1.header
            vol = np.concatenate((img1.get_fdata(), img2.get_fdata()), 0)
            data2d = np.reshape(vol, [vol.shape[0], vol.shape[3]], order='F')
            del vol
            NvoxPerHemi = data2d.shape[0] / 2
            fs_path = opts['fs_path']
            path_to_fsavg = opts['path_to_fsavg']
            if NvoxPerHemi == 2562:
                labelSrcDir = fs_path + '/subjects/fsaverage4/label/'
            if NvoxPerHemi == 10242:
                labelSrcDir = fs_path + '/subjects/fsaverage5/label/'
            if NvoxPerHemi == 40962:
                labelSrcDir = fs_path + '/subjects/fsaverage6/label/'
            if NvoxPerHemi == 163842:
                labelSrcDir = path_to_fsavg + '/label/'
            mask[i] = np.ones(np.shape(data2d)[0])
            needed_labels = ['lh.cortex.label', 'lh.Medial_wall.label', 'rh.cortex.label', 'rh.Medial_wall.label']
            tmp = [np.array(a) for a in range(4)]
            for fi in range(0, 4):
                tmp[fi] = nib.freesurfer.io.read_label(labelSrcDir + needed_labels[fi], read_scalars=True)
                tmp[fi] = tmp[fi][0] + 1 + (needed_labels[fi][0] == 'r') * NvoxPerHemi
            mask[i][tmp[1].astype('int')] = 0
            mask[i][tmp[3].astype('int')] = 0
            mask_mgh = mask[i]
            Data_Modality[i] = data2d[np.argwhere(mask[i][:] != 0)[:, 0], :]
            del data2d
        if '_Train_' in fileName:
            print(f'Training group: Removing {n_removed_train} subjects with missing behavioral data.')
            Data_Modality[i] = Data_Modality[i][:, valid_train]
        elif '_Test_' in fileName:
            print(f'Test group: Removing {n_removed_test} subjects with missing behavioral data.')
            Data_Modality[i] = Data_Modality[i][:, valid_test]
        elif '_Validation_' in fileName:
            print(f'Validation group: Removing {n_removed_validation} subjects with missing behavioral data.')
            Data_Modality[i] = Data_Modality[i][:, valid_validation]
    behavioral_train = behavioral_train[valid_train, :]
    behavioral_test = behavioral_test[valid_test, :]
    behavioral_validation = behavioral_validation[valid_validation, :]
    scaling_data_transform = [np.ndarray(a) for a in range(len(paths2data))]
    for k in range(len(paths2data)):
        Data_Modality[k] = Data_Modality[k] - np.matrix(np.mean(Data_Modality[k], 1)).T
        scaling_data_transform[k] = rms(Data_Modality[k], [], [])
        Data_Modality[k] = np.divide(Data_Modality[k], scaling_data_transform[k])
    Data_Modality = [matrix.T for matrix in Data_Modality]
    modalities = [os.path.basename(os.path.dirname(path)) for path in names]
    masks_for_SBF_inputs = mask
    masks_for_SBF_outputs = {}
    if 'mask_mgh' in locals():
        masks_for_SBF_outputs['mask_for_mgh'] = mask_mgh
    if 'mask_gz' in locals():
        masks_for_SBF_outputs['mask_for_gz'] = mask_gz
    fileinfo = {'data_directories': data_directories, 'masks_for_SBF_inputs': masks_for_SBF_inputs, 'scaling_data_transform': scaling_data_transform, 'filetype': filetypes, 'names': names, 'affine': affine, 'header': header, 'shapes': shapes, 'masks_for_SBF_outputs': masks_for_SBF_outputs, 'behavioral_data': {'train': behavioral_train, 'test': behavioral_test, 'validation': behavioral_validation}}
    return (Data_Modality, fileinfo, modalities)

def rms(IN, dim, options):
    if dim == []:
        out = np.sqrt(np.sum(np.square(IN)) / IN.size)
    else:
        out = np.sqrt(np.divide(np.sum(np.square(IN), dim), IN.shape[dim]))
    return out

def SBF_save_everything(output_dir, spatial_loadings, img_info, opts, dictionaries):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    Z = spatial_loadings
    for k, z_map_dict in enumerate(Z):
        file_extension = opts['modalities_order_filetypes'][k]
        D = dictionaries[k]
        z_map_voxels = np.dot(D, z_map_dict)
        if file_extension == '.gz':
            modality_name = opts['modalities_order'][k]
            out_file_name = '/SBFOut_' + modality_name + '.nii.gz'
            tmp_mask = img_info['masks_for_SBF_outputs']['mask_for_gz']
            modality_key = modality_name + '_train'
            if modality_key not in img_info['shapes']:
                raise ValueError(f"Shape information not found for modality '{modality_key}' in img_info['shapes'].")
            shape_MNI = img_info['shapes'][modality_key]
            reshaped_out = np.zeros((*shape_MNI, z_map_voxels.shape[1]))
            for i in range(z_map_voxels.shape[1]):
                reshaped_out[:, :, :, i] = np.reshape(z_map_voxels[:, i], shape_MNI, order='F')
            original_header = img_info['header'][modality_key]
            img2save = nib.Nifti1Image(reshaped_out, affine=img_info['affine'][modality_key], header=original_header)
            voxel_sizes = original_header.get_zooms()
            img2save.header.set_zooms(voxel_sizes)
            img2save = np.nan_to_num(img2save, nan=0.0)
            outname = output_dir + out_file_name
            nib.save(img2save, outname)
            del tmp_mask
        if file_extension == '.mgh':
            modality_name = opts['modalities_order'][k]
            out_file_name = modality_name + '.mgh'
            tmp_mask = img_info['masks_for_SBF_outputs']['mask_for_mgh']
            non_zero_vertex = np.where(tmp_mask != 0)[0]
            out = np.tile(np.matrix(tmp_mask).T, z_map_voxels.shape[1])
            out[non_zero_vertex, :] = z_map_voxels
            N_vertex_per_side = tmp_mask.shape[0] / 2
            out_left = np.expand_dims(np.expand_dims(out[range(int(N_vertex_per_side)), :], 1), 1)
            out_file_name_lh = '/SBFOut_lh_' + out_file_name
            outname = output_dir + out_file_name_lh
            out_left = np.nan_to_num(out_left, nan=0.0)
            img2save = nib.freesurfer.mghformat.MGHImage(out_left, affine=opts['SBFOut_affine'][k], header=opts['SBFOut_headers'][k], extra=None, file_map=None)
            nib.save(img2save, outname)
            out_right = np.expand_dims(np.expand_dims(out[int(N_vertex_per_side):int(2 * N_vertex_per_side), :], 1), 1)
            out_file_name_rh = '/SBFOut_rh_' + out_file_name
            outname = output_dir + out_file_name_rh
            out_right = np.nan_to_num(out_right, nan=0.0)
            img2save = nib.freesurfer.mghformat.MGHImage(out_right, affine=opts['SBFOut_affine'][k], header=opts['SBFOut_headers'][k], extra=None, file_map=None)
            nib.save(img2save, outname)
    return spatial_loadings


def clean_modality_name(modality_name):
    # Remove common suffixes and return the base modality name
    for suffix in ['_test', '_train', '_validation']:
        modality_name = modality_name.replace(suffix, '')
    return modality_name.strip()
def sort_and_filter_by_modality_order(data_list, order):
    # Normalize and clean the modality names in the order list
    modality_index = {clean_modality_name(modality): index for index, modality in enumerate(order)}

    # If there is a problem do print the normalized modality index for debugging
    #print("Normalized Modality Index Map:", modality_index)

    # Filter and sort the list
    filtered_sorted_list = sorted(
        [item for item in data_list if clean_modality_name(item['modality']) in modality_index],
        key=lambda x: modality_index[clean_modality_name(x['modality'])]
    )

    # If your data lists come out in the wrong order, can use this for debugging to show what's being processed
    #print("Items processed:",
    #      [item['modality'] for item in data_list if clean_modality_name(item['modality']) in modality_index])

    return filtered_sorted_list
