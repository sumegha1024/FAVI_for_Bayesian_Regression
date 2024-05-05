'''Linear Regression with NUTS, Gibbs, MF-VI and FAVI''' 
import torch
import numpy as np
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join('/mnt/ufs18/home-104/premchan/FAVI_for_Bayesian_Regression/','Flows_scripts'))
import nn as nn_
import torch.nn as nn
from tqdm import tqdm
import math
import scipy.stats as ss
import random
import time
from torch.distributions import MultivariateNormal
import argparse
import flows
from utils import set_all_seeds 
import dill
import pymc as pm
import output
import matplotlib.pyplot as plt
#import arviz as az

parser = argparse.ArgumentParser(description='inputs_regression')
parser.add_argument('--mcmc_diag', nargs='+',default=None,help='list of strings. use NUTS if you want MCMC diagnostics for NUTS and MCMC for any other mcmc method')
parser.add_argument('--vi', default=True, action='store_false', help='Whether we want results for MF-VI')
parser.add_argument('--nuts', default=False, action='store_false', help='Whether we want results for NUTS')
parser.add_argument('--rho',default =0.6, type=float, help='Correlation coefficient for design matrix')
parser.add_argument('--sigma',default =1.0, type=float, help='Standard dev for simulated data')
parser.add_argument('--tau',default =1.0, type=float, help='Standard dev for beta prior')
parser.add_argument('--sigma_prior', default=False, action='store_true', help='Whether we want sigma to be treated as constant or use a prior')
parser.add_argument('--pr_roc', default=False, action='store_true', help='Whether we want pr and roc plots to be generated')
parser.add_argument('--lr', default = 5e-3, type=float, help='learning rate for FAVI')
parser.add_argument('--seed',default =5, type=int, help='seed for simulation')
parser.add_argument('--flow_samps',default =64, type=int, help='number of flow samples while training')
parser.add_argument('--flow_type',default = 'IAF', type=str, help='type of flow transform to be used. DSF, DDSF or IAF')
parser.add_argument('--bsize_tr',default =0, type=int, help='default 0: sets it to training data size. i.e only one batch')
parser.add_argument('--out',default = '/mnt/home/premchan/FAVI_for_Bayesian_Regression/Out/', type=str, help='path to results')
parser.add_argument('--pretrain',default = 0, type=int, help='epoch corresponding to pre-trained model')
parser.add_argument('--data_dim',default =2, type=int, help='dimension of beta vector')
parser.add_argument('--n_data',default =200, type=int, help='number of samples for y')
parser.add_argument('--dsf_dim',default =4, type=int, help='hidden dim for dsf/ddsf flow')
parser.add_argument('--dsf_l',default =1, type=int, help='num of layers for dsf/ddsf flow')
parser.add_argument('--cmade_dim',default =1, type=int, help='hidden dim for cmade network')
parser.add_argument('--cmade_l',default =2, type=int, help='num of layers for cmade network plus 1')
parser.add_argument('--sparse',default =0.2, type=float, help='sparsity level for true beta0 in %')
parser.add_argument('--writecsv', default=False, action='store_true', help='Whether we want to write performance metrics to csv file')
args = parser.parse_args()

args.out=args.out+'Out_flowtype_'+args.flow_type+'_fsamps'+str(args.flow_samps)+'_lr_'+str(args.lr)+'_rho'+str(args.rho)+'_cdim'+str(args.cmade_dim)+'/'
isExist = os.path.exists(args.out)
if not isExist:
    os.makedirs(args.out+'MH_Diagnostics/')

print("Simulation set-up #############")
print("n_data",args.n_data)
print("data_dim p",args.data_dim)
print("seed",args.seed)
print("rho",args.rho)

print("Training hyper-params #############")
print("bsize",args.bsize_tr)
print("flow samps",args.flow_samps)

print("NAF hyper-params ##############")
print("dsf_dim, cmade_dim, ds_layers, cmade_layers", args.dsf_dim, args.cmade_dim, args.dsf_l, args.cmade_l)


set_all_seeds(0)

epochs_gibbs=10000
epochs_flows=4000
epochs_vi =4000
if args.n_data>=2000:
    epochs_vi=20000
thin = 1

############################################################################################################################
'''Simulate dataset for Bayesian Regression experiments'''
global n,nt,p,y,yt,X,Xt,tau
tau=args.tau
nw=args.n_data
p=args.data_dim
beta0=np.random.uniform(0.5,2,p)
idx0=set([item for item in range(p)])
S0=random.sample(idx0,int(p*(1-args.sparse)))
if p > 10:
    beta0[S0]=0.0
print("beta0",beta0)
mean=np.zeros((p,))
rho=args.rho
cov=(1-rho)*np.identity(p)+rho*np.ones((p,p))
Xw= np.random.multivariate_normal(mean, cov, nw)
ew=np.random.normal(0,1,nw)
sigma = args.sigma
yw=np.dot(Xw,beta0)+sigma*ew
   
#Split into train and test sets
idx=set([item for item in range(nw)])
S=random.sample(idx,int(nw*0.8))
St=list(idx.difference(S))
X=torch.tensor(Xw[S,])
Xt=torch.tensor(Xw[St,])
y=torch.tensor(yw[S])
yt=torch.tensor(yw[St])
n=X.shape[0]
nt=Xt.shape[0]

y_y=torch.sum(y**2)
X_y=torch.sum(X.T*y,1)
X_X=torch.mm(X.T,X)
X_diag=torch.diag(X_X)
beta_ols = np.array(torch.matmul(torch.inverse(X_X),X_y))

#######################################################################################################################

#function to calculate model predictive rmse
def mod_rmse(y,x,beta):
    ypred = np.dot(x,beta.mean(0))
    return np.sqrt(np.mean((y - ypred)**2))

set_all_seeds(args.seed)

rmse = {"MCMC":None,"Flows":None,"MF-VI":None,"NUTS":None}
Time = {"MCMC":None,"Flows":None,"NUTS":None}


##################################################################MCMC - Gibbs#############################################################################################
'''Gibbs Sampling or True Posterior: Depending on whether sigma is unknown or known respectively'''

start = time.time()
if args.sigma_prior == False:
    sigma_MCMC = 1
    Sig_MCMC=torch.inverse(X_X/(sigma_MCMC**2)+torch.eye(p)/tau**2) #inversion
    mu_MCMC=torch.sum(Sig_MCMC*X_y/(sigma_MCMC**2),1)
    beta_samples_MCMC_store=ss.multivariate_normal.rvs(mu_MCMC,Sig_MCMC,10000)
else:
    nsweep = epochs_gibbs
    sigma_samples_MCMC_store = np.zeros(nsweep)
    beta_samples_MCMC_store = np.zeros((nsweep,p))
    sigma_MCMC = ss.invgamma.rvs(a=2.5/2,scale=2.5/2)
    beta_MCMC = beta_ols
    for i in range(nsweep):
        #Update steps
        ytilda = y - np.dot(X,beta_MCMC)
        sigma_MCMCsq = ss.invgamma.rvs(a=(2.5/2+n/2),scale=(2.5/2+torch.sum(ytilda**2)/2))
        Sig_MCMC=torch.inverse(X_X/sigma_MCMCsq+torch.eye(p)/tau**2)
        mu_MCMC=torch.sum(Sig_MCMC*X_y/sigma_MCMCsq,1)
        beta_MCMC=ss.multivariate_normal.rvs(mu_MCMC,Sig_MCMC)
        #store values
        beta_samples_MCMC_store[i,] = beta_MCMC
        sigma_samples_MCMC_store[i] = sigma_MCMCsq
        
        burn_in=1000
        idx = [i for i in range(burn_in,beta_samples_MCMC_store.shape[0],thin)]
        beta_samples_MCMC_store=beta_samples_MCMC_store[idx]
        sigma_samples_MCMC_store=sigma_samples_MCMC_store[idx]
 
Time['MCMC'] = time.time()-start
rmse['MCMC']= mod_rmse(np.array(yt),np.array(Xt),beta_samples_MCMC_store)

#################################MCMC-NUTS#####################################################

'''NUTS'''
if args.nuts==True:
    basic_model = pm.Model()
    with basic_model:
        # Priors for unknown model parameters
        beta = pm.Normal("beta", mu=0, sigma=args.tau, shape=p)
        if args.sigma_prior==True:
            sigma = pm.Gamma("sigma", alpha=2.5/2, beta=2.5/2)
        # Expected value of outcome
        mu = pm.math.dot(X.numpy(),beta)
        # Likelihood (sampling distribution) of observations
        if args.sigma_prior==True:
            Y_obs = pm.Normal("Y_obs", mu=mu, tau=sigma, observed=y)
        else: 
            Y_obs = pm.Normal("Y_obs", mu=mu, tau=1.0, observed=y)

    start=time.time()
    with basic_model:
        # draw posterior samples
        idata = pm.sample(30000,chains=1)
    Time["NUTS"] = time.time()-start
    #print(az.summary(idata))

    beta_samples_NUTS_store=idata.posterior["beta"].to_numpy()[0,:,:]
    idx = [i for i in range(0,beta_samples_NUTS_store.shape[0],3)]
    beta_samples_NUTS_store=beta_samples_NUTS_store[idx,:]
    rmse["NUTS"]= mod_rmse(np.array(yt),np.array(Xt),beta_samples_NUTS_store)

    if args.sigma_prior==True:
        sigma_samples_NUTS_store=1.0/idata.posterior["sigma"].to_numpy()[0,:]
        sigma_samples_NUTS_store=sigma_samples_NUTS_store[idx]




#############################################Auxillary functions for FAVI###########################################################################################################
'''Some useful functions'''
X_y_f = X_y.type(torch.FloatTensor)
X_X_f = X_X.type(torch.FloatTensor)
y_y_f = y_y.type(torch.FloatTensor)



def gaussian_log_pdf(z):
    """
    Arguments:
    ----------
        - z: a batch of m data points (size: m x no. of params) tensor
    """
    return -.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(1)

def invgamma_log_pdf(z,a,b):
    """
    Arguments:
    ----------
        - z: a batch of m data points (size: m x no. of params) tensor
    """
    a = torch.Tensor([a])
    b = torch.Tensor([b])
    return a*torch.log(b) - torch.lgamma(a) -(a+1)*torch.log(z) - b/z

def U5(y_in,X_in,z,s=None): #s is sigma_squared
    y_in=y_in.type(torch.FloatTensor)
    X_in=X_in.type(torch.FloatTensor)
    y_y_fl=torch.sum(y_in**2)
    X_y_fl=torch.sum(X_in.T*y_in,1)
    X_X_fl=torch.mm(X_in.T,X_in)
    X_diag_fl=torch.diag(X_X)
    if s is None:
        return 0.5*(torch.sum(torch.mm(z, X_X_fl)*z,1)-2*torch.sum(z*X_y_fl,1)+y_y_fl) + n/2*torch.log(torch.tensor([math.pi * 2]))
    else:
        return 0.5*n*torch.log(s) + 0.5*(torch.sum(torch.mm(z, X_X_fl)*z,1)/s-2*torch.sum(z*X_y_fl,1)/s+y_y_fl/s)

def U6(z,s = None):
    if s is None:
        return 0.5*torch.sum(z**2,1)/tau**2 + p/2*torch.log(torch.tensor([math.pi * 2]))
    else:
        return 0.5*torch.sum(z**2,1)/tau**2 - invgamma_log_pdf(s,2.5/2,2.5/2)

exact_log_density = lambda y_in,X_in,z,s=None: -U5(y_in,X_in,z,s)-U6(z,s)


class model(object):
    
    def __init__(self, target_energy, flow_type, ds_dim,ds_l, cmade_dim, cmade_l):

        self.cmade_dim=cmade_dim
        if args.sigma_prior==False:
            self.nparams = p
        else: 
            self.nparams = p+1

        self.ds_dim = ds_dim

        if flow_type=='IAF':
            self.mdl = flows.IAF(self.nparams, self.cmade_dim, 1, cmade_l,activation=nn.ReLU(),realify = torch.exp) 
        elif flow_type == "DSF":
            self.mdl = flows.IAF_DSF(self.nparams, self.cmade_dim, 1, cmade_l,
                num_ds_dim=self.ds_dim, num_ds_layers=ds_l)
        elif flow_type == "DDSF":
            self.mdl = flows.IAF_DDSF(self.nparams, self.cmade_dim, 1, cmade_l,
                num_ds_dim=self.ds_dim, num_ds_layers=ds_l)

        
        self.optim = optim.Adam(self.mdl.parameters(), lr=args.lr, 
                                betas=(0.9, 0.999))
        
        
        self.target_energy = target_energy
        
    def train(self,epochs,y_in,X_in,bsize_tr=n,fsamps=64):
        
        total = epochs
        loss_store=[]
        num_batches = n / bsize_tr
        permutation = torch.randperm(n)
        for it in range(total):

            for bh in range(0,n,bsize_tr):
                self.optim.zero_grad()
                indices=permutation[bh:bh + bsize_tr]
                batch_x, batch_y = X_in[indices], y_in[indices]
            
                spl, logdet, _ = self.mdl.sample(fsamps)
                if args.sigma_prior == False:
                    losses = - self.target_energy(batch_y,batch_x,spl) - logdet - p/2*torch.log(torch.tensor([math.pi * 2])) -torch.Tensor([p/2])
                else:
                    slices = list(range(0,p))
                    losses = - self.target_energy(batch_y,batch_x,spl[:,slices],(torch.log(1+torch.exp(spl[:,-1])))**2) - logdet - torch.log(torch.exp(spl[:,-1])/(1+torch.exp(spl[:,-1]))) -torch.log(2*torch.log(1+torch.exp(spl[:,-1])))
                loss = losses.mean()
            
                loss.backward()
                self.optim.step()

                loss_store.append(loss.detach().numpy())


            
            if ((it + 1) % 1000) == 0:
                print ('Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data))
        return loss_store

#######################################################MEAN-FIELD########################################################################################


y_y=torch.sum(y**2)
X_y=torch.sum(X.T*y,1)
X_X=torch.mm(X.T,X)
X_diag=torch.diag(X_X)

def elbo(params):
    tau_t = torch.Tensor([tau])
    beta_mu = params[0:p]
    beta_rho = params[(p):(2*p)]
    beta_sig = torch.log(1 + torch.exp(beta_rho))

    nll = 0
    nll = nll + 0.5*y_y
    nll = nll + 0.5*torch.sum(torch.sum(X*beta_mu,1)**2)
    nll = nll - torch.sum(beta_mu*X_y) 
    nll = nll + 0.5*torch.sum(X_diag*(beta_sig**2))
    nll = nll + n/2*torch.log(torch.tensor([math.pi * 2]))
    kl_beta = 0.5*(torch.sum(beta_sig**2)/tau_t**2-2*torch.sum(torch.log(beta_sig)) + 2*p*torch.log(tau_t)+torch.sum(beta_mu**2)/tau_t**2-p)
    kl_sigma = 0
   
    if args.sigma_prior == True:
        sigma_mu = params[2*p]
        sigma_rho = params[2*p+1]
        sigma_rho_trans = torch.log(1 + torch.exp(sigma_rho))

        nll = nll*torch.exp(-sigma_mu + 0.5*sigma_rho_trans**2)
        nll = nll + 0.5*n*sigma_mu
        a = b = 2.5/2
        kl_sigma = a*sigma_mu - torch.log(sigma_rho_trans) + b*torch.exp(-sigma_mu + 0.5*sigma_rho_trans**2)
 
    kl = kl_sigma + kl_beta
    loss= nll+kl
    return loss

if args.vi==True:

    set_all_seeds(args.seed)

    params1 = torch.Tensor(np.array(torch.matmul(torch.inverse(X_X),X_y)))
    params2 = torch.Tensor(p).uniform_(-1,-1)
    if args.sigma_prior == True:
        params3 = torch.Tensor(1).uniform_(-1,1)
        params4 = torch.Tensor(1).uniform_(-1,-1)
        params = torch.cat((params1,params2,params3,params4),0)
    else:
        params = torch.cat((params1,params2),0)


    start=time.time()
    params.requires_grad_()
    n_optim_steps = int(epochs_vi)
    optimizer = torch.optim.Adam([params], 0.005)
    start = time.time()
    loss_store_vi=[]
    for ii in range(n_optim_steps):
        optimizer.zero_grad()
        loss = elbo(params)
        loss_store_vi.append(loss.detach().numpy())
        #print('Step # {}, loss: {}'.format(ii, loss.item()))
        loss.backward()
        # Access gradient if necessary
        optimizer.step()
    Time["MF-VI"] = time.time()-start

    mu_VI=np.array(params[0:p].detach())
    Sig_VI=np.array((torch.eye(p)*(torch.log(1+torch.exp(params[p:(2*p)])))**2).detach())
    beta_samples_VI=ss.multivariate_normal.rvs(mu_VI,Sig_VI,10000)
    rmse["MF-VI"]= mod_rmse(np.array(yt),np.array(Xt),beta_samples_VI)

    if args.sigma_prior==True:
        ln_sigma_samples_VI = np.log(1+np.exp(params[2*p+1].detach().numpy()))*np.random.normal(size=10000) + params[2*p].detach().numpy()
        sigma_samples_VI = np.exp(ln_sigma_samples_VI)

    plt.plot(loss_store_vi[len(loss_store_vi)-500:len(loss_store_vi)])
    plt.savefig(args.out+'vi_loss_seed_'+str(args.seed)+'_ndata_'+str(args.n_data)+'_p_'+str(args.data_dim)+'_rho_'+str(args.rho)+'.png')
    plt.clf()
    #print("VI Loss",loss_store_vi[len(loss_store_vi)-50:len(loss_store_vi)])


################################################FLOWS###########################################################################
#Setting the batch size for training
if args.bsize_tr==0:
    args.bsize_tr=n

set_all_seeds(args.seed)
mdl = model(exact_log_density, args.flow_type, args.dsf_dim, args.dsf_l, args.cmade_dim, args.cmade_l)
#Loading pre-trained model if applicable 
if args.pretrain>0:
    mdl.mdl.load_state_dict(torch.load('Pre-trained_fsamps'+str(args.flow_samps)+'_lr_'+str(args.lr)+'/model_epoch_'+str(args.pretrain)+'_ndata_'+str(args.n_data)+'_p_'+str(args.data_dim)+'.pth'))
start = time.time()
loss_store_favi = mdl.train(epochs_flows,y,X,args.bsize_tr,args.flow_samps)
Time["Flows"]=time.time() - start
#torch.save(mdl.mdl.state_dict(), 'Pre-trained_fsamps'+str(args.flow_samps)+'_lr_'+str(args.lr)+'/model_epoch_'+str(args.pretrain+epochs_flows)+'_ndata_'+str(args.n_data)+'_p_'+str(args.data_dim)+'.pth',pickle_module=dill)

data = mdl.mdl.sample(10000)[0].data.numpy()

rmse["Flows"]= mod_rmse(np.array(yt),np.array(Xt),data[:,list(range(0,p))])



#################################################PLOTS#################################################
color_dict={'Truth':'red','NUTS':'purple','MF-VI':'blue','FAVI':'green'}
hyper_params={'seed':args.seed,'n':args.n_data,'p':args.data_dim,'dsfdim':args.dsf_dim,'dsl':args.dsf_l,'cdim':args.cmade_dim,'cl':args.cmade_l,'bsize':args.bsize_tr,'rho':args.rho}

mu_post_smps={'Truth':beta_samples_MCMC_store,'FAVI':data[:,0:p]}  
if args.vi==True:
    mu_post_smps['MF-VI']=beta_samples_VI
if args.nuts==True:
    mu_post_smps['NUTS']=beta_samples_NUTS_store

if args.sigma_prior==False:
    output.generate_plots(mu_post_smps, color_dict, hyper_params, args.out, beta0, mcmc_diag=args.mcmc_diag)
else:
    sigma_post_smps={'MCMC':sigma_samples_MCMC_store,'Flows':(np.log(1+np.exp(data[:,-1])))**2}
    if args.vi==True:
        sigma_post_smps['MF-VI']=sigma_samples_VI
    if args.nuts==True:
        sigma_post_smps['NUTS']=beta_samples_NUTS_store
    output.generate_plots(mu_post_smps, color_dict, hyper_params, args.out, beta0, sigma_post_smps,mcmc_diag=args.mcmc_diag) 


#Flows Loss
plt.plot(loss_store_favi[len(loss_store_favi)-800:len(loss_store_favi)])
plt.savefig(args.out+'flows_loss_seed_'+str(args.seed)+'_ndata_'+str(args.n_data)+'_p_'+str(args.data_dim)+'_dsdim_'+str(args.dsf_dim)+'_dsl_'+str(args.dsf_l)+'_cdim_'+str(args.cmade_dim)+'_cl_'+str(args.cmade_l)+'_bsize_'+str(args.bsize_tr)+'_rho_'+str(args.rho)+'.png')
plt.clf()

#print("Flows Lossmean",np.mean(loss_store[len(loss_store)-500:len(loss_store)]))


#####################################################INFERENCE##########################################################################

#Calculating sample KL div when \sigma_2 is known using elbo obtained after training
#Note the same function to calculate sample kl will not apply when number of batches>1.

if args.sigma_prior==False and args.bsize_tr==n:
    kl={}
    H=np.linalg.inv(X_X.numpy()+np.identity(p)/(tau**2))
    X_y_n=X_y.numpy()
    #Log marginal 
    mxy = -n*0.5*np.log(2*math.pi) -0.5*(y_y.numpy()) + 0.5*((X_y_n*((H*(X_y_n)).sum(1))).sum())
    if args.vi==True:
        kl["MF-VI"] = np.mean(loss_store_vi[len(loss_store_vi)-500:len(loss_store_vi)]) + mxy
    kl["Flows"] = np.mean(loss_store_favi[len(loss_store_favi)-500:len(loss_store_favi)]) + mxy
    kl["MCMC"] = None
    kl["NUTS"] = None

output.generate_metrics(mu_post_smps, color_dict, hyper_params, rmse, Time, args.out, mu0=beta0, pr_roc=args.pr_roc, writecsv=args.writecsv,kl=kl)


