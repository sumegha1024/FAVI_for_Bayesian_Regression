'''KL divergence hard coded p=2 IAF with relu''' 

#This script provides a validation at each stage of the derivation of KL divergence in the SI for the paper. It is not required to reproduce any results in the paper.
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch.nn as nn
import math
import scipy.stats as ss
import random
from torch.distributions import MultivariateNormal, Normal
import argparse
sys.path.append('/mnt/ufs18/home-104/premchan/Extensions_Normalizing_Flows_Review/')
from utils import set_all_seeds 
from output import generate_plots


parser = argparse.ArgumentParser(description='inputs_kl_iaf')
parser.add_argument('--rho',default =-0.8, type=float, help='Correlation coefficient for design matrix')
parser.add_argument('--sigma',default =1.0, type=float, help='Standard dev for simulated data')
parser.add_argument('--seed',default =0, type=int, help='seed for init')
parser.add_argument('--out',default = '/mnt/home/premchan/FAVI_for_Bayesian_Regression/Out/KL_testing/', type=str, help='path to results')
parser.add_argument('--n_data',default =100, type=int, help='number of samples for y')
args = parser.parse_args()

print("Set-up params")
print("seed",args.seed)
print("rho",args.rho)

set_all_seeds(args.seed)

isExist = os.path.exists(args.out)
if not isExist:
    os.makedirs(args.out)

############################################################################################################################
'''Simulate dataset for Bayesian Regression'''
global n,nt,p,y,yt,X,Xt, X_X, X_y, y_y, Sig_MCMC, mu_MCMC
nw=args.n_data
p=2
beta0=np.random.uniform(0.5,2,p)
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

'''True Posterior'''
Sig_MCMC=torch.inverse(X_X+torch.eye(p)) #inversion
mu_MCMC=torch.sum(Sig_MCMC*X_y,1)
beta_samples_MCMC_store=ss.multivariate_normal.rvs(mu_MCMC,Sig_MCMC,10000)

######################################################Simplified KL as function of r1 = b1/a1##############################################

#tt should a Tensor of size 1
def fn(tt):
    normal_d = Normal(0, 1)
    if X_X[0,1]<0:   
        num = (1-normal_d.cdf(-tt))
        den = (1+tt**2)* (1-normal_d.cdf(-tt)) +tt* normal_d.log_prob(-tt).exp()
        den+= -(tt*(1-normal_d.cdf(-tt)) + normal_d.log_prob(-tt).exp())**2
    else:
        num = (1-normal_d.cdf(tt))
        den = (1+tt**2)* (1-normal_d.cdf(tt)) -tt* normal_d.log_prob(tt).exp()
        den+= -(tt*(1-normal_d.cdf(tt)) - normal_d.log_prob(tt).exp())**2
    return num, den

#plotting f(tt)
tt_array=np.linspace(-4,4,100)
fnr0=fn(torch.Tensor(tt_array))[0]**2/(fn(torch.Tensor(tt_array))[1]+1e-10)
plt.plot(tt_array,fnr0.numpy(),linestyle='dotted')
plt.savefig(args.out+'fnr0_'+str(args.rho)+'.png')
plt.clf()

plt.plot(tt_array,fn(torch.Tensor(tt_array))[1].numpy()-fn(torch.Tensor(tt_array))[0].numpy()**2,linestyle='dotted',color='black')
plt.savefig(args.out+'num_den_'+str(args.rho)+'.png')
plt.clf()

#Optimal value for a1 
def out_a1(tt):

    x1_tilda = (X[:,0]**2).sum()+1
    x2_tilda = (X[:,1]**2).sum()+1
    x1x2 = ((X[:,0]*X[:,1]).sum())
    
    num, den =fn(torch.Tensor(tt))
    
    a1 = x2_tilda*(den)/(x1_tilda*x2_tilda*den - (x1x2**2)*(num**2))
    
    return a1

#plotting a1(tt)
a1=out_a1(torch.Tensor(tt_array))
plt.plot(tt_array,a1,linestyle='dotted',label='a1')
plt.legend()
plt.tight_layout()
plt.savefig(args.out+'a1_'+str(args.rho)+'.png')
plt.clf()


#Kl divergence as function of tt
def kld(tt):
    kl_sum_l=[-1]*tt.size()[0]
    x1_tilda = (X[:,0]**2).sum()+1
    x2_tilda = (X[:,1]**2).sum()+1
    x1x2 = ((X[:,0]*X[:,1]).sum())
    pi2 = torch.Tensor([2*math.pi])
    b1 = (x2_tilda*X_y[0] - x1x2*X_y[1])/(x1_tilda*x2_tilda - x1x2**2)

    num, den =fn(torch.Tensor(tt))
    a1_l = out_a1(tt)
    
    
    #KL divergence
    kl_sum0=  -0.5*torch.log(x1_tilda*x2_tilda - x1x2**2) + .5*torch.log(x2_tilda)
    kl_sum0+= 0.5*(X_y*((torch.inverse(X_X+torch.eye(p))*X_y).sum(1))).sum()
    kl_sum0+= 0.5*(b1**2)*x1_tilda - X_y[0]*b1 - 0.5*(X_y[1] - x1x2*b1)**2/x2_tilda
    
    for i in range(tt.shape[0]): 
        if a1_l[i]>0 and den[i]>0:
            kl_sum1 = kl_sum0 - 0.5*torch.log(x2_tilda)-0.5*torch.log((den[i])) + 0.5*torch.log(x1_tilda*x2_tilda*den[i] - (x1x2**2)*(num[i]**2))
            kl_sum_l[i]=kl_sum1

    return kl_sum_l



kl_mfvi = -0.5*torch.log(((X[:,0]**2).sum()+1)*((X[:,1]**2).sum()+1) - ((X[:,0]*X[:,1]).sum())**2) + 0.5*torch.log((X[:,0]**2).sum()+1) + 0.5*torch.log((X[:,1]**2).sum()+1)
print("kl_mfvi",kl_mfvi)
#plotting kl
kl_out = kld(torch.Tensor(tt_array))
idx = [i for i, x in enumerate(kl_out) if x != -1]
print("min simplified kld",min(np.array(kl_out)[idx]))
plt.plot(tt_array[idx],np.array(kl_out)[idx],linestyle='dotted',label='KL IAF k=1',color='green')
plt.plot(tt_array[idx],[kl_mfvi]*len(idx),label='KL MF-VI',color='blue')
plt.ylabel("KL")
plt.xlabel(r'$\tilde{t}$')
plt.legend()
plt.tight_layout()
plt.savefig(args.out+'kl_'+str(args.rho)+'.png')

########################################################Auxilary Functions for KL ##################################################

##Checking to see if un-simplified KL matches optimized kl at optimal parameters

print("mu_true",mu_MCMC)
print("Sig_true",Sig_MCMC)

def kappa1(c,tt):
    normal_d=Normal(0,1)
    out=c*tt*(1-normal_d.cdf(-(c*tt)/(torch.abs(c)+1e-7)))
    out+= torch.abs(c)*(normal_d.log_prob(tt).exp())
    return out

def kappa2(c,tt):
    normal_d = Normal(0,1)
    out= c**2*(tt**2+1)*(1-normal_d.cdf(-(c*tt)/(torch.abs(c)+1e-7)))
    out+= c*torch.abs(c)*tt*(normal_d.log_prob(tt).exp())
    return out

def kappa3(c,tt):
    normal_d = Normal(0,1)
    out = c*(1-normal_d.cdf(-tt*c/(torch.abs(c)+1e-7)))
    return out

def kld_raw(a,r,ta,tb,c,tt):
    x1_tilda=X_X[0,0]+1
    x2_tilda=X_X[1,1]+1
    x1x2= X_X[0,1]

    kl_sum0= -1 -ta - torch.log(a) -0.5*torch.log(x1_tilda*x2_tilda - x1x2**2)
    kl_sum0+= 0.5*x2_tilda*torch.exp(2*ta) + 0.5*x1_tilda*(a**2)
    kl_sum0+= 0.5*(X_y*((torch.inverse((X_X+torch.eye(p)))*X_y).sum(1))).sum()
    kl_sum0+= 0.5*x1_tilda*(a**2)*(r**2) 
    kl_sum0+= 0.5*x2_tilda*(tb**2 + 2*tb*kappa1(c,tt) + kappa2(c,tt))
    kl_sum0+= - X_y[0]*a*r - X_y[1]*tb - X_y[1]*kappa1(c,tt)
    kl_sum0+= x1x2*tb*r*a + x1x2*r*a*kappa1(c,tt) + x1x2*a*kappa3(c,tt)

    return kl_sum0

################################Some checks#################################################################
#print("mu_true",mu_MCMC)
#print("Sig_MCMC",Sig_MCMC)
#ind=np.argmin(kl_out[1]) 
#print("optimal r",r1_array[ind])
#print("min simplified kl",kl_out[1][ind])

#FAVI optimal
#b1=mu_MCMC[0]
#t_a= 0.5*torch.log(1/(X_X[1,1]+1))
#c_b= -X_X[0,1]*fnr0[ind]/(X_X[1,1]+1)
#t_b = (X_y[1] -b1*X_X[0,1])/(X_X[1,1]+1) - kappa1(c_b,a1[1][ind],b1)

#Favi empirical - from stochastic gradient descent when xtx=-0.8
c_b=torch.Tensor([0.1346])
t_b=torch.Tensor([4.0866e-01])
b1=torch.Tensor([1.0791])
t_a=torch.log(torch.Tensor([0.1184]))
a1=torch.Tensor([0.1729])
t_0 = torch.Tensor([0.9949])
#print("kl_raw empirical",kld_raw(a1,b1/a1,t_a,t_b,c_b,t_0/(c_b+1e-7)))

#Sampling IAF distribution based on optimal params
nsamps=10000
set_all_seeds(5)
std=torch.Tensor([a1,torch.exp(t_a)])
mean=torch.Tensor([b1,0]).repeat(nsamps,1)
zspl = torch.FloatTensor(nsamps,p).normal_()
g=nn.ReLU()
mean[:,1]=g(c_b*zspl[:,0] + t_0) + t_b
data=std*zspl + mean


###################Re-creating Density Plots#######################

color_dict={'MCMC':'red','Flows':'green'}
hyper_params={'seed':args.seed,'n':args.n_data,'p':p,'dsfdim':1,'dsl':1,'cdim':1,'cl':2,'bsize':80,'rho':args.rho}

mu_post_smps={'MCMC':beta_samples_MCMC_store,'Flows':data[:,0:p].detach().numpy()}  
#generate_plots(mu_post_smps, color_dict, hyper_params, args.out, beta0, mcmc_diag=None)


