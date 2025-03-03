import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as ss
import torch
from torch.distributions import MultivariateNormal, Normal
######Relu function intuition plot
def g(x):
    return np.maximum(x, 0)

x = np.linspace(-5, 5, 400)
# relu
g_x = g(x)

#Gaussian distribution plot
mean = 2
sd = 1
gaussian = norm.pdf(x, mean, sd)

# Plot g(x) and the Gaussian distribution
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian, color='red',label='Samples from N('+str(mean)+', '+str(sd**2)+') distribution.')
plt.plot(x, g_x, color='blue',label='ReLU function')
plt.xlabel('x',fontsize='20')
plt.ylabel('g(x)',fontsize=20)
plt.xticks(size=14)
plt.yticks(size=14)
plt.title(r'$c<0$',fontsize=22)
plt.grid(True)
plt.legend(fontsize=20)
plt.savefig('relu_cplus.png')


########KL upper bound plots #########

def kl_upper_bound(K,p,rho):
    #return rho*p*(p-K/2-2)/((1-rho)*(1-rho + rho*p))
    return (p-K/2-1)*(-rho*p/(1-rho + rho*p) - np.log((1 - rho)/(1-rho + rho*p)))

x = np.linspace(0.0, 0.8, 400)
y1=kl_upper_bound(K=4, p=5, rho=x)
y2=kl_upper_bound(K=4, p=10, rho=x)
plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='red', label=r"$K=2, p=5$")
plt.plot(x, y2, color='blue', label=r"$K=2, p=10$")
plt.xlabel('Correlation '+r'$\ \rho$',fontsize=18)
plt.ylabel(r'$U_{K,p}(\rho)$',fontsize=22)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title('Plot of KL upper bound as function of correlation '+r'$\rho$',fontsize=20)
plt.legend(fontsize=22)
plt.savefig('upper_bound_rho.png')



dim = np.linspace(4, 20, 17)
print("dim",dim)
y1=kl_upper_bound(K=4, p=dim, rho=0.2)
y2=kl_upper_bound(K=4, p=dim, rho=0.5)
plt.figure(figsize=(10, 6))
plt.plot(dim, y1, color='red', label=r"$K=2, \rho=0.2$")
plt.plot(dim, y2, color='blue', label=r"$K=2, \rho=0.5$")
plt.xlabel('Dimension '+r'$\ p$',fontsize=18)
plt.ylabel(r'$U_{K,\rho}(p)$',fontsize=22)
plt.xticks(size=18)
plt.yticks(size=18)
plt.title('Plot of KL upper bound as function of dimension '+r'$p$',fontsize=20)
plt.legend(fontsize=22)
plt.savefig('upper_bound_p.png')


#######Loss in coverage as functoin of \rho.#######


def fn(tt,xx):
    normal_d = Normal(0, 1)  
    num = (1-normal_d.cdf(torch.sgn(xx)*tt))
    den = (1+tt**2)* (1-normal_d.cdf(torch.sgn(xx)*tt)) -torch.sgn(xx)*tt* normal_d.log_prob(tt).exp()
    den+= -(tt*(1-normal_d.cdf(torch.sgn(xx)*tt)) - torch.sgn(xx)*(normal_d.log_prob(tt).exp()))**2
    return num, den

rho=np.linspace(0.0,0.99,100)
alpha=0.05
alpha_pi_mfvi = 2*(1-ss.norm.cdf(ss.norm.ppf(1-alpha/2)*np.sqrt(1-rho**2))) 
tt = torch.Tensor([-4])
num, den =fn(torch.Tensor(tt),torch.Tensor(rho))
num=num.numpy()
den=den.numpy()
C_rho_tt = ((1 - rho**2)*num)/(num - (rho**2)*(den**2))
alpha_pi_favi1 = 2*(1 - ss.norm.cdf(ss.norm.ppf(1-alpha/2)*np.sqrt(C_rho_tt)))

#Plot
fig = plt.figure()


plt.plot(rho, alpha_pi_mfvi -alpha  ,label=r'MF-VI',color='blue')
plt.plot(rho, alpha_pi_favi1 -alpha,label=r'FAVI',color='green')
plt.ylabel(r'$\Delta^{(1)}_{\alpha}$',fontsize=18)
plt.xlabel(r'$\rho$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('loss_in_coverage.pdf')
plt.clf()