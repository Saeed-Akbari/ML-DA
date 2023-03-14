#%%

import random
random.seed(10)

import numpy as np
np.random.seed(10)

import tensorflow as tf
# tf.random.set_seed(0)

from numpy import linalg as LA

import time as tm

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

font = {'family' : 'Times New Roman',
        'size'   : 16}    
plt.rc('font', **font)

#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from tqdm import tqdm as tqdm
import yaml
import os


from preprocess import loadData, loadMesh, lstmTest, transform, scale, probeT
from visualization import subplotModeDA, subplotProbeDA

#%%
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#%%
# inputs

loc2 = 'input/'
with open(loc2+'inputPOD.yaml') as file:    
    inputData = yaml.load(file, Loader=yaml.FullLoader)
file.close()

mode = inputData['mode']
var = inputData['var']

trainStartTime = inputData['trainStartTime']
trainEndTime = inputData['trainEndTime']
testStartTime = inputData['testStartTime']
testEndTime = inputData['testEndTime']
figTimeTest = np.array(inputData['figTimeTest'])

lstm_seed = inputData['POD']['lstm_seed']
epochsLSTM = inputData['POD']['epochsLSTM']
batchSizeLSTM = inputData['POD']['batchSizeLSTM']
LrateLSTM = float(inputData['POD']['LrateLSTM'])
validationLSTM = inputData['POD']['validationLSTM']
windowSize = inputData['POD']['windowSize']
timeScale = inputData['POD']['timeScale']
numLstmNeu = inputData['POD']['numLstmNeu']
PODmode = inputData['POD']['PODmode']

px = inputData['POD']['px']
py = inputData['POD']['py']
lx = inputData['POD']['lx']
ly = inputData['POD']['ly']


#%%
# loading data

# loading data obtained from full-order simulation
#The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
loc = '../FOM/'
#flattenedPsi, flattenedTh, time, dt, mesh = loadData(loc, var)
flattened, dt, ra, timeStep, mesh = loadData(loc, var)

Xmesh, Ymesh = loadMesh(loc)
mytime = np.arange(trainStartTime, testEndTime, timeStep)

# retrieving data with its original shape
# (number of snapshots, first dimension of the mesh, second dimension)
data = flattened.reshape(flattened.shape[0], (mesh[0]+1), (mesh[1]+1))

trainStartTime = np.argwhere(mytime>trainStartTime)[0, 0] - 1
trainEndTime = np.argwhere(mytime<trainEndTime)[-1, 0] + 1
testStartTime = np.argwhere(mytime>testStartTime)[0, 0] - 1
testEndTime = np.argwhere(mytime<testEndTime)[-1, 0] + 1

flattened = flattened[trainStartTime:testEndTime]

flattened = flattened.T


#%%
nr = PODmode # number of POD modes
sp = 1    # percent of mesh used to locate sensors
me = int(mesh[0]*mesh[1]*0.01*sp) # number of observations
nes = flattened.shape[0] # number of grid points corresponding to Sea (Ns = 1914)
ns_train_rom = 1501 # number of samples to be used for LSTM training 
lookback = windowSize # lookback for LSTM
epochs = epochsLSTM # number of iterations for training
batch_size = batchSizeLSTM # batch size for training
test_size = validationLSTM # validation split
npe = 20 # number of ensembles
mu = 0.0 # mean of the noise to be added in initial condition uncertainty and observations
sd2_ic = 1.0*0.2 # variance for initial condition uncertainty
sd1_ic = np.sqrt(sd2_ic) # standard deviation for initial condition uncertainty
sd2_obs = 1.0*0.02 # variance for measurements noise
sd1_obs = np.sqrt(sd2_obs) # variance for measurements noise
sd2_obs_modes = 0.01 # variance for measurements noise to be added to POD modes
sd1_obs_modes = np.sqrt(sd2_obs_modes) # standard deviation for measurements noise to be added to POD modes
lambda_ = 1.4 #inflation factor
fic = 2 #type of initialization of ensembles
fob = 1 # reconstructed data from sparse sensors
nf = 3 # frequency of observation

nt = flattened.shape[1]
t = np.linspace(1, nt, nt)

ns_train = 1501 # start forecast from N = 1000
ns_test = nt - ns_train
nb = int((ns_test)/nf)# - 1 # number of observation time
#t_test = np.linspace(1, ns_test, ns_test)
t_test = mytime[ns_train:]

#%%

#dirResultSVD = f'result/svd/Ra{ra}'
#fileSVD = dirResultSVD+'/'+var+'.npz'
#fileSVDTemporal = dirResultSVD+'/'+var+'Temporal.npz'

#dataSVD = np.load(fileSVD)
#dataSVDTemporal = np.load(fileSVDTemporal)

#flatMeanTrain = dataSVD['mean']
#PhidTrain = dataSVD['bases']

#alphaTrain = dataSVDTemporal['coeffTrain']
#alphaTrain = alphaTrain.T
#lstmScaler = MinMaxScaler(feature_range=(-1,1))
#alphaTrain = alphaTrain / ((mesh[0]+1) * (mesh[1]))
#scaledTrain = scale(alphaTrain, lstmScaler)

#alphaTest = dataSVDTemporal['coeffTest']
#alphaTest = alphaTest.T
#alphaTestSC = alphaTest / ((mesh[0]+1) * (mesh[1]))
#scaledTest = transform(alphaTestSC, lstmScaler)

##scaledTest = transform(alphaTestSC, sc)

#print("alphaTrain shape = ", alphaTrain.shape)

#flatMeanTrain = np.mean(flattened[:,:ns_train],axis=1)

#myflucTrain = flattened[:,:ns_train] - np.expand_dims(flatMeanTrain, axis=1)

#errTrainData = np.linalg.norm(myflucTrain - sst_fluc_train)/np.sqrt(np.size(sst_fluc_train))
#print('errTrainData = ', errTrainData)


#%%

# compute the mean
sst_avg = np.mean(flattened[:,:ns_train_rom], axis=1, keepdims=True)
# fluctuations
sst_fluc_train = flattened[:,:ns_train_rom] - sst_avg
sst_fluc = flattened - sst_avg

# compute the POD basis functions
PHIw, L, RIC  = POD(sst_fluc_train, nr)

#%%
#L_per = np.cumsum(L, axis=0)*100/np.sum(L)
# k = np.linspace(1,ns,ns)
# fig, axs = plt.subplots(1, 1, figsize=(7,5))#, constrained_layout=True)
# axs.loglog(k,L_per, lw = 2, marker="o", linestyle='-', label=r'$y_'+str(1)+'$'+' (True)', zorder=5)
# axs.set_xlim([1,ns])
# axs.axvspan(0, nr, alpha=0.2, color='red')
# fig.tight_layout()
# plt.show()

# compute the POD modes by projecting the data on the basis functions
at = PODproj(sst_fluc_train, PHIw)
at_signs = np.sign(at[0,:]).reshape([1,-1])
at = at/at_signs
PHIw = PHIw/at_signs

# sample observation locations
oin = sorted(random.sample(range(nes), me))

sst_obs = np.zeros_like(sst_fluc)
sst_obs_noise = np.zeros_like(sst_fluc)
obs_noise = np.zeros_like(sst_fluc)

obs_noise_sampled = np.random.normal(mu,sd1_obs,[me,nt])
sst_obs[oin,:] = sst_fluc[oin,:] 
# add noise to observations to mimic measurement errors
sst_obs_noise[oin,:] = sst_fluc[oin,:] + obs_noise_sampled

obs_noise[oin,:] = obs_noise_sampled

atrue = PODproj(sst_fluc, PHIw)
at_obs = PODproj(sst_obs, PHIw)
at_obs_noise = PODproj(sst_obs_noise, PHIw)

# sst_obs = sst_fluc[oin,:]

# PHIw_obs, L_obs, RIC_obs = POD(sst_obs, nr)     

# at_obs = PODproj(sst_obs, PHIw_obs)
# at_obs_signs = np.sign(at_obs[0,:]).reshape([1,-1])
# at_obs = at_obs/at_obs_signs
# PHIw_obs = PHIw_obs/at_obs_signs

# we need to scale the modal coefficients based on the number of points used 
# in thr projection
at = at/nes
atrue = atrue/nes
at_obs = at_obs/me
at_obs_noise = at_obs_noise/me

#%%
# check if the true modal coefficeints and the one obtained from observations
# are matching well
#fig, ax = plt.subplots(nrows=nr,ncols=1,figsize=(10,8),sharex=True)
#ax = ax.flat
#ns_plot = 1500
#for i in range(nr):
#    # ax[i].plot(at[:,i], lw=3)
#    ax[i].plot(atrue[:ns_plot,i], 'k', lw=2)
#    ax[i].plot(at_obs[:ns_plot,i],'r--')
#    ax[i].plot(at_obs_noise[:ns_plot,i],'go',fillstyle='none',ms=4)
#    # ax[i].axvspan(0, ns, color='gray', alpha=0.3)    

#fig.tight_layout()
#plt.show()
    
#%%
# scaling parameters 
sc = MinMaxScaler(feature_range=(-1,1))
sc.fit(at)
training_set = sc.transform(at)

#atrain_max = np.max(at, axis=0, keepdims=True)
#atrain_min = np.min(at, axis=0, keepdims=True)

#%%

# load the trained model
print('#-------------- Loading the trained model --------------#')
dirModel = f'model/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
model = tf.keras.models.load_model(dirModel+'/PODlstmModel.h5')

#%%
xtest = np.zeros((npe,1,lookback,nr))
ue = np.zeros((nr,npe,ns_test)) # evolution for all ensembles
ua = np.zeros((nr,ns_test)) # analysis state
uu = np.zeros((nr,ns_test)) # uncertainty in ensembles

me_modes = nr # umber of modes to be used as observations for data assimilation
freq = int(nr/me_modes) # if the POD modes observations are sparse in space
oin = [freq*i-1 for i in range(1,me_modes+1)]
roin = np.linspace(0, me_modes-1, me_modes, dtype=int)

# observation operator
dh = np.zeros((me_modes,nr))
dh[roin,oin] = 1.0

# virtual observations for twin experiment
oib = [nf*k for k in range(nb+1)]

at_obs_sc = sc.transform(at_obs)
at_obs_noise_sc = sc.transform(at_obs_noise)
atrue_sc = sc.transform(atrue)

# with this option, we add noise to measurements and the observation modes are
# obtained by projected noisy measurement os the POD basis functions
if fob == 1:    
    uobsfull = at_obs_noise_sc[ns_train:,:].T

# with this option, we add noise to POD modes obtained by projecting non-noisy
# observations of the POD basis functions
elif fob == 2:
    uobsfull = at_obs_sc + np.random.normal(mu,sd1_obs_modes,[nt,nr])
    uobsfull = uobsfull.T

# temporal sparse observations
z = np.zeros((me_modes,nb+1))
z[:,:] = uobsfull[oin,:][:,oib]

#%%
#fig, ax = plt.subplots(nrows=nr,ncols=1,figsize=(10,8),sharex=True)
#ax = ax.flat
#ns_plot = 500
#ns_plot_obs = int(ns_plot/nf)
#for i in range(nr):
#    ax[i].plot(t_test[:ns_plot], atrue_sc[ns_train:ns_train+ns_plot,i],'k-',label='True')
#    # ax[i].plot(t_test[:ns_plot], uobsfull.T[:ns_plot,i],'go',fillstyle='none',ms=8,label='Noisy observatios')
#    ax[i].plot(t_test[oib][:ns_plot_obs], z.T[:ns_plot_obs,i],'bo',fillstyle='none',ms=6,label='Temporally sparse')

## ax[0].legend()
#fig.tight_layout()
#plt.show()

#%%
# equation 25 in PRF paper
ic_snapshots = np.zeros(npe, dtype=int)

testing_set = np.copy(atrue[ns_train:,:])
testing_set = sc.transform(testing_set)

# we will not be using this option
if fic == 1:
    for ne in range(npe):
        print(ne, ic_snapshots[ne])
        nsnap = ic_snapshots[ne]
        ue[:,ne,:lookback] = testing_set[nsnap:nsnap+lookback,:] + np.random.normal(mu,sd1_ic,[lookback,nr])
        xtest[ne,0,:,:] = ue[:,ne,:lookback]

# with this option, we add noise to the initial condition and then project 
# this noisy initial condition on the POD basis functions        
elif fic == 2:
    for ne in range(npe):
        for k in range(lookback):    
            print(ne, k)
            us = sst_fluc[:,ns_train+k] + np.random.normal(mu,sd1_ic,[nes])
            us = np.reshape(us,[-1,1])
            ak = PODproj(us, PHIw)
            ue[:,ne,k] = sc.transform(ak/nes)
            xtest[ne,0,k,:] = ue[:,ne,k]            

# compute the analysis and uncertainty    
ua[:,:lookback] = np.average(ue[:,:,:lookback], axis=1)
uu[:,:lookback] = np.std(ue[:,:,:lookback], axis=1)

#%%
# plot initial conditions for checking
#fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(10,8),sharex=True)
#ax = ax.flat
#ns_plot = lookback
#for i in range(nr):
#    ax[i].plot(t_test[:ns_plot], testing_set[:ns_plot,i], 'k', lw=2, label='True')
#    ax[i].plot(t_test[:ns_plot], ua.T[:ns_plot,i],'r--', lw=2, label='Pred')
#    yp = ua.T[:ns_plot,i] + uu.T[:ns_plot,i]
#    yn = ua.T[:ns_plot,i] - uu.T[:ns_plot,i]
#    ax[i].fill_between(t_test[:ns_plot], yp,yn,'k', alpha=0.5, label='Pred')
#    # ax[i].axvspan(0, ns, color='gray', alpha=0.3)    

#ax[0].legend()
#fig.tight_layout()
#plt.show()

#%%
print('#-------------- Starting data assimilation --------------#')
kobs = 2
stKobs = kobs

for k in range(lookback,ns_test):
    # evaluate all ensembles, equation 26 in PRF paper 
    for ne in range(npe):
        ue[:,ne,k] = model.predict(xtest[ne],verbose=0)
        xtest[ne,0,:-1,:] = xtest[ne,0,1:,:]
        xtest[ne,0,-1,:] = ue[:,ne,k]
    
    # compute the analysis and uncertainty
    ua[:,k] = np.average(ue[:,:,k], axis=1)    
    uu[:,k] = np.std(ue[:,:,k], axis=1)
    
    # do the data assimilation when you get observationws
    if k % nf == 0:
        print("k, kobs = ", k, kobs)
        # compute mean of the forecast fields, equation 27 in PRF paper 
        uf = np.average(ue[:,:,k], axis=1)
        
        # compute Af data, equation 35 in PRF paper 
        Af = ue[:,:,k] - uf.reshape(-1,1)
        
        # da = HA
        da = dh @ Af
        
        # cc = (HA)(HA)^T
        cc = da @ da.T/(npe-1)  
        
        diag = np.arange(nr)
        cc[diag,diag] = cc[diag,diag] + sd2_obs_modes
        
        # the below is the term inside the square bracket in equation 37 of PRF paper
        ci = np.linalg.pinv(cc)
        
        # equation 37 in PRF paper
        km = Af @ da.T @ ci/(npe-1)
        
        # analysis update, equation 31 in the PRF paper    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        # equation 39 in PRF paper
        ue[:,:,k] = Af - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        # compute the analysis and uncertainty    
        ua[:,k] = np.average(ue[:,:,k], axis=1)    
        uu[:,k] = np.std(ue[:,:,k], axis=1)
    
        # multiplicative inflation: set lambda=1.0 for no inflation, equation 40 in PRF paper
        ue[:,:,k] = ua[:,k].reshape(-1,1) + lambda_*(ue[:,:,k] - ua[:,k].reshape(-1,1))
        
        # used the analyzed state for future prediction
        for ne in range(npe):
            xtest[ne,0,-1,:] = ue[:,ne,k]
            
        kobs += 1
  
#%%
# prediction without data assimilation
print('#-------------- Prediction without data assimilation --------------#')
''''''
xtest = np.zeros((1,lookback,nr))
ypred = np.zeros((ns_test,nr))

# create input at t = 0 for the model testing
for i in range(lookback):
    xtest[0,i,:] = testing_set[i]
    ypred[i] = testing_set[i]

# predict results recursively using the model
for i in range(lookback,ns_test):
    ypred[i] = model.predict([xtest],verbose=0)
    xtest[0,:-1,:] = xtest[0,1:,:]
    xtest[0,lookback-1,:] = ypred[i]

#%%
''''''
uobsfull = uobsfull.T
ua = ua.T
uu = uu.T

# mean plus and minus standard deviation
yp = ua + 1*uu
yn = ua - 1*uu

#%%

# Creating a directory for plots.
dirPlot1 = f'plot/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
if not os.path.exists(dirPlot1):
    os.makedirs(dirPlot1)

dirPlot = dirPlot1 + f'/randomDA/sp{sp}_var{sd2_obs}_npe{npe}_nf{nf}_stKobs{stKobs}'
if not os.path.exists(dirPlot):
    os.makedirs(dirPlot)

filename = dirPlot + f'/PODmodesScDA{sp}_{sd2_obs}_{npe}.pdf'
tpLabel='TP'
mlLabel='ML'
daLabel='ML-DA'

subplotModeDA(t_test[:ns_test], testing_set[:ns_test], ypred[:ns_test], ua[:ns_test],\
                tpLabel, mlLabel, daLabel, filename, px, py)


#%%
atest = atrue[ns_train:,:]
apred = sc.inverse_transform(ypred)
apred_da = sc.inverse_transform(ua)

yp_usc = sc.inverse_transform(yp)
yn_usc = sc.inverse_transform(yn)

atest = atest*nes
apred = apred*nes
apred_da = apred_da*nes
yp_usc = yp_usc*nes
yn_usc = yn_usc*nes


filename = dirPlot + f'/PODmodesDA{sp}_{sd2_obs}_{npe}.pdf'
tpLabel='TP'
mlLabel='ML'
daLabel='ML-DA'


subplotModeDA(t_test[:ns_test], atest[:ns_test], apred[:ns_test], apred_da[:ns_test],\
                tpLabel, mlLabel, daLabel, filename, px, py)


#%%

# Creating a directory for result data.
dirResult1 = f'result/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
if not os.path.exists(dirResult1):
    os.makedirs(dirResult1)

dirResult = dirResult1 + f'/randomDA/sp{sp}_var{sd2_obs}_npe{npe}_nf{nf}_stKobs{stKobs}'
if not os.path.exists(dirResult):
    os.makedirs(dirResult)

filename = dirResult + f'/randomDA.npz'
np.savez(filename, t = t, atrue = atrue, at=at, pivot = np.array(oin), 
         PHIw = PHIw, sst_avg = sst_avg,
         apred = apred, apred_da = apred_da, 
         yp = yp_usc, yn = yn_usc, t_test = t_test,
         z = z, oib = np.array(oib))

#%%
ufom = flattened[:,ns_train:]

utrue = PODrec(atest,PHIw)
utrue = utrue + sst_avg

upred = PODrec(apred,PHIw)
upred = upred + sst_avg

upred_da = PODrec(apred_da,PHIw)
upred_da = upred_da + sst_avg        

rmse_fom_true = np.linalg.norm(ufom - utrue, axis=0)/np.sqrt(nes)
rmse_fom_ml = np.linalg.norm(ufom - upred, axis=0)/np.sqrt(nes)
rmse_fom_ml_denkf = np.linalg.norm(ufom - upred_da, axis=0)/np.sqrt(nes)

rmse_true_ml = np.linalg.norm(utrue - upred, axis=0)/np.sqrt(nes)
rmse_true_ml_denkf = np.linalg.norm(utrue - upred_da, axis=0)/np.sqrt(nes)

#%%


fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,5),sharex=True)
ax = ax.flat
FOMLabel = "FOM"
tpLabel = "TP"
mlLabel = "ML"
daLabel = "ML-DA"
filename = dirPlot + f'/randomProb{sp}_{sd2_obs}_{npe}.pdf'

ufom2d = ufom[:,:ns_test].reshape((mesh[0]+1),(mesh[1]+1), ns_test)
utrue2d = utrue[:,:ns_test].reshape((mesh[0]+1),(mesh[1]+1), ns_test)
upred_da2d = upred_da[:,:ns_test].reshape((mesh[0]+1),(mesh[1]+1), ns_test)
upred2d = upred[:,:ns_test].reshape((mesh[0]+1),(mesh[1]+1), ns_test)

FOMProbe = probeT(ufom2d, lx, ly, mesh[0], mesh[1])
TPprobe = probeT(utrue2d, lx, ly, mesh[0], mesh[1])
DAProbe = probeT(upred_da2d, lx, ly, mesh[0], mesh[1])
MLProbe = probeT(upred2d, lx, ly, mesh[0], mesh[1])


subplotProbeDA(t_test, FOMProbe, TPprobe, MLProbe, DAProbe,\
                FOMLabel, tpLabel, mlLabel, daLabel, filename, var)


#%%

filename = dirPlot + f'/error{sp}_{sd2_obs}_{npe}.pdf'
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,5),sharex=True)

ax[0].plot(t_test, rmse_fom_true, label='FOM-TP')
ax[0].plot(t_test, rmse_fom_ml, label='FOM-ML')
ax[0].plot(t_test, rmse_fom_ml_denkf, label='FOM-ML-DA')

ax[1].plot(t_test, rmse_true_ml, label='TP-ML')
ax[1].plot(t_test, rmse_true_ml_denkf, label='TP-ML-DA')

for i in range(2):    
    ax[i].legend()
    ax[i].set_xlabel('Time (s)',fontsize=22)
    ax[i].set_ylabel('$||\epsilon||$',fontsize=22)
    ax[i].tick_params(axis='both', which='major', labelsize=20, labelbottom=True)
    
fig.tight_layout()
plt.savefig(filename, dpi = 500, bbox_inches = 'tight')

#%%

import seaborn as sns

# matplotlib histogram
# plt.hist(rmse_true_ml, color = 'blue', edgecolor = 'black',
#          bins = int(180/5))

# seaborn histogram
# sns.distplot(rmse_true_ml, hist=True, kde=False, 
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})

filename = dirPlot + f'/distribution{sp}_{sd2_obs}_{npe}.pdf'
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4),sharex=True)

sns.distplot(rmse_true_ml, hist=False, kde=True, 
              bins=int(180/2), color = 'darkblue', 
              hist_kws={}, #'edgecolor':'black'
              kde_kws={'shade':True,'linewidth': 2},
              norm_hist=False,
              label='ML')

sns.distplot(rmse_fom_ml_denkf, hist=False, kde=True, 
             bins=int(180/2), color = 'red', 
             hist_kws={}, #'edgecolor':'red'
             kde_kws={'shade':True,'linewidth': 2},
             norm_hist=False,
             label='ML-DA')

ax.legend()
ax.set_xlabel('Forecast RMSE')
ax.set_ylabel('Probability Density Function')
ax.set_yscale('linear')
ax.set_xlim([0,0.1])

fig.tight_layout()
plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
