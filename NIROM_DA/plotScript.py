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

from visualization import contourSubPlot

import warnings
warnings.filterwarnings('ignore')

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#'weight' : 'bold'

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from tqdm import tqdm as tqdm
import yaml
import os


from preprocess import loadData, loadMesh, probeT
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
sp = 10    # percent of mesh used to locate sensors
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
sd2_obs = 1.0*0.2 # variance for measurements noise
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

stKobs = 2

dirResult1 = f'result/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
dirResult = dirResult1 + f'/randomDA/sp{sp}_var{sd2_obs}_npe{npe}_nf{nf}_stKobs{stKobs}'

dirPlot1 = f'plot/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
dirPlot = dirPlot1 + f'/randomDA/sp{sp}_var{sd2_obs}_npe{npe}_nf{nf}_stKobs{stKobs}'

'''
#%%

# compute the mean
sst_avg = np.mean(flattened[:,:ns_train_rom], axis=1, keepdims=True)
# fluctuations
sst_fluc_train = flattened[:,:ns_train_rom] - sst_avg
sst_fluc = flattened - sst_avg

# compute the POD basis functions
PHIw, L, RIC  = POD(sst_fluc_train, nr)


#%%

# compute the POD modes by projecting the data on the basis functions
at = PODproj(sst_fluc_train, PHIw)
at_signs = np.sign(at[0,:]).reshape([1,-1])
at = at/at_signs
PHIw = PHIw/at_signs

# we need to scale the modal coefficients based on the number of points used 
# in thr projection
at = at/nes

filename = dirResult + '/at'
np.save(filename, at)
'''
#%%


#filename = dirResult + '/at.npy'
#at = np.load(filename)

#%%
# scaling parameters 
#sc = MinMaxScaler(feature_range=(-1,1))
#sc.fit(at)

#%% load data

filename = dirResult + f'/randomDA.npz'
loadedData = np.load(filename)

t = loadedData['t']
atrue = loadedData['atrue']
PHIw = loadedData['PHIw']
sst_avg = loadedData['sst_avg']
apred = loadedData['apred']
apred_da = loadedData['apred_da']
yp_usc = loadedData['yp']
yn_usc = loadedData['yn']
z = loadedData['z']
oib = loadedData['oib']

# at = loadedData['at']
# sc = MinMaxScaler(feature_range=(-1,1))
# sc.fit(at)

#%%
atest = atrue[ns_train:,:]
atest = atest*nes

filename = dirPlot + f'/PODmodesDA{sp}_{npe}_{nf}_{stKobs}.pdf'
tpLabel='TP'
mlLabel='Pred-LSTM'
daLabel='Pred-DA'


subplotModeDA(t_test[:ns_test], atest[:ns_test], apred[:ns_test], apred_da[:ns_test],\
                tpLabel, mlLabel, daLabel, filename, px, py)


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
mlLabel = "Pred-LSTM"
daLabel = "Pred-DA"
filename = dirPlot + f'/randomProb{sp}_{npe}_{nf}_{stKobs}.pdf'

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


ind1 = 0
ind2 = 490

barRange = np.linspace(0.0, 1.0, 21, endpoint=True)
figTimeTest = [ind1, ind2]
filename = dirPlot + f'/contour{sp}_{npe}_{nf}_{stKobs}.pdf'
contourSubPlot(Xmesh, Ymesh, ufom2d[:,:,ind1], ufom2d[:,:,ind2], utrue2d[:,:,ind1], utrue2d[:,:,ind2],\
                upred2d[:,:,ind1], upred2d[:,:,ind2], upred_da2d[:,:,ind1], upred_da2d[:,:,ind2],\
                mytime, figTimeTest,\
                    1500, barRange, filename, figSize=(10,9))

#%%

filename = dirPlot + f'/error{sp}_{npe}_{nf}_{stKobs}.pdf'
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,5),sharex=True)

ax[0].plot(t_test, rmse_fom_true, label='FOM-True')
ax[0].plot(t_test, rmse_fom_ml, label='FOM-ML')
ax[0].plot(t_test, rmse_fom_ml_denkf, label='FOM-ML-DEnKF')
0
ax[1].plot(t_test, rmse_true_ml, label='True-ML')
ax[1].plot(t_test, rmse_true_ml_denkf, label='True-ML-DEnKF')

for i in range(2):    
    ax[i].legend()
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$||\epsilon||$')
    
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

filename = dirPlot + f'/distribution{sp}_{npe}_{nf}_{stKobs}.pdf'
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
             label='ML-DEnKF')

ax.legend()
ax.set_xlabel('Forecast RMSE')
ax.set_ylabel('Probability Density Function')
ax.set_yscale('linear')
ax.set_xlim([0,0.1])

fig.tight_layout()
plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
