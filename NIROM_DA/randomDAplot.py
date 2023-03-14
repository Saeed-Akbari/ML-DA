import numpy as np

import os
import yaml

from preprocess import loadData, loadMesh, probeT
from visualization import subplotModeDA, subplotProbeDA


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

stKobs = 2

# Creating a directory for result data.
dirResult1 = f'result/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
dirResult = dirResult1 + f'/randomDA/sp{sp}_npe{npe}_nf{nf}_stKobs{stKobs}'

filename = dirResult + f'/randomDA.npz'

dataDA = np.load(filename)

t = dataDA['t']
atrue = dataDA['atrue']
pivot = dataDA['pivot']
PHIw = dataDA['PHIw']
sst_avg = dataDA['sst_avg']
apred = dataDA['apred']
apred_da = dataDA['apred_da']
yp = dataDA['yp']
yn = dataDA['yn']
t_test = dataDA['t_test']
z = dataDA['z']
oib = dataDA['oib']



nt = t.shape[0]
ns_train = 1501 # start forecast from N = 1000
ns_test = nt - ns_train

#testing_set = np.copy(atrue[ns_train:,:])
#testing_set = sc.transform(testing_set)

# Creating a directory for plots.
dirPlot1 = f'plot/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
if not os.path.exists(dirPlot1):
    os.makedirs(dirPlot1)

dirPlot = dirPlot1 + f'/randomDA/sp{sp}_npe{npe}_nf{nf}_stKobs{stKobs}'
if not os.path.exists(dirPlot):
    os.makedirs(dirPlot)

filename = dirPlot + f'/PODmodesScDA.pdf'
tpLabel='TP'
mlLabel='Pred-LSTM'
daLabel='Pred-DA'

subplotModeDA(t_test[:ns_test], testing_set[:ns_test], ypred[:ns_test], ua[:ns_test],\
                tpLabel, mlLabel, daLabel, filename, px, py)