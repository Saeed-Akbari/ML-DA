import os

import numpy as np
from scipy.fftpack import dst, idst
import yaml

from preprocess import loadData, loadMesh, splitData
from visualization import plotPODcontent


def poisson_fst(nx,ny,dx,dy,w):

    f = np.copy(-w[1:nx,1:ny])

    #DST: forward transform
    ff = np.zeros([nx-1,ny-1])
    ff = dst(f, axis = 1, type = 1)
    ff = dst(ff, axis = 0, type = 1) 
    
    m = np.linspace(1,nx-1,nx-1).reshape([-1,1])
    n = np.linspace(1,ny-1,ny-1).reshape([1,-1])
    
    alpha = (2.0/(dx*dx))*(np.cos(np.pi*m/nx) - 1.0) + (2.0/(dy*dy))*(np.cos(np.pi*n/ny) - 1.0)           
    u1 = ff/alpha
        
    #IDST: inverse transform
    u = idst(u1, axis = 1, type = 1)
    u = idst(u, axis = 0, type = 1)
    u = u/((2.0*nx)*(2.0*ny))

    ue = np.zeros([nx+1,ny+1])
    ue[1:nx,1:ny] = u
    
    return ue


############# temperature #############
def svdTheta():


    with open('input/input.yaml') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    var = "theta"
    mode = input_data['mode']

    podContent = input_data['POD-AE']['podContent']
    podModeContent = input_data['POD-AE']['podModeContent']
    AEmode = input_data['POD-AE']['AEmode']
    PODmode = input_data['POD-AE']['PODmode']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']

    loc = '../FOM/'
    print('loading start')
    flattened, dt, ra, timeStep, mesh = loadData(loc, var)
    print('loading finish')
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime, timeStep)
    
    #data = flattened.reshape(flattened.shape[0], (mesh[0]+1), (mesh[1]+1))

    # Creating a directory for plots.
    dirPlot = f'plot/{var}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)

    # Creating a directory for result data.
    dirResult1 = f'result/svd/Ra{ra}'
    if not os.path.exists(dirResult1):
        os.makedirs(dirResult1)
    dirResult2 = dirResult1+'/'+var
    dirResult3 = dirResult1+'/'+var+'Temporal'

    # Extraction of indices for seleted times.
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0] - 1
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0] + 1
    testStartTime = np.argwhere(time>testStartTime)[0, 0] - 1
    testEndTime = np.argwhere(time<testEndTime)[-1, 0] + 1

    # Length of the training set
    trainDataLen = trainEndTime - trainStartTime
    # Length of the test set
    testDataLen = testEndTime - testStartTime
    
    # data splitting
    #dataTest = splitData(data, testStartTime, testEndTime)
    flattened = flattened[trainStartTime:testEndTime]
    flattenedTrain = splitData(flattened, trainStartTime, trainEndTime).T
    flattenedTest = splitData(flattened, testStartTime, testEndTime).T
    
    # mean subtraction
    flatMeanTrain = np.mean(flattenedTrain,axis=1)
    flatMTrain = (flattenedTrain - np.expand_dims(flatMeanTrain, axis=1))
    flatMTest = (flattenedTest - np.expand_dims(flatMeanTrain, axis=1))

    print('SVD start')
    # singular value decomposition
    Ud, Sd, _ = np.linalg.svd(flatMTrain, full_matrices=False)
    print('SVD finish')
    # compute RIC (relative importance index)
    Ld = Sd**2
    RICd = np.cumsum(Ld)/np.sum(Ld)*100

    if podContent:
        plotPODcontent(RICd, AEmode, dirPlot, podModeContent)
        np.savetxt(dirPlot+'/content.txt', RICd, delimiter=',')

    #PODmode = np.min(np.argwhere(RICd>podModeContent))
    #PODmode = 6

    PhidTrain = Ud[:,:PODmode]
    alphaTrain = np.dot(PhidTrain.T,flatMTrain)
    alphaTest = np.dot(PhidTrain.T,flatMTest)

    newflatMTest = np.dot(PhidTrain,np.dot(PhidTrain.T,flatMTest))
    temp = np.expand_dims(flatMeanTrain, axis=1)
    TPflattend = (newflatMTest + temp)
    #TPdata = TPflattend.T.reshape(testDataLen, (mesh[0]+1),
    #                                                (mesh[1]+1))

    np.savez(dirResult2, mean=flatMeanTrain, bases=PhidTrain)
    np.savez(dirResult3, coeffTrain=alphaTrain, coeffTest=alphaTest,\
                tp=TPflattend, Ld=Ld, ric=RICd)
############# temperature #############

############# stream function vorticity#############
def svdPsiOmega():


    with open('input/input.yaml') as file:
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    var1 = "Omega"
    var2 = "Psi"
    mode = input_data['mode']

    podContent = input_data['POD-AE']['podContent']
    podModeContent = input_data['POD-AE']['podModeContent']
    AEmode = input_data['POD-AE']['AEmode']
    PODmode = input_data['POD-AE']['PODmode']
    
    trainStartTime = input_data['trainStartTime']
    trainEndTime = input_data['trainEndTime']
    testStartTime = input_data['testStartTime']
    testEndTime = input_data['testEndTime']

    loc = '../FOM/'
    print('second loading start')
    flattenedOmega, dt, ra, timeStep, mesh = loadData(loc, var1)
    print('second loading finish')
    print('third loading start')
    flattenedPsi, _, _, _, _ = loadData(loc, var2)
    print('third loading finish')
    Xmesh, Ymesh = loadMesh(loc)
    time = np.arange(trainStartTime, testEndTime, timeStep)

    #dataOmega = flattenedOmega.reshape(flattenedOmega.shape[0], (mesh[0]+1), (mesh[1]+1))

    # Creating a directory for plots.
    dirPlot = f'plot/{var1}_PODAE_{AEmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
    if not os.path.exists(dirPlot):
        os.makedirs(dirPlot)

    # Creating a directory for result data.
    dirResult1 = f'result/svd/Ra{ra}'
    if not os.path.exists(dirResult1):
        os.makedirs(dirResult1)
    dirResult2 = dirResult1+'/'+var1
    dirResult3 = dirResult1+'/'+var1+'Temporal'
    dirResult4 = dirResult1+'/'+var2

    # Extraction of indices for seleted times.
    trainStartTime = np.argwhere(time>trainStartTime)[0, 0] - 1
    trainEndTime = np.argwhere(time<trainEndTime)[-1, 0] + 1
    testStartTime = np.argwhere(time>testStartTime)[0, 0] - 1
    testEndTime = np.argwhere(time<testEndTime)[-1, 0] + 1

    # Length of the training set
    trainDataLen = trainEndTime - trainStartTime
    # Length of the test set
    testDataLen = testEndTime - testStartTime

    # data splitting
    #dataTestOmega = splitData(dataOmega, testStartTime, testEndTime)
    flattenedOmega = flattenedOmega[trainStartTime:testEndTime]
    flattenedTrainOmega = splitData(flattenedOmega, trainStartTime, trainEndTime).T
    flattenedTestOmega = splitData(flattenedOmega, testStartTime, testEndTime).T

    # mean subtraction
    flatMeanTrainOmega = np.mean(flattenedTrainOmega,axis=1)
    flatMTrainOmega = (flattenedTrainOmega - np.expand_dims(flatMeanTrainOmega, axis=1))
    flatMTestOmega = (flattenedTestOmega - np.expand_dims(flatMeanTrainOmega, axis=1))

    print('second SVD start')
    # singular value decomposition
    Ud, Sd, _ = np.linalg.svd(flatMTrainOmega, full_matrices=False)
    print('second SVD finish')
    # compute RIC (relative importance index)
    Ld = Sd**2
    RICd = np.cumsum(Ld)/np.sum(Ld)*100

    if podContent:
        plotPODcontent(RICd, AEmode, dirPlot, podModeContent)
        np.savetxt(dirPlot+'/content.txt', RICd, delimiter=',')

    #PODmode = np.min(np.argwhere(RICd>podModeContent))
    #PODmode = 6

    PhidTrainOmega = Ud[:,:PODmode]
    alphaTrain = np.dot(PhidTrainOmega.T,flatMTrainOmega)
    alphaTest = np.dot(PhidTrainOmega.T,flatMTestOmega)

    newflatMTestOmega = np.dot(PhidTrainOmega,np.dot(PhidTrainOmega.T,flatMTestOmega))
    temp = np.expand_dims(flatMeanTrainOmega, axis=1)
    TPflattend = (newflatMTestOmega + temp)
    #TPdata = TPflattend.T.reshape(testDataLen, (mesh[0]+1),
    #                                                (mesh[1]+1))

    nx, ny = mesh[0], mesh[1]
    dx = Xmesh[1,1] - Xmesh[0,1]
    dy = Ymesh[1,1] - Ymesh[1,0]
    
    tmp = flatMeanTrainOmega.reshape([nx+1,ny+1])
    tmp = poisson_fst(nx,ny,dx,dy,tmp)
    flatMeanTrainPsi = tmp.flatten()
    
    PhidTrainPsi = np.zeros([(nx+1)*(ny+1),PODmode])
    for k in range(PODmode):
        tmp = np.copy(PhidTrainOmega[:,k]).reshape([nx+1,ny+1])
        tmp = poisson_fst(nx,ny,dx,dy,tmp)
        PhidTrainPsi[:,k] = tmp.flatten()

    np.savez(dirResult2, mean=flatMeanTrainOmega, bases=PhidTrainOmega)

    np.savez(dirResult3, coeffTrain=alphaTrain, coeffTest=alphaTest,\
                tp=TPflattend, Ld=Ld, ric=RICd)

    np.savez(dirResult4, mean=flatMeanTrainPsi, bases=PhidTrainPsi)

############# stream function vorticity#############

if __name__ == "__main__":
    svdTheta()
    #svdPsiOmega()