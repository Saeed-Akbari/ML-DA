""" @author: Saeed  """

import os
#from os import times
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from preprocess import loadData, loadMesh, splitData,\
                        scale, transform, inverseTransform,\
                        windowAdapDataSet, lstmTest, probe
from visualization import contourSubPlot, plot, subplotProbe, subplotMode

from LSTMmodel import createLSTMSeq
from activation import my_swish

def main():

    with open('input/inputPOD.yaml') as file:    
        input_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

    mode = input_data['mode']

    mynetwork = network(input_data)

    if mode == 'pod':
        mynetwork.podTrain()

    elif mode == 'podTest':
        mynetwork.podTest()

    else:
        exit()


class network():

    def __init__(self,input_data):

        self.input_data = input_data

        var = input_data['var']

        PODmode = input_data['POD']['PODmode']

        trainStartTime = input_data['trainStartTime']
        trainEndTime = input_data['trainEndTime']
        testStartTime = input_data['testStartTime']
        testEndTime = input_data['testEndTime']
        figTimeTest = np.array(input_data['figTimeTest'])

        # loading data obtained from full-order simulation
        #The flattened data has the shape of (number of snapshots, muliplication of two dimensions of the mesh, which here is 4096=64*64)
        loc = '../FOM/'
        #flattenedPsi, flattenedTh, time, dt, mesh = loadData(loc, var)
        flattened, dt, ra, timeStep, mesh = loadData(loc, var)
        self.Xmesh, self.Ymesh = loadMesh(loc)
        self.mytime = np.arange(trainStartTime, testEndTime+timeStep, timeStep)

        # Make a decition on which variable (temperature or stream funcion) must be trained
        if var == 'Psi':
            #flattened = np.copy(flattenedPsi)
            self.barRange = np.linspace(-0.60, 0.5, 30, endpoint=True)       # bar range for drawing contours
        elif var == 'theta':
            #flattened = np.copy(flattenedTh)
            self.barRange = np.linspace(0.0, 1.0, 15, endpoint=True)

        # retrieving data with its original shape (number of snapshots, first dimension of the mesh, second dimension)
        self.data = flattened.reshape(flattened.shape[0], (mesh[0]+1), (mesh[1]+1))
        #animationGif(Xmesh, Ymesh, data, fileName=var, figSize=(14,7))

        # Creating a directory for plots.
        self.dirPlot = f'plot/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
        if not os.path.exists(self.dirPlot):
            os.makedirs(self.dirPlot)
        # Creating a directory for models.
        self.dirModel = f'model/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
        if not os.path.exists(self.dirModel):
            os.makedirs(self.dirModel)
        # Creating a directory for result data.
        self.dirResult = f'result/{var}_POD_{PODmode}_{mesh[0]+1}_{mesh[1]+1}_{dt}_{ra}'
        if not os.path.exists(self.dirResult):
            os.makedirs(self.dirResult)

        # Extraction of indices for seleted times.
        self.trainStartTime = np.argwhere(self.mytime>trainStartTime)[0, 0] - 1
        self.trainEndTime = np.argwhere(self.mytime<trainEndTime)[-1, 0] + 1
        self.testStartTime = np.argwhere(self.mytime>testStartTime)[0, 0] - 1
        self.testEndTime = np.argwhere(self.mytime<testEndTime)[-1, 0] + 1
        
        # Length of the training set
        trainDataLen = self.trainEndTime - self.trainStartTime
        
        # obtaining indices to plot the results
        for i in range(figTimeTest.shape[0]):
            figTimeTest[i] = np.argwhere(self.mytime>figTimeTest[i])[0, 0]
        self.figTimeTest = figTimeTest - self.testStartTime

        dirResultSVD = f'result/svd/Ra{ra}'
        fileSVD = dirResultSVD+'/'+var+'.npz'
        fileSVDTemporal = dirResultSVD+'/'+var+'Temporal.npz'

        self.dataSVD = np.load(fileSVD)
        self.dataSVDTemporal = np.load(fileSVDTemporal)

        alphaTrain = self.dataSVDTemporal['coeffTrain']
        #tpt = self.dataSVDTemporal['tp']
        #Ld = self.dataSVDTemporal['Ld']
        #RICd = self.dataSVDTemporal['ric']

        # Scale the training data
        alphaTrain = alphaTrain.T

        self.at_signs = np.sign(alphaTrain[0,:]).reshape([1,-1])
        alphaTrain = alphaTrain/self.at_signs

        self.lstmScaler = MinMaxScaler(feature_range=(-1,1))
        alphaTrain = alphaTrain / ((mesh[0]+1) * (mesh[1]))
        self.scaledTrain = scale(alphaTrain, self.lstmScaler)       # fit and transform the training set

        self.mesh = mesh

    def podTrain(self):


        lstm_seed = self.input_data['POD']['lstm_seed']
        epochsLSTM = self.input_data['POD']['epochsLSTM']
        batchSizeLSTM = self.input_data['POD']['batchSizeLSTM']
        LrateLSTM = float(self.input_data['POD']['LrateLSTM'])
        validationLSTM = self.input_data['POD']['validationLSTM']
        windowSize = self.input_data['POD']['windowSize']
        timeScale = self.input_data['POD']['timeScale']
        numLstmNeu = self.input_data['POD']['numLstmNeu']
        PODmode = self.input_data['POD']['PODmode']

        # data shape is (number of example, features), but for POD calculations based on SVD it must be (features, number of examples)
        # after chaning the SVD to covariance matrix may not need shape change

        xtrainLSTM, ytrainLSTM = windowAdapDataSet(self.scaledTrain, windowSize, timeScale)

        #Shuffling data
        np.random.seed(lstm_seed)
        perm = np.random.permutation(ytrainLSTM.shape[0])
        xtrainLSTM = xtrainLSTM[perm,:,:]
        ytrainLSTM = ytrainLSTM[perm,:]

        # creating LSTM model
        lstmModel = createLSTMSeq(LrateLSTM, PODmode, numLstmNeu)
        
        # training the model
        history = lstmModel.fit(xtrainLSTM, ytrainLSTM, epochs=epochsLSTM, batch_size=batchSizeLSTM, validation_split=validationLSTM)

        # saving the trained LSTM model
        lstmModel.save(self.dirModel + f'/PODlstmModel.h5')

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        epochs = np.arange(len(loss)) + 1

        plt.figure().clear()
        figNum = 1
        trainLabel='Training MSE'
        validLabel='Validation MSE'
        plotTitle = 'Training and validation MSE'
        fileName = self.dirPlot + f'/PODlstmModelMSE.png'
        plot(figNum, epochs, loss, val_loss, trainLabel, validLabel, plotTitle, fileName)

        figNum = 2
        trainLabel='Training MAE'
        validLabel='Validation MAE'
        plotTitle = 'Training and validation MAE'
        fileName = self.dirPlot + f'/PODlstmModelMAE.png'
        plot(figNum, epochs, mae, val_mae, trainLabel, validLabel, plotTitle, fileName)
                
    
    def podTest(self):

        mesh = self.mesh

        var = self.input_data['var']
        px = self.input_data['POD']['px']
        py = self.input_data['POD']['py']
        lx = self.input_data['POD']['lx']
        ly = self.input_data['POD']['ly']
        windowSize = self.input_data['POD']['windowSize']
        timeScale = self.input_data['POD']['timeScale']

        flatMeanTrain = self.dataSVD['mean']

        alphaTest = self.dataSVDTemporal['coeffTest']
        alphaTest = alphaTest.T
        alphaTest = alphaTest/self.at_signs

        # Length of the test set
        testDataLen = self.testEndTime - self.testStartTime

        dataTest = splitData(self.data, self.testStartTime, self.testEndTime)
        
        # load trained LSTM model
        lstmModel = tf.keras.models.load_model(self.dirModel + f'/PODlstmModel.h5')

        # predicting future data using LSTM model for test set
        alphaTestSC = alphaTest / ((mesh[0]+1) * (mesh[1]))
        scaledTest = transform(alphaTestSC, self.lstmScaler)     # transform the test set
        ytest = lstmTest(scaledTest, lstmModel, windowSize, timeScale)

        err = np.linalg.norm(ytest - scaledTest)/np.sqrt(np.size(ytest))
        print('err = ', err)
        # inverse transform of lstm prediction before decoding
        pred = inverseTransform(ytest, self.lstmScaler)
        pred = pred * ((mesh[0]+1) * (mesh[1]))

        #Reconstruction
        PhidTrain = self.dataSVD['bases']
        PhidTrain = PhidTrain/self.at_signs

        pred = pred.T 
        dataRecons = np.dot(PhidTrain,pred)
        
        temp = np.expand_dims(flatMeanTrain, axis=1)

        dataRecons = (dataRecons + temp)
        reshapedData = dataRecons.T.reshape(testDataLen, (mesh[0]+1),
                                                        (mesh[1]+1))
        err = np.linalg.norm(reshapedData - dataTest)/np.sqrt(np.size(reshapedData))
        print('err = ', err)

        # plot the countor for POD-LSTM prediction for selected time
        contourSubPlot(self.Xmesh, self.Ymesh, dataTest[self.figTimeTest[0], :, :], reshapedData[self.figTimeTest[0], :, :],
                    dataTest[self.figTimeTest[1], :, :], reshapedData[self.figTimeTest[1], :, :], self.barRange,\
                    fileName= self.dirPlot + f'/PODLSTM.png', figSize=(14,7))
        
        # plotting the evolution of modes in time
        n=0
        fileName = self.dirPlot + f'/PODlstmOutput{n}.png'
        testLabel='LSTM'
        tpLabel='TP'
        aveTime = int (0.75 * (self.trainStartTime + self.trainEndTime))
        subplotMode(self.mytime[aveTime:self.trainEndTime], self.mytime[self.testStartTime:self.testEndTime],\
                    scaledTest, ytest, tpLabel, testLabel, fileName, px, py, n)

        n=1
        fileName = self.dirPlot + f'/PODlstmOutput{n}.png'
        subplotMode(self.mytime[aveTime:self.trainEndTime], self.mytime[self.testStartTime:self.testEndTime],\
                    scaledTest, ytest, tpLabel, testLabel, fileName, px, py, n)

        n=2
        fileName = self.dirPlot + f'/PODlstmOutput{n}.png'
        subplotMode(self.mytime[aveTime:self.trainEndTime], self.mytime[self.testStartTime:self.testEndTime],\
                    scaledTest, ytest, tpLabel, testLabel, fileName, px, py, n)


        tpData = np.dot(PhidTrain,alphaTest.T)
        tpData = (tpData + temp)
        tpData = tpData.T.reshape(testDataLen, (mesh[0]+1),(mesh[1]+1))

        TPprobe = probe(tpData, lx, ly, mesh[0], mesh[1])
        PODProbe = probe(reshapedData, lx, ly, mesh[0], mesh[1])
        FOMProbe = probe(dataTest, lx, ly, mesh[0], mesh[1])
        TPLabel='TP'
        PODLabel='POD'
        FOMLabel='FOM'
        fileName = self.dirPlot + f'/PODprobe.png'
        
        subplotProbe(self.mytime[self.testStartTime:self.testEndTime], self.mytime[aveTime:self.trainEndTime],\
                    TPprobe, PODProbe, FOMProbe,\
                    TPLabel, PODLabel, FOMLabel, fileName, len(lx), len(ly), var)

        # saving the AE-LSTM prediction
        filename = self.dirResult + f"/PODROM"
        np.save(filename, reshapedData)
        
    
if __name__ == "__main__":
    main()
