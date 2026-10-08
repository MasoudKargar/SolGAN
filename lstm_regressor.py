# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
# convert an array of values into a dataset matrix
'''
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])     
    return numpy.array(dataX), numpy.array(dataY)
''' 
  
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in dataset:
        dataX = dataset[:,0:3]
        dataY = dataset[:,3]
    return numpy.array(dataX), numpy.array(dataY)
   
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset

dataframe = pandas.read_csv('new2.csv',engine= 'python')#,delimiter='[;]',usecols=[1]
#dataframe = pandas.read_csv('new3_gan_287.csv',engine= 'python')#,delimiter='[;]',usecols=[1]

dataset = dataframe.values
dataset = dataset.astype( 'float32' )


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(dataset),len(train), len(test))
# reshape dataset
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0],1, testX.shape[1]))

#dataset shape
print ("00___",train[0])

print("0___",(trainX.shape))
print("1___",(trainX[0] ))
print("2___",(trainX[0].shape ))
print("3___",(trainX[1] ))
print("4___",(trainX[1].shape ))
print("5___",(type(trainX[0] )))




# create and fit the LSTM network
model1 = Sequential()
model1.add(LSTM(10, input_dim=3))
model1.add(Dense(1))

model2 = Sequential()
model2.add(LSTM(10, input_dim=3))
model2.add(Dense(1))

model1.compile(loss= 'mean_absolute_error' , optimizer= 'adam' )
model2.compile(loss= 'mean_squared_error' , optimizer= 'adam' )


model1.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
model2.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
# make predictions
trainPredict1 = model1.predict(trainX)
testPredict1 = model1.predict(testX)
print("8___",(trainY.shape ))
print("8___",(trainPredict1.shape ))
print("8___",(testY.shape ))
print("8___",(testPredict1.shape ))
trainPredict1=trainPredict1.reshape(-1)
testPredict1=testPredict1.reshape(-1)
print("8___",(trainY.shape ))
print("8___",(trainPredict1.shape ))
print("8___",(testY.shape ))
print("8___",(testPredict1.shape ))


trainPredict2 = model2.predict(trainX)
testPredict2 = model2.predict(testX)
print("8___",(trainY.shape ))
print("8___",(trainPredict2.shape ))
print("8___",(testY.shape ))
print("8___",(testPredict2.shape ))
trainPredict2=trainPredict2.reshape(-1)
testPredict2=testPredict2.reshape(-1)
print("8___",(trainY.shape ))
print("8___",(trainPredict2.shape ))
print("8___",(testY.shape ))
print("8___",(testPredict2.shape ))


# invert predictions

# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

#calculate root mean squared error
trainScoreMSE = mean_squared_error(trainY, trainPredict2)
print( 'Train ScoreMSE: %.5f MSE' % (trainScoreMSE))
trainScoreMAE = mean_absolute_error(trainY, trainPredict1)
print( 'Train ScoreMAE: %.5f MAE' % (trainScoreMAE))
r12=trainScoreMAE/trainScoreMSE
print(r12)
print( 'R2: %.5f ' % (1-r12))


'''
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''
'''
277 185 92
00___ [0.18777058 0.08301826 0.9        0.7629097 ]
0___ (185, 1, 3)
1___ [[0.18777058 0.08301826 0.9       ]]
2___ (1, 3)
3___ [[0.18777058 0.08301826 0.79999995]]
4___ (1, 3)


'''
'''
Train ScoreRMSE: 0.14207 RMSE
Train ScoreMSE: 0.02019 MSE
Test ScoreRMSE: 0.12485 RMSE
Test ScoreMSE: 0.01559 MSE

'''


#GAN
'''
Train ScoreRMSE: 0.00360 RMSE
Train ScoreMSE: 0.00001 MSE
Test ScoreRMSE: 0.00368 RMSE
Test ScoreMSE: 0.00001 MSE
'''