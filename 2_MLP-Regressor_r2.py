# Regression : Standardized and Wider
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


# load dataset
dataframe = pandas.read_csv("new2.csv", header=None)
#dataframe = pandas.read_csv("new3_gan_287.csv", header=None)
dataset=dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:3]
Y = dataset[:,3]


#define baseline model
def baseline_model1():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=3 , activation= 'relu' ))    
    model.add(Dense(1 ))
    # Compile model
    model.compile(loss= 'mean_squared_error' , optimizer= 'adam' )
    return model
def baseline_model2():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=3 , activation= 'relu' ))    
    model.add(Dense(1 ))
    # Compile model
    model.compile(loss= 'mean_absolute_error' , optimizer= 'adam' , metrics=['r2'] )
    return model    
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#evaluate model with standardized dataset
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=baseline_model1,epochs=10,batch_size=5,verbose=1)))
pipeline1=Pipeline(estimators)
kfold=KFold(n_splits=5,shuffle=True , random_state=seed)
results1=cross_val_score(pipeline1 , X,Y , cv=kfold)


estimators1=[]
estimators1.append(('standardize',StandardScaler()))
estimators1.append(('mlp',KerasRegressor(build_fn=baseline_model2,epochs=10,batch_size=5,verbose=1)))
pipeline2=Pipeline(estimators1)
kfold=KFold(n_splits=5,shuffle=True , random_state=seed)
results2=cross_val_score(pipeline2 , X,Y , cv=kfold)
print(results2)
input()
print("R2: %.2f " % (1-(results2.mean()/results1.mean())))


'''
output
 split 10   : standardized: -1225.23 (334.40) MSE
split 7 :     standardized: -1242.30 (318.64) MSE
split 5 :     standardized: -1461.13 (345.82) MSE


'''
'''
 split 10   : standardized: -0.32 (0.13) MSE
split 7 :     standardized: -0.35 (0.12) MSE
split 5 :   standardized: -0.54 (0.20) MSE


'''
