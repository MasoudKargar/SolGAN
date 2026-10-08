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
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=3 , activation= 'relu' ))    
    model.add(Dense(1 ))
    # Compile model
    model.compile(loss= 'mean_absolute_error' , optimizer= 'adam' )
    return model
    
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#evaluate model with standardized dataset
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=5,verbose=1)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=5,shuffle=True , random_state=seed)
results=cross_val_score(pipeline , X,Y , cv=kfold)

print("standardized: %.2f (%.2f) MAE" % (results.mean(), results.std()))


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
