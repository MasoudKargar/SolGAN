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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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
    model.compile(loss= 'mean_squared_error' , optimizer= 'adam', metrics=['mae'])
    return model
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
model=baseline_model()
history = model.fit(X, Y, epochs=100)
pred = model.predict(X)
print(numpy.shape(Y))
print(numpy.shape(pred))
pred=pred.reshape(-1)
print(numpy.shape(pred))

print(Y[:10])
print(pred[:10])
print(r2_score(Y,pred))
print(mean_squared_error(pred,Y))
print(mean_absolute_error(pred,Y))

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))



original = c( -2,  1, -3, 2, 3, 5, 4, 6, 5, 6, 7)
predicted = c(-1, -1, -2, 2, 3, 4, 4, 5, 5, 7, 7)
x=1:length(original)
plot(x, original,pch=19, col="blue")
lines(x, predicted, col="red")
legend("topleft", legend = c("y-original", "y-predicted"),
       col = c("blue", "red"), pch = c(19,NA), lty = c(NA,1),  cex = 0.7)

d = original-predicted
mse = mean((d)^2)
mae = mean(abs(d))
rmse = sqrt(mse)
R2 = 1-(sum((d)^2)/sum((original-mean(original))^2))

cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
    "RMSE:", rmse, "\n", "R-squared:", R2)