import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dx = pd.read_csv('C:/Users/Sam/Desktop/Deep_Learning_Class/HousePrices_Regression/train-v3.csv',header=None)
datasetx = dx.values

tX = datasetx[ 1:12968 , 2:23 ]
tY = datasetx[ 1:12968 , 1 ]

v = pd.read_csv('C:/Users/Sam/Desktop/Deep_Learning_Class/HousePrices_Regression/valid-v3.csv',header=None)
datasetv = v.values
vx = datasetv[ 1:2162 , 2:23 ]
vy = datasetv[ 1:2162 , 1 ]


vz = pd.read_csv('C:/Users/Sam/Desktop/Deep_Learning_Class/HousePrices_Regression/test-v3.csv',header=None)
datasetvz = vz.values
vz = datasetvz[ 1:6486 , 1:22 ]


tX=np.vstack((tX,tX,tX,tX,tX,tX,tX,tX))
tY=np.hstack((tY,tY,tY,tY,tY,tY,tY,tY))


from keras.models import Sequential
from keras.layers import Dense


def build_nn():
    model = Sequential()
    
    model.add(Dense(1024, input_dim=21,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(2048 , kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024 , kernel_initializer='normal', activation='relu'))
    model.add(Dense(512 ,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(256 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(128 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(64 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(32 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(16 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(8 ,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')    

    return model
     
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import seaborn as sns

estimators = []
estimators.append(('standardise', StandardScaler()))

estimators.append(('multiLayerPerceptron', KerasRegressor(build_fn=build_nn, nb_epoch=150, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

pipeline.fit(tX,tY)

print (pipeline.predict(vx))

score = r2_score(vy, pipeline.predict(vx))
print (score)
score = r2_score(tY, pipeline.predict(tX))
print (score)

predict = pipeline.predict(vx)
predict = predict.astype(int)
vy = vy.astype(int)

sample_df = pd.DataFrame(predict,vy).reset_index()
sample_df.columns = ['True Value', 'Prediction']
sns.regplot('True Value', 'Prediction', sample_df)

print (pipeline.predict(vz))
predict1 = pipeline.predict(vz)
predict1 = predict1.astype(int)
np.savetxt("predict1.csv",predict1,delimiter=",",header = "'id',price")
vzy=len(predict1)
idz = np.arange(vzy, dtype=np.int32).reshape( vzy,1 )

predict1=np.vstack(idz,predict1)

np.savetxt("predict1.csv",predict1,delimiter=",",header = "'id',price")




