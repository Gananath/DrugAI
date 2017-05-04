
'''
Author: Gananath R
DrugAI-gen: Drug like molecule generator
'''


import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.wrappers import TimeDistributed
import pandas as pd

##seed for random number reproduction
np.random.seed(2017)

##text to sequence converter

    
##time step addtition to feature 
def dimX(x,ts):
    x=np.asarray(x)
    newX=[]
    for i, c in enumerate(x):
        newX.append([])
        for j in range(ts):
            newX[i].append(c)
    return np.array(newX)

##time step addtition to target
def dimY(Y,ts):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            #print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)
#prediction of argmax
def prediction(preds):
    y_pred=[]
    for i,c in enumerate(preds):
        y_pred.append([])
        for j in c:
            y_pred[i].append(np.argmax(j))
    return np.array(y_pred)
##sequence to text conversion
def seq_txt(y_pred):
    newY=[]
    for i,c in enumerate(y_pred):
        newY.append([])
        for j in c:
            newY[i].append(idx_char[j])
    
    return np.array(newY)
    
##joined smiles output
def smiles_output(s):
    smiles=np.array([])
    for i in s:
        j=''.join(str(k) for k in i)
        smiles=np.append(smiles,j)
    return smiles

##read csv file
data = pd.read_csv('stahl.csv')
data=data.reindex(np.random.permutation(data.index))
Y=data.SMILES
Y.head()
X=data.ix[:,1:7]
X=X.values
X=X.astype('int')
type(X)

##padding smiles to same length by adding "|" at the end of smiles

maxY=Y.str.len().max()
#maxY=Y.str.len().max()+10  #for adding custom length

y=Y.str.ljust(maxY, fillchar='|')

ts=y.str.len().max()

##CharToIndex and IndexToChar functions
chars = sorted(list( set("".join(y.values.flatten()))))
print('total chars:', len(chars))
char_idx= dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))


y_dash=dimY(y,ts)

x_dash=dimX(X,ts)       

##LSTM model   
model = Sequential()
model.add(TimeDistributed(Dense(x_dash.shape[2]), input_shape=(x_dash.shape[1],x_dash.shape[2])))
model.add(LSTM(216, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(216, return_sequences=True))
model.add(TimeDistributed(Dense(y_dash.shape[2], activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print model.input_shape
print model.output_shape


model.load_weights("drugai.h5")

##For Training uncomment the training section
#start training
'''
model.fit(x_dash,y_dash, nb_epoch=20)
##Epoch 20/20
##335/335 [==============================] - 31s - loss: 1.3319

model.save('drugai.h5')

'''
#End training

##For Prediction


#start Prediction
x_pred=[[0,0,0,1,0,0],
        [0,1,0,0,0,0],
        [0,0,0,0,0,1]]
x_pred=dimX(x_pred,ts)      
preds=model.predict(x_pred)
y_pred=prediction(preds)
y_pred=seq_txt(y_pred)


s=smiles_output(y_pred)
print s
#end prediction

    
