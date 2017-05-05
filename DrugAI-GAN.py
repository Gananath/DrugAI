'''
Author: Gananath R
DrugAI-GAN: Experiments with GAN for drug like molecule generation
Contact: https://github.com/gananath
'''
#Some helpful links
#https://stats.stackexchange.com/questions/153285/derivative-of-softmax-and-squared-error
#https://github.com/soumith/ganhacks
#https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Flatten
from keras.callbacks import History 
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
import pandas as pd
import os.path

##seed for random number reproduction
np.random.seed(2017)
episodes=30000

if not os.path.exists(os.getcwd()+'/output/'):
    os.makedirs(os.getcwd()+'/output/')

def shuffle3D(arr):
    for a in arr:
        np.random.shuffle(a)
    
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
#prediction of sequence with argmax
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

def Gen():
	#Generator model
    G = Sequential()
    G.add(TimeDistributed(Dense(x_dash.shape[2]), input_shape=(x_dash.shape[1],x_dash.shape[2])))
    G.add(LSTM(216, return_sequences=True))
    G.add(Dropout(0.3))
    G.add(LSTM(216, return_sequences=True))
    G.add(Dropout(0.3))
    G.add(LSTM(216, return_sequences=True))
    #G.add(BatchNormalization(momentum=0.9))
    G.add(TimeDistributed(Dense(y_dash.shape[2], activation='softmax')))
    G.compile(loss='categorical_crossentropy', optimizer=Adam(lr=2e-4))
    return G

def Dis():
	#Discriminator model
    D = Sequential()
    D.add(TimeDistributed(Dense(y_dash.shape[2]), input_shape=(y_dash.shape[1],y_dash.shape[2])))
    D.add(LSTM(216, return_sequences=True))
    D.add(Dropout(0.3))
    D.add(LSTM(60, return_sequences=True))
    D.add(Flatten())
    D.add(Dense(1, activation='sigmoid'))
    D.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001))
    return D
    
def Gan():
    GAN=Sequential()
    GAN.add(G)
    D.trainable=False
    GAN.add(D)
    GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4))
    return GAN

def trainDis(data=None,mc=None):
    if data is None and mc is None:
        # Train on fake data        
        fake_data= G.predict(x_dash)
        targets = np.zeros(x_dash.shape[0]).astype(int)
        Dloss=D.fit(fake_data, targets,nb_epoch=1) 
             
    elif data is None and mc=="mc":
        #preventing mode collapse
        #artificial noice training 
        fake_ydata=np.copy(y_dash)
        shuffle3D(fake_ydata)
        targets = np.zeros(x_dash.shape[0]).astype(int)
        Dloss=D.fit(fake_ydata, targets,nb_epoch=1)    
            
    else:
        # Train on real data
        targets = np.ones(x_dash.shape[0]).astype(int)
        Dloss=D.fit(data,targets,nb_epoch=1)    
           
    #print Dloss.history.keys()
    return Dloss.history['loss'][0]

def trainGAN():  	
	#train Generator    
	target = np.ones(x_dash.shape[0]).astype(int)
	gan_loss = GAN.fit(x_dash, target,nb_epoch=1)
	
	return gan_loss.history['loss'][0]
    
##read csv file
data = pd.read_csv('stahl.csv')
data=data.reindex(np.random.permutation(data.index))
#data=data.head(30)  
Y=data.SMILES
Y.head()
X=data.ix[:,1:7]
X=X.values
X=X.astype('int')
type(X)

##padding smiles to same length by adding "|" at the end of smiles
maxY=Y.str.len().max()
y=Y.str.ljust(maxY, fillchar='|')

ts=y.str.len().max()

##CharToIndex and IndexToChar functions
chars = sorted(list( set("".join(y.values.flatten()))))
print('total chars:', len(chars))
char_idx= dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))


y_dash=dimY(y,ts)
x_dash=dimX(X,ts)   

#initializing models
G=Gen()
D=Dis()
GAN=Gan()


'''
if os.path.exists(os.getcwd()+"/output/Gen.h5")==True and os.path.exists(os.getcwd()+"/output/Dis.h5")==True and os.path.exists(os.getcwd()+"/output/Gan.h5")==True:
        #loading weights if exits
        G.load_weights(os.getcwd()+"/output/Gen.h5")
        D.load_weights(os.getcwd()+"/output/Dis.h5")
        GAN.load_weights(os.getcwd()+"/output/Gan.h5")
'''        

print("GAN input "+str(GAN.input_shape))
print("GAN output "+str(GAN.output_shape))
print("Gen input "+str(G.input_shape))
print("Gen output "+str(G.output_shape))
print("Dis input "+str(D.input_shape))
print("Dis output "+str(D.output_shape))
#print D.summary()
#print G.summary()

#enable training in discrimator
D.trainable=True

#pre training
for i in range(20):
    shuffleData = np.random.permutation(y_dash)
    trainDis()
    dloss=trainDis(shuffleData)
    print("Pre Training Discrimator "+str(dloss)+"\n")
 

for episode in range(episodes):
    print("Epoch "+str(episode)+"/"+str(episodes))
    trainDis()
    shuffleData = np.random.permutation(y_dash)
    disloss=trainDis(y_dash)
    disloss=trainDis(mc="mc")      
    ganloss=trainGAN()    
    
	
    print("D loss="+str(disloss)+" GAN loss="+str(ganloss))
    
    
   
    if episode%(episodes/100)==0:
        
        #G.save(os.getcwd()+'/output/Gen.h5')
        #D.save(os.getcwd()+'/output/Dis.h5')
        #GAN.save(os.getcwd()+'/output/Gan.h5')
        
        #for saving files in floydhub output directory              
        G.save("/output/Gen_mc.h5")
        D.save("/output/Dis_mc.h5")
        GAN.save("/output/Gan_mc.h5")
        
        
    if episode%(episodes/600)==0:
        print("Predicting Molecule")
        x_pred=[[0,0,0,1,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]]
        x_pred=dimX(x_pred,ts)   
        preds=G.predict(x_pred)
        y_pred=prediction(preds)
        y_pred=seq_txt(y_pred)
        s=smiles_output(y_pred)
        print(s)
        
	 



##For Prediction

'''
#start Prediction
Ghash=Gen()
Ghash.load_weights('Gen_mc.h5')

x_pred=[[0,0,0,1,0,0],
        [0,1,0,0,0,0],
        [0,0,0,0,0,1]]

x_pred=[[0.6,0,0,0,0,0],
        [.3,0,0,0,0,0],
        [0.7,0,0,0,0,0]]
	
x_pred=dimX(x_pred,ts)      
preds=Ghash.predict(x_pred)
y_pred=prediction(preds)
y_pred=seq_txt(y_pred)


s=smiles_output(y_pred)
print s
#end prediction

'''


