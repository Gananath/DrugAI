#PLEASE REMEMBER TO CHANGE THE DATASET PATH AND FILE SAVING PATH(fpath)

from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
import numpy as np
from time import sleep
import h5py
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split



#seed for random number reproduction
np.random.seed(2016)


#ngrams range
#minimum range
mi=1
#maximum range
mx=1
#epoch
epoch=20
#batch size
bs=15
#number of cross validation
ncv=10


# read file into pandas from the working directory
data = pd.read_csv('/home/gananath/Desktop/stahl-dataset.csv')
data=data.reindex(np.random.permutation(data.index))
X=data.SMILES
#X=X.astype('S32')
data.shape

X.head()
Y=data.ix[:,1:7]
Y=Y.values
Y=Y.astype('int')
type(Y)

#empty variables
cvscore=[]
lscore=[]
auroc=[]

#custom made k fold cv
#kfold cross validation loop
for i in range(0,ncv):
	
    #spliting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3, random_state=i)
	
    # instantiate the vectorizer    
    vect = CountVectorizer(analyzer=u'char',lowercase=False,ngram_range=(mi,mx))
	
    # learn training data vocabulary, then use it to create a document-term matrix
    X_train_dtm = vect.fit_transform(X_train)

    # creating model
    model = Sequential()
    model.add(Dense(X_train_dtm.shape[1], input_dim=X_train_dtm.shape[1], init='normal', activation='relu'))
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150, init='normal', activation='relu'))
    model.add(Dense(y_train.shape[1],init='normal', activation='softmax'))

    #compiling model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])



    # transform testing data (using fitted vocabulary) into a document-term matrix
    X_test_dtm = vect.transform(X_test)
    X_test_dtm

    # Fit the model
    history=model.fit(X_train_dtm.toarray(), y_train,validation_data=(X_test_dtm.toarray(),y_test), shuffle=True, nb_epoch=epoch, batch_size=bs)

    # make class predictions for X_test_dtm
    #y_pred_class = model.predict(X_test_dtm.toarray())

    #np.argmax(y_pred_class)
    #vect.vocabulary_

    #ROC_AUC metrics calculation
    pred_prob = model.predict(X_test_dtm.toarray())
    preds = np.zeros_like(pred_prob)
    preds[np.arange(pred_prob.shape[0]), np.argmax(pred_prob, axis=1)] = 1
    
    iarc=roc_auc_score(y_test,preds)
	
    #accuracy,loss calculation
    score=np.mean(np.array(history.history['val_acc']))
    los=np.mean(np.array(history.history['val_loss']))
    cvscore=np.append(score,cvscore)
    lscore=np.append(los,lscore)
    auroc=np.append(auroc,iarc)
    
    print("kfold: " + str(i+1) + " WAIT..." )
    
    #Prevents "I/O opertion on closed file" value error by adding sleep function
    sleep(5)

#Saving models as h5 files
nam="Smiles_model" + str(mi) + "-" + str(mx) + ".h5"
fpath="/home/gananath/Desktop/" + str(nam)
model.save(fpath)

#Mean calculation
avg=np.mean(cvscore)
lavg=np.mean(lscore)
avg_auroc=np.mean(auroc)

print("\nCV Average Accuracy: "+ str(avg) +"\nCV Average Loss: " + str(lavg) + "\nAUROC: " + str(avg_auroc))

#summary of the model
print(model.summary())

