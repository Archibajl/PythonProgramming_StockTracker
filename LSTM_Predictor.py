import tensorflow as tf
import csv
import numpy as np
#import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

def TrainData():
    with open('upload_DJIA_table.csv', newline='\n') as price_data_file:
        stock_price=list(csv.reader(price_data_file, delimiter=',', skipinitialspace=True))
        price_data_file.close()
    with open('Combined_News_DJIA.csv', 'r',newline='') as DJIA_news_file:
        stock_newsdata=list(csv.reader(DJIA_news_file, delimiter=',', skipinitialspace=True))
        DJIA_news_file.close()
    training_data=[]
    j=0
    k=1
    training_targets = []
    training_data.append(stock_price[0])
    for i in range(1, len(stock_price)):
        if(((i+1)%2)==0):
            training_targets.append(stock_price[i])
        else:
            training_data.append(stock_price[i])
    training_data[0].append('sentiment value')

    for i in range(1, (len(stock_newsdata))):
        #if((k<1592)or((training_data[k][0]==stock_newsdata[i][0])or(training_targets[j][0]==stock_newsdata[i][0]))):
        if(((i+1)%2)==0):
            training_targets[j].append(stock_newsdata[i][1])
            training_targets[j][0] = np.int(np.char.replace(training_targets[j][0], '-', ''))  # .remove(stock_newsdata[j][0])
            training_targets[j] = np.round(np.array(training_targets[j], dtype=np.float))
            j+=1
        else:
            training_data[k].append(stock_newsdata[i][1])
            training_data[k][0] = np.int(np.char.replace(training_data[k][0], '-', ''))#.remove(stock_newsdata[j][0])
            training_data[k] = np.round(np.array(training_data[k], dtype=np.float))
            k+=1
    training_data[0] = training_data[1]
    sc=MinMaxScaler(feature_range=(0, 10))
    training_targets=sc.fit_transform(training_targets)
    training_data=sc.fit_transform(training_data)
    #training_data = np.reshape(training_data, (398, 4, 8))
    return training_data, training_targets

def TrainingDataConc(training_data):
    training_target = list()
    #training_data = training_data.resize((199, 10, 8))
    training_data = np.reshape(training_data, (398, 5, 8))
    training_data = np.array(training_data, dtype=float)
    #print(training_data)
    print(len(training_data))
    print(len(training_data[0]))
    print(len(training_data[0][0]))
    for i in range(0, len(training_data)):
        #print(data[i][4][2],  data[i][4][3], data[i][4][4], data[i][4][5])
        inputval = np.array([training_data[i][4][2],  training_data[i][4][3],
                    training_data[i][4][4], training_data[i][4][5]])
        #training_data[i]=training_data[0:4]
        training_target.append(inputval)
        #np.delete(training_data[i], 4)
    print(len(training_data))
    print(len(training_data[0]))
    print(len(training_data[0][0]))
    print(training_data)
    np.resize(training_data, (398, 4, 8))
    print(training_data)
    print(len(training_data))
    print(len(training_data[0]))
    print(len(training_data[0][0]))
    #temp = training_data
    #training_data = np.resize(training_data, (398, 4, 8))
    #training_target = np.reshape(training_target, (398, 4))
    #for i in range(0, len(training_data)):
    #    for j in range(0, 4):
    #        training_data[i][j]=temp[i][j]
    #print(training_data)
    #print(training_target)
    return(training_data, training_target)

def TestingData(train_data, train_targets):
    testing_data = list()
    testing_targets = list()
    for i in range(0, 90):
        testing_data.append(train_data[i])
        testing_targets.append(train_targets[i])
        np.delete(train_data, i)
        np.delete(train_targets, i)

    return ((train_data, train_targets), (testing_data, testing_targets))
#    return (train_dat, trainval_data),(testing_data, testingval_data)

def SplitTargets(targets):
    target1 = list()
    target2 = list()
    target3 = list()
    target4 = list()
    target5 = list()
    target6 = list()
    target7 = list()
    target8 = list()
    for each in targets:
        target1.append(int(each[0]))
        target2.append(int(each[1]))
        target3.append(int(each[2]))
        target4.append(int(each[3]))
        target5.append(int(each[4]))
        target6.append(int(each[5]))
        target7.append(int(each[6]))
        target8.append(int(each[7]))

    return np.array(target1), np.array(target2), np.array(target3), np.array(target4), np.array(target5), \
           np.array(target6), np.array(target7), np.array(target8)

def TrainModel(training_data, training_targets, testing_data, testing_targets):
    target1, target2, target3, target4, target5, target6, target7, target8 = SplitTargets(training_targets)
    test_target1, test_target2, test_target3, test_target4, test_target5,\
        test_target6, test_target7, test_target8 = SplitTargets(testing_targets)
    training_data = np.reshape(training_data, (995, 8, 1))
    testing_data = np.reshape(testing_data, (90, 8, 1))

    asemptotic_input = Input(shape=(8, 1), dtype=float)
    input_layer = layers.Conv1D(64, kernel_size=4, activation='relu')(asemptotic_input)
    x = layers.MaxPooling1D(3, 3)(input_layer)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(256)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    #x1 = layers.Dense(32, activation='sigmoid')(x)
    #date = layers.Dense(1, activation='softmax')(y1)
    x2 = layers.Dense(32, activation='linear')(x)
    open = layers.Dense(1, name='open', activation='linear')(x2)
    x3 = layers.Dense(32, activation='linear')(x)
    high = layers.Dense(1, name='high', activation='linear')(x3)
    x4 = layers.Dense(32, activation='linear')(x)
    low = layers.Dense(1, name='low', activation='linear')(x4)
    x5 = layers.Dense(32, activation='linear')(x)
    close = layers.Dense(1, name='close', activation='linear')(x5)
    x6 = layers.Dense(32, activation='linear')(x)
    volume = layers.Dense(1, name='volume', activation='linear')(x6)
    x7 = layers.Dense(32, activation='linear')(x)
    adj_close = layers.Dense(1,  name='adj_close', activation='linear')(x7)
    #x8 = layers.Dense(32, activation='sigmoid')(x)
    #close = layers.Dense(1, activation='softmax')(y8)

    SPLSTM = Model(asemptotic_input, [open, high, low, close, volume, adj_close])
    SPLSTM.compile(optimizer='adam',
                   loss=['mean_squared_error', 'mean_squared_error', 'mean_squared_error',
                         'mean_squared_error', 'mean_squared_error', 'mean_squared_error'],
                   metrics=['accuracy']) #accuracy, MAPE
    SPLSTM.fit(training_data, [target2, target3, target4, target5, target6, target7],
               epochs=50,
               batch_size=5,
               validation_split=0.1)

    SPLSTM.evaluate(testing_data, [test_target2, test_target3, test_target4, test_target5,\
        test_target6, test_target7])

training_data, training_targets=TrainData()

(training_data, training_targets),(testing_data, testing_targets)=\
    TestingData(training_data, training_targets)

#target1, target2, target3, target4 = SplitTargets(training_targets)
#test_target1, test_target2, test_target3, test_target4 = SplitTargets(testing_targets)

TrainModel(training_data, training_targets, testing_data, testing_targets)