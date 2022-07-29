# The examples in this notebook use a set of nine benchmarks described in our publication.
# These benchmarks can be downloaded via FTP from: ftp.cs.huji.ac.il/users/nadavb/protein_bert/protein_benchmarks
# Download the benchmarks into a directory on your machine and set the following variable to the path of that directory.
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, SimpleRNN, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics


def read_file():
    os.chdir("../dataset/benchmark_dataset")
    BENCHMARKS_DIR = os.getcwd()
    BENCHMARK_NAME = "Dataset"
    dataset_file_path = os.path.join(BENCHMARKS_DIR, '%s.fasta' % BENCHMARK_NAME)
    dataset = pd.read_csv(dataset_file_path, header=None).dropna().drop_duplicates()
    pos_size = len(dataset.index[dataset[0].str.contains(">pos")])
    neg_size = len(dataset.index[dataset[0].str.contains(">neg")])
    print(f"number of positive samples: {pos_size}")
    print(f"number of negative samples: {neg_size}")
    label_array = np.concatenate((np.ones(pos_size), np.zeros(neg_size)))
    dataset.drop(dataset.index[dataset[0].str.contains(">")], inplace=True)
    dataset.columns = ['seq']
    dataset['label'] = label_array
    return dataset

def AAC(frag):
    lines = frag
    L=len(lines[0])
    n=int(len(lines))
    AAs='ACUG'
    m=len(AAs)
    aac=np.zeros((n,4))
    for i in range(n):
        for j in range(m):
            frequency=lines[i].count(AAs[j])
            frequency=float('%.2f'%frequency)
            aac[i][j]=frequency/L
    aac=aac[:,0:4]
    return aac

def k_space(frag):
    new_lines = frag
    L = len(new_lines[0])
    n = int(len(new_lines))
    AAs = 'ACUG'
    m = len(AAs)
    pair = []
    for i in range(m):
        for j in range(m):
            pair.append(AAs[i] + AAs[j])

    for k in range(5):
        k_space = np.zeros((n, 16))
        for t in range(n):
            for i in range(L - k - 1):
                AApair = new_lines[t][i] + new_lines[t][i + k + 1]
                for j in range(16):
                    if AApair == pair[j]:
                        k_space[t][j] += 1
        if k == 0:
            Kspace = k_space
        else:
            Kspace = np.concatenate((Kspace, k_space), axis=1)
    return Kspace

def PWAA(frag):
    lines = frag
    L=len(lines[0])
    n=int(len(lines))
    AAs='ACUG'
    l=int((L-1)/2)
    data=np.zeros((n,4))
    for i in range(n):
        for k in range(len(AAs)):
            pos=[ii for ii,v in enumerate(lines[i]) if v==AAs[k]]
            pos2=[jj+1 for jj in pos]
            p=[]
            c=[]
            for j in pos2:
                if j>=1 and j<=l:
                    p.append(j-l-1)
                if j>l and j<=L:
                    p.append(j-l-1)
            for m in p:
                if m>=-l and m<=l:
                    S1=float('%.2f'%abs(m))
                    c.append(m+S1/l)
            S2=float('%.2f'%sum(c))
            data[i][k]=S2/(l*(l+1))
    return data

def ExtractFeatures(frag, *var):
    # print(var)
    n = len(var)
    feature = {}
    for i in range(n):
        if var[i]==1:
            feature[i]=AAC(frag)
            print(f"shape of AAC:: {feature[i].shape}")
        if var[i]==2:
            feature[i]=k_space(frag)
            print(f"shape of C5SAAP:: {feature[i].shape}")
        if var[i]==3:
            feature[i]=PWAA(frag)
            print(f"shape of PWAA:: {feature[i].shape}")
        # if var[i] == 4:
        #     feature[i] = DBPB(frag)
        # if var[i] == 5:
        #     feature[i] = EBGW(frag)
        # if var[i] == 6:
        #     feature[i] = KNN(frag)

    for i in range(n):
        if i == 0:
            features = feature[i]
        else:
            features = np.concatenate((features, feature[i]), axis=1)
    # print(features)
    return features

def cross_validation(X_train, y_train):
    #########train model#######################

    # k-fold
    # Convolution
    # filter_length = 3
    nb_filter = 64
    pool_length = 2

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 128
    nb_epoch = 60

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    acc_score = []
    auc_score = []
    sn_score = []
    sp_score = []
    mcc_score = []

    for i, (train, test) in enumerate(kfold.split(X_train, y_train)):
        print('\n\n%d' % i)

        model = Sequential()
        # model.add(Dropout(0.5))
        model.add(Convolution1D(filters=nb_filter,
                                kernel_size=10,
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D(pool_size=pool_length))
        model.add(Dropout(0.3))
        model.add(Convolution1D(filters=nb_filter,
                                kernel_size=5,
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D(pool_size=pool_length))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        print('Train...')

        # 早停法
        checkpoint = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1, mode='auto')

        # early stopping
        
        model.fit(X_train[train], y_train[train], epochs=nb_epoch, batch_size=batch_size,
                  validation_data=(X_train[test], y_train[test]), shuffle=True, callbacks=[checkpoint], verbose=1)
        print(model.summary())
        # model.fit(X_train[train], y_train[train], epochs = nb_epoch, batch_size = batch_size, validation_data = (X_train[test], y_train[test]), shuffle = True)
        ##########################
        prd_acc = model.predict(X_train[test])
        pre_acc2 = []
        for i in prd_acc:
            pre_acc2.append(i[0])

        prd_lable = []
        for i in pre_acc2:
            if i > 0.5:
                prd_lable.append(1)
            else:
                prd_lable.append(0)
        prd_lable = np.array(prd_lable)
        # prd_lable = np.argmax(prd_acc, axis=1)
        obj = confusion_matrix(y_train[test], prd_lable)
        tp = obj[0][0]
        fn = obj[0][1]
        fp = obj[1][0]
        tn = obj[1][1]
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        mcc = (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
        sn_score.append(sn)
        sp_score.append(sp)
        mcc_score.append(mcc)
        ###########################
        pre_test_y = model.predict(X_train[test], batch_size=batch_size)
        test_auc = metrics.roc_auc_score(y_train[test], pre_test_y)
        auc_score.append(test_auc)
        print("test_auc: ", test_auc)

        # score, acc = model.evaluate(X_train[test], y_train[test], batch_size=batch_size)
        acc = (tp + tn) / (tp + fp + tn + fn)
        acc_score.append(acc)
        # print('Test score:', score)
        print('Test accuracy:', acc)
        print('***********************************************************************\n')
    print('***********************print final result*****************************')
    print(acc_score, auc_score)
    mean_acc = np.mean(acc_score)
    mean_auc = np.mean(auc_score)
    mean_sn = np.mean(sn_score)
    mean_sp = np.mean(sp_score)
    mean_mcc = np.mean(mcc_score)
    # print('mean acc:%f\tmean auc:%f'%(mean_acc,mean_auc))

    line = 'acc\tsn\tsp\tmcc\tauc:\n%.2f\t%.2f\t%.2f\t%.4f\t%.4f' % (
        100 * mean_acc, 100 * mean_sn, 100 * mean_sp, mean_mcc, mean_auc)
    print('5-fold result\n' + line)

def main():
    dataset = read_file()
    X_train = ExtractFeatures(dataset['seq'].values, 1, 2, 3)[:,:, np.newaxis]
    y_train = dataset['label'].values
    
    print(X_train.shape, y_train.shape)
    cross_validation(X_train=X_train, y_train=y_train)

if __name__ == "__main__":
    main()