from __future__ import print_function
import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
from keras.layers import Embedding
import random

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class ExtractEmbeddingFeatures:

    def __init__(self):
        os.chdir("../dataset/benchmark_dataset")
        self.data_folder = os.getcwd()
        self.fasta_file = self.data_folder + "//Dataset.fasta"
        os.chdir("../../generated_files")
        self.generated_files_folder = os.getcwd()
        self.corpus_file = self.generated_files_folder + '//2UN.txt'
        self.modelfile_string = "word2vec_model"
        self.posnum = 662
        self.pklfile = "data.pkl"
        self.model_save = 'model_final_word2Vec.h5'
        self.kmer = 2

    def twoTupleDic(self):
        AA_list_sort = ['A', 'U', 'C', 'G']
        AA_dict = {}
        numm = 1
        for i in AA_list_sort:
            for j in AA_list_sort:
                AA_dict[i + j] = numm
                numm += 1
        return AA_dict

    def DNA2Sentence(self,dna, K):
        sentence = ""
        length = len(dna)
        for i in range(length - K + 1):
            sentence += dna[i: i + K] + " "
        # delete extra space
        sentence = sentence[0: len(sentence) - 1]
        return sentence

    def Get_Unsupervised(self,fname, gname, kmer):
        f = open(fname, 'r')
        g = open(gname, 'w')
        seq = []
        k = kmer
        for i in f:
            if '>' not in i:
                i = i.strip('\n').upper()
                line = self.DNA2Sentence(i, k)
                seq.append(line)
                g.write(line + '\n')
        f.close()
        return seq

    def createTrainTestData(self, str_path, nb_words=None, skip_top=0,
                            maxlen=None, test_split=0.1, seed=113,
                            start_char=1, oov_char=2, index_from=3):
        X, labels = cPickle.load(open(str_path, "rb"))
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)
        X_train = np.array(X[:int(len(X) * (1 - test_split))])
        y_train = np.array(labels[:int(len(X) * (1 - test_split))])

        X_test = np.array(X[int(len(X) * (1 - test_split)):])
        y_test = np.array(labels[int(len(X) * (1 - test_split)):])

        return (X_train, y_train), (X_test, y_test)

    def getDNA_split(self, DNAdata, word):
        DNAlist1 = []
        # DNAlist2 = []
        counter = 0
        for DNA in DNAdata["seq"]:
            # if counter % 100 == 0:
            # print ("DNA %d of %d\r" % (counter, 2*len(DNAdata)))
            # sys.stdout.flush()

            DNA = str(DNA).upper()
            DNAlist1.append(
                self.DNA2Sentence(DNA, word).split(
                    " "))  # [['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC'],['ACG', 'CGT', 'GTC']]

            counter += 1
        return DNAlist1

    def getWord_model(self, word, num_features, min_count):
        word_model = ""
        if not os.path.isfile(self.generated_files_folder + "//" + self.modelfile_string):
            sentence = LineSentence(self.generated_files_folder + "//2UN.txt", max_sentence_length=15000)
            print("Start Training Word2Vec model...")
            # Set values for various parameters
            num_features = int(num_features)  # Word vector dimensionality
            min_word_count = int(min_count)  # Minimum word count
            num_workers = 20  # Number of threads to run in parallel????????????????????????
            context = 20  # Context window size?????????????????????
            downsampling = 1e-3  # Downsample setting for frequent words???????????????????????????

            # Initialize and train the model????????????????????????
            print("Training Word2Vec model...")
            word_model = Word2Vec(sentence, workers=num_workers, \
                                  vector_size=num_features, min_count=min_word_count, \
                                  window=context, sample=downsampling, seed=1, epochs=50)
            word_model.save(self.generated_files_folder + "//" + self.modelfile_string)
            # print word_model.most_similar("CATAGT")
        else:
            print("Loading Word2Vec model...")
            word_model = Word2Vec.load(self.generated_files_folder + "//" + self.modelfile_string)
        return word_model

    def extract(self):
        #########deal data, split to train and test################

        # ???index?????????????????????
        texts = self.Get_Unsupervised(self.fasta_file, self.corpus_file, self.kmer)

        word_index1 = self.twoTupleDic()
        sequences = []
        for each in texts:
            each_index_list = []
            each = each.split(' ')
            for i in each:
                each_index_list.append(word_index1[i])
            sequences.append(each_index_list)

        # ?????????????????????????????????1????????????0
        labels = []
        for i in range(0, self.posnum):
            labels.append(1)
        for j in range(self.posnum, len(texts)):
            labels.append(0)

        # ??????????????????3???7??????test???train
        MAX_SEQUENCE_LENGTH = 100
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        t = (data, labels)
        cPickle.dump(t, open(self.pklfile, "wb"))
        (X_train, y_train), (X_test, y_test) = self.createTrainTestData(self.pklfile, nb_words=None, test_split=0.1)

        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        #########obtain embedding_matrix###########

        tokenizer = Tokenizer(num_words=None)  # ??????MAX_NB_WORDS
        tokenizer.fit_on_texts(texts)
        sequences2 = tokenizer.texts_to_sequences(texts)  # ???num_words??????

        word_index2 = tokenizer.word_index  # ???_??????
        print(len(word_index1), len(word_index2))

        # word2vec
        EMBEDDING_DIM = 100
        self.getWord_model(texts, EMBEDDING_DIM, 1)
        Word2VecModel = Word2Vec.load(self.modelfile_string)
        embedding_matrix = np.zeros((len(word_index1) + 1, EMBEDDING_DIM))
        for word, i in word_index1.items():
            if word.lower() in word_index2:
                embedding_vector = Word2VecModel.wv[word]
                # print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix)

        return (X_train, y_train), (X_test, y_test), embedding_matrix

class TrainingWithEmbeddingFeatures:
    #########train model#######################

    def __init__(self, X_train, y_train, X_test, y_test, embedding_matrix):
        self.checkpoint = EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=3,
                                        verbose=1, mode='auto')
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.embedding_matrix = embedding_matrix
        # k-fold
        self.embedding_size = 100

        # Convolution
        # filter_length = 3
        self.nb_filter = 64
        self.pool_length = 2

        # LSTM
        self.lstm_output_size = 70

        # Training
        self.batch_size = 128
        self.nb_epoch = 60

    def cross_validation(self):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        acc_score = []
        auc_score = []
        sn_score = []
        sp_score = []
        mcc_score = []

        for i, (train, test) in enumerate(kfold.split(self.X_train, self.y_train)):
            print('\n\n%d' % i)

            model = Sequential()
            model.add(
                Embedding(len(self.embedding_matrix), self.embedding_size, weights=[self.embedding_matrix],
                          input_length=100,
                          trainable=False))
            model.add(Dropout(0.5))
            model.add(Convolution1D(filters=self.nb_filter,
                                    kernel_size=10,
                                    padding='valid',
                                    activation='relu',
                                    strides=1))
            model.add(MaxPooling1D(pool_size=self.pool_length))
            model.add(Convolution1D(filters=self.nb_filter,
                                    kernel_size=5,
                                    padding='valid',
                                    activation='relu',
                                    strides=1))
            model.add(MaxPooling1D(pool_size=self.pool_length))

            model.add(LSTM(self.lstm_output_size))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            print('Train...')

            # ?????????

            # early stopping
            model.fit(self.X_train[train], self.y_train[train], epochs=self.nb_epoch, batch_size=self.batch_size,
                      validation_data=(self.X_train[test], self.y_train[test]), shuffle=True, callbacks=[
                    self.checkpoint], verbose=2)
            # model.fit(X_train[train], y_train[train], epochs = nb_epoch, batch_size = batch_size, validation_data = (X_train[test], y_train[test]), shuffle = True)
            ##########################
            prd_acc = model.predict(self.X_train[test])
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
            obj = confusion_matrix(self.y_train[test], prd_lable)
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
            pre_test_y = model.predict(self.X_train[test], batch_size=self.batch_size)
            test_auc = metrics.roc_auc_score(self.y_train[test], pre_test_y)
            auc_score.append(test_auc)
            print("test_auc: ", test_auc)

            score, acc = model.evaluate(self.X_train[test], self.y_train[test], batch_size=self.batch_size)
            acc_score.append(acc)
            print('Test score:', score)
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

    def save_final_model(self):
        print('***************************save model********************************************')

        model = Sequential()
        model.add(Embedding(len(self.embedding_matrix), self.embedding_size, weights=[self.embedding_matrix],
                            input_length=100,
                            trainable=False))  # input_length????????????????????????embedding_size????????????????????????????????????????????????
        model.add(Dropout(0.5))
        # nb_filter : ?????????????????????????????????????????????filter_length : ???????????????????????????
        model.add(Convolution1D(filters=self.nb_filter,
                                kernel_size=10,
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_length))
        model.add(Convolution1D(filters=self.nb_filter,
                                kernel_size=5,
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_length))

        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        # optimizer??????????????????Adam???loss???????????????????????????????????????????????????metrics: ????????????????????????????????????????????????????????????????????????????????????metrics=[???accuracy???]?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????metrics={???output_a???: ???accuracy???}
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # model.fit(X_train, y_train, epochs = nb_epoch, batch_size = batch_size, validation_data = (X_test, y_test), shuffle = True)
        # early stopping
        model.fit(self.X_train, self.y_train, epochs=self.nb_epoch, batch_size=self.batch_size,
                  validation_data=(self.X_test, self.y_test), shuffle=True,
                  callbacks=[self.checkpoint], verbose=2)

        model.save(self.model_save)


def main():
    train_data, test_data, embedding_matrix = ExtractEmbeddingFeatures().extract()
    classifier = TrainingWithEmbeddingFeatures(
        X_train=train_data[0], y_train=train_data[1], X_test=test_data[0], y_test=test_data[1], embedding_matrix=embedding_matrix
    )
    classifier.cross_validation()

if __name__ == "__main__":
    main()
