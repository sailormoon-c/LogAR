import sys
sys.path.append('../')
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from keras.layers import *
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

struct_log = '../data/HDFS/HDFS_word.npz' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

def shuffle2(d):
    len_ = len(d)
    times = 1  # 设置打乱顺序次数
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]], d[index[1]] = d[index[1]], d[index[0]]
    return d


def dropout(d, p=0.1):  # noise  设置删除单词个数
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d


def dataaugment(X):
    l = len(X)
    seq = []
    for i in range(l):
        item = X[i]
        item = list(item)
        d1 = shuffle2(item)
        # d2 = dropout(d1)
        # d22 = ' '.join(d2)
        d22 = d1
        seq.append(d22)
    return seq


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    sess = tf.compat.v1.Session(config=config)
    window_size = 10
    set_session(sess)

    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                           window='session',
                                                           train_ratio=0.5,
                                                           split_type='uniform')
    # print(x_train[0])
    # x_train = x_train[0:100]
    # y_train = y_train[0:100]
    # x_test = x_test[0:100]
    # y_test = y_test[0:100]
    train_text = []
    for i in range(len(x_train)):
        temp = ""
        for j in range(len(x_train[i])):
            if j != len(x_train[i])-1:
                temp += x_train[i][j]+","
            else:
                temp += x_train[i][j]
        train_text.append(temp)

    test_text = []
    for i in range(len(x_test)):
        temp = ""
        for j in range(len(x_test[i])):
            if j != len(x_test[i]) - 1:
                temp += x_test[i][j] + ","
            else:
                temp += x_test[i][j]
        test_text.append(temp)

    x_train = dataaugment(x_train)
    x_train = np.array(x_train)  # data load
    train_text = x_train.tolist()
    x_test = dataaugment(x_test)
    x_test = np.array(x_test)
    test_text = x_test.tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    train_x = tokenizer.texts_to_sequences(train_text)
    val_x = tokenizer.texts_to_sequences(test_text)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100
    train_x = pad_sequences(train_x, padding='post', maxlen=maxlen)
    val_x = pad_sequences(val_x, padding='post', maxlen=maxlen)
    embedding_dim = 100
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(path):
        with open(path, encoding='utf-8') as f:
            return dict(get_coefs(*line.strip().split(' ')) for line in f)

    def build_matrix(word_index, path):
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 100))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                pass
        return embedding_matrix

    embedding_matrix = build_matrix(tokenizer.word_index, '../.vector_cache/glove.6B.100d.txt')

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False))
    model.add(layers.LSTM(16))
    model.add(layers.Dense(16, activation="sigmoid"))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = tf.keras.utils.to_categorical(y_train)
    val_y = tf.keras.utils.to_categorical(y_test)
    print(type(train_x), type(y_train))
    model.fit(train_x, y_train,epochs=20,batch_size=32) #input: (3969, 14)  label:(3969,)
    y_pred = model.predict(val_x).argmax(axis=-1)
    precision, recall, f1, _  = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
