import sys
sys.path.append('../')
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing
import os
import tensorflow
# import keras.backend.tensorflow_backend as KTF
# from keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from tensorflow.keras.models import Model
# from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import *
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

struct_log = '../data/HDFS/HDFS_word.npz' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file

def shuffle2(d):
    len_ = len(d)
    times = 10  # 次数
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]], d[index[1]] = d[index[1]], d[index[0]]
    return d

def dropout(d, p=0.1):  # noise
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
        d2 = dropout(d1)
        d22 = ' '.join(d2)
        seq.append(d22)
    return seq

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    disable_eager_execution()
    window_size = 10

    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                           window='session',
                                                           train_ratio=0.5,
                                                           split_type='uniform')

    # x_train = x_train[0:100]
    # y_train = y_train[0:100]
    # x_test = x_test[0:100]
    # y_test = y_test[0:100]
    train_text = []
    target_text = []
    for i in range(len(x_train)):
        temp = ""
        for j in range(len(x_train[i])):
            if j != len(x_train[i])-1:
                temp += x_train[i][j]+","
            else:
                temp += x_train[i][j]
        train_text.append(temp)
        target_text.append(temp[1:-1])

    test_text = []
    for i in range(len(x_test)):
        temp = ""
        for j in range(len(x_test[i])):
            if j != len(x_test[i]) - 1:
                temp += x_test[i][j] + ","
            else:
                temp += x_test[i][j]
        test_text.append(temp)

    train_now_text = train_text
    train_text = dataaugment(train_text)
    test_text = dataaugment(test_text)
    train_text = np.array(train_text) # data load
    train_text = train_text.tolist()
    test_text = np.array(test_text)
    test_text = test_text.tolist()
    target_text = np.array(target_text)
    target_text = target_text.tolist()
    train_now_text = np.array(train_now_text)
    train_new = train_now_text.tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    train_x = tokenizer.texts_to_sequences(train_text)
    val_x = tokenizer.texts_to_sequences(test_text)
    target_x = tokenizer.texts_to_sequences(target_text)
    # 处理数据，方便重构
    maxlen = 100

    vocab_size = len(tokenizer.word_index) + 1

    train_x = pad_sequences(train_x, padding='post', maxlen=maxlen)
    val_x = pad_sequences(val_x, padding='post', maxlen=maxlen)
    target_x = pad_sequences(target_x, padding='post', maxlen=maxlen)
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

    ##编码器##
    encoder_inputs = Input(shape = (None,),name="input1")
    embeddings = Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False, name="embedding1")
    encoder_embedding = embeddings(encoder_inputs)
    encoder = LSTM(32, return_state=True,name="lstm1")
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)
    encoder_dense = Dense(32, activation="sigmoid",name="fc1")
    outputs = encoder_dense(encoder_outputs)
    output_dense = Dense(2, activation="softmax",name="fc2")
    output = output_dense(outputs)

    model = Model(encoder_inputs,output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = tensorflow.keras.utils.to_categorical(y_train)
    val_y = tensorflow.keras.utils.to_categorical(y_test)
    model.fit(train_x, y_train, epochs=30, batch_size=32) #input: (3969, 14)  label:(3969,)
    y_pred = model.predict(val_x).argmax(axis=-1)
    precision, recall, f1, _  = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

