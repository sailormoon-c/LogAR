import sys
sys.path.append('../')
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing
import os
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)
# import keras.backend.tensorflow_backend as KTF
# from keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import *
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

struct_log = '../data/HDFS/HDFS_word.npz'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True   
    sess = tf.compat.v1.Session(config=config)
    window_size = 10
    tf.compat.v1.keras.backend.set_session(sess)
    disable_eager_execution()

    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                           window='session',
                                                           train_ratio=0.5,
                                                           split_type='uniform')
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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    train_x = tokenizer.texts_to_sequences(train_text)
    val_x = tokenizer.texts_to_sequences(test_text)
    target_x = tokenizer.texts_to_sequences(target_text)
    # data processing and reconstruction
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

    ##Encoder##
    encoder_inputs = Input(shape = (None,),name="input1")
    embeddings = Embedding(vocab_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False, name="embedding1")
    encoder_embedding = embeddings(encoder_inputs)
    encoder = LSTM(32, return_state=True, name="lstm1")
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)
    encoder_dense = Dense(32, activation="sigmoid", name="fc1")
    outputs = encoder_dense(encoder_outputs)
    output_dense = Dense(2, activation="softmax", name="fc2")
    output = output_dense(outputs)
    encoder_state = [state_h, state_c]
    ##Decoder##
    decoder_inputs = Input(shape=(None,),name="input2")
    embeddings = Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=False,name="embedding2")
    decoder_embedding = embeddings(decoder_inputs)
    decoder = LSTM(32, return_sequences=True, return_state=True, name="lstm2")
    decoder_outputs, _, _ = decoder(decoder_embedding,initial_state=encoder_state)
    decoder_dense = Dense(vocab_size, activation="softmax", name="fc3")
    decoder_output = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs,decoder_inputs],[output,decoder_output])
    model.summary()
    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy','categorical_crossentropy'],
                  metrics=['accuracy'], experimental_run_tf_function=False)

    y_train = tf.keras.utils.to_categorical(y_train)
    val_y = tf.keras.utils.to_categorical(y_test)
    target_x = target_x.reshape(287530, 100, 1)
    target = np.zeros((287530, 100, 75))
    for i in range(target_x.shape[0]):
        for j in range(target_x.shape[1]):
            temp = tf.keras.utils.to_categorical(target_x[i][j], num_classes=75)
            target[i][j] = temp
    # Input:[None,maxlen]

    # class LossHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = []
    #
    #     def on_batch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('loss'))
    # history = LossHistory()
    model.fit([train_x, train_x], [y_train, target], epochs=10, batch_size=32, verbose=0)  # input: (3969, 14)  label:(3969,)
    # print(history.losses)
    # np.save('log.npy', history.losses)
    seqmodel = Model(encoder_inputs, output)
    y_pred = seqmodel.predict(val_x).argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
