import sys

sys.path.append("../")
from loglizer.models import DecisionTree
from loglizer import dataloader, preprocessing
import os
import tensorflow.compat.v1 as tf

tf.compat.v1.experimental.output_all_intermediates(True)
# import keras.backend.tensorflow_backend as KTF
# from keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import keras
from tensorflow.keras.models import Model

# from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution


struct_log = "../data/HDFS/HDFS_word.npz"  # The structured log file
label_file = "../data/HDFS/anomaly_label.csv"  # The anomaly label file


def shuffle2(d):
    len_ = len(d)
    times = 10  # shuffle the ordering
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]], d[index[1]] = d[index[1]], d[index[0]]
    return d


def dropout(d, p=0.7):  # noise  
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = " "
    return d


def dataaugment(X):
    l = len(X)
    seq = []
    for i in range(l):
        item = X[i]
        item = list(item)
        d1 = shuffle2(item)
        d2 = dropout(d1)
        d22 = " ".join(d2)
        # d22 = d1
        seq.append(d22)
    return seq


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  
    sess = tf.compat.v1.Session(config=config)
    window_size = 10
    tf.compat.v1.keras.backend.set_session(sess)
    disable_eager_execution()
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(
        struct_log, window="session", train_ratio=0.5, split_type="uniform"
    )
    # x_train = x_train[0:100]
    # y_train = y_train[0:100]
    # x_test = x_test[0:100]
    # y_test = y_test[0:100]
    train_text = []
    target_text = []
    for i in range(len(x_train)):
        temp = ""
        for j in range(len(x_train[i])):
            if j != len(x_train[i]) - 1:
                temp += x_train[i][j] + ","
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
    train_text = np.array(train_text)  # data load
    train_text = train_text.tolist()
    test_text = dataaugment(test_text)
    test_text = np.array(test_text)
    test_text = test_text.tolist()
    target_text = np.array(target_text)
    target_text = target_text.tolist()
    train_now_text = np.array(train_now_text)
    train_new = train_now_text.tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    train_x = tokenizer.texts_to_sequences(train_text)
    train_x_1 = tokenizer.texts_to_sequences(train_new)
    val_x = tokenizer.texts_to_sequences(test_text)
    target_x = tokenizer.texts_to_sequences(target_text)
    # data processing and reconstruction
    maxlen = 100

    vocab_size = len(tokenizer.word_index) + 1

    train_x = pad_sequences(train_x, padding="post", maxlen=maxlen)
    train_x_1 = pad_sequences(train_x_1, padding="post", maxlen=maxlen)
    val_x = pad_sequences(val_x, padding="post", maxlen=maxlen)
    target_x = pad_sequences(target_x, padding="post", maxlen=maxlen)
    embedding_dim = 100

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    def load_embeddings(path):
        with open(path, encoding="utf-8") as f:
            return dict(get_coefs(*line.strip().split(" ")) for line in f)

    def build_matrix(word_index, path):
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 100))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                pass
        return embedding_matrix

    embedding_matrix = build_matrix(
        tokenizer.word_index, "../.vector_cache/glove.6B.100d.txt"
    )

    ##Encoder##
    encoder_inputs = Input(shape=(None,), name="input1")
    embeddings = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False,
        name="embedding1",
    )
    encoder_embedding = embeddings(encoder_inputs)  # embedding
    encoder = LSTM(32, return_state=True, name="lstm1")
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)  # feature extraction
    # classification
    encoder_dense = Dense(32, activation="sigmoid", name="fc1")
    outputs = encoder_dense(encoder_outputs)
    output_dense = Dense(2, activation="softmax", name="fc2")
    output = output_dense(outputs)  # prediction outcomes
    batchsize = K.shape(encoder_inputs)
    # print(batchsize.shape)
    coef = tf.Variable(1.0e-4 * tf.ones([10, 10], tf.float32), name="coef")
    sess.run(coef.initializer)
    z_state_h = tf.matmul(coef, state_h)
    z_state_c = tf.matmul(coef, state_c)
    encoder_state = [z_state_h, z_state_c]
    ##Decoder##
    decoder_inputs = Input(shape=(None,), name="input2")
    embeddings = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False,
        name="embedding2",
    )
    decoder_embedding = embeddings(decoder_inputs)
    decoder = LSTM(32, return_sequences=True, return_state=True, name="lstm2")
    decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_state)
    decoder_dense = Dense(vocab_size, activation="softmax", name="fc3")
    decoder_output = decoder_dense(decoder_outputs)  # Decoding
    model = Model([encoder_inputs, decoder_inputs], [output, decoder_output])
    # del model
    # model = tf.keras.models.Model([encoder_inputs, decoder_inputs], [output, decoder_output])
    # model.summary()

    def my_loss(y_true, y_pred):
        loss = K.mean(K.square(y_pred - y_true), -1)
        selfexpress_loss = 0.5 * tf.reduce_sum(
            tf.pow(tf.subtract(z_state_h, state_h), 2.0)
        )
        selfexpress_loss = selfexpress_loss + 0.5 * tf.reduce_sum(
            tf.pow(tf.subtract(z_state_c, state_c), 2.0)
        )
        return loss + 0.0001 * selfexpress_loss

    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy", my_loss],
        loss_weights=[1.0, 10.0],
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    y_train = tf.keras.utils.to_categorical(y_train)
    val_y = tf.keras.utils.to_categorical(y_test)
    target_x = target_x.reshape(train_x.shape[0], 100, 1)
    target = np.zeros((train_x.shape[0], 100, vocab_size))
    for i in range(target_x.shape[0]):
        for j in range(target_x.shape[1]):
            temp = tf.keras.utils.to_categorical(target_x[i][j], num_classes=vocab_size)
            target[i][j] = temp

    # class LossHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = []
    #
    #     def on_batch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('loss'))
    # history = LossHistory()

    model.fit(
        [train_x, train_x_1], [y_train, target], batch_size=10, epochs=20, verbose=0
    )  # input: (3969, 14)  label:(3969,)
    # np.save('log1.npy', history.losses)
    seqmodel = Model(encoder_inputs, output)
    y_pred = seqmodel.predict(val_x).argmax(axis=-1)
    # y_pred,_ = model.predict([val_x,val_x])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    print(
        "Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n".format(
            precision, recall, f1
        )
    )
