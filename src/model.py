from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, GRU, LSTM, Bidirectional, Input, TimeDistributed, Flatten, Activation, CuDNNGRU, Conv1D, Concatenate, BatchNormalization, add, MaxPooling1D, GlobalMaxPool1D, Reshape
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.utils import to_categorical
from keras.models import Model, Input

def conv_block(x, activation=True, batch_norm=True, drop_out=0.5, res=True, num_filters = 128):
    cnn = Conv1D(num_filters, 3, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(drop_out))(cnn)    
    return cnn

def get_cnn_model(
    num_filters = 100, 
    embedding_dim = embedding_dim, 
    embedding_matrix = embedding_matrix, 
    maxlen_seq = maxlen_seq, 
    n_words = n_words, 
    dr = 0.5):
    inp = Input(shape=(maxlen_seq,))
    x = Embedding(n_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen_seq)(inp) 
    x = conv_block(x, drop_out=dr, num_filters = num_filters)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_filters//2, activation="relu")(x)  # Should be 4096
    outy = Dense(3, activation="softmax", name="dense2")(x)
    model = Model(inputs=inp, outputs=outy)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_gru_model(
    num_hidden = 100, 
    embedding_dim = embedding_dim, 
    embedding_matrix = embedding_matrix, 
    maxlen_seq = maxlen_seq, 
    n_words = n_words, 
    dr = 0.5):
    inp = Input(shape=(maxlen_seq,))
    x = Embedding(n_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen_seq)(inp) 
    x = Bidirectional(GRU(num_hidden))(x)
    x = Dropout(0.5)(x)
    x = Dense(num_hidden//2, activation="relu")(x)  # Should be 4096
    outy = Dense(3, activation="softmax", name="dense2")(x)
    model = Model(inputs=inp, outputs=outy)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
  
def get_maxpool_model(
    num_hidden = 100, 
    embedding_dim = embedding_dim, 
    embedding_matrix = embedding_matrix, 
    maxlen_seq = maxlen_seq, 
    n_words = n_words, 
    dr = 0.5):
    inp = Input(shape=(maxlen_seq,))
    x = Embedding(n_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen_seq)(inp) 
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    outy = Dense(3, activation="softmax", name="dense2")(x)
    model = Model(inputs=inp, outputs=outy)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def get_avgpool_model(
    num_hidden = 100, 
    embedding_dim = embedding_dim, 
    embedding_matrix = embedding_matrix, 
    maxlen_seq = maxlen_seq, 
    n_words = n_words, 
    dr = 0.5):
    inp = Input(shape=(maxlen_seq,))
    x = Embedding(n_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen_seq)(inp) 
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outy = Dense(3, activation="softmax", name="dense2")(x)
    model = Model(inputs=inp, outputs=outy)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
