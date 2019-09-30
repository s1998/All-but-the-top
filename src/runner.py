from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
from src.loder import *
from src.embeddings_processor import *
from src.model import *

processed = get_processed_embeddings(embedding_matrix)

K.clear_session()
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
mcp_save = ModelCheckpoint("cnn_" + 'pre.{val_acc:.4f}-{epoch:02d}-'+ str(embedding_dim) + '.hdf5.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model = get_cnn_model(
                num_filters = 100, 
                embedding_dim = embedding_dim, 
                embedding_matrix = embedding_matrix, 
                maxlen_seq = maxlen_seq, 
                n_words = n_words, 
                dr = 0.5)
batch_size = 128
n_epochs = 20
for epoch_ in range(n_epochs):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=2,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_dev, y_dev))

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report



K.clear_session()
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
mcp_save = ModelCheckpoint("cnn_" + 'post.{val_acc:.4f}-{epoch:02d}-'+ str(embedding_dim) + '.hdf5.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model = get_cnn_model(
                num_filters = 100, 
                embedding_dim = embedding_dim, 
                embedding_matrix = processed, 
                maxlen_seq = maxlen_seq, 
                n_words = n_words, 
                dr = 0.5)
batch_size = 128
n_epochs = 20
for epoch_ in range(n_epochs):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=2,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_dev, y_dev))

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report



K.clear_session()
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
mcp_save = ModelCheckpoint("gru_" + 'pre.{val_acc:.4f}-{epoch:02d}-'+ str(embedding_dim) + '.hdf5.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model = get_gru_model(
                num_hidden = 100, 
                embedding_dim = embedding_dim, 
                embedding_matrix = embedding_matrix, 
                maxlen_seq = maxlen_seq, 
                n_words = n_words, 
                dr = 0.5)
batch_size = 128
n_epochs = 20
for epoch_ in range(n_epochs):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=2,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_dev, y_dev))

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report



K.clear_session()
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
mcp_save = ModelCheckpoint("gru_" + 'post.{val_acc:.4f}-{epoch:02d}-'+ str(embedding_dim) + '.hdf5.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model = get_gru_model(
                num_hidden = 100, 
                embedding_dim = embedding_dim, 
                embedding_matrix = processed, 
                maxlen_seq = maxlen_seq, 
                n_words = n_words, 
                dr = 0.5)
batch_size = 128
n_epochs = 20
for epoch_ in range(n_epochs):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=2,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_dev, y_dev))

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report



K.clear_session()
earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='max')
mcp_save = ModelCheckpoint("avgpool_" + 'post.{val_acc:.4f}-{epoch:02d}-'+ str(embedding_dim) + '.hdf5.mdl_wts.hdf5', save_best_only=True, monitor='val_acc', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model = get_avgpool_model(
                num_hidden = 100, 
                embedding_dim = embedding_dim, 
                embedding_matrix = processed, 
                maxlen_seq = maxlen_seq, 
                n_words = n_words, 
                dr = 0.5)
batch_size = 128
n_epochs = 20
for epoch_ in range(n_epochs):
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=2,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_dev, y_dev))
    y_pred = model.predict(x_test)
    print(precision_recall_fscore_support(y_test, np.argmax(y_pred, axis=1), average='weighted'))
    print(accuracy_score(y_test, np.argmax(y_pred, axis=1)))