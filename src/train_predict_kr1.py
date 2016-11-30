from __future__ import absolute_import, division, print_function
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error as MAE

import argparse
import logging
import keras.backend as K
import numpy as np
import os
import pandas as pd
import time


from kaggler.data_io import load_data
from const import N_CLASS, SEED


np.random.seed(SEED) # for reproducibility
offset = 200


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


def nn_model(dims):
    model = Sequential()
    
    model.add(Dense(400, input_dim=dims, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init='he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, retrain=True, n_fold=5):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    y = np.log(y + offset)

    X_tst, _ = load_data(test_file)

    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tst = scaler.transform(X_tst)
    """

    dims = X.shape[1]
    logging.info('{} class(es), {} dims'.format(N_CLASS, dims))

    logging.info('Loading CV Ids')
    cv = KFold(len(y), n_folds=n_fold, shuffle=True, random_state=SEED)

    p_val = np.zeros_like(y)
    p_tst = np.zeros((X_tst.shape[0],))
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}'.format(i))
        clf = nn_model(dims)
        clf.fit_generator(generator=batch_generator(X[i_trn],
                                                    y[i_trn],
                                                    128,
                                                    True),
                          nb_epoch=n_est,
                          samples_per_epoch=X[i_trn].shape[0],
                          verbose=0)

        p_val[i_val] = clf.predict_generator(generator=batch_generatorp(X[i_val], 800, False),
                                             val_samples=X[i_val].shape[0])[:, 0]
        logging.info('CV #{}: {:.4f}'.format(i, MAE(np.exp(y[i_val]),
                                                    np.exp(p_val[i_val]))))

        if not retrain:
            p_tst += clf.predict_generator(generator=batch_generatorp(X_tst, 800, False),
                                           val_samples=X_tst.shape[0])[:, 0] / n_fold

    logging.info('Saving validation predictions...')
    logging.info('CV: {:.4f}'.format(MAE(np.exp(y), np.exp(p_val))))
    np.savetxt(predict_valid_file, np.exp(p_val) - offset, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        clf = nn_model(dims)
        clf.fit_generator(generator=batch_generator(X, Y, 128, True),
                          nb_epoch=n_est)
        p_tst = clf.predict_generator(generator=batch_generatorp(X_tst, 800, False),
                                      val_samples=X_tst.shape[0])[:, 0]

    logging.info('Saving normalized test predictions...')
    np.savetxt(predict_test_file, np.exp(p_tst) - offset, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', default=10, type=int, dest='n_est')
    parser.add_argument('--batch-size', default=64, type=int,
                        dest='batch_size')
    parser.add_argument('--hiddens', default=2, type=int)
    parser.add_argument('--neurons', default=512, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                 60.))
