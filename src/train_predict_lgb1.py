#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error as MAE

import argparse
import ctypes
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from kaggler.data_io import load_data

import lightgbm as lgb


offset = 200.


def logcoshobj(preds, dtrain):
    labels = dtrain.get_label()
    grad = np.tanh(preds - labels)
    hess = 1 - grad * grad
    return grad, hess


def fairobj(labels, preds):
    c = 2.
    e = preds - labels
    grad = c * e / (np.abs(e) + c)
    hess = c ** 2 / (np.abs(e) + c) ** 2
    return grad, hess


def eval_mae(preds, dataset):
    labels = dataset.get_label()
    return 'mae', MAE(np.exp(labels), np.exp(preds)), False


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.3, subrow=.8,
                  subrow_freq=100, n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    y = np.log(y + offset)

    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = KFold(len(y), n_folds=N_FOLD, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}'.format(i))
        watchlist = [(X[i_val], y[i_val])]

        logging.info('Training with early stopping')
        clf = lgb.LGBMRegressor(n_estimators=n_est,
                                num_leaves=n_leaf,
                                learning_rate=lrate,
                                min_child_samples=n_min,
                                subsample=subrow,
                                subsample_freq=subrow_freq,
                                colsample_bytree=subcol,
                                objective=fairobj,
                                nthread=1,
                                seed=SEED)
        clf = clf.fit(X[i_trn], y[i_trn], eval_set=watchlist,
                      eval_metric=eval_mae, early_stopping_rounds=n_stop,
                      verbose=10)
        n_best = clf.best_iteration
        n_bests.append(n_best)
        logging.info('best iteration={}'.format(n_best))

        p_val[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.4f}'.format(i, MAE(np.exp(y[i_val]),
                                                    np.exp(p_val[i_val]))))

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.4f}'.format(MAE(np.exp(y), np.exp(p_val))))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, np.exp(p_val) - offset, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        n_best = sum(n_bests) // N_FOLD
        clf = lgb.LGBMRegressor(n_estimators=n_best,
                                num_leaves=n_leaf,
                                learning_rate=lrate,
                                min_child_samples=n_min,
                                subsample=subrow,
                                subsample_freq=subrow_freq,
                                colsample_bytree=subcol,
                                objective=fairobj,
                                nthread=1,
                                seed=SEED)

        clf = clf.fit(X, y)
        p_tst = clf.predict(X_tst)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, np.exp(p_tst) - offset, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--n-leaf', type=int, dest='n_leaf')
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--subrow-freq', type=int, default=100,
                        dest='subrow_freq')
    parser.add_argument('--n-min', type=int, default=1, dest='n_min')
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  n_leaf=args.n_leaf,
                  lrate=args.lrate,
                  n_min=args.n_min,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  subrow_freq=args.subrow_freq,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
