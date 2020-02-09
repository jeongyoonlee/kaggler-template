#!/usr/bin/env python

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from kaggler.data_io import load_data

import lightgbm as lgb


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.3, subrow=.8,
                  subrow_freq=100, n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    params = {'random_state': SEED,
              'n_jobs': -1,
              'objective': 'binary',
              'boosting': 'gbdt',
              'learning_rate': lrate,
              'num_leaves': n_leaf,
              'feature_fraction': subcol,
              'bagging_fraction': subrow,
              'bagging_freq': subrow_freq,
              'verbosity': -1,
              'min_child_samples': n_min,
              'metric': 'auc'}

    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))
        trn_lgb = lgb.Dataset(X[i_trn], label=y[i_trn])
        val_lgb = lgb.Dataset(X[i_val], label=y[i_val])

        logging.info('Training with early stopping')
        clf = lgb.train(params, trn_lgb, n_est, val_lgb, early_stopping_rounds=n_stop, verbose_eval=100)
        n_best = clf.best_iteration
        n_bests.append(n_best)
        logging.info('best iteration={}'.format(n_best))

        p[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.4f}'.format(i, AUC(y[i_val], p[i_val])))

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.4f}'.format(AUC(y, p)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')

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
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


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
