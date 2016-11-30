#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error as MAE

import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import SEED
from kaggler.data_io import load_data

import xgboost as xgb


offset = 200.


def logcoshobj(preds, dtrain):
    labels = dtrain.get_label()
    grad = np.tanh(preds - labels)
    hess = 1 - grad * grad
    return grad, hess


def fairobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 2.
    e = preds - labels
    grad = c * e / (np.abs(e) + c)
    hess = c ** 2 / (np.abs(e) + c) ** 2
    return grad, hess


def eval_mae(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', MAE(np.exp(labels), np.exp(preds))


def train_predict(train_file, test_file, feature_map_file, predict_valid_file,
                  predict_test_file, feature_importance_file, n_est=100,
                  depth=4, lrate=.1, subcol=.5, subrow=.5, sublev=1, weight=1,
                  n_stop=100, retrain=True, n_fold=5):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    # set xgb parameters
    params = {'objective': "reg:linear",
              'max_depth': depth,
              'eta': lrate,
              'subsample': subrow,
              'colsample_bytree': subcol,
              'colsample_bylevel': sublev,
              'min_child_weight': weight,
              'silent': 1,
              'nthread': 10,
              'seed': SEED}

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    y = np.log(y + offset)

    X_tst, _ = load_data(test_file)
    xgtst = xgb.DMatrix(X_tst)

    logging.info('Loading CV Ids')
    cv = KFold(len(y), n_folds=n_fold, shuffle=True, random_state=SEED)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    for i, (i_trn, i_val) in enumerate(cv, 1):
        xgtrn = xgb.DMatrix(X[i_trn], label=y[i_trn])
        xgval = xgb.DMatrix(X[i_val], label=y[i_val])

        logging.info('Training model #{}'.format(i))
        watchlist = [(xgtrn, 'train'), (xgval, 'val')]

        if i == 1:
            logging.info('Training with early stopping')
            clf = xgb.train(params, xgtrn, n_est, watchlist, fairobj, eval_mae,
                            early_stopping_rounds=n_stop)
            n_best = clf.best_iteration
            logging.info('best iteration={}'.format(n_best))

            importance = clf.get_fscore(feature_map_file)
            df = pd.DataFrame.from_dict(importance, 'index')
            df.index.name = 'feature'
            df.columns = ['fscore']
            df.ix[:, 'fscore'] = df.fscore / df.fscore.sum()
            df.sort_values('fscore', axis=0, ascending=False, inplace=True)
            df.to_csv(feature_importance_file, index=True)
            logging.info('feature importance is saved in {}'.format(feature_importance_file))
        else:
            clf = xgb.train(params, xgtrn, n_best, watchlist, fairobj, eval_mae)

        p_val[i_val] = clf.predict(xgval, ntree_limit=n_best)
        logging.info('CV #{}: {:.4f}'.format(i, MAE(np.exp(y[i_val]),
                                                    np.exp(p_val[i_val]))))

        if not retrain:
            p_tst += clf.predict(xgtst, ntree_limit=n_best) / n_fold

    logging.info('CV: {:.4f}'.format(MAE(np.exp(y), np.exp(p_val))))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, np.exp(p_val) - offset, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        xgtrn = xgb.DMatrix(X, label=y)
        watchlist = [(xgtrn, 'train')]
        clf = xgb.train(params, xgtrn, n_best, watchlist, fairobj, eval_mae)
        p_tst = clf.predict(xgtst, ntree_limit=n_best)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, np.exp(p_tst) - offset, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True,
                        dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--feature-importance-file', required=True,
                        dest='feature_importance_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--sublev', type=float, default=1.)
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_importance_file=args.feature_importance_file,
                  n_est=args.n_est,
                  depth=args.depth,
                  lrate=args.lrate,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  sublev=args.sublev,
                  weight=args.weight,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
