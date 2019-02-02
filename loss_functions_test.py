import unittest
from loss_funs import *
import numpy as np
import torch.nn as nn
from torch import FloatTensor


BATCH_SIZE = 2
NUM_SCALES = 5
MAX_P      = 5
MAX_ALPHA  = 1

# noinspection PyClassHasNoInit
class TestBCELoss(unittest.TestCase):
    def test_false_correct(self):
        y_n = 1e-7
        targets = FloatTensor(np.zeros([5, 1]))
        preds = FloatTensor(y_n * np.ones([5, 1]))
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()
        # res = sess.run(bce_loss(preds, targets))

        log_con = np.log(1 - y_n)
        # res_tru = -1 * np.sum(np.array([log_con] * 5))
        res_tru = -1 * np.sum(log_con * np.ones([5, 1]))
        np.testing.assert_array_almost_equal(res, res_tru)

    def test_false_incorrect(self):
        targets = FloatTensor(np.zeros([5, 1]))
        preds = FloatTensor(np.ones([5, 1]) - 1e-7)
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()

        log_con = np.log(1e-7)
        res_tru = -1 * np.sum(np.array(log_con * np.ones([5, 1])))
        np.testing.assert_array_almost_equal(res, res_tru, decimal=0)

    def test_false_half(self):
        targets = FloatTensor(np.zeros([5, 1]))
        preds = 0.5 * FloatTensor(np.ones([5, 1]))
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()

        log_con = np.log(0.5)
        res_tru = -1 * np.sum(np.array([log_con] * 5))
        np.testing.assert_array_almost_equal(res, res_tru, decimal=7)

    def test_true_correct(self):
        targets = FloatTensor(np.ones([5, 1]))
        preds = FloatTensor(np.ones([5, 1]) - 1e-7)
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()

        log = np.log(1 - 1e-7)
        res_tru = -1 * np.sum(np.array([log] * 5))
        np.testing.assert_array_almost_equal(res, res_tru, decimal=6)

    def test_true_incorrect(self):
        targets = FloatTensor(np.ones([5, 1]))
        preds = 1e-7 * FloatTensor(np.ones([5, 1]))
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()

        log = np.log(1e-7)
        res_tru = -1 * np.sum(np.array([log] * 5))
        np.testing.assert_approx_equal(res, res_tru, significant=7)

    def test_true_half(self):
        targets = FloatTensor(np.ones([5, 1]))
        preds = 0.5 * FloatTensor(np.ones([5, 1]))
        loss = nn.BCELoss(reduction='sum')
        res = loss(preds, targets).numpy()

        log_ = np.log(0.5)
        res_tru = -1 * np.sum(np.array([log_] * 5))
        np.testing.assert_array_almost_equal(res, res_tru, decimal=7)

if __name__ == '__main__':
    unittest.main()
