import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import numpy as np

from tfutils import log10
import constants as c

def combined_loss(gen_frames, gt_frames, d_preds, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
    """
    Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
    for training the generative model.
    @param gen_frames: A list of tensors of the generated frames at each scale.
    @param gt_frames: A list of tensors of the ground truth frames at each scale.
    @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                    scale.
    @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
    @param lam_lp: The percentage of the lp loss to use in the combined loss.
    @param lam_gdl: The percentage of the GDL loss to use in the combined loss.
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @param alpha: The power to which each gradient term is raised in GDL loss.
    @return: The combined adversarial, lp and GDL losses.
    """

    # TODO: get the batch size in pytorch
    # batch_size = tf.shape(gen_frames[0])[0]  # variable batch size as a tensor

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, torch.ones([batch_size, 1]))

    return loss


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.
    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.
    @return: The sum of binary cross-entropy losses.
    """
    return nn.BCELoss(preds, targets)
    # return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
    #                         tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def lp_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @return: The lp loss.
    """
    # calculate the loss for each scale
    if l_num == 1:
        return nn.L1Loss(gen_frames, gt_frames)
    elif l_num == 2:
        return nn.MSELoss(gen_frames, gt_frames)
    else:
        print("Not supported!")
        exit()


def gdl_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.
    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.
    @return: The GDL loss.
    """
    filter_x_values = np.array(
        [[[[-1, 1]], [[0, 0]], [[0, 0]]],
        [[[0, 0]], [[-1, 1]], [[0, 0]]],
        [[[0, 0]], [[0, 0]], [[-1, 1]]]], dtype=np.float32)
    filter_x = nn.Conv2d(3, 3, (1, 2))
    filter_x.weight = nn.Parameter(torch.from_numpy(filter_x_values))

    filter_y_values = np.array(
        [[[[-1], [1]], [[0], [0]], [[0], [0]]],
        [[[0], [0]], [[-1], [1]], [[0], [0]]],
        [[[0], [0]], [[0], [0]], [[-1], [1]]]], dtype=np.float32)
    filter_y = nn.Conv2d(3, 3, (2, 1))
    filter_y.weight = nn.Parameter(torch.from_numpy(filter_y_values))

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total = torch.stack(grad_diff_x, grad_diff_y)

    return torch.mean(grad_total)


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.
    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).
    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    loss = nn.BCE_loss(size_average=True)
    scale_losses = []
    for i in xrange(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    # condense into one tensor and avg
    return torch.mean(scale_losses)
# return tf.reduce_mean(tf.pack(scale_losses))