import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import config

dtype = config.dtype


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
    batch_size = gen_frames.shape[0]

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    # if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, torch.ones([batch_size, 1]))
    loss += lam_adv * adv_loss(d_preds, torch.ones([batch_size, 4]).type(dtype))

    return loss


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
        loss = nn.L1Loss()
        return loss(gen_frames, gt_frames)
    elif l_num == 2:
        loss = nn.MSELoss()
        return loss(gen_frames, gt_frames)
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
        [[[[-1, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]],
        [[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]],
        [[[0, 0, 0]], [[0, 0, 0]], [[-1, 1, 0]]]], dtype=np.float32)
    filter_x = nn.Conv2d(3, 3, (1, 3), padding=(0, 1))
    filter_x.weight = nn.Parameter(torch.from_numpy(filter_x_values))

    filter_y_values = np.array(
        [[[[-1], [1], [0]], [[0], [0], [0]], [[0], [0], [0]]],
        [[[0], [0], [0]], [[-1], [1], [0]], [[0], [0], [0]]],
        [[[0], [0], [0]], [[0], [0], [0]], [[-1], [1], [0]]]], dtype=np.float32)
    filter_y = nn.Conv2d(3, 3, (3, 1), padding=(1, 0))
    filter_y.weight = nn.Parameter(torch.from_numpy(filter_y_values))

    filter_x = filter_x.type(dtype)
    filter_y = filter_y.type(dtype)

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total = torch.stack([grad_diff_x, grad_diff_y])

    return torch.mean(grad_total)


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.
    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).
    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    loss = nn.BCELoss(size_average=True)
    return loss(preds, labels)
