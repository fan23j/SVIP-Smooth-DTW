from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

import numpy as np
from utils.config import CONFIG
from utils.loss import compute_cls_loss
import pdb

# This code is shared under the
# Attribution-NonCommercial-ShareAlike 4.0 International
# Please find the full license in the main directory under LICENSE.MD

"""
Definition of our SmoothDTW-based alignment loss
@author: isma.hadji
@email: isma.hadji@samsung.com
"""

"""definition of the loss"""

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def DTW_loss(logits, loss_type, batch_size, beta=1):
    """calculate loss function given DTW table"""
    if 'D2TW' in loss_type:
        """non-discriminative DTW loss"""
        loss = torch.mean(logits)
    else:
        raise ValueError('%s is an unsupported DTW loss' % loss_type)
    return loss


def assign2Tensor(tensor, i, j, new_val):
    """ function to deal with torch.Tensors being non-assignable """
    # create mask
    mask = np.ones(tensor.shape, dtype=np.float32)
    # hack to assign a new value to tensor at position (i,j)
    mask[i, j] = 0
    mask = torch.tensor(mask, dtype=torch.float32).to(device)

    tensor = (tensor.to(device)*mask) + (new_val * (1-mask))
    return tensor


def smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f):
    """function to obtain a soft (differentiable version of DTW)"""
    # first get a pairwise distance matrix
    if distance_type == 'cosine':
        dist = torch.matmul(embs1, embs2.t())
    else:
        raise ValueError(
            'distance_type % s not supported for now' % distance_type)
    # normalize distance column-wise
    if CONFIG.D2TW_NORM:
        dist = -torch.log(torch.softmax(dist/gamma_f, dim=0))
    nrows, ncols = dist.shape
    # calculate soft-DTW table
    sdtw = torch.zeros((nrows+1, ncols+1), dtype=torch.float32)
    # obtain dtw table using min_gamma or prob relaxation
    for i in range(0, nrows+1):
        for j in range(0, ncols+1):
            if (i == 0) and (j == 0):
                new_val = torch.tensor(0.0, dtype=torch.float32)
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            elif (i == 0) and (j != 0):
                new_val = torch.finfo(torch.float32).max
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            elif (i != 0) and (j == 0):
                new_val = torch.finfo(torch.float32).max
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            else:
                neighbors = torch.stack(
                    [sdtw[i, j-1], sdtw[i-1, j-1], sdtw[i-1, j]])
                if softning == 'dtw_minGamma':
                    new_val = dist[i-1, j-1] + minGamma(neighbors, gamma_s)
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                elif softning == 'dtw_prob':
                    probs = torch.softmax((-neighbors)/gamma_s, dim=0)
                    new_val = dist[i-1, j-1] + (probs[0] * sdtw[i, j-1]) + (
                        probs[1] * sdtw[i-1, j-1]) + (probs[2] * sdtw[i-1, j])
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                elif softning == 'non-diff':
                    new_val = dist[i-1, j-1] + torch.min(torch.stack(
                        [sdtw[i, j-1], sdtw[i-1, j-1], sdtw[i-1, j]]))
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                else:
                    raise ValueError(
                        'only softning based on dtw_minGamma or dtw_prob supported for now.')
    return sdtw, dist


def minGamma(inputs, gamma=1):
    """ continuous relaxation of min defined in the D3TW paper"""
    if gamma == 0:
        minG, _ = torch.min(inputs, dim=1)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi, _ = torch.max(zi, dim=1, keepdim=True)
        log_sum_G = max_zi + \
            torch.log(torch.sum(torch.exp(zi-max_zi),
                      dim=1, keepdim=True))  # + 1e-10)
        minG = -gamma * log_sum_G.squeeze()
    return minG


def compute_dtw_alignment_loss(embs,
                               batch_size,
                               loss_type,
                               distance_type,
                               softning,
                               gamma_s,
                               gamma_f):
    """Compute d2tw loss for all steps in each sequence.
    Args:
        embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
            batch size, T is the number of timesteps in the sequence, D is the size
            of the embeddings.
        loss_type: define the loss type used in our dtw alignment
        distance_type: String, Currently supported distance metrics: ‘cosine’
        softning: relaxation used for dtw. currently supported: ‘dtw_minGamma’ and ‘dtw_prob’
    Returns:
        loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
            dtw loss.
    """
    logits_list = []
    i = 0
    for j in range(i+1, batch_size):
        logits, _ = smoothDTW(
            embs[i], embs[j], distance_type, softning, gamma_s, gamma_f)
        logits_list.append(logits[-1, -1])
    logits = torch.stack(logits_list, dim=0)
    # calculate the loss
    loss = DTW_loss(logits, loss_type, batch_size)
    return loss


# def compute_dtw_alignment_consistency_loss(embs, batch_size, loss_type, distance_type, softning, gamma_s, gamma_f, label_smoothing):
#     """Compute d2tw loss with Global Cycle Consistency for all steps in each sequence.
#     Args:
#         embs: Tensor, sequential embeddings of the shape[N, T, D] where N is the
#         batch size, T is the number of timesteps in the sequence, D is the size
#         of the embeddings.
#         loss_type: define the loss type used in our dtw alignment
#         distance_type: String, Currently supported distance metrics: 'cosine'
#         softning: relaxation used for dtw. currently supported: 'dtw_minGamma' and 'dtw_prob'
#     Returns:
#         loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
#         dtw loss.
#     """
#     logits_list = []
#     logits_ij_list = []
#     logits_ji_list = []
#     labels_list = []
#     i = 0
#     if CONFIG.MODE == 'train':
#         skip = CONFIG.TRAIN.SKIP_FRAMES
#     else:
#         skip = 1
#     for j in range(i+1, batch_size):
#         logits_ij, _ = smoothDTW(
#             embs[i, ::skip, :], embs[j], distance_type, softning, gamma_s, gamma_f)
#         logits_ij_list.append(logits_ij[-1, -1])
#         logits_ij = F.softmax(-logits_ij[1:, 1:], dim=0)
#         logits_ji, _ = smoothDTW(
#             embs[j], embs[i, ::skip, :], distance_type, softning, gamma_s, gamma_f)
#         logits_ji_list.append(logits_ji[-1, -1])
#         logits_ji = F.softmax(-logits_ji[1:, 1:], dim=0)
#         if CONFIG.REVCONS:
#             logits = torch.matmul(logits_ij, logits_ji)
#             # transpose to make sure that the each row sums to 1 (to use categorical cross entropy loss that reads tensors by rows)
#             logits = logits.T
#             logits_list.append(logits)
#             labels = torch.eye(logits.shape[0])
#             labels_list.append(labels)
#     if CONFIG.REVCONS:
#         logits = torch.cat(logits_list, dim=0)
#         labels = torch.cat(labels_list, dim=0)
#     logits_ij_list = torch.stack(logits_ij_list, dim=0)
#     logits_ji_list = torch.stack(logits_ji_list, dim=0)
#     # calculate the loss
#     loss_sdtw_ij = DTW_loss(logits_ij_list, loss_type, batch_size)
#     loss_sdtw_ji = DTW_loss(logits_ji_list, loss_type, batch_size)
#     if CONFIG.REVCONS:
#         loss_con = compute_cls_loss(logits, labels, label_smoothing)
#         loss = loss_con + 0.1*loss_sdtw_ij + 0.1*loss_sdtw_ji
#     else:
#         loss = 0.1*loss_sdtw_ij + 0.1*loss_sdtw_ji
#     return loss

def compute_dtw_alignment_consistency_loss(embs1, embs2, batch_size, loss_type, distance_type, softning, gamma_s, gamma_f, label_smoothing):
    """Compute d2tw loss with Global Cycle Consistency for all steps in each sequence.
    Args:
        embs: Tensor, sequential embeddings of the shape[N, T, D] where N is the
        batch size, T is the number of timesteps in the sequence, D is the size
        of the embeddings.
        loss_type: define the loss type used in our dtw alignment
        distance_type: String, Currently supported distance metrics: 'cosine'
        softning: relaxation used for dtw. currently supported: 'dtw_minGamma' and 'dtw_prob'
    Returns:
        loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        dtw loss.
    """
    logits_list = []
    logits_ij_list = []
    logits_ji_list = []
    labels_list = []
    # pudb.set_trace()
    i = 0
    if CONFIG.MODE == 'train':
        skip = CONFIG.TRAIN.SKIP_FRAMES
    else:
        skip = 1
    for j in range(i+1, batch_size):
        # embs1[i, ::skip, :] => size [16,2048] tensor, equivalent to extracting a single
        # item in the batch. We compare this to another sample from the other sequence.
       # pdb.set_trace()
        logits_ij, _ = smoothDTW(
            embs1[i, ::skip, :], embs2[j], distance_type, softning, gamma_s, gamma_f)
        logits_ij_list.append(logits_ij[-1, -1])
        logits_ij = F.softmax(-logits_ij[1:, 1:], dim=0)
        logits_ji, _ = smoothDTW(
            embs1[j], embs2[i, ::skip, :], distance_type, softning, gamma_s, gamma_f)
        logits_ji_list.append(logits_ji[-1, -1])
        logits_ji = F.softmax(-logits_ji[1:, 1:], dim=0)
        logits = torch.matmul(logits_ij, logits_ji)
        # transpose to make sure that the each row sums to 1 (to use categorical cross entropy loss that reads tensors by rows)
        logits = logits.T
        logits_list.append(logits)
    logits = torch.cat(logits_list, dim=0)
    logits_ij_list = torch.stack(logits_ij_list, dim=0)
    logits_ji_list = torch.stack(logits_ji_list, dim=0)
    # calculate the loss
    loss_sdtw_ij = DTW_loss(logits_ij_list, loss_type, batch_size)
    loss_sdtw_ji = DTW_loss(logits_ji_list, loss_type, batch_size)
    loss = 0.1*loss_sdtw_ij + 0.1*loss_sdtw_ji
    return loss

# %% COMPUTE ALIGNMENT LOSS


def compute_alignment_loss(embs1,
                           embs2,
                           batch_size,
                           alignment_type='D2TW_consistency',
                           loss_type='D2TW_consistency',
                           similarity_type='cosine',
                           label_smoothing=0.1,
                           softning='dtw_prob',
                           gamma_s=50.0,
                           gamma_f=50.0):
    """Computes DTW alignment loss between sequences of embeddings."""

    loss = compute_dtw_alignment_consistency_loss(embs1=embs1,
                                                  embs2=embs2,
                                                  batch_size=batch_size,
                                                  loss_type=loss_type,
                                                  distance_type=similarity_type,
                                                  softning=softning,
                                                  gamma_s=gamma_s,
                                                  gamma_f=gamma_f,
                                                  label_smoothing=label_smoothing
                                                  )
    return loss
