import torch

import os
import argparse
from configs.defaults import get_cfg_defaults

from utils.logger import setup_logger
from utils.preprocess import frames_preprocess
from models.model import CAT

from data.dataset import load_dataset
import torch.nn.functional as F
from utils.visualization import seq_similarity
from utils.loss import compute_cls_loss, compute_seq_loss
from utils.smoothDTW import compute_alignment_loss, smoothDTW

from smoothDTW_demo import softDTW

import pdb
import matplotlib.pyplot as plt
import time

# from sklearn.metrics import auc
# from sklearn.metrics import roc_curve

from tqdm import tqdm
import numpy as np


um_class_pairs = [[i, j]
                  for i in np.arange(25) for j in np.arange(25) if i < j]


def predict(model, threshold=1000, dist='L2'):

    # pdb.set_trace()
    start_time = time.time()

    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    auc_value = 0

    # auc metric
    with torch.no_grad():

        for iter, sample in enumerate(tqdm(test_loader)):
            # pdb.set_trace()
            frames_list1 = sample["clips1"]
            frames_list2 = sample["clips2"]
            # frames_list2 = sample["clips2"]
            assert len(frames_list1) == len(frames_list2)

            labels1 = sample["labels1"]
            labels2 = sample["labels2"]
            label = torch.tensor(np.array(labels1) ==
                                 np.array(labels2)).to(device)
            pred1 = 0
            pred2 = 0

            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i]).to(device)
                frames2 = frames_preprocess(frames_list2[i]).to(device)

                # Save frames as plot
                _, axs = plt.subplots(2, 16, figsize=(12, 2))

                _frames1 = frames1.cpu()
                _frames2 = frames2.cpu()

                for i in range(len(_frames1)):
                    axs[0, i].imshow(_frames1[i].T.permute(
                        1, 0, 2))
                    axs[1, i].imshow(_frames2[i].T.permute(
                        1, 0, 2))

                # one liner to remove *all axes in all subplots*
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.text(15, 300, labels2)
                plt.text(15, -200, labels1)
                plt.savefig("figs/frames_" + str(iter) + ".png",
                            dpi=300, bbox_inches='tight')

                _pred1, seq_features1 = model(frames1)
                _pred2, seq_features2 = model(frames2)
                pred1 += _pred1
                pred2 += _pred2

                dtm, dist = smoothDTW(
                    seq_features1[0], seq_features2[0], distance_type='cosine', softning='dtw_prob', gamma_s=10.0, gamma_f=10.0)

                seq_similarities = seq_similarity(seq_features1, seq_features2)

                _dtm = dtm[1:, 1:].cpu()
                _dist = dist.cpu()

                # Plot the dist matrix as a heatmap
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(_dtm, cmap='coolwarm')
                # Add grid lines and labels
                ax.set_xticks(np.arange(16))
                ax.set_yticks(np.arange(16))
                ax.set_xticklabels(np.arange(16))
                ax.set_yticklabels(np.arange(16))
                ax.tick_params(axis='both', labelsize=10, labelcolor='black')
                # Add the actual values to each grid item
                for i in range(16):
                    for j in range(16):
                        ax.text(j, i, np.round(
                            _dtm[i, j].item(), 2), ha='center', va='center', color='black', fontsize=4)
                # Loop over the rows of the tensor and highlight the minimum value in each row
                for i in range(16):
                    min_idx = np.argmin(_dtm[i, :])
                    ax.add_patch(plt.Rectangle(
                        (min_idx - 0.5, i-0.5), 1, 1, fill=True, color='green', alpha=1, zorder=1))
                # Set plot title and save the plot
                ax.set_title(
                    "DTW Matrix", fontsize=12)
                plt.savefig("figs/dtw_matrix_" + str(iter) + ".png",
                            dpi=300, bbox_inches='tight')

                # Plot the dist matrix as a heatmap
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(_dist, cmap='coolwarm')
                # Add grid lines and labels
                ax.set_xticks(np.arange(16))
                ax.set_yticks(np.arange(16))
                ax.set_xticklabels(np.arange(16))
                ax.set_yticklabels(np.arange(16))
                ax.tick_params(axis='both', labelsize=10, labelcolor='black')
                # Add the actual values to each grid item
                for i in range(16):
                    for j in range(16):
                        ax.text(j, i, np.round(
                            _dist[i, j].item(), 2), ha='center', va='center', color='black', fontsize=4)
                # Loop over the rows of the tensor and highlight the minimum value in each row
                for i in range(16):
                    min_idx = np.argmin(_dist[i, :])
                    ax.add_patch(plt.Rectangle(
                        (min_idx - 0.5, i-0.5), 1, 1, fill=True, color='green', alpha=1, zorder=1))
                # Set plot title and save the plot
                ax.set_title(
                    "Dist Matrix", fontsize=12)
                plt.savefig("figs/dist_matrix_" + str(iter) + ".png",
                            dpi=300, bbox_inches='tight')


def launch():

    model = CAT(num_class=cfg.DATASET.NUM_CLASS,
                num_clip=cfg.DATASET.NUM_CLIP,
                dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                pretrain=cfg.MODEL.PRETRAIN,
                dropout=cfg.TRAIN.DROPOUT,
                use_TE=cfg.MODEL.TRANSFORMER,
                use_SeqAlign=cfg.MODEL.ALIGNMENT).to(device)

    # assert args.root_path, logger.info('Please appoint the root path')

    if args.model_path == None:
        model_path = os.path.join(args.root_path, 'save_models')
    else:
        model_path = args.model_path

    start_time = time.time()

    if os.path.isfile(model_path):
        logger.info('To evaluate 1 models in %s with threshold %.4f' %
                    (model_path, args.threshold))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        auc_value = predict(model, args.threshold, args.dist)

    else:
        logger.info('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/test_config.yml',
                        help='config file path [default: configs/test_config.yml]')
    parser.add_argument('--model_path', default=None,
                        help='path to load one model [default: None]')
    parser.add_argument('--log_name', default='predict_log', help='log name')
    parser.add_argument('--threshold', type=float, default=1000.0,
                        help='threshold to distinguish match/unmatch pairs')
    parser.add_argument('--dist', default='NormL2')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    torch.manual_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logger = setup_logger("ActionVerification", 'temp_log', args.log_name, 0)
    logger.info("Running with config:\n{}\n".format(cfg))

    test_loader = load_dataset(cfg)

    launch()
