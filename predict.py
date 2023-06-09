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
from utils.smoothDTW import compute_alignment_loss

import pdb
import pudb
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
            frames_list2 = sample["clips1"]
            # frames_list2 = sample["clips2"]
            assert len(frames_list1) == len(frames_list2)

            labels1 = sample["labels1"]
            labels2 = sample["labels2"]
            label = torch.tensor(np.array(labels1) ==
                                 np.array(labels2)).to(device)
            pred1 = 0
            pred2 = 0

            seq_similarities = []

            for i in range(len(frames_list1)):

                frames1 = frames_preprocess(frames_list1[i]).to(device)
                frames2 = frames_preprocess(frames_list2[i]).to(device)

                _pred1, seq_features1 = model(frames1)
                _pred2, seq_features2 = model(frames2)
                pred1 += _pred1
                pred2 += _pred2

                seq_similarities = seq_similarity(seq_features1, seq_features2)
                pudb.set_trace()
                # loss_cls = compute_cls_loss(
                #     _pred1, labels1) + compute_cls_loss(_pred2, labels2)
                loss = compute_alignment_loss(
                    seq_features1, seq_features2, 1
                )

            pudb.set_trace()
            pred1 /= len(frames_list1)
            pred2 /= len(frames_list2)

            if dist == 'L1':
                # L1 distance
                pred = torch.sum(torch.abs(pred1 - pred2), dim=1)
            elif dist == 'L2':
                # L2 distance
                pred = torch.sum((pred1 - pred2) ** 2, dim=1)
            elif dist == 'NormL2':
                # L2 distance between normalized embeddings
                pred = torch.sum(
                    (F.normalize(pred1, p=2, dim=1) - F.normalize(pred2, p=2, dim=1)) ** 2, dim=1)
            elif dist == 'cos':
                # Cosine similarity
                pred = torch.cosine_similarity(pred1, pred2, dim=1)

            if iter == 0:
                preds = pred
                # labels = label
                labels1_list = labels1
                labels2_list = labels2
            else:
                preds = torch.cat([preds, pred])
                # labels = torch.cat([labels, label])
                labels1_list += labels1
                labels2_list += labels2

    # match indices & unmatch indices
    m_idx = labels1_list == labels2_list
    um_idx = labels1_list != labels2_list
    m_preds = preds[m_idx]
    um_preds = preds[um_idx]
    m_labels = labels1_list[m_idx]
    # um_labels = torch.cat([labels1_list[um_idx].unsqueeze(0), labels2_list[um_idx].unsqueeze(0)]).transpose(0, 1)
    um_labels = torch.cat(
        [labels1_list[um_idx], labels2_list[um_idx]]).transpose(0, 1)
    labels = labels1_list == labels2_list

    # Predict on all matched pairs
    m_accs = []
    for label in range(25):
        selected_preds = m_preds[m_labels == label]
        acc = torch.sum(selected_preds < threshold) / selected_preds.size(0)
        m_accs.append(acc.item())
        logger.info('[MATCH] accuracy %.4f with label %d over %d samples' % (
            acc.item(), label, selected_preds.size(0)))
    logger.info('[MATCH] TOTAL match accuracy %.4f over %d samples' % (
        torch.sum(m_preds < threshold)/m_preds.size(0), m_preds.size(0)))
    m_accs = np.nan_to_num(m_accs)

    # Print top-5 verification accuracy
    for i in range(5):
        logger.info('[MATCH] RANK %d accuracy is %.4f with label %d' %
                    (i+1, np.sort(m_accs)[-i-1], np.argsort(m_accs)[-i-1]))

    # Predict on all unmatched pairs
    um_accs = []
    for label in um_class_pairs:
        index = um_labels.cpu() == torch.tensor(label)
        index = index[:, 0] * index[:, 1]
        selected_preds = um_preds[index]
        acc = torch.sum(selected_preds > threshold) / selected_preds.size(0)
        um_accs.append(acc.item())
        logger.info('[UNMATCH] accuracy %.4f with label (%d, %d) caculated from %d samples' % (
            acc.item(), label[0], label[1], selected_preds.size(0)))
    logger.info('[UNMATCH] TOTAL unmatch accuracy %.4f over %d samples' % (
        torch.sum(um_preds > threshold)/um_preds.size(0), um_preds.size(0)))
    um_accs = np.nan_to_num(um_accs)

    # Print top-20 verification accuracy
    for i in range(20):
        label = um_class_pairs[np.argsort(um_accs)[-i-1]]
        logger.info('[UNMATCH] RANK %d accuracy is %.4f with label (%d, %d)' % (
            i+1, np.sort(um_accs)[-i-1], label[0], label[1]))

    # Print top-5 intra-group verification accuracy
    counter = 0
    for i in range(um_preds.size(0)):
        if counter == 10:
            break
        label = um_class_pairs[np.argsort(um_accs)[-i-1]]
        if label[0] // 5 == label[1] // 5:
            # Inter-group pair
            logger.info('[UNMATCH-Intra] RANK %d accuracy is %.4f with label (%d, %d)' % (
                counter + 1, np.sort(um_accs)[-i - 1], label[0], label[1]))
            counter += 1
        else:
            continue

    # Print top-5 inter-group verification accuracy
    counter = 0
    for i in range(um_preds.size(0)):
        if counter == 10:
            break
        label = um_class_pairs[np.argsort(um_accs)[-i - 1]]
        if label[0] // 5 != label[1] // 5:
            # Inter-group pair
            logger.info('[UNMATCH-Inter] RANK %d accuracy is %.4f with label (%d, %d)' % (
                counter + 1, np.sort(um_accs)[-i - 1], label[0], label[1]))
            counter += 1
        else:
            continue

    logger.info('[ALL] OVERALL accuracy %.4f over %d samples' % (
        torch.sum((preds < threshold) == labels)/labels.size(0), labels.size(0)))

    fpr, tpr, thresholds = roc_curve(labels.cpu().detach(
    ).numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value = auc(fpr, tpr)
    logger.info('[ALL] OVERALL auc is %.4f over %d samples' %
                (auc_value, labels.size(0)))

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Predict costs %dh%dm%ds' % (hour, min, sec))
    pdb.set_trace()

    return auc_value


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
        logger.info("Model is %s, AUC is %.4f" % (model_path, auc_value))

    else:
        logger.info('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Evaluate %d models cost %dh%dm%ds' %
                (len(os.listdir(model_path)), hour, min, sec))


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
