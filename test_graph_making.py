import os
from torch.autograd import Variable
import sys
from test_dataset import LungDataset
from unet import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
from coatention_model import CoattentionModel
from unet import UNet

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = img - (min)
    img = img / (max - min)
    return img


def evaluate_metrics(gt, pred_mask):
    intersect = np.sum(pred_mask * gt)
    union = np.sum(pred_mask) + np.sum(gt) - intersect
    sum_gt = np.sum(gt)
    sum_pred = np.sum(pred_mask)
    total_sum = sum_pred + sum_gt
    xor = np.sum(gt == pred_mask)
    gt_invers = 1 - gt
    TN = np.sum(pred_mask * gt_invers)
    # dice = round(np.mean(2 * intersect / total_sum), 3)
    # iou = round(np.mean(intersect / union), 3)
    # acc = round(np.mean(xor / (union + xor - intersect)), 3)
    # recall = round(np.mean(intersect / sum_gt), 3)
    # precision = round(np.mean(intersect / sum_pred), 3)

    dice = np.mean(2 * intersect / total_sum)
    iou = np.mean(intersect / union)
    acc = np.mean(xor / (union + xor - intersect))
    recall = np.mean(intersect / sum_gt)
    precision = np.mean(intersect / sum_pred)

    return {'dice': dice, 'iou': iou, 'acc': acc, 'recall': recall, 'precision': precision}


train_path = Path("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train")

val_path = Path("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/val")
val_dataset = LungDataset(val_path, None,num_nodes=3)
# val_dataset = LungDataset(val_path, None,is_val=True,one_patient='3160145')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

model = CoattentionModel().cuda()



model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

evaluate = open('./logs/evaluate_graph_making.txt', 'a')
des_addr = './logs/seg_result/'

loss_fn = torch.nn.BCEWithLogitsLoss()
max_dice = 0
min_BCE = 100
THRESHOLD = 0.5
for e in range(17,30):
    cnt = 0
    BCE_log = []
    weights = torch.load('/home/fumcomp/Desktop/poorsoltani/min_distance_neighbour/epoch_' + str(e) + '.pth', map_location='cuda')['model']
    model.load_state_dict(weights)
    preds = []
    labels = []
    step = 0
    for img, label in tqdm(val_loader):
        step += 1
        with torch.no_grad():
            node0 = torch.tensor(img[0]).cuda()
            node1 = torch.tensor(img[1]).cuda()
            node2 = torch.tensor(img[2]).cuda()

            label0 = torch.tensor(label[0]).cuda()
            label1 = torch.tensor(label[1]).cuda()
            label2 = torch.tensor(label[2]).cuda()

            pred0, pred1, pred2 = model(node0, node1, node2)
            pred = pred0

        pred = pred > 0.5
        pred = pred.squeeze().cpu().numpy()
        preds.append(pred)
        labels.append(label[0].squeeze().cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    eval_metrics = evaluate_metrics(labels, preds)

    # mean_BCE = np.mean(BCE_log)
    evaluate.write('################### epoch {} ###############################\n'.format(e))
    for key in eval_metrics:
        evaluate.write('{}: {}\t'.format(key, eval_metrics[key]))
    evaluate.write('\n')
    evaluate.write('############################################################\n')
    evaluate.flush()

    evaluate.flush()
    if eval_metrics['dice'] > max_dice:
        max_dice = eval_metrics['dice']

evaluate.write('max dice is= {}\t min BCE is= {}\n'.format(max_dice, min_BCE))
evaluate.flush()
