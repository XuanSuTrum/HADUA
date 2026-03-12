import argparse
import os
import utils
import numpy as np
import torch
from get_dataset import get_dataset
from load_data2 import load_data
# from load_data2_multi_eeg import create_domain_loaders
from load_data2_multi_eye import create_domain_loaders
import math
import torch
import SDA_DDA
import SDA_DDA_2
import SDA_DDA_3
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn import init
# import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score

# pd.set_option('display.max_rows', None)  # 显示全部行
# pd.set_option('display.max_columns', None)  # 显示全部列
np.set_printoptions(threshold=np.inf)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(-1)
cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log = []

# Command setting
parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--model', type=str, default='CFE')
parser.add_argument('--batchsize', type=int, default=256)
# parser.add_argument('--src', type=str, default='amazon')
# parser.add_argument('--tar', type=str, default='webcam')
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--lamb', type=float, default=0.5)
parser.add_argument('--trans_loss', type=str, default='mmd')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
args = parser.parse_args(args=[])


def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weigth_init(m):  ## model parameter initialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()


def segmented_function(epoch):
    if epoch <= 10:
        value = 1
    elif 10 < epoch <= 40:
        # 在10-30之间逐渐减小值，你可以根据需要调整
        value = 2 / (1 + math.exp(-10 * (args.n_epoch) / args.n_epoch)) - 1
    elif 40 < epoch <= 85:
        # 在10-30之间逐渐减小值，你可以根据需要调整
        value = 1 * np.exp(-0.6 * epoch)
    else:
        value = 0.01
    return value


def segmented_function_1(epoch):
    if epoch <= 40:
        value = 0.65
    else:
        value = 1
    return value


def tt(model, target_test_loader):
    model.eval()
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    num_classes = args.n_class
    conf_matrix = np.zeros((num_classes, num_classes))
    all_preds = []
    all_targets = []
    all_probs = []  # For AUC calculation

    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            probs = F.softmax(s_output, dim=1)  # Get softmax probabilities for AUC
            pred = torch.max(s_output, 1)[1]
            target = torch.argmax(target, dim=1)

            correct += torch.sum(pred == target)
            conf_matrix += confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=np.arange(num_classes))

            # Collect predictions, targets, and probabilities
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    acc = 100. * correct / len_target_dataset

    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    # Calculate AUC (multi-class, one-vs-rest)
    try:
        auc = roc_auc_score(all_targets, np.array(all_probs), multi_class='ovr')
    except ValueError:
        auc = 0.0  # Handle cases where AUC cannot be computed

    return acc, pred, conf_matrix, precision, f1, auc


def train(source_loader, target_train_loader, target_test_loader, model, optimizer):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    best_acc = 0
    best_precision = 0
    best_f1 = 0
    best_auc = 0
    stop = 0
    best_confusion_matrix = None
    for e in range(args.n_epoch):
        data_target_ = []
        data_source_ = []
        t_label_s = []
        s_label_s = []
        all_pseudo_labels = []
        stop += 1
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_cmmd_loss = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)

        criterion = torch.nn.CrossEntropyLoss()
        for mlen in range(n_batch):
            data_source, label_source = next(iter_source)
            data_target, label_target = next(iter_target)
            if mlen % len(target_train_loader) == 0:
                iter_target = iter(target_train_loader)
            if cuda:
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()

            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target, label_target = data_target.to(DEVICE), label_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss, cmmd_loss = model(e, data_source, data_target, label_source)

            clf_loss = criterion(label_source_pred, label_source.float())

            if args.gamma == 1:
                gamma = 2 / (1 + math.exp(-10 * (args.n_epoch) / args.n_epoch)) - 1
            if args.gamma == 2:
                gamma = args.n_epoch / args.n_epoch

            beta = segmented_function(e)
            beta_1 = segmented_function_1(e)

            # 添加条件判断，如果clf_loss低于0.4，cmmd_loss前的权重为1，否则为0.1
            if clf_loss <= 0.1:
                cmmd_weight = 1
            elif 0.1 < clf_loss < 0.15:
                cmmd_weight = 0.5
            else:
                cmmd_weight = 0

            # loss = clf_loss + beta * transfer_loss + cmmd_weight * cmmd_loss
            loss = clf_loss + transfer_loss

            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_cmmd_loss.update(cmmd_loss.item())
            train_loss_total.update(loss.item())

            data_source_.append(data_source)
            data_target_.append(data_target)
            s_label_s.append(label_source)
            t_label_s.append(label_target)

        # Test
        acc, pred, conf_matrix, precision, f1, auc = tt(model, target_test_loader)
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_cmmd_loss.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('F:\\Emotion_datasets\\SEED\\train_log.csv', np_log, delimiter=',', fmt='%.6f')

        if best_acc < acc:
            best_acc = acc
            best_confusion_matrix = conf_matrix
            best_precision = precision
            best_f1 = f1
            best_auc = auc

    print('Transfer result: Acc: {:.4f}, Precision: {:.4f}, F1: {:.4f}, AUC: {:.4f}'.format(
        best_acc, best_precision, best_f1, best_auc))
    print('Confusion Matrix:\n', best_confusion_matrix)
    return best_acc, best_confusion_matrix, best_precision, best_f1, best_auc

if __name__ == '__main__':
    all_test_results = []
    all_matrix = []
    all_precision = []
    all_f1 = []
    all_auc = []

    for test_id in range(1, 13):  # 12 subjects
        print(f"\nProcessing test_id: {test_id}")
        torch.manual_seed(0)
        SESSION = 1
        batch_size = 128
        source_loader, target_train_loader, target_test_loader = create_domain_loaders(test_id, batch_size)

        model = SDA_DDA_3.Transfer_Net(
            args.n_class,
            transfer_loss=args.trans_loss,
            base_net=args.model,
            base_net_eye='CFE_eye',
            num_hiddens=128,
            num_heads=16
        ).to(DEVICE)

        optimizer = torch.optim.Adam([
            {'params': model.base_network.parameters(), 'lr': args.lr * 0.1},
            {'params': model.base_network_eye.parameters(), 'lr': args.lr * 0.1},
            {'params': model.self_attention_eeg.parameters(), 'lr': args.lr},
            {'params': model.self_attention_eye.parameters(), 'lr': args.lr},
            {'params': model.cross_attention.parameters(), 'lr': args.lr},
        ], lr=args.lr, weight_decay=1e-3)

        transfer_results, matrix, precision, f1, auc = train(source_loader, target_train_loader, target_test_loader,
                                                             model, optimizer)
        all_test_results.append(transfer_results)
        all_matrix.append(matrix)
        all_precision.append(precision)
        all_f1.append(f1)
        all_auc.append(auc)

    # Aggregate results
    stacked_results = torch.tensor(all_test_results)
    average_result = torch.mean(stacked_results)
    std_result = torch.std(stacked_results)
    avg_precision = np.mean(all_precision)
    std_precision = np.std(all_precision)
    avg_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1)
    avg_auc = np.mean(all_auc)
    std_auc = np.std(all_auc)
    avg_conf_matrix = sum(all_matrix) / len(all_matrix)

    # Print results
    print("\nFinal Results:")
    print(f"Average Accuracy: {average_result:.4f} ± {std_result:.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print("All Accuracies:", stacked_results.tolist())
    print("Average Confusion Matrix:\n", avg_conf_matrix)

    # Calculate percentage confusion matrix
    row_sums = avg_conf_matrix.sum(axis=1, keepdims=True)
    percentage_conf_matrix = (avg_conf_matrix / row_sums) * 100
    print("Percentage Confusion Matrix:\n", percentage_conf_matrix)