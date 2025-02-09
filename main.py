import torch
import numpy as np
import argparse
from mask import *
from load_and_split_data import set_seed, prepare_cross_validation_data
from model.DVAModel import DVA, VGAEEncoder1, VGAEEncoder2, GCNEncoder, GINEncoder, GATEncoder, SAGEEncoder, LPDecoder


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
from functools import partial

def parse_arguments():
    """
    此函数用于解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for model and dataset.")
    parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
    parser.add_argument('--p', type=float, default=0.5, help='Mask ratio')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=5)
    return parser.parse_args()

def initialize_DVAmodel_and_optimizer(args, num_features, model_name):
    """
    此函数用于初始化模型和优化器
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #network_input = torch.eye(N).to(device)
    #model_name = 'puregcn'
#    encoder = GCN(in_channels = num_features, hidden_channels=128, out_channels=128, num_layers = 1, ln = True, res = True, max_x = -1, jk = True, dropout=0.3, edrop = 0.3, xdropout=0.2, taildropout=0.3, noinputlin=False, conv_fn=model_name).to(device)
    
    if model_name == "SAGEEncoder":
        encoder = SAGEEncoder(in_channels = num_features, hidden_channels=128, out_channels=128).to(device)
    elif model_name == "GINEncoder":
        encoder = GINEncoder(in_channels = num_features, hidden_channels=128, out_channels=128).to(device)
    elif model_name == "GATEncoder":
        encoder = GATEncoder(in_channels = num_features, hidden_channels=128, out_channels=128).to(device)
    elif model_name == "GCNEncoder":
        encoder = GCNEncoder(in_channels = num_features, hidden_channels=128, out_channels=128).to(device)

    
    VEEncoder1 = VGAEEncoder1(in_channels=128, out_channels=128).to(device)
    VEEncoder2 = VGAEEncoder2(in_channels=num_features, out_channels=128).to(device)
    

    predictor = LPDecoder(in_channels=128 * 3, hidden_channels=64, out_channels=1, encoder_layer=1, num_layers=args.num_layers,
                        dropout=args.dropout)

    model = DVA(encoder, VEEncoder1, VEEncoder2, predictor, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    return model, optimizer, device


def initialize_model_and_optimizer(model_name, args, num_features, N):

    if model_name == 'DVA':
        return initialize_DVAmodel_and_optimizer(args, num_features, "SAGEEncoder")
    else:
        raise ValueError("Invalid database name. Choose from 'DVA'")

def train_model(splits_data, model_name, args, dim):

    fold_metrics = {'AUC': [], 'AUPR': [], "Acc": [], "Pre": [], "SEN": [], "F1": [], "MCC": []}
    all_tpr = []
    all_fpr = np.linspace(0, 1, 100)
    fold_aucs = []
    fold_auprcs = []
    mean_tpr = np.zeros_like(all_fpr)
    num_folds = len(splits_data)
    for fold_index, splits in enumerate(splits_data):
        model, optimizer, device = initialize_model_and_optimizer(model_name, args, dim, splits['train'].num_nodes)
        epoch_metrics_sum = {'AUC': 0, 'AUPR': 0, "Acc": 0, "Pre": 0, "SEN": 0, "F1": 0, "MCC": 0}
        best_loss = 1000
        for epoch in range(100):
            model.train()
            train_data = splits['train'].to(device)
            test_data = splits['test'].to(device)
            #x, edge_index = train_data.x, train_data.edge_index
            loss = model.train_epoch(splits['train'], test_data, optimizer, alpha=args.alpha, epoch=epoch)
            
        for epoch in range(1000):
            model.train()
            train_data = splits['train'].to(device)
            test_data = splits['test'].to(device)
            #x, edge_index = train_data.x, train_data.edge_index
            loss = model.train_epoch(splits['train'], test_data, optimizer, alpha=args.alpha, epoch=epoch)
            
            model.eval()
            #z = model.encoder(train_data.x, train_data.pos_edge_label_index)
            #test_auc, test_aupr, acc, pre, sen, F1, mcc, y_true, y_scores = model.test(z, test_data.pos_edge_label_index,
            #                                                         test_data.neg_edge_label_index)
            test_auc, test_aupr, acc, pre, sen, F1, mcc, y_true, y_scores = model.test(train_data, test_data, epoch)
            
            results = {
                'AUC': "{:.6f}".format(test_auc),
                'AUPR': "{:.6f}".format(test_aupr),
                "Acc": "{:.6f}".format(acc),
                "Pre": "{:.6f}".format(pre),
                "SEN": "{:.6f}".format(sen),
                "F1": "{:.6f}".format(F1),
                "MCC": "{:.6f}".format(mcc),
            }
            
            results1 = {
                'AUC': "{:.6f}".format(test_auc),
                'AUPR': "{:.6f}".format(test_aupr),
                "Acc": "{:.6f}".format(acc),
                "Pre": "{:.6f}".format(pre),
                "SEN": "{:.6f}".format(sen),
                "F1": "{:.6f}".format(F1),
                "MCC": "{:.6f}".format(mcc),
                "LOSS": "{:.6f}".format(loss),
            }
            #if loss < best_loss:
            #    best_loss = loss
            #    best_metrics = results
            #    print(results1)
            fold_metrics['AUC'].append(test_auc)
            fold_metrics['AUPR'].append(test_aupr)
            fold_metrics['Acc'].append(acc)
            fold_metrics['Pre'].append(pre)
            fold_metrics['SEN'].append(sen)
            fold_metrics['F1'].append(F1)
            fold_metrics['MCC'].append(mcc)
            for metric in epoch_metrics_sum:
                epoch_metrics_sum[metric] += float(results[metric])
        fold_avg_metrics = {metric: epoch_metrics_sum[metric] / 1000 for metric in epoch_metrics_sum}
        print(f'Fold {fold_index + 1} average metrics: {fold_avg_metrics}')
        
        for metric, avg_value in fold_avg_metrics.items():
            fold_metrics[metric].append(avg_value)
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        fold_aucs.append(roc_auc)
        all_tpr.append(tpr)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    avg_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
    print(f'Average metrics: {avg_metrics}')
    return fold_metrics, all_tpr, all_fpr, fold_aucs, mean_tpr


def plot_roc_curve(all_tpr, all_fpr, fold_aucs, mean_tpr, num_folds):

    plt.figure(figsize=(10, 8))
    for fold_index in range(num_folds):
        plt.plot(all_fpr, all_tpr[fold_index], label=f'Fold {fold_index + 1} (AUC = {fold_aucs[fold_index]:.4f})')
    mean_tpr /= num_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='red', label=f'Mean ROC ', lw=2, alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each fold and Mean Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():

    warnings.filterwarnings("ignore")
    databases = ['Disbiome', 'HMDAD']
    is_processes = [False]
    model_names = ['DVA']
    for database in databases:
        for is_process in is_processes:
            for model_name in model_names:
                print(database, is_process, model_name)
                args = parse_arguments()
                set_seed(args.seed)
                splits_data, dim = prepare_cross_validation_data(database, is_process)
                fold_metrics, all_tpr, all_fpr, fold_aucs, mean_tpr = train_model(splits_data, model_name, args, dim)


if __name__ == "__main__":
    main()