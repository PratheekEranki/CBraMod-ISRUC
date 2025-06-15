import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader, device=None):
        self.params = params
        self.data_loader = data_loader
        self.device = device or torch.device("cpu")

    def get_metrics_for_multiclass(self, model):
        model.eval()
        model.to(self.device)

        truths = []
        preds = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()
        model.to(self.device)

        truths = []
        preds = []
        scores = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)

        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()
        model.to(self.device)

        truths = []
        preds = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)

        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse
