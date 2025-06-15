import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")
        print(f"Trainer using device: {self.device}")

        self.val_eval = Evaluator(params, self.data_loader['val'], self.device)
        self.test_eval = Evaluator(params, self.data_loader['test'], self.device)

        self.model = model.to(self.device)

        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).to(self.device)
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().to(self.device)
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().to(self.device)

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                param.requires_grad = not params.frozen
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.item())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    f"Epoch {epoch+1} : Training Loss: {np.mean(losses):.5f}, acc: {acc:.5f}, "
                    f"kappa: {kappa:.5f}, f1: {f1:.5f}, LR: {optim_state['param_groups'][0]['lr']:.5f}, "
                    f"Time elapsed {(timer() - start_time) / 60:.2f} mins"
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !!")
                    print(f"Val Evaluation: acc: {acc:.5f}, kappa: {kappa:.5f}, f1: {f1:.5f}")
                    best_f1_epoch = epoch + 1
                    acc_best, kappa_best, f1_best, cm_best = acc, kappa, f1, cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(f"Test Evaluation: acc: {acc:.5f}, kappa: {kappa:.5f}, f1: {f1:.5f}")
            print(cm)

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = os.path.join(
                self.params.model_dir,
                f"epoch{best_f1_epoch}_acc_{acc:.5f}_kappa_{kappa:.5f}_f1_{f1:.5f}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            print("model saved in " + model_path)
