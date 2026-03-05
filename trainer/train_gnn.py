from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from dgl.dataloading import GraphDataLoader

from trainer import Trainer
from evaluator import HomoGraphEvaluator
from parser import parse_optimizer, parse_gnn_model, parse_loss
from utils import acc, metrics
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset
import os


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict, ckpt_path=None):
        super().__init__(config)

        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.gnn.load_state_dict(checkpoint, strict=False )
            print(f"Loaded pretrained weights from {ckpt_path}")

        # Parse loss function
        self.loss_fcn = parse_loss(self.config_train)

        train_path = self.config_data["train_path"]
        self.valid_path = self.config_data["valid_path"]

        name = self.config_data["dataset"]
        normal_path = self.config_data["normal_path"] if (name == "COAD" or name == "BRCA" or name == "ESCA") else ""
        task = self.config_data["task"]
        if name == "COAD" and task == "cancer staging":
            self.average = "macro"
            train_data = TCGACancerStageDataset(train_path, normal_path, 'train')
        elif name == "BRCA" and task == "cancer staging":
            self.average = "macro"
            train_data = TCGACancerStageDataset(train_path, normal_path, 'train')
        elif (name == "BRCA" or name == "ESCA") and task == "cancer typing":
            self.average = "binary"
            train_data = TCGACancerTypingDataset(train_path, normal_path, 'train')
        else:
            self.average = "binary"
            train_data = GraphDataset(train_path, normal_path, name, 'train')

        self.dataloader = GraphDataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def train_one_step(self, graphs, label, epoch):
        self.optimizer.zero_grad()
        label = label.to(self.device)
        graphs = graphs.to(self.device)
        pred, node_attn, he_attn = self.gnn(graphs, return_attention=True)  # 返回节点注意力

        if epoch >= 0:
            with torch.no_grad():
                G_aug = self.gnn.graphsha_augment_hetero(graphs, node_attn=node_attn, he_attn=he_attn,
                                                    epoch=epoch, attn_threshold=0.001, max_new_ratio=0.2)
            pred = self.gnn(G_aug, return_attention=False)

        prob = F.softmax(pred)
        loss = self.loss_fcn(pred, label)

        loss.backward()
        self.optimizer.step()

        accuracy = acc(pred, label)

        pred = pred.detach().cpu().numpy().argmax(axis=1)
        prob = prob.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        return loss.item(), accuracy, pred, prob, label

    from tqdm import tqdm

    def train(self) -> None:
        print(f"Start training Homogeneous GNN")

        epoch_range = tqdm(range(self.n_epoch), nrows=3, desc="Epochs")
        metrics_log = tqdm(total=0, position=1, bar_format='{desc}')

        for epoch in epoch_range:
            self.gnn.train()

            res = 0
            pred_list = []
            prob_list = []
            label_list = []
            accuracy_list = []

            dataloader_tqdm = tqdm(self.dataloader, ncols=100, leave=False, desc=f"Epoch {epoch+1} WSIs")
            for graphs, label in dataloader_tqdm:
                loss, accuracy, pred, prob, label = self.train_one_step(graphs, label, epoch)
                res += loss
                # accuracy_list.append(accuracy)
                # pred_list.append(pred)
                # prob_list.append(prob)
                # label_list.append(label)

                dataloader_tqdm.set_postfix({"loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"})

            # 计算训练指标
            # accuracy = np.mean(accuracy_list)
            # pred_list = np.concatenate(pred_list)
            # prob_list = np.concatenate(prob_list)
            # label_list = np.concatenate(label_list)
            # precision, recall, f1_score, train_auc = metrics(prob_list, label_list, average=self.average)

            # 验证和测试
            self.checkpoint_manager.save_model(self.gnn.state_dict())
            evaluator = HomoGraphEvaluator(self.config, verbose=False)
            # 测试集指标，包括 per-class
            test_acc, test_f1, test_prec, test_recall, test_auc, test_per_class = evaluator.eval_per()
            # evaluator.test_data = evaluator.load_data(self.valid_path)
            # val_acc, val_f1, val_prec, val_recall, val_auc, val_per_class = evaluator.eval_per()

            epoch_range.set_description_str(f"Epoch {epoch+1} | loss: {res:.4f}")

            metrics_list = (test_acc, test_f1, test_prec, test_recall, test_auc)

            # 打印宏指标
            metrics_log.set_description_str(
                "Metrics ==> [Test Acc: {:.4f} | Test F1: {:.4f} | Test Ps: {:.4f} | Test Rec: {:.4f} | Test AUC: {:.4f}]".format(*metrics_list)
            )

            # 打印每个类别指标
            print("\n=== Validation per-class metrics ===")
            for c, m in test_per_class.items():
                print(f"Class {c}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")

            # print("\n=== Test per-class metrics ===")
            # for c, m in test_per_class.items():
            #     print(f"Class {c}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")


            # 保存 checkpoint
            epoch_stats = {
                "Epoch": epoch + 1,
                "Train Loss: ": res,
                "Testing Accuracy": test_acc,
                "Testing F1": test_f1,
                "Testing Precision": test_prec,
                "Testing Recall": test_recall,
                "Testing AUC": test_auc
            }

            self.checkpoint_manager.write_new_version(
                self.config,
                self.gnn.state_dict(),
                epoch_stats
            )
            # self.checkpoint_manager.remove_old_version()
