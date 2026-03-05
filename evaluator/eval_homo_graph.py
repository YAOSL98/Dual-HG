import numpy as np
import torch
from tqdm import tqdm


from .evaluator import Evaluator
from data import GraphDataset, TCGACancerStageDataset, TCGACancerTypingDataset
from utils import metrics
from parser import parse_gnn_model
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class HomoGraphEvaluator(Evaluator):
    def __init__(self, config, verbose=True):
        super().__init__(config, verbose)

        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)

        # Load trained checkpoint
        state_dict = self.checkpoint_manager.load_model()
        self.gnn.load_state_dict(state_dict)
        self.gnn.eval()
        # Load testing data
        test_path = self.config_data["eval_path"]
        self.name = self.config_data["dataset"]
        self.normal_path = self.config_data["normal_path"] if (self.name == "COAD" or self.name == "BRCA" or self.name== "ESCA") else ""
        self.test_data = self.load_data(test_path)

    def load_data(self, path):
        if self.name == "COAD" or self.name == "BRCA" or self.name == "ESCA":
            type = self.config_data["task"]
            if type == "cancer staging":
                self.average = "macro"
                test_data = TCGACancerStageDataset(path, self.normal_path, "eval")
            elif type == "cancer classification":
                self.average = "macro"
                test_data = GraphDataset(path, self.normal_path, self.name, 'eval')
            elif type == "cancer typing":
                self.average = "binary"
                test_data = TCGACancerTypingDataset(path, self.normal_path, self.name, 'eval')
            else:
                raise ValueError("This task not supported")
        else:
            self.average = "binary"
            test_data = GraphDataset(path, self.normal_path, self.name, 'eval')

        return test_data


    def test_one_step(self, g, label, total, correct):
        g = g.to(self.device)
        with torch.no_grad():
            out = self.gnn(g)
            prob = F.softmax(out)
            pred = out.detach().cpu().numpy().argmax(axis=1)[0]
            prob = prob.detach().cpu().numpy()
        correct += 1 if pred == label else 0
        total += 1
        return total, correct, pred, prob, label

    def eval(self):
        # Initialize metrics
        correct = 0
        total = 0

        if self.verbose:
            testing_range = tqdm(range(len(self.test_data)))
        else:
            testing_range = range(len(self.test_data))
        metrics_log = tqdm(total=0, position=0, bar_format='{desc}')

        pred_list = []
        label_list = []
        prob_list = []
        for idx in testing_range:

            graph, label = self.test_data[idx]

            total, correct, pred, prob, label = self.test_one_step(graph, label, total, correct)
            pred_list.append(pred)
            label_list.append(label)
            prob_list.append(prob)
            if self.verbose:
                testing_range.set_description("Index %d | accuracy: %f" % (idx, correct / total))

        pred_list = np.array(pred_list)
        label_list = np.array(label_list)
        prob_list = np.concatenate(prob_list)

        precision, recall, f1_score, auc = metrics(prob_list, label_list, average=self.average)
        metrics_list = (f1_score, precision, recall, auc)

        if self.verbose:
            metrics_log.set_description_str("Metrics ==> [F1 score: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | AUC: {:.4f}]".format(*metrics_list))

        return correct / total, f1_score, precision, recall, auc

    def eval_per(self):
        self.num_classes = 4  # 固定四类
        correct = 0
        total = 0

        if self.verbose:
            testing_range = tqdm(range(len(self.test_data)))
        else:
            testing_range = range(len(self.test_data))

        pred_list = []
        label_list = []
        prob_list = []

        for idx in testing_range:
            graph, label = self.test_data[idx]
            total, correct, pred, prob, label = self.test_one_step(graph, label, total, correct)
            pred_list.append(pred)
            label_list.append(label)
            prob_list.append(prob)
            if self.verbose:
                testing_range.set_description(f"Index {idx} | accuracy: {correct / total:.4f}")

        pred_list = np.array(pred_list)
        label_list = np.array(label_list)

        # 处理 prob_list shape 可能是 [N, 1, num_classes] 的情况
        prob_array = np.concatenate([p.reshape(-1, self.num_classes) for p in prob_list], axis=0)

        # 如果概率列数不够 4 类，补零
        if prob_array.shape[1] < self.num_classes:
            tmp = np.zeros((prob_array.shape[0], self.num_classes))
            tmp[:, :prob_array.shape[1]] = prob_array
            prob_array = tmp

        per_class_metrics = {}
        for c in range(self.num_classes):
            y_true_c = (label_list == c).astype(int)
            y_pred_c = (pred_list == c).astype(int)
            y_prob_c = prob_array[:, c]

            precision_c = precision_score(y_true_c, y_pred_c, zero_division=0)
            recall_c = recall_score(y_true_c, y_pred_c, zero_division=0)
            f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
            try:
                auc_c = roc_auc_score(y_true_c, y_prob_c)
            except ValueError:
                auc_c = float('nan')

            per_class_metrics[c] = {
                'precision': precision_c,
                'recall': recall_c,
                'f1': f1_c,
                'auc': auc_c
            }

        precision_macro = precision_score(label_list, pred_list, average='macro', zero_division=0)
        recall_macro = recall_score(label_list, pred_list, average='macro', zero_division=0)
        f1_macro = f1_score(label_list, pred_list, average='macro', zero_division=0)
        try:
            auc_macro = roc_auc_score(np.eye(self.num_classes)[label_list], prob_array, average='macro', multi_class='ovr')
        except ValueError:
            auc_macro = float('nan')

        if self.verbose:
            print("\n=== Per-class metrics ===")
            for c, m in per_class_metrics.items():
                print(f"Class {c}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")
            print("\n=== Macro metrics ===")
            print(f"F1={f1_macro:.4f}, Precision={precision_macro:.4f}, Recall={recall_macro:.4f}, AUC={auc_macro:.4f}")

        return correct / total, f1_macro, precision_macro, recall_macro, auc_macro, per_class_metrics
