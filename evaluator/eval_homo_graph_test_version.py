import numpy as np
import torch
from tqdm import tqdm


from .evaluator import Evaluator
from data import GraphDataset, TCGACancerStageDataset
from utils import metrics
from parser import parse_gnn_model
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm

import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def specificity_score(y_true, y_pred, labels):
    """计算每类的specificity = TN / (TN + FP)"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp + 1e-8))
    return np.array(specificity)

def metrics_per(prob_list, label_list, average="macro", n_classes=None):
    pred = np.argmax(prob_list, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, pred, average=average, zero_division=0)
    try:
        auc = roc_auc_score(label_list, prob_list, multi_class="ovo", average=average)
    except:
        auc = 0.0
    return precision, recall, f1, auc

class HomoGraphEvaluator(Evaluator):
    def __init__(self, config, checkpoint_manager=None, verbose=True):
        super().__init__(config, verbose)

        # 如果外部没传，才自己新建
        if checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(config['checkpoint']['path'])
        else:
            self.checkpoint_manager = checkpoint_manager

        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)

        state_dict = self.checkpoint_manager.load_model()
        self.gnn.load_state_dict(state_dict, strict=True)
        self.gnn.eval()

        # 加载测试数据
        test_path = self.config_data["eval_path"]
        self.name = self.config_data["dataset"]
        self.normal_path = self.config_data["normal_path"] if (self.name in ["COAD", "BRCA", "ESCA"]) else ""
        self.test_data = self.load_data(test_path)

    def load_data(self, path):
        if self.name == "COAD" or self.name == "BRCA" or self.name == "ESCA":
            type = self.config_data["task"]
            if type == "cancer staging":
                self.average = "macro"
                print('TCGACancerStageDataset')
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
            out = self.gnn(G=g)
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

    def eval_per(self, save_path="./checkpoints/test_he/eval_results.txt", version=None, return_pred=False):
        """
        计算 Overall / Primary / Metastatic 的多分类指标：
        Precision, Recall, F1, AUC, Specificity（macro + per class）
        并保存到 txt 文件，同时生成混淆矩阵图片
        """
        pred_list, label_list, prob_list = [], [], []

        testing_range = tqdm(range(len(self.test_data)), desc="Evaluating")
        for idx in testing_range:
            graph, label = self.test_data[idx]
            _, _, pred, prob, label = self.test_one_step(graph, label, 0, 0)

            # 所有数据
            pred_list.append(pred)
            label_list.append(label)
            prob_list.append(prob)


            # 动态显示累积准确率
            if len(pred_list) > 0:
                acc_so_far = np.mean(np.array(pred_list) == np.array(label_list))
                testing_range.set_postfix({"acc_so_far": f"{acc_so_far:.4f}"})

        # 转为 numpy
        pred_list = np.array(pred_list)
        label_list = np.array(label_list)
        prob_list = np.concatenate(prob_list)

        n_classes = prob_list.shape[1]

        def topk_accuracy(y_true, y_prob, k=1):
            """
            y_true: (N,) 真实标签
            y_prob: (N, C) 每个样本的类别预测概率
            k: top-k
            """
            # 获取每个样本概率最大的 k 个类别
            topk_preds = np.argsort(y_prob, axis=1)[:, -k:]  # shape (N, k)
            # 判断真实标签是否在 top-k 预测中
            correct = [y_true[i] in topk_preds[i] for i in range(len(y_true))]
            return np.mean(correct) if len(y_true) > 0 else 0.0

        def compute_metrics(y_true, y_pred, y_prob):
            top1_acc = topk_accuracy(y_true, y_prob, k=1)
            top2_acc = topk_accuracy(y_true, y_prob, k=2)
            top3_acc = topk_accuracy(y_true, y_prob, k=3)
            # Macro指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            try:
                auc = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr", labels=list(range(n_classes)))
            except Exception:
                auc = float("nan")

            # 每类指标
            class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
            )
            try:
                class_auc = roc_auc_score(y_true, y_prob, average=None, multi_class="ovr", labels=list(range(n_classes)))
            except Exception:
                class_auc = [float("nan")] * n_classes

            # Specificity
            cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
            class_spec = []
            for i in range(n_classes):
                tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fp = cm[:, i].sum() - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                class_spec.append(spec)
            spec_macro = np.mean(class_spec)

            return precision, recall, f1, auc, spec_macro, class_prec, class_rec, class_f1, class_auc, class_spec, cm, top1_acc, top2_acc, top3_acc

        sections = {
            "Overall (All data)": (label_list, pred_list, prob_list),
        }

        results = []
        cm_dict = {}  # 保存每个部分的混淆矩阵
        acc_dict = {}
        for name, (y_true, y_pred, y_prob) in sections.items():
            if len(y_true) == 0:
                continue
            prec, rec, f1, auc, spec_macro, cp, cr, cf1, ca, cs, cm, top1_acc, top2_acc, top3_acc = compute_metrics(y_true, y_pred, y_prob)
            cm_dict[name] = cm
            # acc_dict[name] = accuracy  # 保存准确率
            results.append(f"=== {name} ===")
            results.append(f"Top-1 ACC={top1_acc:.4f}, Top-2 ACC={top2_acc:.4f}, Top-3 ACC={top3_acc:.4f}")
            results.append(f"Macro Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")
            results.append(f"Specificity (macro avg)={spec_macro:.4f}")
            for i in range(n_classes):
                results.append(
                    f"Class {i}: Precision={cp[i]:.4f} | Recall={cr[i]:.4f} | F1={cf1[i]:.4f} | AUC={ca[i]:.4f} | Spec={cs[i]:.4f}"
                )
            results.append("")

        # 保存txt
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "a") as f:
            f.write("\n".join(results) + "\n")

        # 绘制混淆矩阵图片
        # save_dir = os.path.dirname(save_path) or "."
        # if version is not None:
        #     save_dir = os.path.join(save_dir, f"confusion/version_{version}")
        #     os.makedirs(save_dir, exist_ok=True)

        # for name, cm in cm_dict.items():
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        #     plt.title(f"Confusion Matrix: {name}")
        #     plt.ylabel("True label")
        #     plt.xlabel("Predicted label")
        #     # 文件名
        #     fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        #     plt.savefig(os.path.join(save_dir, f"{fname}.png"))
        #     plt.close()

        print("\n".join(results))

        if return_pred:
            return prec, rec, f1, auc, spec_macro, {
                "Overall": (np.array(label_list), np.array(prob_list)),
            }
        else:
            return prec, rec, f1, auc, spec_macro