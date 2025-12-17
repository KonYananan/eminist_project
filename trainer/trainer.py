import os
import torch
import numpy as np
from utils.metrics import accuracy, multiclass_auc


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        writer,
        num_classes,
        save_dir="../checkpoints",
        model_name="model"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.num_classes = num_classes

        self.best_auc = 0.0
        self.save_dir = save_dir
        self.model_name = model_name

        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- 训练一个 epoch ----------
    def train_one_epoch(self, dataloader, epoch):
        self.model.train()

        total_loss = 0
        y_true, y_pred = [], []

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        acc = accuracy(y_true, y_pred)

        # TensorBoard 记录
        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/Accuracy", acc, epoch)

        return avg_loss, acc

    # ---------- 验证阶段 ----------
    def validate(self, dataloader, epoch):
        self.model.eval()

        total_loss = 0
        y_true = []
        y_prob = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)

                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)
                y_true.extend(y.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        auc = multiclass_auc(y_true, y_prob, self.num_classes)

        # TensorBoard 记录
        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/AUC", auc, epoch)

        # 保存最优模型
        self._save_best_model(auc, epoch)

        return avg_loss, auc

    # ---------- 自动保存最优模型 ----------
    def _save_best_model(self, auc, epoch):
        if auc > self.best_auc:
            self.best_auc = auc
            save_path = os.path.join(
                self.save_dir,
                f"{self.model_name}_best.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_auc": self.best_auc
            }, save_path)
