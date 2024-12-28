import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm.auto import tqdm

class Experiment:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device = "cuda",
                 warmup_steps = 0, decay_scheduler = None, experiment_name = "experiment_1",
                 experiment_note = None,compute_metrics = False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.warmup_steps = warmup_steps
        self.decay_scheduler = decay_scheduler
        self.model.to(self.device)
        self.experiment_name = experiment_name
        self.experiment_note = experiment_note
        self.current_step = 0
        self.compute_metrics = compute_metrics

        # TensorBoard setup
        logs_dir = os.path.join("runs", experiment_name)
        self.writer = SummaryWriter(log_dir=logs_dir)

        # Save initial learning rate for warmup
        self.initial_lr = self.optimizer.param_groups[0]["lr"]

    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                self.current_step += 1

                # Warmup Learning Rate
                if self.warmup_steps > 0:
                    self.warmup_lr()

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Log training loss to TensorBoard
                self.writer.add_scalar("Training Loss", loss.item(), self.current_step)

            # Average epoch loss
            epoch_loss = running_loss / len(self.train_loader)

            # Log epoch loss
            self.writer.add_scalar("Epoch Loss", epoch_loss, epoch + 1)

            # Evaluate after each epoch
            val_loss = self.evaluate(epoch + 1)

            # Step the scheduler if provided
            if self.decay_scheduler:
                self.decay_scheduler.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:4f}")

    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

                if self.compute_metrics:
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

        # Compute average evaluation loss
        val_loss = running_loss / len(self.val_loader)

        # Compute metrics
        if self.compute_metrics:
            accuracy = self.compute_accuracy(all_preds, all_targets)
            precision = self.compute_precision(all_preds, all_targets)
            recall = self.compute_recall(all_preds, all_targets)
            f1 = self.compute_f1_score(all_preds, all_targets)
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            
            # Log metrics to TensorBoard
            self.writer.add_scalar("Accuracy", accuracy, epoch)
            self.writer.add_scalar("Precision", precision, epoch)
            self.writer.add_scalar("Recall", recall, epoch)
            self.writer.add_scalar("F1 Score", f1, epoch)

        # Log validation loss
        self.writer.add_scalar("Validation Loss", val_loss, epoch)

        return val_loss

    def warmup_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_lr = self.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

    @staticmethod
    def compute_accuracy(preds, targets):
        correct = sum(p == t for p, t in zip(preds, targets))
        return correct / len(targets)

    @staticmethod
    def compute_precision(preds, targets):
        num_classes = len(set(targets))
        true_positives = [0] * num_classes
        false_positives = [0] * num_classes

        for p, t in zip(preds, targets):
            if p == t:
                true_positives[p] += 1
            else:
                false_positives[p] += 1

        precisions = [
            tp / (tp + fp) if (tp + fp) > 0 else 0
            for tp, fp in zip(true_positives, false_positives)
        ]
        return sum(precisions) / num_classes

    @staticmethod
    def compute_recall(preds, targets):
        num_classes = len(set(targets))
        true_positives = [0] * num_classes
        false_negatives = [0] * num_classes

        for p, t in zip(preds, targets):
            if p == t:
                true_positives[t] += 1
            else:
                false_negatives[t] += 1

        recalls = [
            tp / (tp + fn) if (tp + fn) > 0 else 0
            for tp, fn in zip(true_positives, false_negatives)
        ]
        return sum(recalls) / num_classes

    @staticmethod
    def compute_f1_score(preds, targets):
        precision = Experiment.compute_precision(preds, targets)
        recall = Experiment.compute_recall(preds, targets)
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0
