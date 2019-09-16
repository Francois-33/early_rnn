import torch
from utils.classmetric import ClassMetric
from utils.logger import Logger
from utils.printer import Printer
import os
from utils.loss import loss_cross_entropy, early_loss_cross_entropy, early_loss_linear, loss_early_reward
import numpy as np
from utils.optim import ScheduledOptim
import logging
import tqdm

class Trainer():

    def __init__(self,
                 model,
                 traindataloader,
                 testdataloader,
                 optimizer,
                 criterion,
                 store="/tmp",
                 test_every_n_epochs=1,
                 visdomlogger=None):

        self.batch_size = testdataloader.batch_size
        self.traindataloader = traindataloader
        self.testdataloader = testdataloader
        self.nclasses=traindataloader.dataset.nclasses
        self.store = store
        self.test_every_n_epochs = test_every_n_epochs
        self.logger = Logger(columns=["accuracy"], modes=["train", "test"], rootpath=self.store)

        self.model = model


        self.optimizer=optimizer
        self.criterion=criterion

        self.visdom=visdomlogger


        # only save checkpoint if not previously resumed from it
        self.resumed_run = False

        self.epoch = 0

        #if os.path.exists(self.get_classification_model_name()) and not overwrite:
        #    print("Resuming from snapshot {}.".format(self.get_classification_model_name()))
        #    self.resume(self.get_classification_model_name())
        #    self.resumed_run = True

        logging.debug(self.model)
        logging.debug(self.optimizer)
        logging.debug("traindataloader")
        logging.debug(traindataloader.sampler)

        import sys
        logging.debug(sys.getsizeof(traindataloader))

        logging.debug("validdataloader")
        logging.debug(testdataloader.sampler)
        logging.debug(sys.getsizeof(testdataloader))

    def resume(self, filename):
        snapshot = self.model.load(filename)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.epoch = snapshot["epoch"]
        if self.resume_optimizer:
            print("resuming optimizer state")
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.logger.resume(snapshot["logged_data"])

    def snapshot(self, filename):
        self.model.save(
        filename,
        optimizer_state_dict=self.optimizer.state_dict(),
        epoch=self.epoch,
        logged_data=self.logger.get_data())

    def fit(self, epochs):
        printer = Printer()

        while self.epoch < epochs:
            self.new_epoch() # increments self.epoch

            self.logger.set_mode("train")
            stats = self.train_epoch(self.epoch)
            self.logger.log(stats, self.epoch)
            printer.print(stats, self.epoch, prefix="\ntrain: ")

            if self.epoch % self.test_every_n_epochs == 0:
                self.logger.set_mode("test")
                stats = self.test_epoch(self.testdataloader)
                self.logger.log(stats, self.epoch)
                printer.print(stats, self.epoch, prefix="\ntest: ")
                if hasattr(self,"visdom"): self.visdom_log_test_run(stats)

            if hasattr(self,"visdom"): self.visdom.plot_epochs(self.logger.get_data())

        self.logger.save()

        return self.logger.data

    def new_epoch(self):
        self.epoch += 1

    def visdom_log_test_run(self, stats):
        if hasattr(self, 'visdom'):

            if hasattr(self.traindataloader.dataset,"samplet"):
                self.visdom.plot_boxplot(labels=stats["labels"], t_stops=stats["t_stops"], tmin=0, tmax=self.traindataloader.dataset.samplet)

            self.visdom.confusion_matrix(stats["confusion_matrix"], norm=None, title="Confusion Matrix")
            self.visdom.confusion_matrix(stats["confusion_matrix"], norm=0, title="Recall")
            self.visdom.confusion_matrix(stats["confusion_matrix"], norm=1, title="Precision")
            legend = ["class {}".format(c) for c in range(self.nclasses)]
            targets = stats["targets"]

            for i in range(1):
                classid = targets[i, 0]

                if len(stats["probas"].shape) == 3:
                    self.visdom.plot(stats["probas"][:, i, :], name="sample {} P(y) (class={})".format(i, classid),
                                     fillarea=True,
                                     showlegend=True, legend=legend)
                self.visdom.plot(stats["inputs"][i, :, 0], name="sample {} x (class={})".format(i, classid))
                if "pts" in stats.keys(): self.visdom.bar(stats["pts"][i, :], name="sample {} P(t) (class={})".format(i, classid))
                if "deltas" in stats.keys(): self.visdom.bar(stats["deltas"][i, :], name="sample {} deltas (class={})".format(i, classid))
                if "budget" in stats.keys(): self.visdom.bar(stats["budget"][i, :], name="sample {} budget (class={})".format(i, classid))

    def ending_phase_earliness_event(self):
        print("ending training phase earliness")
        self.snapshot(os.path.join(self.store, "model_{}.pth".format(EARLINESS_PHASE_NAME)))
        log = os.path.join(self.store, "log_{}.csv".format(EARLINESS_PHASE_NAME))
        print("Saving log to {}".format(log))
        self.logger.get_data().to_csv(log)

    def train_epoch(self, epoch):
        # sets the model to train mode: dropout is applied
        self.model.train()

        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.nclasses)

        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, targets, meta = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1,2))

            loss, stats = self.criterion(logprobabilities, pts, targets)
            loss.backward()

            if isinstance(self.optimizer, ScheduledOptim):
                self.optimizer.step_and_update_lr()
                lr = self.optimizer._optimizer.state_dict()["param_groups"][0]["lr"]
            else:
                self.optimizer.step()
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

            prediction, t_stop = self.model.predict(logprobabilities, deltas)

            stats = metric.add(stats)
            stats["lr"] = lr

            accuracy_metrics = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())
            stats["accuracy"] = accuracy_metrics["overall_accuracy"]
            stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
            stats["mean_recall"] = accuracy_metrics["recall"].mean()
            stats["mean_precision"] = accuracy_metrics["precision"].mean()
            stats["mean_f1"] = accuracy_metrics["f1"].mean()
            stats["kappa"] = accuracy_metrics["kappa"]
            if t_stop is not None:
                earliness = (t_stop.float()/(inputs.shape[1]-1)).mean()
                stats["earliness"] = metric.update_earliness(earliness.cpu().detach().numpy())

        return stats

    def test_epoch(self, dataloader):
        # sets the model to train mode: no dropout is applied
        self.model.eval()

        # builds a confusion matrix
        #metric_maxvoted = ClassMetric(num_classes=self.nclasses)
        metric = ClassMetric(num_classes=self.nclasses)
        #metric_all_t = ClassMetric(num_classes=self.nclasses)

        tstops = list()
        predictions = list()
        labels = list()


        with torch.no_grad():
            for iteration, data in enumerate(dataloader):

                inputs, targets, meta = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2))
                loss, stats = self.criterion(logprobabilities, pts, targets)
                prediction, t_stop = self.model.predict(logprobabilities, deltas)

                ## enter numpy world
                prediction = prediction.detach().cpu().numpy()
                label = targets.mode(1)[0].detach().cpu().numpy()
                t_stop = t_stop.cpu().detach().numpy()
                pts = pts.detach().cpu().numpy()
                deltas = deltas.detach().cpu().numpy()
                budget = budget.detach().cpu().numpy()

                tstops.append(t_stop)
                predictions.append(prediction)
                labels.append(label)


                stats = metric.add(stats)

                accuracy_metrics = metric.update_confmat(label,
                                                         prediction)

                stats["accuracy"] = accuracy_metrics["overall_accuracy"]
                stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
                stats["mean_recall"] = accuracy_metrics["recall"].mean()
                stats["mean_precision"] = accuracy_metrics["precision"].mean()
                stats["mean_f1"] = accuracy_metrics["f1"].mean()
                stats["kappa"] = accuracy_metrics["kappa"]
                if t_stop is not None:
                    earliness = (t_stop.astype(float) / (inputs.shape[1] - 1)).mean()
                    stats["earliness"] = metric.update_earliness(earliness)

            stats["confusion_matrix"] = metric.hist
            stats["targets"] = targets.cpu().numpy()
            stats["inputs"] = inputs.cpu().numpy()
            if deltas is not None: stats["deltas"] = deltas
            if pts is not None: stats["pts"] = pts
            if budget is not None: stats["budget"] = budget

            probas = logprobabilities.exp().transpose(0, 1)
            stats["probas"] = probas.detach().cpu().numpy()

            stats["t_stops"] = np.hstack(tstops)
            stats["predictions"] = np.hstack(predictions)
            stats["labels"] = np.hstack(labels)

        return stats
