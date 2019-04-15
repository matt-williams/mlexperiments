import math
import numpy as np
import pickle
import random
from matplotlib import pyplot as plt

class Logger:
    def __init__(self, fname=None, logs=None):
        if fname:
            self.load(fname)
        else:
            self.logs = logs or []

    def log(self, *args):
        self.logs.append(tuple(args))

    def get(self):
        def generator():
            ii = 0
            while True:
                if ii < len(self.logs):
                    yield self.logs[ii]
                    ii += 1
                else:
                    yield None
        return generator()

    def save(self, fname):
        with open(fname, 'wb') as file:
            pickle.Pickler(file).dump(self.logs)

    def load(self, fname):
        with open(fname, 'rb') as file:
            self.logs = pickle.Unpickler(file).load()

class Filter:
    def __init__(self, logs, func):
        self.logs = logs
        self.func = func

    def get(self):
        def generator():
            for log in self.logs.get():
                if not log is None:
                    log = self.func(log)
                yield log
        return generator()

class Viewer:
    def __init__(self, func, init_func=(lambda: ({}, None))):
        self.func = func
        self.init_func = init_func

    def get_view(self, logs):
        result, scratch = self.init_func()
        return View(logs, self.func, result, scratch)

class View:
    def __init__(self, logs, func, view={}, scratch=None):
        self.logs = logs.get()
        self.func = func
        self.result = view
        self.scratch = scratch

    def get(self):
        for log in self.logs:
            if log is None:
                break
            self.func(log, self.result, self.scratch)
        return self.result

class Log:
    def __init__(self, logger, *prefix):
        self.logger = logger
        self.prefix = prefix

    def __call__(self, *args):
        args = self.prefix + args
        self.logger.log(*args)


class UnaggregatedViewer(Viewer):
    def __init__(self, filter_prefix, columns, labels):
        filter_prefix = tuple(filter_prefix)

        def init_func():
            result = {}
            for label in labels:
                result[label] = []
            return (result, None)

        def func(log, result, _):
            if log[:len(filter_prefix)] == filter_prefix:
                for column, label in zip(columns, labels):
                    result[label].append(log[column])

        super(UnaggregatedViewer, self).__init__(func, init_func)

class EpochAggregatedViewer(Viewer):
    def __init__(self, filter_prefix, epoch_column, columns, labels, average=True):
        filter_prefix = tuple(filter_prefix)

        def init_func():
            result = {"epoch": []}
            scratch = {"total": {}, "count": {}}
            for label in labels:
                result[label] = []
                scratch["count"] = []
                scratch["total"][label] = []
            return (result, scratch)

        def func(log, result, scratch):
            if log[:len(filter_prefix)] == filter_prefix:
                epoch = log[epoch_column]
                epochs = result["epoch"]
                totals = scratch["total"]
                try:
                    ii = len(epochs) - 1 if len(epochs) > 0 and epochs[-1] == epoch else epochs.index(epoch)
                except ValueError:
                    ii = len(epochs)
                    epochs.append(epoch)
                    scratch["count"].append(0)
                    for label in labels:
                        if average:
                            totals[label].append(0)
                            result[label].append(math.nan)
                        else:
                            result[label].append([])
                scratch["count"][ii] += 1
                for column, label in zip(columns, labels):
                    if average:
                        totals[label][ii] += log[column]
                        result[label][ii] = totals[label][ii] / scratch["count"][ii]
                    else:
                        result[label][ii].append(log[column])

        super(EpochAggregatedViewer, self).__init__(func, init_func)

class BatchAggregatedViewer(Viewer):
    def __init__(self, filter_prefix, epoch_column, batch_column, columns, labels, batches_per_epoch=100):
        filter_prefix = tuple(filter_prefix)

        def init_func():
            result = {"epoch": [], "batch": [], "epoch_batch": []}
            scratch = {"total": {}, "count": {}}
            for label in labels:
                result[label] = []
                scratch["count"] = []
                scratch["total"][label] = []
            return (result, scratch)

        def func(log, result, scratch):
            if log[:len(filter_prefix)] == filter_prefix:
                epoch = log[epoch_column]
                batch = log[batch_column]
                epochs = result["epoch"]
                batches = result["batch"]
                epoch_batches = result["epoch_batch"]
                totals = scratch["total"]
                try:
                    if len(epochs) > 0 and (epochs[-1] == epoch and batches[-1] == batch):
                        ii = len(epochs) - 1
                    else:
                        ii = epochs.index(epoch)
                        ii = batches.index(batch, ii)
                        while epochs[ii] != epoch or batches[ii] != batch:
                            ii = epochs.index(epoch, ii)
                            ii = batches.index(batch, ii)
                except ValueError:
                    ii = len(epochs)
                    epochs.append(epoch)
                    batches.append(batch)
                    epoch_batches.append(epoch + batch / batches_per_epoch)
                    scratch["count"].append(0)
                    for label in labels:
                        totals[label].append(0)
                        result[label].append(math.nan)
                scratch["count"][ii] += 1
                for column, label in zip(columns, labels):
                    totals[label][ii] += log[column]
                    result[label][ii] = totals[label][ii] / scratch["count"][ii]

        super(BatchAggregatedViewer, self).__init__(func, init_func)

class LastEpochViewer(Viewer):
    def __init__(self, filter_prefix, epoch_column, columns, labels):
        filter_prefix = tuple(filter_prefix)

        def init_func():
            result = {"epoch": -1}
            scratch = {"total": {}, "count": {}}
            for label in labels:
                result[label] = math.nan
                scratch["count"] = 0
                scratch["total"][label] = 0
            return (result, scratch)

        def func(log, result, scratch):
            if log[:len(filter_prefix)] == filter_prefix:
                epoch = log[epoch_column]
                totals = scratch["total"]
                if epoch > result["epoch"]:
                    result["epoch"] = epoch
                    scratch["count"] = 0
                    for label in labels:
                        totals[label] = 0
                if epoch == result["epoch"]:
                    scratch["count"] += 1
                    for column, label in zip(columns, labels):
                        totals[label] += log[column]
                        result[label] = totals[label] / scratch["count"]

        super(LastEpochViewer, self).__init__(func, init_func)


class EpochGraph:
    def __init__(self, views, labels, colors, epoch_range=(0, -1), epoch_label="epoch", sub_epoch_label=None, expected_interval=1, yscale="linear"):
        self.views = [views] if not isinstance(views, (list, tuple)) else views
        self.labels = [labels] if not isinstance(labels, (list, tuple)) else labels
        self.colors = [[colors]] if not isinstance(colors, (list, tuple)) else colors
        self.epoch_range = epoch_range
        self.epoch_label = epoch_label
        self.sub_epoch_label = sub_epoch_label or epoch_label
        self.expected_interval = expected_interval
        self.yscale = yscale

    def get_data(self):
        return [view.get() for view in self.views]

    def get_epochs(self):
        data = self.get_data()
        last_epoch = max([max(data[self.epoch_label]) if len(data[self.epoch_label]) > 0 else 0 for data in data])
        epoch_range = (self.epoch_range[0] if self.epoch_range[0] >= 0 else max(last_epoch + 1 + self.epoch_range[0], 0), self.epoch_range[1] if self.epoch_range[1] >= 0 else max(last_epoch + 1 + self.epoch_range[1], 0))
        epoch_range = (epoch_range[0], max(epoch_range[0], epoch_range[1]))
        epoch_range = (min(epoch_range[0], last_epoch), min(epoch_range[1], last_epoch))
        return epoch_range

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        epoch_range = self.get_epochs()
        data = self.get_data()
        ax.set_xlim((epoch_range[0], max(epoch_range[0] + 1, epoch_range[1])))
        ax.set_yscale(self.yscale)
        ax.grid(color='black')
        for label, colors in zip(self.labels, self.colors):
            for data2, color in zip(data, colors):
                epochs = data2[self.epoch_label]
                sub_epochs = data2[self.sub_epoch_label]
                label_data = data2[label]
                combined = [(sub_epochs[ii], label_data[ii]) for (ii, epoch) in enumerate(epochs) if epoch_range[0] <= epoch <= epoch_range[1]]
                combined.sort()
                ii_off = 0
                if len(label_data) > 1:
                    if isinstance(combined[0][1], (list, tuple)):
                        epoch, label_data = zip(*combined)
                        label_data = [random.sample(x, 100) if len(x) > 100 else x for x in label_data]
                        parts = ax.violinplot(label_data, positions=epoch, showmeans=False, showmedians=False, showextrema=False)
                        for part in parts['bodies']:
                            part.set_facecolor(color)
                    else:
                        for ii in range(len(combined) - 1):
                            if combined[ii + ii_off][0] + 1.5 * self.expected_interval < combined[ii + ii_off + 1][0]:
                                empty_val = -1 if self.yscale == 'log' else np.nan # log graph plots don't support np.nan, but do at least handle negatives!
                                combined.insert(ii + ii_off + 1, (combined[ii + ii_off][0] + self.expected_interval, empty_val))
                                ii_off += 1
                        epoch, label_data = zip(*combined)
                        ax.plot(epoch, label_data, c=color)
                if len(data2[label]) > 0 and not isinstance(combined[0][1], (list, tuple)) and (self.yscale != 'log' or min(data2[label]) > 0):
                    ax.axhline(min(data2[label]), c=color, dashes=[5, 5])
        return ax

class EpochLRGraph:
    def __init__(self, view, label, color="black", epoch_range=(0, -1), epoch_label="epoch"):
        self.view = view
        self.label = label
        self.color = color
        self.epoch_range = epoch_range
        self.epoch_label = epoch_label

    def get_data(self):
        return self.view.get()

    def get_epochs(self):
        data = self.get_data()
        last_epoch = max(data[self.epoch_label]) if len(data[self.epoch_label]) > 0 else 0
        epoch_range = (self.epoch_range[0] if self.epoch_range[0] >= 0 else max(last_epoch + 1 + self.epoch_range[0], 0), self.epoch_range[1] if self.epoch_range[1] >= 0 else max(last_epoch + 1 + self.epoch_range[1], 0))
        epoch_range = (epoch_range[0], max(epoch_range[0] + 1, epoch_range[1]))
        return epoch_range

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        epoch_range = self.get_epochs()
        data = self.get_data()
        if data[self.epoch_label]:
            ax.set_xlim(epoch_range)
            ax.grid(color='black')
            prev_value = np.nan
            for epoch in range(epoch_range[0], epoch_range[1] + 1):
                try:
                    label_value = data[self.label][data[self.epoch_label].index(epoch)]
                except ValueError:
                    label_value = np.nan
                if not math.isnan(label_value) and label_value != prev_value:
                    ax.axvline(epoch, linewidth=3, color=self.color)
                    ax.text(epoch, ax.get_ybound()[1], "{} = {}".format(self.label, label_value), rotation=270, verticalalignment='top', fontsize=20, color='black')
                prev_value = label_value
        return ax
