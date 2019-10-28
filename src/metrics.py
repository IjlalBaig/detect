from ignite.metrics import Metric, EpochMetric
from ignite.engine import Engine, Events
import torch

class EpochAverage(Metric):
    """Compute running average of a metric or the output of process function.

    Args:
        src (Metric or None): input source: an instance of :class:`~ignite.metrics.Metric` or None. The latter
            corresponds to `engine.state.output` which holds the output of process function.
        alpha (float, optional): running average decay factor, default 0.98
        output_transform (callable, optional): a function to use to transform the output if `src` is None and
            corresponds the output of process function. Otherwise it should be None.

    Examples:

    .. code-block:: python

        alpha = 0.98
        acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
        acc_metric.attach(trainer, 'running_avg_accuracy')

        avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_running_avg_metrics(engine):
            print("running avg accuracy:", engine.state.metrics['running_avg_accuracy'])
            print("running avg loss:", engine.state.metrics['running_avg_loss'])

    """

    def __init__(self, src=None, output_transform=None):
        self._iterations = 0
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError("Argument output_transform should not be None if src corresponds "
                                 "to the output of process function.")
            self._get_src_value = self._get_output_value
            self.update = self._output_update
        super(EpochAverage, self).__init__(output_transform=output_transform)

    def reset(self):
        self._value = None
        self._iterations = 0


    def update(self, output):
        # Implement abstract method

        pass

    def compute(self):
        if self._value is None:
            self._value = self._get_src_value().mean(dim=0)
        else:
            self._value = self._value + self._get_src_value().mean(dim=0)
        return self._value

    def attach(self, engine, name):
        # avg error
        engine.add_event_handler(Events.COMPLETED, self.ended, name)
        # restart average every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_count)

    def completed(self, engine, name):
        result = self.compute()
        if torch.is_tensor(result) and len(result.shape) == 0:
            result = result.item()

    def iteration_count(self, engine):
        # because half of size is for query and half for compute
        # self._iterations += engine.state.batch[list(engine.state.batch.keys())[0]].size(0) / 2
        self._iterations += 1

    def ended(self, engine, name):
        if torch.is_tensor(self._value) and len(self._value.shape) == 0:
            engine.state.metrics[name] = self._value / self._iterations

    def _get_metric_value(self):
        return self.src.compute()

    def _get_output_value(self):
        return self.src

    def _metric_iteration_completed(self, engine):
        self.src.started(engine)
        self.src.iteration_completed(engine)

    def _output_update(self, output):
        self.src = output


class EpochMax(Metric):
    """Compute running average of a metric or the output of process function.

    Args:
        src (Metric or None): input source: an instance of :class:`~ignite.metrics.Metric` or None. The latter
            corresponds to `engine.state.output` which holds the output of process function.
        alpha (float, optional): running average decay factor, default 0.98
        output_transform (callable, optional): a function to use to transform the output if `src` is None and
            corresponds the output of process function. Otherwise it should be None.

    Examples:

    .. code-block:: python

        alpha = 0.98
        acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
        acc_metric.attach(trainer, 'running_avg_accuracy')

        avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_running_avg_metrics(engine):
            print("running avg accuracy:", engine.state.metrics['running_avg_accuracy'])
            print("running avg loss:", engine.state.metrics['running_avg_loss'])

    """

    def __init__(self, src=None, output_transform=None):
        self._iterations = 0
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError("Argument output_transform should not be None if src corresponds "
                                 "to the output of process function.")
            self._get_src_value = self._get_output_value
            self.update = self._output_update
        super(EpochMax, self).__init__(output_transform=output_transform)

    def reset(self):
        self._value = None
        self._iterations = 0

    def update(self, output):
        # Implement abstract method
        pass

    def compute(self):
        if self._value is None:
            self._value, _ = self._get_src_value().max(dim=0)
        else:
            new_value = self._get_src_value().max(0)[0]
            if new_value > self._value:
                self._value = new_value
        return self._value

    def attach(self, engine, name):
        # restart average every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self):
        return self.src.compute()

    def _get_output_value(self):
        return self.src

    def _metric_iteration_completed(self, engine):
        self.src.started(engine)
        self.src.iteration_completed(engine)

    def _output_update(self, output):
        self.src = output
