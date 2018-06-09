import sys
import time
import numpy as np
import psutil
import GPUtil

def get_cpu_loads():
    '''
    Get current load (0.0~1.0) for all CPUs
    '''
    return [f / 100.0 for f in psutil.cpu_percent(percpu=True)]

def get_gpu_loads_and_efficiency(gpu_indices):
    '''
    Get current load (0.0~1.0) for specified GPUs
    And overall efficiency rating (0.0~1.0), 1.0 for perfect efficiency

    Args:
        gpu_indices: indices of GPUs
    '''

    assert len(gpu_indices) > 0

    # get new load data
    gpus = GPUtil.getGPUs()
    out = []

    for i in gpu_indices:
        out.append(gpus[i].load)

    return out, float(sum(out)) / float(len(gpu_indices))

def minibatches(data, minibatch_size, always_fill = False):
    '''
    Args:
        data: generator of object batches
        minibatch_size: (int)
        always_fill: act like circular queue
    Returns: 
        list of objects
    '''
    out = []
    for d in data:
        out.append(d)
        if len(out) == minibatch_size:
            yield out
            out = []

    if always_fill:
        if len(out) != 0: # is there even a partial batch?
            # if it was a full batch it would have been emitted above, but it
            # wasn't, so we need to wrap around and add more data to it and make
            # it a full batch
            # number of remaining items to fill
            rem_size = minibatch_size - len(out)
            yield out + data[:rem_size]
    else:
        # last remainder batch
        if len(out) != 0: # is there even a partial batch?
            yield out


class Progbar(object):
    '''Progbar class copied from keras (https://github.com/fchollet/keras/)
    
    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    '''

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        '''
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        '''

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
