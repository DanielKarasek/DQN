import math
import os

import numpy as np
import tensorflow as tf


class Memory(object):

    def __init__(self, size):
        self.max_size = size

    def add_transition(self, transition):
        pass

    def is_full(self):
        pass

    def get_n_samples(self, N=None):
        pass


class MemoryRankBased(Memory):

    def __init__(self, RELOAD_FLAG=False, **conf):
        # Making max size power of so the last row is full
        # and has size of 2^(exponent-1).
        # Max Size gonna be (2^exponent) -1
        if RELOAD_FLAG:
            self.load(conf["logdir"])
        else:
            self.alpha = conf["alpha"] if "alpha" in conf.keys() else 0.5

            self.beta_start = conf["beta"] if "beta" in conf.keys() else 0.0
            self.total_beta_time = conf["total_beta_time"] if "total_beta_time" in conf.keys() else 5e4
            self.beta_gradient = (1 - self.beta_start) / self.total_beta_time

            exponent = math.ceil(math.log2(conf["size"] if "size" in conf.keys() else 1e5))
            self.max_size = (2 ** exponent) - 1
            self.last_row_size = 2 ** (exponent - 1)

            self.current_size = 0
            self.heap = []

            self.NEED_UPDATE_FLAG = False
            self.mem_idxs = []

            self.base_batch_size = conf["batch_size"] if "batch_size" in conf.keys() else 64

            self._compute_probs_CDF_span()

    def __repr__(self):
        if self.current_size == 0:
            return "Heap is currently empty !!!"

        max_level = math.ceil(math.log(self.current_size, 2))
        tmp_level = -1

        to_string = "The Heap has {} levels\n".format(max_level)

        for i in range(self.current_size):
            curr_level = math.ceil(math.log(i + 2, 2))
            if curr_level != tmp_level:
                to_string += ("\n" if tmp_level != -1 else "") \
                             + "level {}: ".format(curr_level)
                tmp_level = curr_level
            to_string += "{}:{}  ".format(self.heap[i][0], self.heap[i][1])
        return to_string

    def _insert(self, transition, priority):
        if not self.is_full():
            priority = float(priority)
            self.heap.append([transition, priority])
            self._perc_up(self.current_size)
            self.current_size += 1

        else:
            #
            #             pos = np.random.randint(0,self.last_row_size)
            #             self.heap[-pos] = [transition,priority]
            #             self._perc_up(pos)
            priority = float(priority)
            rand_pos = np.random.randint(0, self.max_size)
            self.heap[rand_pos] = [transition, priority]
            self._update_node(rand_pos)

    def _perc_up(self, idx):
        parent_idx = (idx + 1) // 2 - 1
        while parent_idx >= 0:
            if self.heap[idx][1] > self.heap[parent_idx][1]:
                self.heap[idx], self.heap[parent_idx] = self.heap[parent_idx], self.heap[idx]
                idx = parent_idx
                parent_idx = (idx + 1) // 2 - 1
            else:
                break

    def _perc_down(self, idx):
        idx_child, max_child = self._max_child(idx)
        while self.heap[idx][1] < max_child != -1:
                self.heap[idx_child], self.heap[idx] = self.heap[idx], self.heap[idx_child]
                idx = idx_child
                idx_child, max_child = self._max_child(idx)

    def _max_child(self, idx):
        left_child_idx = idx*2 + 1
        right_child_idx = idx*2 + 2
        if left_child_idx >= self.current_size:
            return -1, -1

        elif right_child_idx >= self.current_size:
            return left_child_idx, self.heap[left_child_idx][1]

        left_priority = self.heap[left_child_idx][1]
        right_priority = self.heap[right_child_idx][1]

        if left_priority >= right_priority:
            return left_child_idx, left_priority

        return right_child_idx, right_priority

    def _extract_max_sort(self):
        value = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.current_size -= 1
        self._perc_down(0)
        return value

    def _update_node(self, idx):
        self._perc_up(idx)
        self._perc_down(idx)

    def _compute_probs_CDF_span(self):
        nomi = list(map(lambda x: x ** (-self.alpha), range(1, self.max_size + 1)))
        denom = math.fsum(nomi)

        self.probabilities = [x / denom for x in nomi]

        self.CDF_reversed = [math.fsum(self.probabilities)]
        for probability in self.probabilities[:-1]:
            new_value = self.CDF_reversed[-1] - probability
            self.CDF_reversed.append(new_value)

        prob_segment = 1 / self.base_batch_size
        prob_rest = 1
        self.span_indicis = []
        eps = 1e-10

        for idx, current in enumerate(self.CDF_reversed):
            if prob_rest >= current:
                prob_rest -= prob_segment
                self.span_indicis.append(idx)
                if prob_rest <= 0 + eps:
                    self.span_indicis.append(self.max_size)
                    break

    def _compute_spans(self, batch_size):
        prob_segment = 1 / batch_size
        prob_rest = 1
        span_indicis = []
        eps = 1e-10

        for current, idx in zip(self.CDF_reversed, range(self.max_size)):
            if prob_rest >= current:
                prob_rest -= prob_segment
                span_indicis.append(idx)
                if prob_rest <= 0 + eps:
                    span_indicis.append(self.max_size)
                    break

        return span_indicis

    def heap_sort(self):
        arr = []
        current_size_backup = self.current_size
        if self.current_size > 0:
            for _ in range(self.current_size - 1):
                arr.append(self._extract_max_sort())
            arr.append(self.heap[0])
        self.heap = arr
        self.current_size = current_size_backup

    def save(self, logdir):
        try:
            os.mkdir(logdir + "/numpy_save")
        except:
            pass
        np.save(logdir + "/numpy_save/memory", self.heap)
        np.save(logdir + "/numpy_save/memory_spans", self.span_indicis)
        np.save(logdir + "/numpy_save/probabilities", self.probabilities)
        np.save(logdir + "/numpy_save/CDF_reversed", self.CDF_reversed)
        f = open(logdir + "/numpy_save/constants", "w")
        f.write(str(self.alpha) + "\n")
        f.write(str(self.beta_start) + "\n")
        f.write(str(self.total_beta_time) + "\n")
        f.write(str(self.beta_gradient) + "\n")
        f.write(str(self.max_size) + "\n")
        f.write(str(self.last_row_size) + "\n")
        f.write(str(self.current_size) + "\n")
        f.write(str(self.base_batch_size) + "\n")
        f.close()

    def load(self, logdir):
        self.heap = list(np.load(logdir + "/numpy_save/memory.npy"))
        self.span_indicis = list(np.load(logdir + "/numpy_save/memory_spans.npy"))
        self.probabilities = list(np.load(logdir + "/numpy_save/probabilities.npy"))
        self.CDF_reversed = list(np.load(logdir + "/numpy_save/CDF_reversed.npy"))
        f = open(logdir + "/numpy_save/constants", "r")
        self.alpha = float(f.readline())
        self.beta_start = float(f.readline())
        self.total_beta_time = float(f.readline())
        self.beta_gradient = float(f.readline())
        self.max_size = int(f.readline())
        self.last_row_size = int(f.readline())
        self.current_size = int(f.readline())
        self.base_batch_size = int(f.readline())

        f.close()

        self.NEED_UPDATE_FLAG = False
        self.mem_idxs = []

    def get_beta_plus_weight(self, sess):
        global_step = sess.run(tf.train.get_global_step())
        if global_step <= self.total_beta_time:
            beta = self.beta_start + global_step * self.beta_gradient
            weighting_constant = self.probabilities[-1]
            return beta, weighting_constant
        else:
            beta = 1
            weighting_constant = self.probabilities[-1]
            return beta, weighting_constant

    def unset_need_update(self):
        self.mem_idxs = []
        self.NEED_UPDATE_FLAG = False

    def update(self, priorities=None):
        if not self.is_full():
            raise RankBasedMemoryError('''Memory has to be first fully filled for rank based Memory''')
        if priorities is None:
            pass

        elif len(priorities) != len(self.mem_idxs):
            raise RankBasedMemoryError('''Number of given priorities doesn't match the size of last given batch''')

        else:
            for priority, idx in zip(priorities, self.mem_idxs):
                priority = float(priority)
                self.heap[idx][1] = priority
                self._update_node(idx)

        self.unset_need_update()

    def add_transition(self, transition, priority):
        self._insert(transition, priority)

    def is_full(self):
        return self.current_size > self.max_size

    def get_n_samples(self, batch_size=None):
        if not self.is_full():
            raise RankBasedMemoryError('''Memory has to be first fully filled for rank based Memory''')
        if self.NEED_UPDATE_FLAG:
            raise RankBasedMemoryError('''Memory hasn't been updated after get_n_samples''')

        experiences = []
        probabilities = []
        if batch_size is None:
            for idx, next_idx in zip(self.span_indicis, self.span_indicis[1:]):
                position = np.random.randint(idx, next_idx)
                experiences.append(self.heap[position][0])
                probabilities.append(self.probabilities[position])
                self.mem_idxs.append(position)

        else:
            span_indicis = self._compute_spans(batch_size)
            for idx, next_idx in zip(span_indicis, span_indicis[1:]):
                position = np.random.randint(idx, next_idx)
                experiences.append(self.heap[position][0])
                probabilities.append(self.probabilities[position])
                self.mem_idxs.append(position)

        self.NEED_UPDATE_FLAG = True

        return np.column_stack(experiences), probabilities


class MemoryUniform(Memory):
    '''
    Unoptimized implementation of Uniform Memory, use on your own danger 
    Doesnt support saving yet.
    '''

    def __init__(self, **conf):
        self.max_size = conf["size"]
        self.base_batch_size = conf["batch_size"]
        self.memory_arr = []
        self.oldest_memory = 0

    def add_transition(self, transition):
        if self.is_full():
            if self.oldest_memory == self.max_size:
                self.oldest_memory = 0
            self.memory_arr[self.oldest_memory] = transition
            self.oldest_memory += 1
        else:
            self.memory_arr.append(transition)

    def is_full(self):
        return len(self.memory_arr) >= self.max_size

    def get_n_samples(self, N=None):
        if N is None:
            N = self.base_batch_size
        lenArr = len(self.memory_arr)
        N = min(N, lenArr)
        if lenArr > 0:
            indices = np.random.choice(lenArr, N)
            return np.column_stack([self.memory_arr[ind] for ind in indices])
        else:
            return -1


class RankBasedMemoryError(Exception):

    def __init__(self, message=None):
        self.message = message
