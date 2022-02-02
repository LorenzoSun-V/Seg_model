import numpy as np


class RandomSampler(object):
    """random sampler to yield a mini-batch of indices."""
    def __init__(self, data_iter, batch_size, drop_last=False):
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.num_imgs = len(data_iter)
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.random.permutation(self.num_imgs)
        batch = []
        for i in indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        ## if images not to yield a batch
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_imgs // self.batch_size
        else:
            return (self.num_imgs + self.batch_size - 1) // self.batch_size