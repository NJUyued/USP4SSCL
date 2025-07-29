import torch
import numpy as np
import copy


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            # finally sample n_batch*  n_cls(way)* n_per(shot) instances. per bacth.

class BasePreserverCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            classes=torch.arange(len(self.m_ind))
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)

            yield batch


class NewCategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per,):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
        self.classlist=np.arange(np.min(label),np.max(label)+1)
        #print(self.classlist)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.classlist:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        data_source,
        world_size,
        rank,
        batch_size=1,
        classes_per_batch=10,
        epochs=1,
        seed=0,
        unique_classes=False
    ):
        """
        ClassStratifiedSampler

        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)

        :param data_source: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe

      

if __name__ == '__main__':
    q=np.arange(5,10)
    print(q)
    y=torch.tensor([5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,5,5,55,])
    label = np.array(y)  # all data label
    m_ind = []  # the data index of each class
    for i in range(max(label) + 1):
        ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
        ind = torch.from_numpy(ind)
        m_ind.append(ind)
    print(m_ind, len(m_ind))
