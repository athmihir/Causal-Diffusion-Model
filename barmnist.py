import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils


def gen_scm(cg, n):
    if cg == 'backdoor':
        # ud = torch.randint(10, size=(n, 1))
        # d = ud.int()
        # b = torch.bernoulli(0.1 * (ud+0.5)).int()
        # # t_1 = torch.logical_xor(torch.bernoulli(0.1 * (d+0.5)), torch.bernoulli(0.1 * torch.ones((n, 1)))).int()
        # t_1 = torch.logical_xor(d >= 5, torch.bernoulli(0.1 * torch.ones((n, 1)))).int()
        # t_2 = torch.logical_xor(b, torch.bernoulli(0.8 * torch.ones((n, 1)))).int()
        # t_3 = torch.bernoulli(0.7 * torch.ones((n, 1)))
        # c = torch.logical_and(torch.logical_or(t_1, t_2), t_3).int()

        ud = torch.randint(10, size=(n, 1))
        d = ud.int()
        c = torch.bernoulli(0.1 * (ud+0.5)).int()
        # # t_1 = torch.logical_xor(torch.bernoulli(0.1 * (d+0.5)), torch.bernoulli(0.1 * torch.ones((n, 1)))).int()
        t_1 = torch.logical_xor(d >= 5, torch.bernoulli(0.8 * torch.ones((n, 1)))).int()
        t_2 = torch.logical_xor(c, torch.bernoulli(0.9 * torch.ones((n, 1)))).int()
        t_3 = torch.bernoulli(0.75 * torch.ones((n, 1)))
        b = torch.logical_and(torch.logical_or(t_1, t_2), t_3).int()
        v = dict()
        v['Digit'] = d
        v['Bar'] = b
        v['Color'] = c
        return v

    elif cg == 'frontdoor':
        ud = torch.randint(10, size=(n, 1))
        d = ud.int()
        c = torch.bernoulli(0.1 * (9.5 - d)).int()
        t_1 = torch.logical_xor(ud < 3, torch.bernoulli(0.9 * torch.ones((n, 1)))).int()
        t_2 = torch.logical_xor(c, torch.bernoulli(0.8 * torch.ones((n, 1)))).int()
        t_3 = torch.bernoulli(0.7 * torch.ones((n, 1)))
        b = torch.logical_and(torch.logical_or(t_1, t_2), t_3).int()
        v = dict()
        v['Digit'] = d
        v['Bar'] = b
        v['Color'] = c

        return v

    else:
        raise RuntimeError('The SCM is not implemented.')


def barmnistfI(d, b, c, n, raw_mnist):
    total = len(raw_mnist[d.item()])
    ind = torch.randint(total, size=[1])
    I = raw_mnist[d.item()][ind].clone().numpy().reshape(28, 28, 1)
    dtype = I.dtype

    if c == 0:
        arr = np.concatenate([np.zeros((28, 28, 1), dtype=dtype),
                              I,
                              np.zeros((28, 28, 1), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([I,
                              np.zeros((28, 28, 1), dtype=dtype),
                              np.zeros((28, 28, 1), dtype=dtype)], axis=2)

    if b == 1:
        arr[0:4, :, 2] = 255
    return arr


class BarMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
    """
    def __init__(self, cg, root='./', env='train', transform=None, target_transform=None, ow=False):
        super(BarMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.prepare_bar_mnist(ow, cg, env)
        if env in ['train', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'BARMNIST', cg, env) + '.pt', weights_only=False)
        else:
            raise RuntimeError(f'{env} unknown. Valid envs are train, test')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_bar_mnist(self, ow, cg, env, n=60000):
        if not os.path.isdir(os.path.join(self.root, 'BARMNIST')):
            os.mkdir(os.path.join(self.root, 'BARMNIST'))
        bar_mnist_dir = os.path.join(self.root, 'BARMNIST', cg)
        if os.path.exists(os.path.join(bar_mnist_dir, env+'.pt')) and not ow:

            print('BAR MNIST' + cg + ' dataset already exists')
            return

        print('Preparing BAR MNIST')
        print('Causal Diagram:' + cg)

        if env not in ['train', 'test']:
            raise RuntimeError(f'{env} unknown. Valid envs are train and test')

        if env == 'train':
            mnist_data = datasets.mnist.MNIST(self.root, train=True, download=True)
        else:
            mnist_data = datasets.mnist.MNIST(self.root, train=False, download=True)

        images = mnist_data.data
        labels = mnist_data.targets

        raw_mnist_n = len(images)
        raw_mnist_images = dict()
        for i in range(len(labels)):
            if labels[i].item() not in raw_mnist_images:
                raw_mnist_images[labels[i].item()] = []
            raw_mnist_images[labels[i].item()].append(images[i])

        v_sample = gen_scm(cg, raw_mnist_n)

        dat_set = []
        for i in range(raw_mnist_n):
            arr = barmnistfI(v_sample['Digit'][i], v_sample['Bar'][i], v_sample['Color'][i],
                             raw_mnist_n, raw_mnist_images)
            label_oh = torch.zeros(10)
            label_oh[int(v_sample['Digit'][i])] = 1
            bar_oh = torch.zeros(2)
            bar_oh[int(v_sample['Bar'][i])] = 1
            color_oh = torch.zeros(2)
            color_oh[int(v_sample['Color'][i])] = 1
            dat_set.append((Image.fromarray(arr),
                              torch.cat((label_oh.view(-1), bar_oh.view(-1), color_oh.view(-1)))))

        if not os.path.isdir(bar_mnist_dir):
            os.mkdir(bar_mnist_dir)
        torch.save(dat_set, os.path.join(bar_mnist_dir, env+'.pt'))


if __name__ == "__main__":

    cg = 'frontdoor'
    BarMNIST(cg=cg, ow=True)
    trans_f = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ])
    train_set = BarMNIST(cg=cg, root='./', env='train', transform=trans_f)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=40, shuffle=True,
                                               drop_last=True)

    for idx, (x, y) in enumerate(train_loader):
        save_image(
            x,
            cg + "_samples.png",
            pad_value=2
        )
        print(y)
        break

    # trans_f = transforms.Compose([transforms.ToTensor(),
    #                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                               ])
    #
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True,
    #                                            drop_last=True)
    #
    # print('!')
    #
    # for idx, (x, y) in enumerate(train_loader):
    #     print(y)