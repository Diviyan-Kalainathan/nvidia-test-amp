#!/usr/bin/env python

import torch as th
from cos_dataset import generate_pair
from sklearn.preprocessing import minmax_scale, scale
from joblib import Parallel, delayed
from itertools import product
import numpy as np
from tqdm import trange
import shutil
from torch.utils import data
from apex import amp
amp_handle = amp.init()


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datas, labels):
        'Initialization'
        self.labels = labels
        self.datas = datas

  def __len__(self):
        'Denotes the total number of samples'
        return self.datas.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        X = self.datas[index]
        y = self.labels[index]

        return X, y

def compute_pixel_gaussian(i, j, dataset, resolution):
    out = 0
    for x, y in dataset:
        out += np.exp(-((x-i/resolution[0])**2 + (y-j/resolution[1])**2)/.001)
    return out


def compute_pixel(i, j, dataset, resolution):
    _i = i/resolution[0]
    i_ = (i+1)/resolution[0]
    _j = j/resolution[1]
    j_ = (j+1)/resolution[1]
    step = 1/resolution[0]
    for x, y in dataset:
        if x - _i > 0 and x - i_ < 0  and y - _j > 0 and y - j_ < 0:
            return 1
    return 0

def build_image(pair, shape=(64, 64), pixel_compute="gaussian"):
    compute = {'raw': compute_pixel, 'gaussian': compute_pixel_gaussian}
    data = minmax_scale(pair.reshape(-1, 1)).reshape(-1, 2)
    pixels = Parallel(n_jobs=32)(delayed(compute[pixel_compute])(i, j, data, shape)
                                for i, j in product(range(shape[0]), range(shape[1])))
    return np.float32(scale(np.array(pixels)).reshape(shape))

def build_dataset(n):
    data_points = []
    out = []
    for i in trange(n):
        pair = generate_pair(500, noisef=np.random.choice(['uniform', 'gaussian']),
                             causef=np.random.choice(['gmm', 'normal']))
        pair = np.array(pair).transpose()
        data_points.append(np.float32(pair))
        out.append(build_image(pair))
    return np.stack(data_points, 0) , np.stack(out, 0)


class CNN_model(th.nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()  # size 1x64x64
        layers = []
        layers.append(th.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2))
        layers.append(th.nn.BatchNorm2d(16))
        layers.append(th.nn.ReLU())
        layers.append(th.nn.MaxPool2d(kernel_size=2, stride=2))
        # size 16x32x32
        layers.append(th.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2))
        layers.append(th.nn.BatchNorm2d(32))
        layers.append(th.nn.ReLU())
        layers.append(th.nn.MaxPool2d(kernel_size=2, stride=2))
        # size 32x16x16
        layers.append(th.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2))
        layers.append(th.nn.BatchNorm2d(64))
        layers.append(th.nn.ReLU())
        layers.append(th.nn.MaxPool2d(kernel_size=2, stride=2))
        # size 64x8x8
        dense = []
        dense.append(th.nn.Linear(64*8*8, 64*8))
        dense.append(th.nn.ReLU())
        dense.append(th.nn.Linear(64*8, 64))
        dense.append(th.nn.ReLU())
        dense.append(th.nn.Linear(64, 1))

        self.conv = th.nn.Sequential(*layers)
        self.dense = th.nn.Sequential(*dense)

    def forward(self, x):
        return th.sigmoid(self.dense(self.conv(x).view(x.shape[0], -1))), x

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main(input , labels, batchsize=32, epochs=15000, device="cpu"):
    input = th.FloatTensor(input).unsqueeze(1).to(device)
    labels = th.FloatTensor(labels).unsqueeze(1).to(device)
    dataset = Dataset(input, labels)
    model = CNN_model().to(device)
    train_loader = data.DataLoader(dataset, batch_size=batchsize, shuffle=True,
                                   pin_memory=True, drop_last=True)

    optim = amp_handle.wrap_optimizer(th.optim.Adam(model.parameters(),
                                                    lr=0.01))
    criterion = th.nn.BCELoss(reduce=True, size_average=True)
    try:
        with trange(epochs) as t:
            for epoch in t:
                for idx, (batch, b_labels) in enumerate(train_loader):
                    out = model(batch)[0]
                    loss = criterion(out, b_labels)
                    with amp_handle.scale_loss(loss, optim) as scaled_loss:
                        scaled_loss.backward()
                    optim.step()
                    t.set_postfix(loss=loss.data)
    except KeyboardInterrupt:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optim.state_dict(),
        }, False)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optim.state_dict(),
        }, False)

def generate_scatter_dataset():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    pair = generate_pair(500, noisef=np.random.choice(['uniform', 'gaussian']),
                                   causef=np.random.choice(['gmm', 'normal']))
    pair = np.array(pair).transpose()
    print(pair.shape)
    im, im2 = build_image(pair, pixel_compute='raw'), build_image(pair)
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax = axs[0]
    image = ax.imshow(im)
    ax.set_title(r"$\textrm{a) raw scatter plot}$")
    ax.axis('off')
    ax2 = axs[1]
    image2 = ax2.imshow(im2)
    ax2.set_title(r"$\textrm{b) Gaussian diffusion}$")
    plt.axis('off')
    plt.show()
    print(im)
    print("Done ! ")
    dataset, scatters = build_dataset(5000)

    np.save("sine_dataset.npy", dataset, allow_pickle=True)
    np.save("sine_scatter.npy", scatters, allow_pickle=True)


if __name__ == "__main__":
    print("Starting experiment...")
    training_set = np.load("sine_scatter.npy")
    # shuffle it !

    labels = np.ones(training_set.shape[0])
    for i in trange(training_set.shape[0]):
        if np.random.choice([True, False]):
            labels[i] = 0
            training_set[i, :, :] = np.flip(training_set[i, :, :], 1)
    print("Starting training...")
    main(training_set, labels, device="cpu")
