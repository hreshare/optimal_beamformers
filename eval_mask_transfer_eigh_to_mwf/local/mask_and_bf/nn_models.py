import torch
import torch.nn as nn
import torch.nn.functional as F

import mbbf.beamforming

import numpy as np
from matplotlib import pylab as pl
def imshow_specgram(ax, spectrogram, cmap='hot', rm_low=0.1):
    spectrogram = np.abs(spectrogram)
    max_value = spectrogram.max()
    ### amp to dbFS
    db_spec = np.lib.scimath.log10(spectrogram / float(max_value)) * 20
    hist, bin_edges = np.histogram(db_spec.flatten(), bins = 1000, density = True)
    hist /= float(hist.sum())
    ax.hist(hist)
    S = 0
    ii = 0
    while S < rm_low:
        S += hist[ii]
        ii += 1
    vmin = bin_edges[ii]
    vmax = db_spec.max()
    ax.imshow(db_spec, origin="lower", aspect="auto", cmap=cmap, vmax=vmax, vmin=vmin)

def to_np(x):
    return x.cpu().detach().numpy().copy()

def plot_specgrams(X, S, masks, bf_out):
    fig = pl.figure(0)
    fig.clf()
    ax = fig.add_subplot(4, 1, 1)
    imshow_specgram(ax, to_np(X[:,0,:]).T, 'hot')
    ax = fig.add_subplot(4, 1, 2)
    ax.imshow(to_np(masks[:,0,:]).T, origin="lower", aspect="auto", cmap='gray', vmax=1, vmin=0)
    ax = fig.add_subplot(4, 1, 3)
    imshow_specgram(ax, to_np(bf_out[:,0,:]).T, 'hot')
    ax = fig.add_subplot(4, 1, 4)
    imshow_specgram(ax, to_np(S[:,0,:]).T, 'hot')
    pl.pause(0.000001)
    #input('Hit ENTER')




class SequenceBLSTM(nn.Module):
    """
        A batchnorm wrapper for single RNN layer.
    """
    def __init__(self, input_size, output_size, normalized=True, concat=False, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.blstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=output_size,
            bidirectional=True,
        )
        self.batchnorm = None
        if normalized:
            if concat:
                insize = output_size * 2
            else:
                insize = output_size
            self.batchnorm = nn.BatchNorm1d(insize)
        self.concat = concat

    def forward(self, x):
        """
            Parameter x: T x N x F
                T: number of frames
                N: number of channels
                F: number of frequency bins
        """
        # go through RNN
        (T, N, _) = x.shape
        x = self.dropout(x)
        x, _ = self.blstm(x)
        if not self.concat:
            border = x.shape[-1] // 2
            x_former = x[...,:border]
            x_latter = x[...,border:]
            x = x_former + x_latter
        x = x.view(T*N, -1)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = x.view(T, N, -1)
        return x


class SequenceLinear(nn.Module):
    """
        A single fully connected layer with batchnorm
    """
    def __init__(self, input_size, output_size, normalized=True, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_transform = nn.Linear(input_size, output_size)
        self.batchnorm = None
        if normalized:
            self.batchnorm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        """
            linear transform => batchnorm
        """
        x = self.dropout(x)
        x = self.linear_transform(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        return x



class BLSTMMaskBfEstimator(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.blstm_layer = SequenceBLSTM(513, 256, normalized=True, dropout=dropout)
        self.rest_layers = nn.Sequential(
            SequenceLinear(256, 513, normalized=True, dropout=dropout),
            nn.ReLU(),
            SequenceLinear(513, 513, normalized=True, dropout=dropout),
            nn.ReLU(),
            SequenceLinear(513, 513, normalized=True),
            nn.Sigmoid(),
        )

    def _calc_masks(self, x):
        (T, N, _) = x.shape
        x = x.abs()
        x = self.blstm_layer(x)
        x = x.view(T*N, -1)
        x = self.rest_layers(x)
        masks = x.view(T, N, -1) 
        return masks

    def _mask_based_bf(self, x, masks, rescale_targets):
        """
        x.shape = (frames, mics, bins)
        masks.shape = (frames, C, bins), where C is number of microphones or 1.
        rescale_targets.shape = (frames, C, bins)
        """
        x = x.T.unsqueeze(1)            # (bins, 1, mics, frames)
        masks = masks.T.unsqueeze(-2)   # (bins, C, 1, frames)
        rescale_targets = rescale_targets.T.unsqueeze(-2) # (bins, C, 1, frames)
        bf_out = mbbf.beamforming.mask_based_bf(x, masks, rescale_targets)  # bf_out.shape = (bins, C, 1, frames)
        bf_out = bf_out.squeeze(-2).T   # (frames, C, frames)
        return bf_out

    def calc_mask_and_bf(self, x, target_mic=0):
        with torch.no_grad():  ## disable autograd
            masks = self._calc_masks(x)
            mask = masks[:,target_mic:target_mic+1]
            rescale_target = x[:,target_mic:target_mic+1]
            bf_out = self._mask_based_bf(x, mask, rescale_target)
        return bf_out, masks

    def train_and_cv(self, X, S, plot=False):
        masks = self._calc_masks(X)
        bf_out = self._mask_based_bf(X, masks, X)
        if plot:
            plot_specgrams(X, S, masks, bf_out)
        loss = F.mse_loss(bf_out.real, S.real) + F.mse_loss(bf_out.imag, S.imag)
        return loss
