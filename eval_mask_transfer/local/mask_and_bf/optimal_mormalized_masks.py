import argparse
import os

import numpy as np
import scipy.linalg
from tqdm import tqdm
import sys
from distutils.util import strtobool

from chime_data import gen_flist_simu, gen_flist_2ch,\
    gen_flist_real, get_audio_data, get_audio_data_1ch, get_audio_data_with_context
from mbbf.signal_processing import audiowrite, stft, istft
from mbbf.utils import Timer
from mbbf.utils import mkdir_p
#import mbbf.beamforming

MARKER_SIZE = 10

MODE_EIG_LEFT_MAX = 'eig_left_max'
MODE_EIG_LEFT_MIN = 'eig_left_min'
MODE_EIG_RIGHT_MAX = 'eig_right_max'
MODE_EIG_RIGHT_MIN = 'eig_right_min'
MODE_MWF = 'mwf'


from matplotlib import pylab as pl
import nn_models
def plot_specgrams(X, masks, Y, S, N, fig_no=0):
    X = np.abs(X)
    Y = np.abs(Y)
    S = np.abs(S)
    N = np.abs(N)
    masks = np.abs(masks)
    num_masks = masks.shape[1]

    fig = pl.figure(fig_no)
    fig.clf()
    ax = fig.add_subplot(3, 3, 1)
    nn_models.imshow_specgram(ax, S.T, 'hot')
    ax.set_title('Target')

    ax = fig.add_subplot(3, 3, 4)
    nn_models.imshow_specgram(ax, N.T, 'hot')
    ax.set_title('Jammer')

    ax = fig.add_subplot(3, 3, 7)
    nn_models.imshow_specgram(ax, X.T, 'hot')
    ax.set_title('Observation')

    if num_masks > 0:
        m = masks[:,0]
        vmax = 1
        #vmax = np.median(m) * 3
        #vmax = np.median(m)
        #vmax = np.max(m) / 2
        ax = fig.add_subplot(3, 3, 2)
        ax.imshow(m.T, origin="lower", aspect="auto", cmap='gray', vmin=0, vmax=vmax)
        ax.set_title('Mask1')
        ax = fig.add_subplot(3, 3, 3)
        nn_models.imshow_specgram(ax, (X*m+1e-6).T, 'hot')
        ax.set_title('Observation*mask1')

    if num_masks > 1:
        m = masks[:,1]
        vmax = 1
        #vmax = np.median(m) * 3
        #vmax = np.median(m)
        #vmax = np.max(m) / 2
        vmax = np.max(m) / 2
        ax = fig.add_subplot(3, 3, 5)
        ax.imshow(m, origin="lower", aspect="auto", cmap='gray', vmin=0, vmax=vmax)
        ax.set_title('Mask2')
        ax = fig.add_subplot(3, 3, 6)
        nn_models.imshow_specgram(ax, (X*m+1e-6).T, 'hot')
        ax.set_title('Observation*mask2')

    ax = fig.add_subplot(3, 3, 9)
    nn_models.imshow_specgram(ax, Y.T, 'hot')
    ax.set_title('BF output')

    fig.tight_layout()
    #pl.pause(0.00001)
    #input('Hit ENTER')
    #pl.show()

def scatter_plot(mask, S, N, Y, X, fig_no=1, marker='o'):
    #mask = np.abs(mask)
    mask = normalize(np.abs(mask))
    S_abs = normalize(np.abs(S))
    N_abs = normalize(np.abs(N))
    Y_abs = normalize(np.abs(Y))
    X_abs = normalize(np.abs(X))
    fig = pl.figure(fig_no)
    fig.clf()

    ax = fig.add_subplot(3, 3, 1)
    y_data = (S_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('Normalized |Target|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 2)
    y_data = (S_abs / X_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('|Target|/|Observation|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_ylim([0, 100])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 3)
    y_data = (S_abs / N_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('|Target|/|Jammer|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_ylim([0, 1000])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 4)
    y_data = (N_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('Normalized |Jammer|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 5)
    y_data = (N_abs / X_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('|Jammer|/|Observation|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_ylim([0, 100])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 6)
    y_data = (N_abs / S_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('|Jammer|/|Target|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_ylim([0, 1000])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 7)
    y_data = (Y_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('Normalized |Extraction|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_xlim([0, 10])

    ax = fig.add_subplot(3, 3, 8)
    y_data = (Y_abs / X_abs).flatten()
    x_data = (mask).flatten()
    ax.scatter(x=x_data, y=y_data, marker=marker, s=MARKER_SIZE, alpha=0.5, linewidths=1, c='#aaaaFF', edgecolors='b')
    ax.set_ylabel('|Extraction|/|Observation|')
    ax.set_xlabel('Weight')
    ax.set_ylim([0, np.max(y_data)])
    #ax.set_xlim([0, np.max(x_data)])
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    #ax.set_ylim([0, 100])
    #ax.set_xlim([0, 10])

    fig.tight_layout()


def scatter_plot_some_bins(mask, S, N, Y, X, fig=2, marker='o'):
    mask = np.abs(mask)
    S_abs = normalize(np.abs(S))
    N_abs = normalize(np.abs(N))
    Y_abs = normalize(np.abs(Y))
    X_abs = normalize(np.abs(X))
    figure = pl.figure(fig)
    figure.clf()

    bins = S.shape[-1]
    freqs = (0, 250, 500, 1000, 2000, 4000, 8000)
    #freqs = (0, 125, 250, 500, 1000, 2000, 8000)

    y_data = S_abs
    x_data = mask
    for i in range(1, len(freqs)):
        ax = figure.add_subplot(3, len(freqs), i+1)
        begin_freq = freqs[i-1]
        end_freq = freqs[i]
        begin_idx = int(bins * begin_freq/8000)
        end_idx = int(bins * end_freq/8000)
        label = '{}-{}Hz'.format(begin_freq, end_freq)
        ax.scatter(x_data[:,begin_idx:end_idx], y_data[:,begin_idx:end_idx], s=MARKER_SIZE, alpha=0.5, linewidths=1, label=label)
        #ax.set_xlim([0, np.max(x_data)])
        ax.set_xlim([np.min(x_data), np.max(x_data)])
        ax.set_ylim([0, np.max(y_data)])
        ax.legend()

    y_data = N_abs
    x_data = mask
    for i in range(1, len(freqs)):
        ax = figure.add_subplot(3, len(freqs), i+1+len(freqs))
        begin_freq = freqs[i-1]
        end_freq = freqs[i]
        begin_idx = int(bins * begin_freq/8000)
        end_idx = int(bins * end_freq/8000)
        label = '{}-{}Hz'.format(begin_freq, end_freq)
        ax.scatter(x_data[:,begin_idx:end_idx], y_data[:,begin_idx:end_idx], s=MARKER_SIZE, alpha=0.5, linewidths=1, label=label)
        #ax.set_xlim([0, np.max(x_data)])
        ax.set_xlim([np.min(x_data), np.max(x_data)])
        ax.set_ylim([0, np.max(y_data)])
        ax.legend()

    y_data = Y_abs
    x_data = mask
    for i in range(1, len(freqs)):
        ax = figure.add_subplot(3, len(freqs), i+1+len(freqs)*2)
        begin_freq = freqs[i-1]
        end_freq = freqs[i]
        begin_idx = int(bins * begin_freq/8000)
        end_idx = int(bins * end_freq/8000)
        label = '{}-{}Hz'.format(begin_freq, end_freq)
        ax.scatter(x_data[:,begin_idx:end_idx], y_data[:,begin_idx:end_idx], s=MARKER_SIZE, alpha=0.5, linewidths=1, label=label)
        #ax.set_xlim([0, np.max(x_data)])
        ax.set_xlim([np.min(x_data), np.max(x_data)])
        ax.set_ylim([0, np.max(y_data)])
        ax.legend()

    figure.tight_layout()

def normalize(x):
    var = np.mean(x*np.conj(x), axis=0, keepdims=True).real
    return x / np.sqrt(var)
    #idx = (np.abs(var) > np.finfo(var.dtype).eps)
    #x_copy = x.copy()
    #x_copy[idx] /= np.sqrt(var[idx])
    #return x_copy

def htp(x):
    return x.swapaxes(-1, -2).conj()

def mask_based_bf(X, S, mask, target_ch, mode):
    X = X.T
    S = S.T
    mask = mask.T
    Y, mask_after = _mask_based_bf(X, S, mask, target_ch, mode)
    return Y.T, mask_after.T

def _mask_based_bf(X, S, mask, target_ch, mode):
    (bins, channels, frames) = X.shape

    if mode in (MODE_EIG_LEFT_MIN, MODE_EIG_RIGHT_MAX):
        mask = mask.max(axis=-1, keepdims=True) - mask

    XX = np.matmul(X, htp(X))
    mXX = np.matmul(mask*X, htp(X))
    if mode == MODE_EIG_LEFT_MAX:
        w = _gevh(mXX, XX, minimize=False)
    elif mode == MODE_EIG_LEFT_MIN:
        w = _gevh(mXX, XX, minimize=True)
    elif mode == MODE_EIG_RIGHT_MAX:
        w = _gevh(XX, mXX, minimize=False)
    elif mode == MODE_EIG_RIGHT_MIN:
        w = _gevh(XX, mXX, minimize=True)
    else: # mode == MODE_MWF
        W = np.matmul(np.linalg.pinv(XX), mXX)
        w = _power_method((W), target_ch, 1)
    Y = np.matmul(htp(w), X)
    Y = _rescale(Y, S[:,target_ch:target_ch+1])

    return Y, mask

def _power_method(X, target_ch, iterations):
    (bins, rows, cols) = X.shape
    w = np.zeros((bins,rows,1), dtype=X.dtype)
    w[:,target_ch,:] = 1
    for _ in range(iterations):
        w = np.matmul(X, w)
        w /= np.sqrt(np.matmul(htp(w),w))
    return w

def _gevh(A, B, minimize):
    return _gevh_np_cholesky(A, B, minimize)
    #return _gevh_scipy(A, B, minimize)
    #return _gevh_np_decorr(A, B, minimize)

def _gevh_np_decorr(A, B, minimize):
    D, Q = np.linalg.eigh(B)
    D = D[...,np.newaxis]
    htp_P = Q / np.sqrt(D)
    P = htp(htp_P)   # P = diag(D)^(-1/2) Q^H
    PAP = np.matmul(np.matmul(P, A), htp_P)
    _, V = np.linalg.eigh(PAP)
    if minimize:
        v = V[...,:1]
    else:
        v = V[...,-1:]
    return np.matmul(htp_P, v)

def _gevh_np_cholesky(A, B, minimize):
    L = np.linalg.cholesky(B)
    #P = np.linalg.pinv(L)
    P = np.linalg.inv(L)
    htp_P = htp(P)
    PAP = np.matmul(np.matmul(P, A), htp_P)
    _, V = np.linalg.eigh(PAP)
    if minimize:
        v = V[...,:1]
    else:
        v = V[...,-1:]
    return np.matmul(htp_P, v)

def _gevh_scipy(A, B, minimize):
    (binf, rows, cols) = A.shape
    V = np.zeros_like(A)
    for f in range(bins):
        _, V[f] = scipy.linalg.eigh(A[f], B[f])
    if minimize:
        v = V[...,:1]
    else:
        v = V[...,-1:]
    return v


def _rescale(y, rescale_target):
    cov = np.matmul(rescale_target, htp(y))
    var = np.matmul(y, htp(y)).real
    scale = cov
    idx = (np.abs(var) > np.finfo(var.dtype).eps)
    scale[idx] /= var[idx]
    y_scaled = y * scale
    return y_scaled


parser = argparse.ArgumentParser(description='Weighted minimum variance BF')
parser.add_argument('flist',
                    help='Name of the flist to process (e.g. tr05_simu)')
parser.add_argument('chime_dir',
                    help='Base directory of the CHiME challenge.')
parser.add_argument('sim_dir',
                    help='Base directory of the CHiME challenge simulated data.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
#parser.add_argument('--single', '-s', default=0, type=int,
#                    help='0 for multi-channel and channel number (1-6) for single channel')
parser.add_argument('--track', '-t', default=6, type=int,
                    help='1, 2 or 6 depending on the data used')
parser.add_argument('--target_mic', '-m', default=5, type=int,
                    help='Index of target microphone')
parser.add_argument('--max_frames', default=None, type=int,
                    help='Max. number of frames of a spectrogram')
parser.add_argument('--bg_gain', default=1., type=float,
                    help='Gain for background noise.')
parser.add_argument('--plot', '-p', default=False, type=strtobool,
                    help='Display spectrograms of observation and estimated target source')
parser.add_argument('--mode',
                    help='Mode. (both_max|both_min|left_max|left_min|right_max|right_min|ideal_max|ideal_min)')
args = parser.parse_args()

if args.track != 6:
    raise ValueError('Only track=6 case is supported.')
#if args.single != 0:
#    raise ValueError('Only single=0 case is supported.')

if args.mode not in (MODE_EIG_LEFT_MAX,  MODE_EIG_LEFT_MIN,
                     MODE_EIG_RIGHT_MAX, MODE_EIG_RIGHT_MIN,
                     MODE_MWF):
    raise ValueError('Unknoen mode:', args.mode)

MASK_BASE_DIR = os.path.join(
    '..',
    'eval_normalized_masks_powm',
    'enhan',
    'normalized_masks_powm_bg-{}_mode-right_powm-1'.format(args.bg_gain),
)


stage = args.flist[:2]
scenario = args.flist.split('_')[-1]

if stage == 'tr' and (args.track == 1 or args.track == 2):
    print("No train data for 1ch track and 2ch track")
    sys.exit(0)

# CHiME data handling
if scenario == 'simu':
    if args.track == 6:
        flist = gen_flist_simu(args.chime_dir, args.sim_dir, stage, ext=True)
    else:
        raise ValueError('Only case of track=6 is supported.')
else:
    raise ValueError('Unknown flist {}'.format(args.flist))

for env in ['caf', 'bus', 'str', 'ped']:
    mkdir_p(os.path.join(args.output_dir, '{}05_{}_{}'.format(
            stage, env, scenario
    )))

target_ch = args.target_mic - 1

t_io = 0
t_bf = 0

last_idx = len(flist)
#last_idx = min(5, last_idx)
# Beamform loop
#for cur_line in tqdm(flist):
for cur_line in tqdm(flist[:last_idx]):
    with Timer() as t:
        if args.track == 6:
            if scenario == 'simu':
                audio_data_clean = get_audio_data(cur_line, postfix='.Clean')
                audio_data_noise = get_audio_data(cur_line, postfix='.Noise')
                audio_data_noise *= args.bg_gain
                audio_data = audio_data_clean + audio_data_noise
                context_samples = 0
            elif scenario == 'real':
                raise ValueError('Scenario \'real\' is not supported.')
        else:
            raise ValueError('Case of track=1 is not supported. Specify 2 or 6.')
    t_io += t.msecs
    X = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    S = stft(audio_data_clean, time_dim=1).transpose((1, 0, 2))
    N = stft(audio_data_noise, time_dim=1).transpose((1, 0, 2))
    if args.max_frames is not None:
        X = X[:args.max_frames]
        S = S[:args.max_frames]
        N = N[:args.max_frames]
    X = X.astype(np.complex64)
    S = S.astype(np.complex64)
    N = N.astype(np.complex64)
    frames, channels, bins = X.shape

    if scenario == 'simu' or args.track == 2:
        wsj_name = cur_line.split('/')[-1].split('_')[1]
        spk = cur_line.split('/')[-1].split('_')[0]
        env = cur_line.split('/')[-1].split('_')[-1]
    elif scenario == 'real':
        raise ValueError('Scenario \'real\' is not supported.')

    mask_filename = os.path.join(
            MASK_BASE_DIR,
            '{}05_{}_{}'.format(stage, env.lower(), scenario),
            'masks_{}_{}_{}.npy'.format(spk, wsj_name, env.upper())
    )
    #print('Reading', mask_filename)
    with Timer() as t:
        mask = np.load(mask_filename)
    t_io += t.msecs

    with Timer() as t:
        Y, mask_after = mask_based_bf(X, S, mask=mask, target_ch=target_ch, mode=args.mode)
    t_bf += t.msecs

    if args.plot:
        scatter_plot(mask[:,0], S[:,target_ch], N[:,target_ch], Y[:,0], X[:,target_ch], fig_no=1)
        plot_specgrams(X[:,target_ch], mask, Y[:,0], S[:,target_ch], N[:,target_ch])
        pl.show()

    Y = Y.squeeze(1)

    filename = os.path.join(
            args.output_dir,
            '{}05_{}_{}'.format(stage, env.lower(), scenario),
            '{}_{}_{}.wav'.format(spk, wsj_name, env.upper())
    )

    with Timer() as t:
        audiowrite(istft(Y)[int(context_samples):], filename, 16000, True, True)
    t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | BF: {:.2f}s'.format(
            t_io / 1000, t_bf / 1000))
