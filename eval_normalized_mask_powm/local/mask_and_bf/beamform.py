import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
import sys
from distutils.util import strtobool

from chime_data import gen_flist_simu, gen_flist_2ch,\
    gen_flist_real, get_audio_data, get_audio_data_1ch, get_audio_data_with_context
#from mbbf.beamforming import gev_wrapper_on_masks
from mbbf.signal_processing import audiowrite, stft, istft
from mbbf.utils import Timer
from mbbf.utils import mkdir_p
from nn_models import BLSTMMaskBfEstimator


from matplotlib import pylab as pl
import nn_models
def plot_specgrams(X, mask, Y):
    X = np.abs(X)
    Y = np.abs(Y)
    mask = np.abs(mask)

    fig = pl.figure(0)
    fig.clf()
    ax = fig.add_subplot(4, 1, 1)
    nn_models.imshow_specgram(ax, X.T, 'hot')
    ax = fig.add_subplot(4, 1, 2)
    ax.imshow(mask.T, origin="lower", aspect="auto", cmap='gray', vmax=1, vmin=0)
    ax = fig.add_subplot(4, 1, 3)
    nn_models.imshow_specgram(ax, (X*mask).T, 'hot')
    if Y is not None:
        ax = fig.add_subplot(4, 1, 4)
        nn_models.imshow_specgram(ax, Y.T, 'hot')
    #pl.pause(5)
    pl.pause(0.00001)
    #input('Hit ENTER')


def to_np(x):
    return x.cpu().detach().numpy().copy()


parser = argparse.ArgumentParser(description='MBBF or softmasking')
parser.add_argument('flist',
                    help='Name of the flist to process (e.g. tr05_simu)')
parser.add_argument('chime_dir',
                    help='Base directory of the CHiME challenge.')
parser.add_argument('sim_dir',
                    help='Base directory of the CHiME challenge simulated data.')
parser.add_argument('output_dir',
                    help='The directory where the enhanced wav files will '
                         'be stored.')
parser.add_argument('model',
                    help='Trained model file')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--single', '-s', default=0, type=int,
                    help='0 for multi-channel and channel number (1-6) for single channel')
parser.add_argument('--track', '-t', default=6, type=int,
                    help='1, 2 or 6 depending on the data used')
parser.add_argument('--target_mic', '-m', default=5, type=int,
                    help='Index of target microphone')
parser.add_argument('--plot', '-p', default=False, type=strtobool,
                    help='Display spectrograms of observation and estimated target source')
args = parser.parse_args()

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskBfEstimator()
else:
    raise ValueError('Unknown model type. Possible is "BLSTM" only')

if args.track == 1:
    raise ValueError('Case of track=1 is not supported. Specify 2 or 6.')


model.load_state_dict(torch.load(args.model))
if args.gpu >= 0:
    model.cuda(args.gpu)
model.eval()  ## torch.nn.Module.eval

stage = args.flist[:2]
scenario = args.flist.split('_')[-1]

if stage == 'tr' and (args.track == 1 or args.track == 2):
    print("No train data for 1ch track and 2ch track");
    sys.exit(0);

# CHiME data handling
if scenario == 'simu':
    if args.track == 6:
        flist = gen_flist_simu(args.chime_dir, args.sim_dir, stage)
    elif args.track == 2:
        flist = gen_flist_2ch(args.chime_dir, stage, scenario)
    else:
        raise ValueError('Case of track=1 is not supported. Specify 2 or 6.')

elif scenario == 'real':
    if args.track == 6:
        flist = gen_flist_real(args.chime_dir, stage)
    elif args.track == 2:
        flist = gen_flist_2ch(args.chime_dir, stage, scenario)
    else:
        raise ValueError('Case of track=1 is not supported. Specify 2 or 6.')
else:
    raise ValueError('Unknown flist {}'.format(args.flist))

for env in ['caf', 'bus', 'str', 'ped']:
    mkdir_p(os.path.join(args.output_dir, '{}05_{}_{}'.format(
            stage, env, scenario
    )))

target_ch = args.target_mic - 1

t_io = 0
t_net = 0
t_beamform = 0
# Beamform loop
for cur_line in tqdm(flist):
    with Timer() as t:
        if args.track == 6:
            if scenario == 'simu':
                audio_data = get_audio_data(cur_line)
                context_samples = 0
            elif scenario == 'real':
                audio_data, context_samples = get_audio_data_with_context(
                        cur_line[0], cur_line[1], cur_line[2])
        elif args.track == 2:
            audio_data = get_audio_data(cur_line)
            context_samples = 0
        else:
            raise ValueError('Case of track=1 is not supported. Specify 2 or 6.')
    t_io += t.msecs
    X = stft(audio_data, time_dim=1).transpose((1, 0, 2))
    X_tensor = torch.from_numpy(X.astype(np.complex64))
    if args.gpu >= 0:
        X_tensor = X_tensor.cuda(args.gpu)
    with Timer() as t:
        with torch.no_grad():  ## disable autograd
            Y, masks = model.calc_mask_and_bf(X_tensor, target_mic=target_ch)
        if args.single >= 1:
            ch = args.single - 1
            Y = masks[:,ch] * X_tensor[:,ch]
        elif args.single == 0:
            Y = Y.squeeze(1)
    t_net += t.msecs

    Y = to_np(Y)

    if args.plot:
        if args.single == 0:
            plot_specgrams(X[:,target_ch], to_np(masks[:,target_ch]), Y)
        elif args.single >= 1:
            ch = args.single - 1
            plot_specgrams(X[:,ch], to_np(masks[:,ch]), Y)


    if scenario == 'simu' or args.track == 2:
        wsj_name = cur_line.split('/')[-1].split('_')[1]
        spk = cur_line.split('/')[-1].split('_')[0]
        env = cur_line.split('/')[-1].split('_')[-1]
    elif scenario == 'real':
        wsj_name = cur_line[3]
        spk = cur_line[0].split('/')[-1].split('_')[0]
        env = cur_line[0].split('/')[-1].split('_')[-1]

    filename = os.path.join(
            args.output_dir,
            '{}05_{}_{}'.format(stage, env.lower(), scenario),
            '{}_{}_{}.wav'.format(spk, wsj_name, env.upper())
    )

    with Timer() as t:
        audiowrite(istft(Y)[int(context_samples):], filename, 16000, True, True)
    t_io += t.msecs

print('Finished')
print('Timings: I/O: {:.2f}s | Net: {:.2f}s | Beamformer: {:.2f}s'.format(
        t_io / 1000, t_net / 1000, t_beamform / 1000
))
