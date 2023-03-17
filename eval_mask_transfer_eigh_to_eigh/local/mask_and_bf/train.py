import argparse
import json
import logging
import os
import pickle

import numpy as np
#from chainer import Variable
#from chainer import cuda
#from chainer import optimizers
#from chainer import serializers
import torch
from tqdm import tqdm

from chime_data import prepare_training_data
from mbbf.utils import Timer
from mbbf.utils import mkdir_p
from nn_models import BLSTMMaskBfEstimator

parser = argparse.ArgumentParser(description='NN GEV training')
parser.add_argument('data_dir', help='Directory used for the training data '
                                     'and to store the model file.')
parser.add_argument('model_type',
                    help='Type of model (BLSTM or FW)')
parser.add_argument('--chime_dir', default='',
                    help='Base directory of the CHiME challenge. This is '
                         'used to create the training data. If not specified, '
                         'the data_dir must contain some training data.')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--max_epochs', default=25, type=int,
                    help='Max. number of epochs to train')
parser.add_argument('--patience', default=5, type=int,
                    help='Max. number of epochs to wait for better CV loss')
parser.add_argument('--dropout', default=.5, type=float,
                    help='Dropout probability')
args = parser.parse_args()

log = logging.getLogger('mbbf')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.data_dir, 'mbbf.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

#if args.chime_dir != '':
#    log.info(
#            'Preparing training data and storing it in {}'.format(
#                    args.data_dir))
#    prepare_training_data(args.chime_dir, args.data_dir)


flists = dict()
for stage in ['tr', 'dt']:
    with open(
            os.path.join(args.data_dir, 'flist_{}.json'.format(stage))) as fid:
        flists[stage] = json.load(fid)
log.debug('Loaded file lists')

# Prepare model
if args.model_type == 'BLSTM':
    model = BLSTMMaskBfEstimator(dropout=args.dropout)
    model_save_dir = os.path.join(args.data_dir, 'BLSTM_model')
    mkdir_p(model_save_dir)
else:
    raise ValueError('Unknown model type. Possible is "BLSTM" only')

if args.gpu >= 0:
    model.cuda(args.gpu)
log.debug('Prepared model')

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters())

# Init/Resume
#if args.initmodel:
#    print('Load model from', args.initmodel)
#    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    #serializers.load_hdf5(args.resume, optimizer)
    model.load_state_dict(torch.load(args.resume))



def _create_batch(file):
    with open(os.path.join(args.data_dir, file), 'rb') as fid:
        data = pickle.load(fid)
    X = torch.from_numpy(data['X'].astype(np.complex64))
    S = torch.from_numpy(data['S'].astype(np.complex64))
    if args.gpu >= 0:
        X = X.cuda(args.gpu)
        S = S.cuda(args.gpu)
    return X, S


# Learning loop
epoch = 0
exhausted = False
best_epoch = 0
best_cv_loss = np.inf
while (epoch < args.max_epochs and not exhausted):
    log.info('Starting epoch {}. Best CV loss was {} at epoch {}'.format(
            epoch, best_cv_loss, best_epoch
    ))

    # training
    perm = np.random.permutation(len(flists['tr']))
    #perm = perm[:1000]
    sum_loss_tr = 0
    t_io = 0
    t_fw = 0
    t_bw = 0
    model.train()
    for i in tqdm(perm, desc='Training epoch {}'.format(epoch)):
        with Timer() as t:
            X, S = _create_batch(flists['tr'][i])
        t_io += t.msecs

        model.zero_grad()
        with Timer() as t:
            loss = model.train_and_cv(X, S)
        t_fw += t.msecs

        with Timer() as t:
            loss.backward()
            optimizer.step()
        t_bw += t.msecs

        sum_loss_tr += float(loss.cpu().detach().numpy())
    
    # cross-validation
    sum_loss_cv = 0
    model.eval()
    for i in tqdm(range(len(flists['dt'])), desc='Cross-validation epoch {}'.format(epoch)):
    #for i in tqdm(range(5), desc='Cross-validation epoch {}'.format(epoch)):
        X, S = _create_batch(flists['dt'][i])
        with torch.no_grad():  ## disable autograd
            loss = model.train_and_cv(X, S, plot=(i==0))
        sum_loss_cv += float(loss.cpu().detach().numpy())

    loss_tr = sum_loss_tr / len(flists['tr'])
    loss_cv = sum_loss_cv / len(flists['dt'])

    log.info(
            'Finished epoch {}. '
            'Mean loss during training/cross-validation: {:.3f}/{:.3f}'.format(
                    epoch, loss_tr, loss_cv))
    log.info('Timings: I/O: {:.2f}s | FW: {:.2f}s | BW: {:.2f}s'.format(
            t_io / 1000, t_fw / 1000, t_bw / 1000))

    if loss_cv < best_cv_loss or (args.patience < 0 and epoch + 1 == args.max_epochs):
        best_epoch = epoch
        best_cv_loss = loss_cv
        model_file = os.path.join(model_save_dir, 'best.nnet')
        log.info('New best loss during cross-validation. Saving model file '
                 'under {}'.format(model_file))
        #serializers.save_hdf5(model_file, model)
        #serializers.save_hdf5(os.path.join(model_save_dir, 'mlp.tr'), optimizer)
        model.cpu()
        torch.save(model.state_dict(), model_file)
        if args.gpu >= 0:
            model.cuda(args.gpu)

    if epoch - best_epoch == args.patience:
        exhausted = True
        log.info('Patience exhausted. Stopping training')

    epoch += 1

log.info('Finished!')
