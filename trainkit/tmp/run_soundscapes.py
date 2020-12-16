import os
os.environ['MKL_NUM_THREADS'] = '4'

import string
from collections import OrderedDict
from functools import partial

import numpy as np
import timm
import torch
import torch.nn as nn
from trainkit import ConfParser, Trainer

from utils import data, models, utils

import warnings
warnings.filterwarnings('ignore', message='PySoundFile')


def setup_training(run_params: dict, hyper_params: dict):
    # define models body and head
    # _body = timm.create_model('resnet34', pretrained=True, num_classes=0)
    _embed_size = 0
    _freq_size = int(1 + hyper_params['dataset']['spec_kws']['n_fft'] / 2)
    _body = nn.Sequential(OrderedDict([
        ('freq_scale', models.FreqScaleEmbed(freq_size=_freq_size)),
        # ('freq_cat', models.ConcativeFreqEmbed(freq_size=_freq_size, embed_size=_embed_size)),
        ('body', timm.create_model('resnet18', pretrained=True, num_classes=0, in_chans=3+_embed_size))
        # ('body', timm.create_model('resnet34', pretrained=False, num_classes=0, in_chans=128))
        # ('body', timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, in_chans=3+_embed_size))
    ]))
    # nn.init.zeros_(_body.freq_cat.cat_freq_embeds)

    _head_lin = nn.Linear(512, 9)
    # _head_lin = nn.Linear(1280, 9)
    nn.init.zeros_(_head_lin.weight), nn.init.ones_(_head_lin.bias)
    _head = nn.Sequential(OrderedDict([('lin1', _head_lin)]))

    # define loss and metrics functions
    _train_labels_weights = np.array(hyper_params['general']['labels_weights'].split(','),
                                     dtype=np.float32)
    _val_labels_weights = np.array(hyper_params['general']['val_labels_weights'].split(','),
                                   dtype=np.float32)

    _loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(_train_labels_weights), reduction='mean')
    # _loss_fn = utils.FocalLoss(alpha=hyper_params['general']['focal_alpha'],
    #                            gamma=hyper_params['general']['focal_gamma'],
    #                            reduction='mean',
    #                            log_loss_weight=torch.tensor(_train_labels_weights))

    # _multi_class_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(_train_labels_weights), reduction='mean')
    # _loss_fn = utils.CrossEntropyRankingLoss(multiclass_loss_fn=_multi_class_loss_fn,
    #                                          multiclass_loss_weight=0.8)
    # _loss_fn = utils.BCERankingLoss(multiclass_loss_fn=_multi_class_loss_fn,
    #                                 multiclass_loss_weight=0.8)

    # _metrics_fn = nn.NLLLoss(weight=torch.tensor(_val_labels_weights), reduction='mean')
    # _metrics_fn = nn.CrossEntropyLoss(weight=torch.tensor(_val_labels_weights), reduction='none')
    _metrics_fn = partial(utils.custom_ova_roc_auc, **{'weight': _val_labels_weights})

    # init model class
    _model_cls = models.NSecondPadSpecNet
    # _model_cls = models.FirstNSecondSpecNet
    # _model_cls = models.WaveSpecCNN

    model = _model_cls(device=run_params['general']['device'],
                       body=_body, head=_head,
                       loss_fn=_loss_fn,
                       metrics_fn=_metrics_fn,
                       flood_level=0.1)
                       # pool_out_size=256

    # CatFreqScaleEmbed_, FreqScaleEmbed_, FreqScaleEmbed_resNet34_Pretrained, WaveSpecCNN_
    hyper_params['extra'].update({'model_body': 'FreqScaleEmbed_EffNetB0_Pretrained',
                                  # randInit, zeroInitWeight_oneInitBias
                                  'model_head': 'singleLinear_zeroInitWeight_oneInitBias',
                                  # cross_entropy, focal, CE_n_rankingCE, CE_n_rankingBCE
                                  'loss': 'cross_entropy',
                                  # softmaxed_logits_mean, full_audio_used, first_30_second_used
                                  'preds_aggregation': 'softmaxed_logits_mean'
                                  })

    # datasets
    _labels_map = dict([(string.ascii_uppercase[i], i) for i in range(9)])

    _dataset_cls = data.NSecondPadDataset
    # _dataset_cls = data.FirstNSecondDataset
    _drop_quiet_clips = False

    train_dataset = _dataset_cls(
        run_params=run_params['dataset'],
        hyper_params=hyper_params['dataset'],
        data_info_csv_path=run_params['dataset']['train_info_csv_path'],
        labels_map=_labels_map,
        img_transforms=None,
        mode='train',
        drop_quiet=_drop_quiet_clips)
    val_dataset = _dataset_cls(
        run_params=run_params['dataset'],
        hyper_params=hyper_params['dataset'],
        data_info_csv_path=run_params['dataset']['val_info_csv_path'],
        labels_map=_labels_map,
        img_transforms=None,
        mode='train',
        drop_quiet=_drop_quiet_clips)
    hyper_params['extra'].update({'train_val_split': 'parent_stratified',
                                  'audio_augs': None,
                                  'img_augs': None,
                                  'drop_quiet_clips': _drop_quiet_clips
                                  })

    # train
    trainer = Trainer(model, run_params, hyper_params)
    trainer.fit(train_dataset, val_dataset)


def parse_args():
    parser = ConfParser()
    run_params, h_params = parser()

    # cv2.setNumThreads(_run_params['general']['libs_threads'])

    return run_params, h_params


def main():
    run_params, h_params = parse_args()
    setup_training(run_params, h_params)


if __name__ == '__main__':
    main()
