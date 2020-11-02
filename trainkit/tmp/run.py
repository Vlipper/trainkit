import os
from collections import OrderedDict

import cv2
import timm
# import pretrainedmodels
# from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from trainkit.core.trainer import Trainer
from trainkit.utils.conf_parser import ConfParser

from utils import data
from utils import models


def main(run_params: dict, hyper_params: dict):
    # define model
    ## pretrainedmodels
    # body = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
    # body.last_linear = nn.Sequential(OrderedDict([('lin1', nn.Linear(512, 1))]))
    ## timm efficientNet
    # body = timm.create_model('efficientnet_b0', pretrained=True)  # num_classes=1
    # body.classifier = nn.Sequential(OrderedDict([('lin1', nn.Linear(1280, 1))]))
    body = timm.create_model('resnet34', pretrained=True, num_classes=0)
    head_lin = nn.Linear(512, 1)
    # nn.init.zeros_(head_lin.weight), nn.init.zeros_(head_lin.bias)
    head = nn.Sequential(OrderedDict([('lin1', head_lin)]))
    ## efficientNet
    # body = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
    # body._fc = nn.Sequential(OrderedDict([('lin1', nn.Linear(1280, 1))]))
    ## init model class
    _pos_weight = torch.tensor(hyper_params['general']['pos_weight'], dtype=torch.float32)
    _loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=_pos_weight)
    model_kw = {'body': body, 'head': head,
                'loss_fn': _loss_fn,
                'metrics_fn': nn.BCEWithLogitsLoss(reduction='none'),
                'device': run_params['general']['device']}
    model = models.PElabelNet(**model_kw)
    hyper_params['extra'].update({'body': 'resNet34',
                                  'head': 'singleLinear_randInit'})

    # datasets
    train_dataset_kw = {'description_csv_path': run_params['dataset']['train_description_csv_path']}
    val_dataset_kw = {'description_csv_path': run_params['dataset']['val_description_csv_path']}
    train_dataset = data.ImageSet(run_params['dataset'], hyper_params['dataset'], **train_dataset_kw)
    val_dataset = data.ImageSet(run_params['dataset'], hyper_params['dataset'], **val_dataset_kw)
    hyper_params['extra'].update({'augs': None})

    ## holdout train
    trainer = Trainer(model, run_params, hyper_params)
    trainer.fit(train_dataset, val_dataset)

    ## cross-val train
    # train_kw.update({'save_best_state': True})
    # trainer = utils.Trainer(model=model, optimizer=optimizer, num_epochs=h_params['num_epochs'],
    #                         models_dir_path=models_path, n_iter_train=run_params['n_iter_train'],
    #                         run_params=run_params, h_params=h_params, **train_kw)
    #
    # # cv_img_trg_split_no_dups | blend_img_trg_cvsplit_no_dups
    # img_trg_split_path = Path(data_mod_path, 'blend_img_trg_cvsplit_no_dups.csv')
    # img_trg_split = np.loadtxt(img_trg_split_path, dtype=str, delimiter=',')
    # num_folds = len(np.unique(img_trg_split[:, 2]))
    #
    # dataset_kw = {'imgs_path': imgs_dir_path, 'mean': mean_imgs, 'std': std_imgs,
    #               'imgs_feats_path': Path(data_mod_path, 'img_sex.csv')}
    # train_val_sets_pairs = []
    # for i in range(num_folds):
    #     mask_train = img_trg_split[:, 2] != str(i)
    #     train_imgs_targets = img_trg_split[mask_train, :2].tolist()
    #     val_imgs_targets = img_trg_split[np.logical_not(mask_train), :2].tolist()
    #
    #     train_data_kw, val_data_kw = dataset_kw.copy(), dataset_kw.copy()
    #     train_data_kw.update({'imgs_targets': train_imgs_targets, 'transforms': train_transforms})
    #     val_data_kw.update({'imgs_targets': val_imgs_targets, 'transforms': val_transforms})
    #     train_val_sets_pairs.append((data.MultiLossSet(**train_data_kw),
    #                                  data.MultiLossSet(**val_data_kw)))
    # trainer.cv_train(train_val_sets_pairs)

    ## cross-val with blending fold
    # img_trg_blend_path = Path(data_mod_path, 'blend_img_trg_no_dups.csv')
    # img_trg_blend = np.loadtxt(img_trg_blend_path, dtype=str, delimiter=',')
    #
    # blend_imgs_targets = img_trg_blend[:, :2].tolist()
    # blend_data_kw = dataset_kw.copy()
    # blend_data_kw.update({'imgs_targets': blend_imgs_targets, 'transforms': val_transforms})
    # blend_dataset = data.MultiLossSet(**blend_data_kw)
    #
    # trainer.cv_train(train_val_sets_pairs, agg_logs=False)
    # trainer.cv_blend(num_folds, blend_dataset)


if __name__ == '__main__':
    parser = ConfParser()
    _run_params, _h_params = parser.parse_n_valid_params()

    # num threads config
    os.environ['MKL_NUM_THREADS'] = str(_run_params['general']['libs_threads'])
    cv2.setNumThreads(_run_params['general']['libs_threads'])

    main(_run_params, _h_params)
