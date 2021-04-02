from collections import OrderedDict

from trainkit import Trainer
import timm
import torch.nn as nn

from .utils import data, models, utils


def train(run_params: dict, hyper_params: dict):
    # define models body and head
    _body = timm.create_model(model_name='resnet18', num_classes=0)

    _head_lin = nn.Linear(512, 10)
    # nn.init.zeros_(_head_lin.weight), nn.init.ones_(_head_lin.bias)
    _head = nn.Sequential(OrderedDict([('lin1', _head_lin)]))

    _loss_fn = nn.CrossEntropyLoss()
    _metrics_fn = utils.pre_accuracy

    # init model class
    model = models.BaseModel(device=run_params['general']['device'],
                             body=_body,
                             head=_head,
                             loss_fn=_loss_fn,
                             metrics_fn=_metrics_fn)
    hyper_params['extra'].update({'model_body': 'resNet18_notPretrained',
                                  'model_head': 'singleLinear',
                                  'loss': 'cross_entropy'})

    # datasets
    train_dataset = data.Dataset(
        run_params=run_params['dataset'],
        hyper_params=hyper_params['dataset'])
    val_dataset = data.Dataset(
        run_params=run_params['dataset'],
        hyper_params=hyper_params['dataset'])
    hyper_params['extra'].update({})

    # train
    trainer = Trainer(model, run_params, hyper_params)
    trainer.fit(train_dataset, val_dataset)
