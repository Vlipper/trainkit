from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainkit import Trainer
from trainkit.utils.hydra_conf_parser import HydraConfigModder

from tests.utils import SplitSampler, TwoMoonsDataset, TwoMoonsModel

if TYPE_CHECKING:
    from typing import Tuple

# disable all tqdm calls from the code underground
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


# pylint: disable=redefined-outer-name
@pytest.fixture(scope='module')
def parameters(tmp_path_factory) -> 'Tuple[dict, dict]':
    with hydra.initialize(config_path='./configs'):
        conf = hydra.compose(config_name='config', overrides=["+experiments=exp_fit"])

    # create temp directories and update paths
    conf.run_params.paths.update({
        'tboard_path': str(tmp_path_factory.mktemp('tboard-logs', numbered=False)),
        'hparam_path': str(tmp_path_factory.mktemp('hparam-logs', numbered=False)),
        'models_path': str(tmp_path_factory.mktemp('models', numbered=False))})
    run_params, hyper_params = HydraConfigModder()(conf)

    return run_params, hyper_params


@pytest.fixture(scope='module')
def dataloaders(parameters) -> 'Tuple[DataLoader, DataLoader]':
    run_params, hyper_params = parameters

    dataset = TwoMoonsDataset(run_params['dataset'], hyper_params['dataset'])
    train_sampler = SplitSampler(dataset, 'train', run_params['sampler'], hyper_params['sampler'])
    val_sampler = SplitSampler(dataset, 'val', run_params['sampler'], hyper_params['sampler'])
    train_loader = DataLoader(dataset=dataset,
                              batch_size=hyper_params['dataset']['batch_size'],
                              sampler=train_sampler,
                              num_workers=run_params['dataset']['num_workers'])
    val_loader = DataLoader(dataset=dataset,
                            batch_size=hyper_params['dataset']['batch_size'],
                            sampler=val_sampler,
                            num_workers=run_params['dataset']['num_workers'])

    return train_loader, val_loader


@pytest.fixture()
def trainer(parameters: 'Tuple[dict, dict]',
            dataloaders: 'Tuple[DataLoader, DataLoader]') -> Trainer:
    run_params, hyper_params = parameters
    train_loader, val_loader = dataloaders

    model = TwoMoonsModel(run_params['general']['device'])
    trainer = Trainer(model, train_loader, val_loader, run_params, hyper_params)

    return trainer


def test_trainer_fit(trainer: Trainer):
    trainer.fit()
    assert trainer.best_val_metrics > 0.95

    accumulator = EventAccumulator(trainer.log_writer.tb_writer.log_dir).Reload()
    expected_tags = ['losses/train', 'losses/val', 'metrics/train', 'metrics/val']
    existed_tags = accumulator.scalars.Keys()
    assert all(tag in existed_tags for tag in expected_tags)

    hparam_path = Path(trainer.run_params['paths']['hparam_path'])
    assert len(list(hparam_path.glob('*.json'))) > 0


def test_find_lr(trainer: Trainer):
    trainer.model.train_preps(trainer)
    trainer.run_find_lr()
    accumulator = EventAccumulator(trainer.log_writer.tb_writer.log_dir).Reload()

    expected_tags = ['lr-rt/lr', 'lr-rt/loss', 'lr-rt/smooth-loss']
    existed_tags = accumulator.scalars.Keys()
    assert all(tag in existed_tags for tag in expected_tags)

    expected_tag_size = trainer.run_params['find_lr']['kwargs']['num_lrs']
    tags_sizes = [len(accumulator.scalars.Items(tag)) for tag in expected_tags]
    assert all(i == expected_tag_size for i in tags_sizes)
