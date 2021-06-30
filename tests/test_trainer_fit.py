from functools import partialmethod

from hydra import compose, initialize
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainkit import Trainer
from trainkit.utils.hydra_conf_parser import HydraConfigModder

from tests.utils import SplitSampler, TwoMoonsDataset, TwoMoonsModel

# disable all tqdm calls from the code underground
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def test_model_fit():
    with initialize(config_path='./configs'):
        conf = compose(config_name='config', overrides=["+experiments=exp_fit"])
    run_params, hyper_params = HydraConfigModder()(conf)

    model = TwoMoonsModel(run_params['general']['device'])
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

    trainer = Trainer(model, run_params, hyper_params)
    trainer.fit(train_loader, val_loader)

    assert trainer.best_val_metrics > 0.85

    # tmp_dir = TemporaryDirectory(dir=Path.cwd())
    # tmp_dir.cleanup()
