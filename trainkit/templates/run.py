"""Runs train function from setup_train.py

First of all, current queueing module is designed to be ran from dir which is root for all projects
and from which RQ worker was initialized. (See examples)

Examples:
    Ex.1:
    runs `/home/user/projects/mnist/src/run.py` with additional config
    `/home/user/projects/mnist/runs/configs/experiments/exp0.yaml`.
    CWD must be `/home/user/projects/`.

    >>> python -m mnist.src.run +experiments=exp0

    Ex.2:
    enqueues job into own RQ with overwritten queue name.

    >>> python -m mnist.src.run +experiments=exp0 +run_params/enqueue=enqueue enqueue.queue_name=big

    Ex.3:
    makes two jobs with different configs: exp0, exp1 and send them into launcher, 'big' queue.

    >>> python -m mnist.src.run -m +experiments=exp0,exp1 hydra.launcher.queue=big

Args:
    All given args will be parsed by hydra
"""

from dotenv import load_dotenv
load_dotenv()

from typing import TYPE_CHECKING

import hydra
from omegaconf import OmegaConf
from trainkit.utils.hydra_conf_parser import HydraConfigModder

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path='../runs/configs/', config_name='config')
def main(conf: 'DictConfig'):
    run_params, hyper_params = HydraConfigModder()(conf)

    enqueue_params = conf.get('enqueue', None)
    if enqueue_params is not None:
        from trainkit.queueing.job_maker import JobMaker

        enqueue_params = OmegaConf.to_container(enqueue_params, resolve=True)
        job_maker = JobMaker(run_params=run_params,
                             hyper_params=hyper_params,
                             proj_root_path=run_params['paths']['proj_root'],
                             **enqueue_params)
        job_maker.__call__()
    else:
        from .setup_train import train

        return train(run_params=run_params, hyper_params=hyper_params)


if __name__ == '__main__':
    main()
