from dotenv import load_dotenv
load_dotenv()

from typing import TYPE_CHECKING

import hydra
from trainkit.utils.hydra_conf_parser import HydraConfigModder

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(config_path='../runs/configs/')
def main(conf: 'DictConfig'):
    run_params, hyper_params = HydraConfigModder()(conf)

    enqueue_params = run_params.get('enqueue', None)
    if enqueue_params:
        from trainkit.queueing.job_maker import JobMaker

        job_maker = JobMaker(run_params=run_params,
                             hyper_params=hyper_params,
                             proj_root_path=run_params['paths']['proj_root'],
                             **enqueue_params)
        job_maker.__call__()
    else:
        from setup_train import train

        return train(run_params=run_params, hyper_params=hyper_params)


if __name__ == '__main__':
    main()
