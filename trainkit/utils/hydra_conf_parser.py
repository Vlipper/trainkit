from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from typing import Tuple
    from omegaconf import DictConfig


class HydraConfigModder:
    def __init__(self):
        self.run_params = None
        self.hyper_params = None

    def __call__(self, config: 'DictConfig') -> 'Tuple[dict, dict]':
        config = OmegaConf.to_container(config, resolve=True)

        self.run_params: dict = config['run_params']
        self.hyper_params: dict = config['hyper_params']

        return self.run_params, self.hyper_params
