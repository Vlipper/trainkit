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
        self.run_params: dict = OmegaConf.to_container(config['run_params'], resolve=True)
        self.hyper_params: dict = OmegaConf.to_container(config['hyper_params'], resolve=True)

        return self.run_params, self.hyper_params
