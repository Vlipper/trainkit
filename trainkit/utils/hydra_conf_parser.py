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

    # from torch import device as torch_device
    # def _mod_run_params(self):
    #     self._mod_param_device()
    #
    # def _mod_param_device(self):
    #     device = self.run_params['general']['device']
    #
    #     self.run_params['general'].update({'device': torch_device(device)})

        # general_params = self.run_params['general']
        #
        # device_split = general_params['device'].split(':')
        # device_name = device_split[0]
        #
        # if device_name == 'cpu':
        #     general_params.update({'device': torch_device('cpu')})
        # elif device_name == 'cuda':
        #     os.environ['CUDA_VISIBLE_DEVICES'] = device_split[1]
        #     general_params.update({'device': torch_device('cuda')})
        # else:
        #     raise ValueError(f"Argument `device` must be one of: 'cpu', 'cuda:N', "
        #                      f"got: '{general_params['device']}'")
