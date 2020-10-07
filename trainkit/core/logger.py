import warnings
from pathlib import Path
from typing import Iterable, Optional

from torch.utils.tensorboard import SummaryWriter


class LogWriter:
    def __init__(self, tboard_dir_path: Path, model_name: str):
        logs_dir_path = Path(tboard_dir_path, model_name)
        self.tb_writer = SummaryWriter(logs_dir_path, flush_secs=30)
        # self.cv_log_paths.append(logs_path)

    def write_log(self, tag: str, value, step: int):  # value: Union[float, np.float]
        self.tb_writer.add_scalar(tag, value, step)

    def write_hparams(self, h_params: dict, best_loss: float, best_metrics: float):
        metrics_dict = {'hparam/best_loss': best_loss, 'hparam/best_metrics': best_metrics}

        unnested_h_params = {}
        unnested_h_params = self.__unpack_hparams(h_params, unnested_h_params)
        unnested_h_params = self.__valid_n_mod_hparams(unnested_h_params)

        self.tb_writer.add_hparams(unnested_h_params, metrics_dict)

    def close(self):
        self.tb_writer.flush()
        self.tb_writer.close()

    @classmethod
    def __unpack_hparams(cls, params: dict, unnested_params: dict,
                         prev_key: Optional[str] = None) -> dict:
        for key, val in params.items():
            # modify out key name (key_mod). Ex.: {'key1/key2': dict[key1][key2]}
            if prev_key is not None:
                key_mod = '{}/{}'.format(prev_key, key)
            else:
                key_mod = key

            if isinstance(val, dict):
                cls.__unpack_hparams(params[key], unnested_params, key_mod)
            else:
                unnested_params.update({key_mod: val})

        return unnested_params

    @staticmethod
    def __valid_n_mod_hparams(unnested_params: dict) -> dict:
        unnested_params_mod = {}

        for key, val in unnested_params.items():
            if isinstance(val, (bool, str, float, int)) or (val is None):
                unnested_params_mod.update({key: val})
            elif isinstance(val, Iterable):
                str_val = ','.join([str(item) for item in val])  # ToDo: может упасть
                unnested_params_mod.update({key: str_val})
            else:
                warnings.warn('Given key "{}" has invalid value type'
                              'to write it into tensorborad hparams'.format(key))

        return unnested_params_mod
