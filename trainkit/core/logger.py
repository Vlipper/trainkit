import json
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class LogWriter:
    def __init__(self, tboard_dir_path: Path,
                 hparam_dir_path: Path,
                 model_name: str,
                 flush_secs=30):
        self.tb_writer = SummaryWriter(log_dir=str(Path(tboard_dir_path, model_name).resolve()),
                                       flush_secs=flush_secs)
        self.hparam_log_file_path = Path(hparam_dir_path, f'{model_name}.json')

        # self.cv_log_paths.append(logs_path)

    def write_scalar(self, tag: str, value: float, step: int):
        self.tb_writer.add_scalar(tag, value, step)

    def write_hparams(self, hparams: dict):
        unnested_hparams = self.__unpack_hparams(hparams)
        unnested_hparams_json_formatted = json.dumps(unnested_hparams, indent=4)
        self.hparam_log_file_path.write_text(unnested_hparams_json_formatted)

    def close(self):
        self.tb_writer.close()

    @classmethod
    def __unpack_hparams(cls, hparams: Dict[str, Any],
                         prev_key: Optional[str] = None) -> dict:
        unnested_params = {}

        for key, val in hparams.items():
            mod_key = key if prev_key is None else f'{prev_key}/{key}'

            if isinstance(val, dict):
                unnested_params.update(cls.__unpack_hparams(hparams=val, prev_key=mod_key))
            else:
                unnested_params.update({mod_key: val})

        return unnested_params
