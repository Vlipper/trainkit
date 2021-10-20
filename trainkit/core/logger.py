import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class LogWriter:
    def __init__(self, tboard_dir_path: Path,
                 hparam_dir_path: Path,
                 model_name: str,
                 flush_secs=10):
        self.tb_writer = SummaryWriter(log_dir=str(Path(tboard_dir_path, model_name).resolve()),
                                       flush_secs=flush_secs)
        self.hparam_log_file_path = Path(hparam_dir_path, f'{model_name}.json')

        # self.cv_log_paths.append(logs_path)

    def write_hparams(self, hparams: dict):
        with self.hparam_log_file_path.open('w') as file:
            json.dump(hparams, file, indent=4)
