import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import yaml
from torch import device as torch_device


class ConfCheckerMixin:
    run_params: dict
    hyper_params: dict

    def mod_run_params(self):
        # check and modify paths args
        proj_root_path = self.__val_n_mod_path_arg(self.run_params['paths']['proj_root'])
        self.run_params['paths']['proj_root'] = proj_root_path
        self.__val_n_mod_paths(self.run_params, proj_root_path)

        # mod model_name to dttm if needed
        if self.run_params['general']['model_name'] == 'dttm':
            cur_dttm = datetime.today().strftime('%y-%m-%d_%H-%M-%S')
            self.run_params['general']['model_name'] = cur_dttm

        # check device param
        self.__val_n_mod_device_arg(self.run_params)

    def mod_hyper_params(self):
        pass

    @classmethod
    def __val_n_mod_paths(cls, params: dict, proj_root_path: Path):
        for arg_name, arg_val in params.items():
            if isinstance(arg_val, dict):
                cls.__val_n_mod_paths(params[arg_name], proj_root_path)
            elif (arg_name != 'proj_root') and arg_name.endswith('_path'):
                arg_val_path = cls.__val_n_mod_path_arg(proj_root_path, arg_val)
                params.update({arg_name: arg_val_path})

    @staticmethod
    def __val_n_mod_path_arg(*path_args: Union[str, Path]) -> Path:
        path_arg_mod = Path(*path_args)

        if not path_arg_mod.exists():
            raise ValueError('Path is not exist: {}'.format(path_arg_mod))

        return path_arg_mod

    @staticmethod
    def __val_n_mod_device_arg(params: dict):
        general_params: dict = params['general']
        device_split = general_params['device'].split(':')

        if device_split[0] == 'cpu':
            general_params.update({'device': torch_device('cpu')})
        elif device_split[0] == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = device_split[1]
            general_params.update({'device': torch_device('cuda')})
        else:
            raise ValueError("Argument 'device' must be one of: 'cpu', 'cuda:N', "
                             "got: '{}'".format(general_params['device']))


class ConfParser(ConfCheckerMixin):
    def __init__(self):
        self.conf_filepath = self.parse_conf_filepath()

    @staticmethod
    def parse_conf_filepath() -> Path:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--conf_file', required=True, type=str, help='path to conf_file')
        conf_filepath = Path(parser.parse_args().conf_file)

        if conf_filepath.exists():
            return conf_filepath
        else:
            raise ValueError('Path to conf_file does not exist')

    def read_conf_file(self):
        """
        Read config file and check presence of meta-groups: run_params, hyper_params.
        If everything is fine, write params into: self.run_params, self.hyper_params.
        """
        with self.conf_filepath.open('r') as file:
            all_params = yaml.safe_load(file)

        for param_group in ['run_params', 'hyper_params']:
            if param_group not in all_params.keys():
                raise ValueError('Group "{}" must be in config'.format(param_group))

        self.run_params = all_params['run_params']
        self.hyper_params = all_params['hyper_params']

    def parse_n_valid_params(self) -> Tuple[dict, dict]:
        """
        Run all underwear funcs and return two dicts with run params and hyper params

        Returns:
            Two unnested dicts: run_params and hyper_params
        """
        self.read_conf_file()

        self.mod_run_params()
        self.mod_hyper_params()

        return self.run_params, self.hyper_params
