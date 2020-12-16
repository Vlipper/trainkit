import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from torch import device as torch_device


class ConfCheckerMixin:
    run_params: Optional[dict]
    hyper_params: Optional[dict]

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
        device_name = device_split[0]

        if device_name == 'cpu':
            general_params.update({'device': torch_device('cpu')})
        elif device_name == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = device_split[1]
            general_params.update({'device': torch_device('cuda')})
        else:
            raise ValueError("Argument 'device' must be one of: 'cpu', 'cuda:N', "
                             "got: '{}'".format(general_params['device']))


class ConfParser(ConfCheckerMixin):
    def __init__(self):
        self.hyper_params = None
        self.run_params = None

    def __call__(self) -> Tuple[dict, dict]:
        """Runs all underwear funcs and return two dicts with run params and hyper params

        Returns:
            Two unnested dicts: run_params and hyper_params
        """
        args = self.parse_args()
        args = self.check_args(args)

        if len(args.keys()) == 1:
            conf_file_path = args['conf_file']
            all_params = self.read_conf_file(conf_file_path)
        else:
            conf_file_path, modification_kwargs = args['conf_file'], args['kwargs']
            all_params = self.read_conf_file(conf_file_path)
            all_params = self.update_params(all_params, modification_kwargs)

        self.hyper_params = all_params['hyper_params']
        self.run_params = all_params['run_params']

        self.mod_run_params()
        self.mod_hyper_params()

        return self.run_params, self.hyper_params

    @staticmethod
    def parse_args() -> Dict[str, Any]:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--conf_file', required=True, type=Path, help='path to conf file')
        parser.add_argument('-k', '--kwargs', type=str, nargs='*',
                            help="kwargs with '=' as separator, which replace config's keys")
        args = vars(parser.parse_args())

        return args

    @staticmethod
    def check_args(args: dict) -> Dict[str, Any]:
        args = args.copy()

        if not args['conf_file'].exists():
            raise ValueError('Path to "conf_file" does not exist.\n'
                             f'Given path: "{args["conf_file"]}"')

        if args['kwargs'] is None:
            args.pop('kwargs')

        return args

    @staticmethod
    def read_conf_file(conf_file_path: Path) -> Dict[str, Any]:
        """Reads config file and checks presence of meta-groups: run_params, hyper_params.

        Returns:
            Dictionary with params
        """
        with conf_file_path.open('r') as file:
            all_params = yaml.safe_load(file)

        for param_group in ['run_params', 'hyper_params']:
            if param_group not in all_params.keys():
                raise ValueError(f'Group "{param_group}" must be in config')

        return all_params

    @classmethod
    def update_params(cls, params: dict, kwargs: List[str]):
        params = params.copy()

        for key_val in kwargs:
            key, val = key_val.split('=')
            key_seq = key.split('/')
            cls._update_key_val(params, key_seq, val)

        return params

    @staticmethod
    def _update_key_val(params: dict, key_seq: list, val: str):
        for key in key_seq[:-1]:
            existed_val = params.get(key, None)

            if existed_val is None:
                params[key] = {}
                params = params[key]
            elif isinstance(existed_val, dict):
                params = params[key]
            else:
                raise Exception('Unexpected key')  # raises when key returns real val (not dict)

        params[key_seq[-1]] = yaml.safe_load(val)
