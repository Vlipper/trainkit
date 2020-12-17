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

    def _mod_run_params(self):
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

    def _mod_hyper_params(self):
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

    def __call__(self, args_raw: Optional[List[str]] = None) -> Tuple[dict, dict]:
        """Runs all underwear funcs and returns two dicts with run params and hyper params

        Returns:
            Two unnested dicts: run_params and hyper_params
        """
        args = self._parse_args(args_raw)
        self._check_args(args)

        all_params = self._read_conf(args)
        if len(args.keys()) > 1:
            all_params = self._update_params(all_params, args['kwargs'])

        self.hyper_params = all_params['hyper_params']
        self.run_params = all_params['run_params']

        self._mod_run_params()
        self._mod_hyper_params()

        return self.run_params, self.hyper_params

    @staticmethod
    def _parse_args(args_raw: Optional[List[str]] = None) -> Dict[str, Any]:
        """Parses argument given with script run

        Args:
            args_raw: list of strings to parse. If None, args will be read from sys.argv.

        Returns:
            Dictionary with parsed args
        """
        parser = argparse.ArgumentParser()

        conf_group = parser.add_mutually_exclusive_group(required=True)
        conf_group.add_argument('-f', '--conf_file', type=Path, help='path to conf file')
        conf_group.add_argument('-s', '--conf_string', type=str, help='conf file as string')
        parser.add_argument('-k', '--kwargs', type=str, nargs='*', action='extend',
                            help="kwargs with '=' as separator, which replace config's keys")

        args = vars(parser.parse_args(args_raw))
        args = {key: val for key, val in args.items() if val is not None}  # drop keys with None values

        return args

    @staticmethod
    def _check_args(args: dict):
        """Checks validity of given args

        Args:
            args: parsed config
        """
        if args.get('conf_file') is not None and not args['conf_file'].exists():
            raise ValueError(f'Given path to "conf_file" does not exist: "{args["conf_file"]}"')

        if args.get('kwargs') is not None and len(args['kwargs']) == 0:
            raise ValueError(f'Given "kwargs" must have more than zero values')

    @staticmethod
    def _read_conf(args: dict) -> Dict[str, Any]:
        """Reads config and checks presence of meta-groups: run_params, hyper_params

        Args:
            args: arguments read by ArgumentParser

        Returns:
            Dictionary with params
        """
        if args.get('conf_file') is not None:
            with args['conf_file'].open('r') as file:
                params = yaml.safe_load(file)
        else:
            params = yaml.safe_load(args['conf_string'])

        for param_group in ['run_params', 'hyper_params']:
            if param_group not in params.keys():
                raise ValueError(f'Group "{param_group}" must be in config')

        return params

    @classmethod
    def _update_params(cls, params: dict, kwargs: List[str]):
        """Update parameters with given kwargs

        Args:
            params: parameters read from given config file
            kwargs: list of string parameters read from given argument

        Returns:
            Updated parameters
        """
        params = params.copy()

        for key_val in kwargs:
            key, val = key_val.split('=')
            key_seq = key.split('/')
            cls.__update_key_val(params, key_seq, val)

        return params

    @staticmethod
    def __update_key_val(params: dict, key_seq: list, val: str):
        """Helper for `_update_params` method
        """
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
