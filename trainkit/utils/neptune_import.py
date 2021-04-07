"""Imports

Args:
    project_name: neptune's project name in format `{NAMESPACE}/{PROJECT_NAME}`.
        Possibly to pass through --project_name or -p.
    params_dir: directory with unnested jsons with hyper parameters.
        Possibly to pass through --params_dir or -d.

Example:
    >>> python -m trainkit.utils.neptune_import
    >>>     --project_name 'vlipper/sandbox'
    >>>     --params_dir 'tmp2'
"""

import argparse
import json
from pathlib import Path

import neptune.new as neptune


def run_neptune(project_name: str,
                params_dir: Path):
    params_dir_path = params_dir.expanduser().resolve()
    if not params_dir_path.exists():
        raise Exception(f'Given "params_dir" does not exist: {str(params_dir_path)}')

    done_exp_file_path = Path(params_dir_path, 'done_experiments.txt')
    if not done_exp_file_path.exists():
        done_exp_file_path.touch()
    done_exp_names = done_exp_file_path.read_text().strip().split('\n')

    for params_file_path in params_dir_path.glob('*.json'):
        params_file_name = params_file_path.name.split('.')[0]
        if params_file_name in done_exp_names:
            continue

        with params_file_path.open('r') as file:
            params = json.load(file)
        # if 'results' not in set([key.split('/', 1)[0] for key in params.keys()]):
        if 'results' not in params.keys():
            continue

        # if params_file_name not in done experiments list and there are results key in params
        #   than create new experiment and add fields into it
        run = neptune.init(project=project_name,
                           name=params_file_name,
                           source_files=[],
                           capture_stdout=False,
                           capture_stderr=False,
                           capture_hardware_metrics=False)
        run.assign(params)
        run.stop()

        with done_exp_file_path.open('a') as file:
            file.write(f'{params_file_name}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', '-p', type=str, required=True)
    parser.add_argument('--params_dir', '-d', type=Path, required=True)

    return vars(parser.parse_args())


def main():
    args_dict = parse_args()
    run_neptune(**args_dict)


if __name__ == '__main__':
    main()
