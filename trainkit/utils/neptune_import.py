"""Imports

Args:
    project_name: neptune's project name in format `{NAMESPACE}/{PROJECT_NAME}`.
        Possibly to pass through --project_name or -p.
    hparams_dir: directory with unnested jsons with hyper parameters.
        Possibly to pass through --hparams_dir or -d.

Example:
    >>> python -m trainkit.utils.neptune_import
    >>>     --project_name 'vlipper/sandbox'
    >>>     --hparams_dir 'tmp2'
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import neptune

if TYPE_CHECKING:
    from neptune.experiments import Experiment


def read_hparams(file_path: Path):
    with file_path.open('r') as file:
        hparams = json.load(file)

    return hparams


def sep_hparams_n_metrics(hparams: Dict[str, Any]):
    # separate hparams and metrics (e.g. best_loss, best_metrics, ...)
    hparams_only, metrics_only = hparams.copy(), {}

    for key, val in hparams.items():
        if key.startswith('results/'):
            metrics_only.update({key.replace('results/', ''): val})
            hparams_only.pop(key)

    return hparams_only, metrics_only


def log_new_experiment(project: neptune.projects.Project,
                       experiment_name: str,
                       hparams: Dict[str, Any]):
    hparams_only, metrics_only = sep_hparams_n_metrics(hparams)

    exp = project.create_experiment(name=experiment_name,
                                    params=hparams_only,
                                    upload_source_files=[],
                                    upload_stdout=False,
                                    upload_stderr=False,
                                    send_hardware_metrics=False,
                                    run_monitoring_thread=False,
                                    handle_uncaught_exceptions=True)

    if len(metrics_only) != 0:
        log_metrics(experiment=exp, metrics=metrics_only)


def log_metrics(experiment: 'Experiment',
                metrics: Dict[str, Any]):
    for key, val in metrics.items():
        experiment.log_metric(log_name=key, x=val)


def get_experiment(name: str,
                   experiments_list: List['Experiment']):
    for exp in experiments_list:
        if exp.name == name:
            return exp


def main(project_name,
         hparams_dir,
         **kwargs):
    project = neptune.init(project_qualified_name=project_name)
    existed_experiments = project.get_experiments()
    existed_experiments_names = [exp.name for exp in existed_experiments]

    hparams_dir_path = Path(hparams_dir).expanduser().resolve()
    if not hparams_dir_path.exists():
        raise Exception(f'Given "hparams_dir" does not exist: {str(hparams_dir_path)}')

    for hparams_file_path in hparams_dir_path.glob('*'):
        hparams = read_hparams(hparams_file_path)
        hparams_file_name = hparams_file_path.name.split('.')[0]

        if hparams_file_name in existed_experiments_names:
            existed_experiment = get_experiment(hparams_file_name, existed_experiments)

            _, metrics_only = sep_hparams_n_metrics(hparams)
            metrics_exists = [key in existed_experiment.get_logs().keys()
                              for key in metrics_only.keys()]

            if sum(metrics_exists) == 0:
                log_metrics(experiment=existed_experiment, metrics=metrics_only)
        else:
            log_new_experiment(project=project, experiment_name=hparams_file_name, hparams=hparams)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', '-p', type=str, required=True)
    parser.add_argument('--hparams_dir', '-d', type=str, required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args_dict = parse_args()

    main(**args_dict)
