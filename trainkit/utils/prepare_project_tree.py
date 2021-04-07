"""Creates project's dir tree inside the `root_dir`.

Examples:
    Ex.1:

    >>> python -m trainkit.utils.prepare_project_tree

    Ex.2:

    >>> python -m trainkit.utils.prepare_project_tree
    >>>     -c 'custom_dir1' 'custom_dir2/in_dir2_1'

    Ex.3:

    >>> python -m trainkit.utils.prepare_project_tree
    >>>     -r ./project/ -s

Args:
    root_dir (optional): path to project root dir (default is current working directory (cwd))
        Possibly to pass through --root_dir or -r.
    custom_dirs (optional): sequence of strings (paths) to create (relative to cwd).
        Possibly to pass through --custom_dirs or -c.
    add_static (optional): whether to add static files (like configs and run templates) into project.
            Possibly to pass through --add_static or -s.
"""

import argparse
import shutil
from importlib import resources
from pathlib import Path


def _copy_static(static_path: Path,
                 proj_root_dir: Path):
    # enrich src
    proj_src_path = Path(proj_root_dir, 'src')
    proj_src_path.mkdir(mode=0o755, exist_ok=False)

    shutil.copy2(Path(static_path, 'run.py'), proj_src_path)
    shutil.copy2(Path(static_path, 'setup_train.py'), proj_src_path)
    shutil.copy2(Path(static_path, '.env'), proj_src_path)

    # enrich configs
    shutil.copytree(Path(static_path, 'configs'),
                    Path(proj_root_dir, 'runs', 'configs'),
                    dirs_exist_ok=True)


def main(**kwargs):
    root_dir = Path(kwargs['root_dir'])

    if not root_dir.exists():
        raise ValueError(f"Given 'root_dir' not exists: '{kwargs['root_dir']}'")

    # data branch
    Path(root_dir, 'data', 'raw').mkdir(parents=True)
    Path(root_dir, 'data', 'mod').mkdir(parents=True)

    # runs branch
    Path(root_dir, 'runs', 'configs').mkdir(parents=True)
    Path(root_dir, 'runs', 'find_lr').mkdir(parents=True)
    Path(root_dir, 'runs', 'models').mkdir(parents=True)
    Path(root_dir, 'runs', 'tboard_logs').mkdir(parents=True)
    Path(root_dir, 'runs', 'hparam_logs').mkdir(parents=True)

    # runs/configs branch
    Path(root_dir, 'runs', 'configs', 'run_params').mkdir(parents=True)
    Path(root_dir, 'runs', 'configs', 'hyper_params').mkdir(parents=True)
    Path(root_dir, 'runs', 'configs', 'experiments').mkdir(parents=True)

    # other dirs in root
    Path(root_dir, 'insides').mkdir()
    Path(root_dir, 'notebooks').mkdir()
    Path(root_dir, 'preds').mkdir()

    # custom dirs
    if kwargs.get('custom_dirs') is not None:
        for custom_path in kwargs['custom_dirs']:
            Path(root_dir, custom_path).mkdir(parents=True)

    # copy static
    if kwargs.get('add_static'):
        with resources.path('trainkit', 'static') as static_path:
            _copy_static(static_path=static_path, proj_root_dir=root_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', '-r', type=str, default='.', help='project root path')
    parser.add_argument('--custom_dirs', '-c', type=str, nargs='*',
                        help='list of custom dirs into root_dir')
    parser.add_argument('--add_static', '-s', action='store_true',
                        help='copy static files (e.g. configs and run templates) into project')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args_dict = parse_args()

    main(**args_dict)
