"""Creates project's dir tree inside current working directory (cwd).

Args:
    custom_dirs: sequence of strings (paths) to create (relative to cwd).
        Possibly to pass through --custom_dirs or -c.

Examples:
    Ex.1:
    >>> python -m trainkit.utils.make_project_tree

    Ex.2:
    >>> python -m trainkit.utils.make_project_tree
    >>>     -c 'custom_dir1' 'custom_dir2/in_dir2_1'
"""

import argparse
from pathlib import Path


def main(**kwargs):
    root_dir = Path.cwd()

    # data branch
    Path(root_dir, 'data', 'raw').mkdir(parents=True)
    Path(root_dir, 'data', 'mod').mkdir(parents=True)

    # runs branch
    Path(root_dir, 'runs', 'confs').mkdir(parents=True)
    Path(root_dir, 'runs', 'find_lr').mkdir(parents=True)
    Path(root_dir, 'runs', 'models').mkdir(parents=True)
    Path(root_dir, 'runs', 'tboard_logs').mkdir(parents=True)
    Path(root_dir, 'runs', 'hparam_logs').mkdir(parents=True)

    # other dirs in root
    Path(root_dir, 'insides').mkdir()
    Path(root_dir, 'notebooks').mkdir()
    Path(root_dir, 'preds').mkdir()

    # custom dirs
    if kwargs.get('custom_dirs') is not None:
        for custom_path in kwargs['custom_dirs']:
            Path(root_dir, custom_path).mkdir(parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_dirs', '-c', type=str, nargs='*',
                        help='list of custom dirs into root_dir')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args_dict = parse_args()

    main(**args_dict)
