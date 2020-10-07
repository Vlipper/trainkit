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

    # other dirs in root
    Path(root_dir, 'insides').mkdir()
    Path(root_dir, 'notebooks').mkdir()
    Path(root_dir, 'preds').mkdir()
    if not Path(root_dir, 'comp_utils').exists():
        Path(root_dir, 'comp_utils').mkdir()
    Path(root_dir, 'comp_utils', 'preps').mkdir()

    # custom dirs
    if kwargs.get('custom_dirs') is not None:
        for cstm_path in kwargs['custom_dirs']:
            Path(root_dir, cstm_path).mkdir(parents=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_dirs', '-c', type=str, nargs='*',
                        help='list of custom dirs into root_dir')
    args_dict = vars(parser.parse_args())

    main(**args_dict)
