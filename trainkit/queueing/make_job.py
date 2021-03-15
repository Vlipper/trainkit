import argparse
from pathlib import Path

from .utils import JobMaker, read_conf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--conf_file', type=Path, required=True)
    parser.add_argument('-k', '--conf_kwargs', type=str, nargs='*')
    parser.add_argument('-q', '--queue_name', type=str, required=True)
    parser.add_argument('-c', '--rq_conf', type=Path, default=Path('rq_conf.yaml'))
    parser.add_argument('-s', '--src_path', type=Path, required=False)

    # rq_conf_group = parser.add_mutually_exclusive_group(required=True)
    # rq_conf_group.add_argument('-ru', '--redis_url', type=str)

    args = vars(parser.parse_args())
    args = {key: val for key, val in args.items() if val is not None}

    return args


def main():
    args = parse_args()
    args_conf = read_conf(args['rq_conf'])
    args.update(args_conf)

    JobMaker(**args)()


if __name__ == '__main__':
    main()
