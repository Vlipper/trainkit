import argparse
from pathlib import Path

from redis import Redis

from .utils import PerCudaJob, PerCudaWorker, read_conf


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--rq_conf', type=Path, default=Path('rq_conf.yaml'))
    parser.add_argument('-n', '--name', type=str, required=True)
    args = vars(parser.parse_args())

    return args


def main():
    args = parse_args()
    conf = read_conf(args['rq_conf'])

    connection = Redis.from_url(conf['url'])

    worker = PerCudaWorker(name=args['name'],
                           queues=conf['queues'],
                           connection=connection,
                           job_class=PerCudaJob,
                           log_job_description=False)
    worker.work()


if __name__ == '__main__':
    main()
