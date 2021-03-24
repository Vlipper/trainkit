"""Initializes and run RQ worker.

Args:
    queue_names (required): list of queue names which will be listened by initialized worker
            Possibly to pass through --queue_names or -q.
    worker_name (optional): worker name
        Possibly to pass through --worker_name or -n.

Examples:
    Ex.1:
    >>> python -m trainkit.queueing.init_worker
    >>>     -q 'small' 'big' -n 'worker_1'

    Ex.2:
    # if you want to use only cuda:0 to execute job
    >>> CUDA_VISIBLE_DEVICES=0 python -m trainkit.queueing.init_worker
    >>>     -q 'small' 'big' -n 'worker_1'
"""

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv('.rq_worker.env', raise_error_if_not_found=True, usecwd=True))

import argparse
import os

from redis import Redis
from rq import Worker


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--queue_names', type=str, nargs='+', required=True)
    parser.add_argument('-n', '--worker_name', type=str)
    args = vars(parser.parse_args())

    return args


def main():
    args = parse_args()

    connection = Redis(host=os.environ['REDIS_HOST'],
                       port=os.environ['REDIS_PORT'],
                       db=os.environ['REDIS_DB'],
                       password=os.environ['REDIS_PASSWORD'])
    worker = Worker(queues=args['queue_names'],
                    name=args['worker_name'],
                    connection=connection,
                    log_job_description=False)

    worker.work()


if __name__ == '__main__':
    main()
