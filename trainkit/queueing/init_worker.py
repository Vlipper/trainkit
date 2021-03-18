from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path

from redis import Redis
from rq import Worker
import yaml


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--rq_conf', type=Path, default=Path('rq_conf.yaml'))
    parser.add_argument('-n', '--worker_name', type=str, required=True)
    parser.add_argument('-q', '--queue_names', type=str, required=True, action='append')
    args = vars(parser.parse_args())

    return args


def main():
    args = parse_args()

    with args['rq_conf'].open('r') as file:
        conf = yaml.safe_load(file)

    connection = Redis.from_url(conf['url'])
    worker = Worker(queues=args['queue_names'],
                    name=args['worker_name'],
                    connection=connection,
                    log_job_description=False)

    worker.work()


if __name__ == '__main__':
    main()
