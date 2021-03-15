import shutil
from pathlib import Path
from typing import Optional

import yaml
from redis import Redis
from rq import Queue


def read_conf(conf_path: Path) -> dict:
    """Reads yaml config and returns parsed dict
    """
    with conf_path.open('r') as file:
        conf = yaml.safe_load(file)

    return conf


class JobMaker:
    def __init__(self,
                 conf_file: Path,
                 url: str,
                 queue_name: str,
                 conf_kwargs: Optional[list] = None,
                 src_path: Optional[Path] = None,
                 cache_dir_path: Optional[Path] = None,
                 **_ignored):
        self.conf_string = conf_file.read_text()
        self.conf_kwargs = conf_kwargs

        self.job_queue = self.get_rq_queue(url=url, name=queue_name)

        self.src_path = src_path if src_path is not None else Path('src')

        if cache_dir_path is None:
            self.cache_dir_path = Path(self.src_path.parent, 'src_cache')
        else:
            self.cache_dir_path = cache_dir_path
        if not self.cache_dir_path.exists():
            self.cache_dir_path.mkdir()

    def __call__(self):
        cache_path = self.copy_src_into_cache_dir()
        self.make_job(cache_path)

    def copy_src_into_cache_dir(self) -> Path:
        dst_dir_name = self._get_dst_dir_id(self.cache_dir_path)
        cache_path = Path(self.cache_dir_path, dst_dir_name, 'src')

        shutil.copytree(self.src_path, cache_path)

        return cache_path

    @staticmethod
    def _get_dst_dir_id(cache_dir_path: Path):
        num_dirs = len(list(cache_dir_path.glob('*')))
        new_id = f'{num_dirs:04d}'

        return new_id

    @staticmethod
    def get_rq_queue(url: str,
                     name: str):
        connection = Redis.from_url(url)
        job_queue = Queue(name=name,
                          default_timeout=432000,  # 5 days (in seconds)
                          connection=connection)

        return job_queue

    def make_job(self, cache_path: Path):
        func_path = '.'.join(cache_path.parts)
        func_path = f'{func_path}.run.main'

        func_args = ['--conf_string', self.conf_string]
        if self.conf_kwargs is not None:
            func_args.append('--kwargs')
            func_args.extend(self.conf_kwargs)

        self.job_queue.enqueue(func_path, func_args, job_id=cache_path.parts[-2])
