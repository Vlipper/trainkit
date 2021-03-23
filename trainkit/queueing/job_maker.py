import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from redis import Redis
from rq import Queue

if TYPE_CHECKING:
    from typing import Optional


class JobMaker:
    def __init__(self,
                 run_params: dict,
                 hyper_params: dict,
                 proj_root_path: str,
                 host: str,
                 port: str,
                 db: str,
                 password: str,
                 queue_name: str,
                 job_timeout: 'Optional[str]' = None,
                 ttl: 'Optional[str]' = None,
                 result_ttl: 'Optional[str]' = None,
                 failure_ttl: 'Optional[str]' = None,
                 func_path: str = 'setup_train.train',  # path to function (must be relative to (importable) src dir)
                 src_path: str = 'src',  # path to src dir (must be relative to it's project root)
                 cache_path: str = 'src_cache',  # path to src cache dir (must be relative to it's project root)
                 ):
        # ToDo: add docstring
        self.proj_root_path = proj_root_path
        self.func_path = func_path
        self.src_path = Path(proj_root_path, src_path)

        self.func_kwargs = {'run_params': run_params, 'hyper_params': hyper_params}
        self.enqueue_kwargs = self._parse_enqueue_kwargs(job_timeout, ttl, result_ttl, failure_ttl)
        self.queue = self._get_queue(host, port, db, password, queue_name)
        self.cache_path = self._get_cache_path(proj_root_path, src_path, cache_path)

    def __call__(self):
        self.cache_path.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.src_path, self.cache_path, dirs_exist_ok=True)
        self._enqueue_job()

    @staticmethod
    def _parse_enqueue_kwargs(job_timeout: 'Optional[str]' = None,
                              ttl: 'Optional[str]' = None,
                              result_ttl: 'Optional[str]' = None,
                              failure_ttl: 'Optional[str]' = None) -> dict:
        job_timeout = -1 if job_timeout is None else job_timeout
        result_ttl = -1 if result_ttl is None else result_ttl
        failure_ttl = -1 if failure_ttl is None else failure_ttl

        enqueue_kwargs = {'job_timeout': job_timeout,
                          'result_ttl': result_ttl,
                          'failure_ttl': failure_ttl,
                          'ttl': ttl}

        return enqueue_kwargs

    @staticmethod
    def _get_queue(host: str,
                   port: str,
                   db: str,
                   password: str,
                   queue_name: str) -> Queue:
        connection = Redis(host=host, port=port, db=db, password=password)
        queue = Queue(name=queue_name, connection=connection)

        return queue

    @staticmethod
    def _get_cache_path(proj_root_path: str,
                        src_path: str,
                        cache_path: str) -> Path:
        cache_root_path = Path(proj_root_path, cache_path)

        num_dirs_in_cache = len(list(cache_root_path.glob('*')))
        cache_dir_name = f'{num_dirs_in_cache:04d}'

        cache_path = Path(cache_root_path, cache_dir_name, src_path)

        return cache_path

    def _enqueue_job(self):
        # `func_path` must be relative to directory with all projects (where Worker works)
        proj_root_dir_name = Path(self.proj_root_path).name
        proj_root_idx = self.cache_path.parts.index(proj_root_dir_name)
        func_path = '.'.join(self.cache_path.parts[proj_root_idx:])
        func_path = f'{func_path}.{self.func_path}'

        self.queue.enqueue(func_path, kwargs=self.func_kwargs, **self.enqueue_kwargs)
#         job_id=cache_path.parts[-2]
