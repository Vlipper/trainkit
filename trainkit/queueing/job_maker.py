import shutil
from pathlib import Path

from redis import Redis
from rq import Queue


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
                 queue_timeout: int = 432000,  # 5 days (in seconds)
                 func_path: str = 'setup_train.train',  # path to function (must be relative to (importable) src dir)
                 src_path: str = 'src',  # path to src dir (must be relative to it's project root)
                 cache_path: str = 'src_cache',  # path to src cache dir (must be relative to it's project root)
                 ):
        self.run_params = run_params
        self.hyper_params = hyper_params
        self.proj_root_path = proj_root_path
        self.src_path = Path(proj_root_path, src_path)
        self.func_path = func_path

        self.queue = self._get_queue(host, port, db, password, queue_name, queue_timeout)
        self.cache_path = self._get_cache_path(proj_root_path, src_path, cache_path)

    def __call__(self):
        self.cache_path.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.src_path, self.cache_path, dirs_exist_ok=True)
        self._enqueue_job()

    @staticmethod
    def _get_queue(host: str,
                   port: str,
                   db: str,
                   password: str,
                   queue_name: str,
                   queue_timeout: int) -> Queue:
        connection = Redis(host=host, port=port, db=db, password=password)
        queue = Queue(name=queue_name, default_timeout=queue_timeout, connection=connection)

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

        func_kwargs = {'run_params': self.run_params, 'hyper_params': self.hyper_params}
        self.queue.enqueue(func_path, kwargs=func_kwargs)
#         job_id=cache_path.parts[-2]
