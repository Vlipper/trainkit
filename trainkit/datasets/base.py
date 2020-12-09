from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        super().__init__(**_ignored)

        self.batch_size = hyper_params['batch_size']
        self.num_workers = run_params['num_workers']

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def get_dataloader(self, is_val: bool) -> DataLoader:
        """Initializes and returns DataLoader instance based on self Dataset

        Args:
            is_val: if True than DataLoader returns shuffled batches

        Returns:
            DataLoader instance
        """
        shuffle = False if is_val else True

        loader = DataLoader(dataset=self,
                            batch_size=self.batch_size,
                            shuffle=shuffle,
                            num_workers=self.num_workers,
                            persistent_workers=True)

        return loader
