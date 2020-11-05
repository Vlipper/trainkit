from abc import abstractmethod

from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset):
    def __init__(self, run_params: dict,
                 hyper_params: dict,
                 **_ignored):
        super().__init__()

        # self.desc = self.__read_data_description(kwargs['description_csv_path'])
        # self.sampler = kwargs['sampler']
        # self.mean, self.std = hyper_params['mean'], hyper_params['std']

        self.batch_size = hyper_params['batch_size']
        self.num_workers = run_params['num_workers']

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def get_dataloader(self, is_val: bool):
        shuffle = False if is_val else True
        # sampler=self.sampler,
        loader = DataLoader(dataset=self, batch_size=self.batch_size, shuffle=shuffle,
                            num_workers=self.num_workers)

        return loader

    # @staticmethod
    # def __read_data_description(desc_path: Path) -> np.ndarray:
    #     desc = pd.read_csv(desc_path)
    #     desc = desc.values
    #
    #     return desc

    # def _standartize(self, img: np.ndarray) -> np.ndarray:
    #     img_mod = (img - self.mean) / self.std
    #
    #     return img_mod

    # @staticmethod
    # def _reorder_img_axes(img: np.ndarray, axes_order: Tuple[int, ...]) -> np.ndarray:
    #     img_mod = np.transpose(img, axes_order)
    #
    #     return img_mod
