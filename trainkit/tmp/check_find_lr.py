import numpy as np
from trainkit.core.lr_finder import LRFinder


if __name__ == '__main__':
    lr_finder = LRFinder.__new__(LRFinder)

    # generate data
    np.random.seed(42)
    part_1 = np.random.uniform(-0.2, 0.2, 30) + np.repeat(1.5, 30)
    part_2 = np.random.uniform(-0.2, 0.2, 30) + np.linspace(1.5, 0.7, 30)
    part_3 = np.random.uniform(-0.2, 0.2, 20) + np.linspace(0.7, 1, 20)
    part_4 = np.random.uniform(-0.2, 0.2, 10) + np.linspace(1, 2, 10)
    part_5 = np.random.uniform(-0.2, 0.2, 10) + np.repeat(1.5, 10)
    loss = np.concatenate((part_1, part_2, part_3, part_4, part_5))
    lr = np.geomspace(1e-7, 1e-1, loss.shape[0])
    avg_loss = [loss[0]]
    for cur_loss in loss[1:]:
        avg_loss.append(0.8*avg_loss[-1] + 0.2*cur_loss)

    lr_finder.logs = {'loss': loss, 'lr': lr, 'avg_loss': avg_loss}
    lr_finder.model_name = 'default'

    lr_finder.logs.update({key: np.array(val) for key, val in lr_finder.logs.items()})
    # print(lr_finder.logs.keys())
    # print([type(i) for i in lr_finder.logs.values()])
    # print([i.shape for i in lr_finder.logs.values()])

    a, b = lr_finder.find_optimal_lr_borders()
    print(a, b)
    lr_finder.plot('show')
