import faiss
import pandas as pd

import time
import numpy as np
import torch
import os
from scipy import stats as s


class knn:
    def __init__(self, datafile, savefile=None, knn_size=10, save_to_file=True, resume=True):

        self.knn_size = knn_size
        self.x_data = None
        self.y_data = None
        self.save_file = datafile if not savefile else savefile
        self.classes = None

        self.save_to_file = save_to_file

        self.faiss_index = None
        # self.faiss_index = faiss.IndexFlatL2()

        if datafile and resume:
            print(f'loading data from file: {datafile}')
            if (os.path.exists(datafile)):
                print('File found')
                data = torch.load(datafile)
                self.x_data = data['x'].numpy()
                self.y_data = data['y']
                print(
                    f'Found {self.x_data.shape[0]} points with {len(set(self.y_data))} classes')
                print(pd.Series(self.y_data).value_counts())
                self.classes = list(set(self.y_data))

                self.faiss_index = faiss.IndexFlatL2(self.x_data.shape[-1])

                self.faiss_index.add(self.x_data)
            else:
                print('File not found')

    def print_info(self):
        print(pd.Series(self.y_data).value_counts())

    def add_points(self, x, y):

        if self.x_data is None:
            self.x_data = np.array(x)
            self.y_data = y
            self.faiss_index = faiss.IndexFlatL2(self.x_data.shape[-1])
        else:
            self.x_data = np.concatenate([self.x_data, x])
            self.y_data = np.concatenate([self.y_data, y])

        self.classes = list(set(self.y_data))

        self.faiss_index.reset()
        self.faiss_index.add(self.x_data)
        if self.save_to_file:
            torch.save({'x': self.x_data.detach().cpu(),
                        'y': self.y_data}, self.save_file)

    def remove_class(self, cl):
        inds_to_keep = [idx for idx, el in enumerate(self.y_data) if el != cl]

        self.x_data = self.x_data[inds_to_keep]
        self.y_data = [self.y_data[i] for i in inds_to_keep]

        self.classes = list(set(self.y_data))

        self.faiss_index.add(self.x_data)
        if self.save_to_file:
            torch.save({'x': self.x_data.detach().cpu(),
                        'y': self.y_data}, self.save_file)

    def classify(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        D, I = self.faiss_index.search(x, self.knn_size)

        # print(self.y_data[I])

        # print(I.min(), I.max())
        # print(self.x_data.shape, I.shape)
        # print(len(self.y_data))

        near_y = np.vectorize(lambda a: self.y_data[a])(I)

        # print(near_y)

        # near_y = [list(map(self.y_data.__getitem__, row)) for row in I.T]
        # print(near_y)

        cl = s.mode(near_y.T)[0][0]
        # print(cl[0][0])

        # print(I.shape)
        # print(cl[0], near_y[0])
        # print(near_y[0].count(cl[0]))
        # print(len(near_y), len(cl))
        frac = [np.count_nonzero(y == row) /
                self.knn_size for y, row in zip(near_y, cl)]

        # print(cl, frac)
        # print(D[..., 0])
        return cl, frac, D[..., 0]


if __name__ == '__main__':
    knn = knn('/home/iiwa/ros_ws/src/grasping_vision/scripts/datafiles/test_data_own_21_09_dino.pth',
              save_to_file=False)

    # knn.add_points()

    knn.print_info()

    print(knn.faiss_index.is_trained)
    print(knn.faiss_index.ntotal)

    test_data = np.ones((5, 384), dtype=np.float32)

    ret = knn.classify(test_data)

    print(ret)
