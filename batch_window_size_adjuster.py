import numpy as np 
import os

class BatchWindowSizeAdjuster:
    def __init__(self, window_size, feature_size, overlapping, batch_size):
        self.window_size = window_size
        self.feature_size = feature_size
        self.overlapping = overlapping
        self.batch_size = batch_size
        self.typ = None

    def set_typ(self, typ):
        self.typ = typ

    def windowing(self, data_tr, label_tr):
        if self.typ is None:
            raise ValueError('You should specify type (train/test/dev)')
        
        PATH_X, PATH_Y = self.get_path('windowing')
        if self.is_exists(PATH_X):
            return self.load(PATH_X), self.load(PATH_Y)

        x_wind, y_wind = [], []
        for x,y in zip(data_tr, label_tr):
            x_list, y_list = self.windowing_(x, y)
            x_wind.append(x_list)         
            y_wind.append(y_list)
        
        self.save(PATH_X, PATH_Y, x_wind, y_wind)
        return x_wind, y_wind

    """" reshape time to the fix size """
    def windowing_(self, x, y): 
        not_exceed = lambda mv, size : (mv+self.window_size) <= size 
        x_list = np.array([]).reshape(-1, self.window_size, self.feature_size)
        y_list = np.array([]).reshape(-1, 1)
        mv_ = 0
        while not_exceed(mv_, x.shape[0]): 
            x_ = x[mv_:(self.window_size+mv_)]
            if x_.shape[0] == self.window_size:
                x_list = np.append(x_list, self.reshape_3D(x_), axis=0)
                y_list = np.append(y_list, y.reshape(-1,1), axis=0)
            mv_ += self.overlapping
        return x_list, y_list

    def reshape_3D(self, x):
        return np.full((1, x.shape[0], x.shape[1]), x)

    def div_batch_size(self, x, y):
        x = np.array([x[i:(i+self.batch_size)] 
                for i in range(0, x.shape[0], self.batch_size)])
        y = np.array([y[i:(i+self.batch_size)]
                for i in range(0, y.shape[0], self.batch_size)])
        return x, y

    def conc_one_matrix(self, x_wind, y_wind):
        x_mat = np.array([]).reshape(-1, self.window_size, self.feature_size)
        y_mat = np.array([]).reshape(-1, 1)
        for x,y in zip(x_wind, y_wind):
            x_mat = np.vstack((x_mat, x))
            y_mat = np.vstack((y_mat, y))
        return x_mat, y_mat 

    def split_into_batch_size(self, data_tr, label_tr):
        if self.typ is None:
            raise ValueError('You should specify type (train/test/dev)')

        PATH_X, PATH_Y = self.get_path('batch')
        if self.is_exists(PATH_X):
            return self.load(PATH_X), self.load(PATH_Y)
        
        x_wind, y_wind = self.windowing(data_tr, label_tr)
        
        # self.split_batch_size(x_wind, y_wind)

        x_mtrx, y_mtrx = self.conc_one_matrix(x_wind, y_wind)  
        x_btch, y_btch = self.div_batch_size(x_mtrx, y_mtrx)
        
        self.save(PATH_X, PATH_Y, x_btch, y_btch)
        return x_btch, y_btch

    def load(self, path):
        return np.load(path, allow_pickle=True)

    def is_exists(self, path):
        return os.path.exists(path)

    def save(self, path_x, path_y, data_x, data_y):
        np.save(path_x, data_x)
        np.save(path_y, data_y)

    def get_path(self, fnc_typ):
        folder_path = 'Dataset_Final_{}_window_size'.format(self.window_size)
        functn_path = fnc_typ
        x = os.path.join(folder_path, functn_path, '{}_x.npy'.format(self.typ))
        y = os.path.join(folder_path, functn_path, '{}_y.npy'.format(self.typ))
        return x, y
    
    def get_dataset_if_exists(self, fnc_typ):
        if self.typ is None:
            raise ValueError('You should specify type (train/test/dev)')
        PATH_X, PATH_Y = self.get_path('batch')
        if self.is_exists(PATH_X):
            return self.load(PATH_X), self.load(PATH_Y)
        else:
            return PATH_X, PATH_Y

    # implement later 
    # (34, 200, 20) ... (42, 30, 20)
    def split_batch_size(self, x, y):
        List_x, List_y = [], []
        for x_, y_ in zip(x, y):
            size = x_.shape[0]
            for idx in x_.shape[0]:
                arr = x_[idx, :, :]
                List_x.append()
            if size <= self.batch_size:
                List_x.append(x_)
                List_y.append(y_)
            else:
                return 0
