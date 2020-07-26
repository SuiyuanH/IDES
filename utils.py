# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaseEnsemble

class one_hot_encoder(object):
    '''
    class one_hot_encoder(object)
    
    Attributes:
        label_unique: 1D-ndarray, unique ones of train labels, the rule of encoding.
    
    Methods:
        __len__(self):
            number of classes.
        to_one_hot(self, sample_label):
            apply one hot encoding.
        to_index(self, sample_label):
            index of labels.
        new_rule(self, sample_label, return_one_hot = True):
            generate new rule and return encoded label.
        from_index(self, index_array):
            return labels according to given index.
    '''
    def __init__(self):
        '''
        one_hot_encoder.__init__(self)
        
        Parameters:
            None
        '''
        self.label_unique = np.array([])
    
    def __len__(self):
        '''
        one_hot_encoder.__len__(self)
        Number of classes.
        
        Parameters:
            None
        
        Return:
            number of classes.
        '''
        return len(self.label_unique)
    
    def to_one_hot(self, sample_label):
        '''
        one_hot_encoder.to_one_hot(self, sample_label)
        Apply one hot encoding.
        
        Parameters:
            sample_label: 1D-ndarray, label to encode.
        
        Return:
            label_encoded: 2D-ndarray, one hot encoded label.
        '''
        label_encoded = np.vstack([self.label_unique == label for label in sample_label]).astype(np.float64)
        return label_encoded
    
    def to_index(self, sample_label):
        '''
        one_hot_encoder.to_index(self, sample_label)
        Index of labels.
        
        Parameters:
            sample_label: 1D-ndarray, label to encode.
        
        Return:
            label_encoded: 1D-ndarray, index of labels.
        '''
        label_encoded = np.zeros_like(sample_label)
        for index, label in enumerate(self.label_unique):
            label_encoded[sample_label == label] = index
        label_encoded = label_encoded.astype(np.int64)
        return label_encoded
    
    def new_rule(self, sample_label, return_one_hot = True):
        '''
        one_hot_encoder.to_index(self, sample_label, return_one_hot = True)
        Generate new encoding rule and return encoded labels.
        
        Parameters:
            sample_label: 1D-ndarray, label to encode.
            return_one_hot: bool, optional, default = True, whether encode in one hot method.
        
        Return:
            label_encoded: 1D or 2D-ndarray, encoded labels.
        '''
        self.label_unique = np.unique(sample_label)
        if return_one_hot:
            label_encoded = self.to_one_hot(sample_label)
        else:
            label_encoded = self.to_index(sample_label)
        return label_encoded
    
    def from_index(self, index_array):
        '''
        one_hot_encoder.to_index(self, sample_label, return_one_hot = True)
        Return labels according to given index.
        
        Parameters:
            index_array: 1D-ndarray, index of labels.
        
        Return:
            label_decoded: 1D-ndarray, decoded labels.
        '''
        label_decoded = self.label_unique[index_array]
        return label_decoded

class data_loader(object):
    '''
    class data_loader(object)
    
    Attributes:
        dataset_name: str, name of dataset. e.g. 'Australian'.
        data: 2D-ndarray, data of samples.
        label: 1D-ndarray, label of samples.
    '''
    def __init__(self, csv_path):
        '''
        data_loader.__init__(self, csv_path)
        
        Parameters:
            csv_path: path of dataset. e.g. 'datasets/Australian.csv'
        '''
        self.dataset_name = csv_path.split('/')[-1][:-4]
        self.data = np.loadtxt(csv_path, delimiter = ',')[:, 1:]
        self.label = self.data[:, -1]
        self.data = self.data[:, :-1]
    
    def iterator_10_fold(self, random_state, save_division = True):
        '''
        (Unfinished)
        data_loader.iterator_10_fold(self, random_state, save_division = True)
        Generate 10-fold iterator.
        '''
        pass
    
    def iterator_random_split(self, iterator_random_state, test_size = 0.25, selection_size = 0.25, save_division = False, save_path = 'data_split'):
        '''
        data_loader.iterator_random_split(self, iterator_random_state, save_division = False, save_path = 'data_split')
        Generator of some random splited data.
        
        Parameters:
            iterator_random_state: iterable object, provide random states to split the dataset.
            save_division: bool, optional, default = False, whether save splited dataset.
            save_path: str, optional, default = 'data_split', directory of saved dataset. e.g. set save_path = 'data_split' then save splited data as 'data_split/Australian_random_0.npz'.
        
        Return:
        generator object, each item is a tuple, whilch includes:
            x_train_base: data to train original ensemble.
            x_train_selection: data to select classifiers.
            x_test: data to calculate accuracy.
            y_train_base: label to train original ensemble.
            y_train_selection: label to select classifiers.
            y_test: label to calculate accuracy.
            random_state: int.
        '''
        selection_size_fixed = selection_size / (1 - test_size)
        for random_state in iterator_random_state:
            x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size = test_size, random_state = random_state)
            x_train_base, x_train_selection, y_train_base, y_train_selection = train_test_split(x_train, y_train, test_size = selection_size_fixed, random_state = random_state)
            if save_division == True:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.savez('{}/{}_random_{}.npz'.format(save_path, self.dataset_name, random_state), x_train_base = x_train_base, x_train_selection = x_train_selection, x_test = x_test, y_train_base = y_train_base, y_train_selection = y_train_selection, y_test = y_test)
            yield x_train_base, x_train_selection, x_test, y_train_base, y_train_selection, y_test, random_state
    
def load_random_split(dataset_name, iterator_random_state, save_path = 'data_split'):
    '''
    Function load_random_split(dataset_name, iterator_random_state, save_path = 'data_split')
    Generator to load some saved random splited data.
    
    Parameters:
        dataset_name: str, name of dataset. e.g. 'Australian'.
        iterator_random_state: iterable object, provide random states to split the dataset.
        save_path: str, optional, default = 'data_split', directory of saved dataset. e.g. for loading 'data_split/Australian_random_0.npz' then we set save_path = 'data_split'.
    
    Return:
        generator object, each item is a tuple, which includes:
            x_train_base: data to train original ensemble.
            x_train_selection: data to select classifiers.
            x_test: data to calculate accuracy.
            y_train_base: label to train original ensemble.
            y_train_selection: label to select classifiers.
            y_test: label to calculate accuracy.
            random_state: int.
    '''
    for random_state in iterator_random_state:
        npz_data = np.load('{}/{}_random_{}.npz'.format(save_path, dataset_name, random_state))
        x_train_base = npz_data['x_train_base']
        x_train_selection = npz_data['x_train_selection']
        x_test = npz_data['x_test']
        y_train_base = npz_data['y_train_base']
        y_train_selection = npz_data['y_train_selection']
        y_test = npz_data['y_test']
        yield x_train_base, x_train_selection, x_test, y_train_base, y_train_selection, y_test, random_state

def classifier_array(classifier_iterable):
    '''
    Function classifier_array(classifier_iterable)
    Generate ndarray of classifiers from sklearn.ensemble or iterable object of classifiers.
    
    Parameters:
        classifier_iterable: sklearn.ensemble or iterable object of classifiers.
    
    Return:
        classifier_pool: ndarray of classifiers.
    '''
    classifier_list = classifier_iterable.estimators_ if isinstance(classifier_iterable, BaseEnsemble) else (classifier_iterable if type(classifier_iterable) == list else list(classifier_iterable))
    classifier_pool = np.array(classifier_list)
    return classifier_pool

def min_max_scaler(input_array, axis = 0, eps = 1e-8):
    '''
    Function min_max_scaler(input_array, axis = 0)
    
    Parameters:
        input_array: ndarray.
        axis: int, the index of axis to which the scaler is applied.
    
    Return:
        result_array: ndarray. The scaled array.
    '''
    array_max = input_array.max(axis = axis, keepdims = True)
    array_min = input_array.min(axis = axis, keepdims = True)
    result_array = np.where(array_max == array_min, 1, (input_array - array_min) / (array_max - array_min + eps))
    return result_array