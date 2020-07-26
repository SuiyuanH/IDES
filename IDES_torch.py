# -*- coding: utf-8 -*-

import torch
import numpy as np
import pickle
from os import makedirs
from os.path import exists

from .functional_torch import L21_norm
from .utils import one_hot_encoder, classifier_array, min_max_scaler

class IDES_rv_torch(object):
    '''
    class IDES_rv_torch(object)
    
    Attributes:
        classifier_pool: ndarray of base learners, essay H.
        classifier_number: int, essay k.
        character_number: int, essay m.
        target_number: int, essay d.
        weight_martix: 2D-tensor, k * m weight martix, essay W.
        encoder: one_hot_encoder.
    '''
    def __init__(self, classifier_pool, base_label):
        '''
        IDES_rv_torch.__init__(self, classifier_pool)
        
        Parameters:
            classifier_pool: an ensemble or a list of base learners.
            base_label: lables that base classifiers use.
        '''
        self.classifier_pool = classifier_array(classifier_pool)
        self.classifier_number = len(classifier_pool)
        self.character_number = None
        self.target_number = None
        self.weight_martix = None
        self.encoder = one_hot_encoder()
        self.encoder.new_rule(base_label, return_one_hot = False)
        self.target_number = len(self.encoder)
    
    def ensemble_probability(self, sample_input):
        '''
        IDES_rv_torch._ensemble_probability(self, sample_input)
        
        Parameters:
            sample_input: 2D-tensor, n * m data martix of input sample.
        
        Return:
            probability_result: 3D-tensors, n * d * k, all probabilities predicted by base learners.
        '''
        probability_result = torch.tensor([base_classifier.predict_proba(sample_input) for base_classifier in self.classifier_pool])
        probability_result = probability_result.permute(1, 2, 0)
        return probability_result
    
    def ensemble_correlation(self, sample_input):
        '''
        IDES_rv_torch._ensemble_correlation(self, sample_input)
        
        Parameters:
            sample_input: 2D-tensor, n * m data martix of input sample.
        
        Return:
            correlation_result: 2D-tensor, n * d data martix, result of correlation, essay phi.
        '''
        sample_number = len(sample_input)
        probability_result = self.ensemble_probability(sample_input)
        correlation_result = torch.matmul(probability_result, self.weight_martix)
        correlation_result = torch.bmm(correlation_result, sample_input.view(-1, self.character_number, 1)).view(sample_number, -1)
        return correlation_result
    
    def criterion_cross_entropy(self, sample_input, sample_target, IDES_lambda):
        '''
        IDES_rv_torch._loss(self, sample_input, sample_target, IDES_lambda = 1.5, need_loss = True)
        Calculate loss and derivative of loss in mode cross entropy.
        
        Parameters:
            sample_input: 2D-tensor, n * m data martix of input sample.
            sample_target: 1D-tensor, n encoded lable vector of input sample.
            IDES_lambda: float, coefficient of regularization.
        
        Return:
            loss: 1 * 1 tensor, loss.
        '''
        correlation_result = self.ensemble_correlation(sample_input)
        cross_entropy_term = torch.nn.CrossEntropyLoss()(correlation_result, sample_target)
        regularization_term = torch.mm(self.weight_martix, sample_input.T)
        regularization_term = L21_norm(regularization_term)
        loss = cross_entropy_term + IDES_lambda * regularization_term
        return loss
    
    def criterion_mean_square(self, sample_input, sample_target, IDES_lambda):
        '''
        IDES_rv_torch._loss(self, sample_input, sample_target, IDES_lambda = 1.5, need_loss = True)
        Calculate loss and derivative of loss in mode mean square.
        
        Parameters:
            sample_input: 2D-tensor, n * m data martix of input sample.
            sample_target: 2D-tensor, n * d one hot encoded label martix of input sample.
            IDES_lambda: float, coefficient of regularization.
        
        Return:
            loss: 1 * 1 tensor, loss.
        '''
        correlation_result = self.ensemble_correlation(sample_input)
        softmax_result = torch.nn.functional.softmax(correlation_result, dim = -1)
        cross_entropy_term = torch.nn.MSELoss()(softmax_result, sample_target)
        regularization_term = torch.mm(self.weight_martix, sample_input.T)
        regularization_term = L21_norm(regularization_term)
        loss = cross_entropy_term + IDES_lambda * regularization_term
        return loss
    
    def fit_once(self, sample_input, sample_target, IDES_lambda, learning_rate, loss_mode):
        '''
        IDES_rv_torch.fit_once(self, sample_input, sample_target, IDES_lambda, learning_rate)
        An iteration during learning.
        
        Parameters:
            sample_input: 2D-tensor, n * m data martix of input sample.
            sample_target: 1D-tensor or 2D tensor, n encoded lable vector of input sample for cross entropy loss, or n * d one hot encoded label martix for mean square loss.
            IDES_lambda: float, coefficient of regularization.
            learning_rate: float, learning rate.
            loss_mode: str, if it is set as 'cross_entropy', use CrossEntropyLoss, else use MSELoss.
        
        Return:
            loss_value: float, value of loss.
        '''
        self.weight_martix.requires_grad_(True)
        loss = self.criterion_cross_entropy(sample_input, sample_target, IDES_lambda) if loss_mode == 'cross_entropy' else self.criterion_mean_square(sample_input, sample_target, IDES_lambda)
        loss.backward()
        self.weight_martix = self.weight_martix.detach() - learning_rate * self.weight_martix.grad
        loss_value = loss.item()
        return loss_value
    
    def fit(self, sample_input, sample_label, IDES_lambda = 1.5, iteration_number = 50, learning_rate = 1e-3, random_state = None, loss_mode = 'cross_entropy'):
        '''
        IDES_rv_torch.fit(self, sample_input, sample_label, IDES_lambda = 1.5, iteration_number = 50, learning_rate = 1e-3)
        Process to train weight martix.
        
        Parameters:
            sample_input: 2D-ndarray, n * m data martix of input sample.
            sample_label: 1D-ndarray, n lable vector of input sample.
            IDES_lambda: float, optional, default = 1.5, coefficient of regularization.
            iteration_number: int, optional, default = 50, number of iterations.
            learning_rate: float, optional, default = 1e-3, learning rate.
            random_state: int or None, optional, default = None, random state to initialize weight martix, if None, initialize it totally randomly.
            loss_mode: str, optional, default = 'cross_entropy', if it is set as 'mean_square', use MSELoss.
        
        Return:
            loss_trend: list of float, the values of loss in all iterations.
        '''
        if self.character_number == None:
            self.character_number = sample_input.shape[1]
            self.weight_martix = torch.from_numpy(np.random.RandomState(random_state).rand(self.classifier_number, self.character_number)).double() if not random_state is None else torch.rand(self.classifier_number, self.character_number).double()
        sample_input = torch.from_numpy(sample_input)
        sample_target = self.encoder.to_index(sample_label) if loss_mode == 'cross_entropy' else self.encoder.to_one_hot(sample_label)
        sample_target = torch.from_numpy(sample_target)
        loss_trend = [self.fit_once(sample_input, sample_target, IDES_lambda, learning_rate, loss_mode) for iteration in range(iteration_number)]
        return loss_trend
        
    def predict(self, sample_input, IDES_rou = 0.2, select_mode = 'value'):
        '''
        IDES_rv_torch.predict(self, sample_input, IDES_rou = 0.2)
        Predict according to input samples.
        
        Parameters:
            sample_input: 2D-ndarray, n1 * m data martix of input sample.
            IDES_rou: float, optional, default = 0.2, threshold of classitier selection.
            select_mode: str, optional, default = 'value', if set it 'value', rou will be the value threshold. IF set it 'ratio', rou will be the ratio threshold.
        
        Return:
            sample_prediction: 1D-ndarray, n1 label vector of predictions.
        '''
        sample_number = len(sample_input)
        sample_input = torch.from_numpy(sample_input)
        predict_rou = torch.mm(self.weight_martix, sample_input.T).numpy()
        if select_mode == 'value':
            predict_rou = min_max_scaler(predict_rou, axis = 0)
            index_prediction = []
            for sample_index in range(sample_number):
                classifier_selected = self.classifier_pool[predict_rou[:, sample_index] >= IDES_rou]
                classifier_prediction = np.array([classifier.predict(sample_input[[sample_index]]) for classifier in classifier_selected])
                prediction_unique, prediction_count = np.unique(classifier_prediction, return_counts = True)
                vote_prediction = prediction_unique[prediction_count.argmax()]
                index_prediction.append(vote_prediction)
        else:
            classifier_selected_number = round(len(self.classifier_pool) * IDES_rou)
            index_prediction = []
            for sample_index in range(sample_number):
                classifier_selected = self.classifier_pool[predict_rou[:, sample_index].argsort()[- classifier_selected_number :]]
                classifier_prediction = np.array([classifier.predict(sample_input[[sample_index]]) for classifier in classifier_selected])
                prediction_unique, prediction_count = np.unique(classifier_prediction, return_counts = True)
                vote_prediction = prediction_unique[prediction_count.argmax()]
                index_prediction.append(vote_prediction)
        index_prediction = np.array(index_prediction, dtype = np.int64)
        sample_prediction = self.encoder.from_index(index_prediction)
        return sample_prediction
    
    def score(self, sample_input, sample_label, IDES_rou = 0.2, select_mode = 'value'):
        '''
        IDES_rv_torch.predict(self, sample_input, IDES_rou = 0.2)
        Predict according to input samples.
        
        Parameters:
            sample_input: 2D-ndarray, n * m data martix of input sample.
            sample_label: 1D-ndarray, n lable vector of input sample.
            IDES_rou: float, optional, default = 0.2, threshold of classitier selection.
            select_mode: str, optional, default = 'value', if set it 'value', rou will be the value threshold. IF set it 'ratio', rou will be the ratio threshold.
        
        Return:
            accuracy: float, accuracy of predictions.
        '''
        sample_prediction = self.predict(sample_input, IDES_rou, select_mode)
        accuracy = np.sum(sample_prediction == sample_label) / len(sample_label)
        return accuracy
    
    def save_model(self, path):
        '''
        IDES_rv_torch.save_model(self, path)
        Save model as pkl file.
        
        Parameters:
            path: str, path to save the model.
        '''
        if '/' in path:
            root_path = '/'.join(path.split('/')[:-1])
            if not exists(root_path):
                makedirs(root_path)
        with open(path, 'wb') as model_file:
            pickle.dump(self, model_file)
    
def load_model(path):
    '''
    load_model(path)
    Load model from pkl file.
    
    Parameters:
        path: str, path to load the model.
    
    Return:
        IDES_model: IDES_rv_torch, the model.
    '''
    with open(path, 'rb') as model_file:
        IDES_model = pickle.load(model_file)
    return IDES_model