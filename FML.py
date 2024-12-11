import os
import sys

import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from copy import deepcopy

class FML:
    def __init__(
            self,
            input_modes=20,
            target_modes=20,
            path=None,
            name="FML",
            base_model=None,
            samp_freq=1000,
            ):
        self.input_modes = input_modes
        self.target_modes = target_modes
        self.path = path
        self.model_name = name
        self.samp_freq = samp_freq # sampling frequency in units of Hertz
        
        # define directory to save model
        self.save_path = os.path.join(self.path, self.model_name)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        # Define the input and target columns
        self.input_cols = [f'sin_{n}' for n in range(1, self.input_modes)] + [f'cos_{n}' for n in range(self.input_modes)]
        self.target_cols = [f'sin_{n}' for n in range(1, self.target_modes)] + [f'cos_{n}' for n in range(self.target_modes)]
        
        # create the model instances
        if base_model is None:
            base_model = SVR(kernel='linear', degree=2, gamma=1e-3, epsilon=1e-3, C=100)
        self.models = {k: deepcopy(base_model) for k in self.target_cols}
        self.scaler = MinMaxScaler()

    def train(self, inputs, targets, input_type='time', target_type='time'):
        """ Train the model using the inputs and targets """

        # check that inputs and targets are in correct form
        self.check_inputs(inputs, input_type)
        self.check_targets(targets, target_type)

        # convert the inputs and targets to frequency domain
        if input_type == 'time':
            inputs_fft_arr = self.time_to_frequency_transform(inputs, self.input_modes, self.input_cols)
            inputs_freq = inputs_fft_arr[self.input_cols]
        elif input_type == 'frequency':
            inputs_freq = inputs[self.input_cols]

        if target_type == 'time':
            targets_fft_arr = self.time_to_frequency_transform(targets, self.target_modes, self.target_cols)
            targets_freq = targets_fft_arr[self.target_cols]
        elif target_type == 'frequency':
            targets_freq = targets[self.target_cols]

        # compute the sample weights for the inputs
        sample_weights = self.calculate_sample_weights(inputs)

        # preprocessing with self.scaler
        inputs_freq = self.scaler.fit_transform(inputs_freq)

        # train the model
        for col in tqdm(self.target_cols, total=len(self.target_cols), desc='Training FML'):
            self.models[col].fit(inputs_freq, targets_freq[col], sample_weight=sample_weights)

    def predict(self, inputs, input_type='time', target_type='time'):
        """ Predict the target using the inputs """

        # check that inputs are in correct form
        self.check_inputs(inputs, input_type)

        # convert the inputs to frequency domain
        if input_type == 'time':
            inputs_fft_arr = self.time_to_frequency_transform(inputs, self.input_modes, self.input_cols)
            inputs_freq = inputs_fft_arr[self.input_cols]
            inputs_len = inputs_fft_arr['len'].astype(int)
        elif input_type == 'frequency':
            inputs_freq = inputs[self.input_cols]
            inputs_len = inputs['len'].astype(int)

        # preprocessing with self.scaler
        inputs_freq = self.scaler.transform(inputs_freq)

        # predict the target
        predictions = {}
        for col in tqdm(self.target_cols, total=len(self.target_cols), desc='Predicting FML'):
            predictions[col] = self.models[col].predict(inputs_freq)
        predictions['len'] = inputs_len # add the length of the input array to the predictions
        predictions = pd.DataFrame(predictions, index=inputs.index)
        
        # convert the predictions to time domain
        if target_type == 'time':
            predictions_time = self.frequency_to_time_transform(predictions)
            return predictions_time
        else:
            return predictions
        
    def check_inputs(self, inputs, input_type):
        """ Check if the inputs shape, type, and index is correct """

        # check that inputs is a pandas dataframe
        if not isinstance(inputs, pd.DataFrame):
            raise ValueError(f"Inputs should be a pandas dataframe, but got {type(inputs)}")
        
        # check that inputs have the correct indexing
        if inputs.index.shape[0] != 2 and inputs.index.names != ['id', 'cycle']:
            raise ValueError(f"Inputs should have a multiindex of ['id', 'cycle'], but got {inputs.index}")

        # check that inputs have the correct shape
        if input_type == 'time':
            
            # check that we have a single column for the waveform
            if inputs.shape[1] != 1:
                raise ValueError(f"Input shape should be (n, 1), but got {inputs.shape}")
            
            # check that the inputs have the correct column name
            if inputs.columns[0] != 'wvf':
                raise ValueError(f"Input column should be 'wvf', but got {inputs.columns}")

        elif input_type == 'frequency':
            
            # check that the inputs have the correct shape
            if inputs.shape[1] != len(self.input_cols):
                raise ValueError(f"Input shape should be (n, {self.input_modes}), but got {inputs.shape}")
            
            # check that the inputs have the correct columns and order
            if not all([col in inputs.columns for col in self.input_cols]):
                raise ValueError(f"Input columns should be {self.input_cols}, but got {inputs.columns}")
        
    def check_targets(self, targets, target_type):
        """ Check if the targets shape, type, and index is correct """

        # check that targets is a pandas dataframe
        if not isinstance(targets, pd.DataFrame):
            raise ValueError(f"Targets should be a pandas dataframe, but got {type(targets)}")
        
        # check that targets have the correct indexing
        if targets.index.shape[0] != 2 and targets.index.names != ['id', 'cycle']:
            raise ValueError(f"targets should have a multiindex of ['id', 'cycle'], but got {targets.index}")

        # check that targets have the correct shape
        if target_type == 'time':
            
            # check that we have a single column for the waveform
            if targets.shape[1] != 1:
                raise ValueError(f"Target shape should be (n, 1), but got {targets.shape}")
            
            # check that the targets have the correct column name
            if targets.columns[0] != 'wvf':
                raise ValueError(f"Target column should be 'wvf', but got {targets.columns}")

        elif target_type == 'frequency':
            
            # check that the targets have the correct shape
            if targets.shape[1] != len(self.target_cols):
                raise ValueError(f"Target shape should be (n, {len(self.target_cols)}), but got {targets.shape}")
            
            # check that the targets have all the correct columns
            if not all([col in targets.columns for col in self.input_cols]):
                raise ValueError(f"Input columns should be {self.input_cols}, but got {targets.columns}")

    def calculate_sample_weights(self, inputs):
        """ Calculate the sample weights for the inputs """
        index_cnt = inputs.index.get_level_values(0).value_counts()
        sample_weights = inputs.apply(lambda x: 1/index_cnt.loc[x.name[0]], axis=1)
        return sample_weights

    def time_to_frequency_transform(self, time_series, num_modes, cols):
        """ Convert the time series to frequency domain """
        
        # compute rfft on the time_series arrays
        arr_ffts = []
        for i in range(len(time_series)):
            arr = time_series.iloc[i]
            arr_wvf = arr['wvf']
            arr_len = len(arr_wvf)
            arr_t = np.arange(0, arr_len)/self.samp_freq
            arr_fft = np.fft.rfft(arr_wvf)/arr_len
            arr_freq = np.fft.rfftfreq(arr_len, d=1/self.samp_freq)[:self.input_modes]
            arr_fft_cos = np.real(arr_fft[:num_modes])
            arr_fft_sin = np.imag(arr_fft[1:num_modes])
            arr_ffts.append(np.hstack([arr_fft_sin, arr_fft_cos, arr_len]))

        # create dataframe from the ffts
        df_ffts = pd.DataFrame(arr_ffts, columns=cols+['len'], index=time_series.index)

        return df_ffts

    def frequency_to_time_transform(self, frequency_series):
        """ Convert the frequency series to time domain """
        
        # compute the irfft on the frequency_series arrays
        times_series = []
        for i in range(len(frequency_series)):
            arr = frequency_series.iloc[i]
            arr_len = arr['len'].astype(int)

            # compose the frequency components into a single array
            freqs = [complex(arr['cos_0'], 0)]
            for j in range(1, self.target_modes):
                freqs.append(complex(arr[f'cos_{j}'], arr[f'sin_{j}']))
            
            # compute the irfft
            time_serie = np.fft.irfft(np.array(freqs)*arr_len, n=arr_len)
            
            # append results
            times_series.append([time_serie])

        # create dataframe from the ffts
        df_times = pd.DataFrame(times_series, columns=['wvf'], index=frequency_series.index)
        return df_times
            
    def save(self):
        """ Save the entire model class in the save_path directory as a .pkl file """
            
        # save the model
        with open(os.path.join(self.save_path, f"{self.model_name}.pkl"), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, name="FML"):
        """ Load the entire class from the specified path """
        
        # load the model
        with open(os.path.join(path, f"{name}.pkl"), 'rb') as f:
            return pickle.load(f)
        