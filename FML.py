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
        """
        Train the model using the inputs and targets.
        
        Parameters:
        -----------
        inputs : pd.DataFrame
            The input data for training. Can be in time or frequency domain.
        targets : pd.DataFrame
            The target data for training. Can be in time or frequency domain.
        input_type : str, optional
            The type of input data, either 'time' or 'frequency'. Default is 'time'.
        target_type : str, optional
            The type of target data, either 'time' or 'frequency'. Default is 'time'.
        
        Returns:
        --------
        None
        """
        # check that input_type and target_type are valid
        if input_type not in ['time', 'frequency']:
            raise ValueError(f"input_type should be 'time' or 'frequency', but got {input_type}")
        if target_type not in ['time', 'frequency']:
            raise ValueError(f"target_type should be 'time' or 'frequency', but got {target_type}")

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
        """
        Predict the target using the inputs.
        
        Parameters:
        -----------
        inputs : pd.DataFrame
            The input data for prediction. It can in either the time or frequency domain.
        input_type : str, optional
            The type of input data, either 'time' or 'frequency'. Default is 'time'.
        target_type : str, optional
            The type of target data, either 'time' or 'frequency'. Default is 'time'.
        
        Returns:
        --------
        pd.DataFrame
            The predicted target data. If target_type is 'time', the predictions are returned in the time domain.
            Otherwise, the predictions are returned in the frequency domain.
        """
        # check that input_type and target_type are valid
        if input_type not in ['time', 'frequency']:
            raise ValueError(f"input_type should be 'time' or 'frequency', but got {input_type}")
        if target_type not in ['time', 'frequency']:
            raise ValueError(f"target_type should be 'time' or 'frequency', but got {target_type}")
        
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
        """
        Check if the inputs shape, type, and index is correct.
        
        The expected dataframe should have the following conditions:
        - a multiindex of ['id', 'cycle'] where 'id' is the unique identifier for each time series and 'cycle' is the index of the time series.
        - a single column 'wvf' if input_type is 'time', where 'wvf' is the single cardiac cycle that is being analyzed.
        - multiple columns if input_type is 'frequency', where each column represents a frequency component of the signal. The columns are the real and imaginary parts of the FFT. Note: excludes first imaginary part as it is always zero. 

        Parameters:
        -----------
        inputs : pd.DataFrame
            The input data to be checked. It should be a pandas DataFrame.
        input_type : str
            The type of input data. It can be either 'time' or 'frequency'.
        
        Raises:
        -------
        ValueError
            If the inputs are not a pandas DataFrame.
            If the inputs do not have a multiindex of ['id', 'cycle'].
            If the input_type is 'time' and the inputs do not have a single column named 'wvf'.
            If the input_type is 'frequency' and the inputs do not have the correct shape or columns.
        
        Returns:
        --------
        None
        """
        
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
        """
        Check if the targets shape, type, and index is correct.
        
        The expected dataframe should have the following conditions:
        - a multiindex of ['id', 'cycle'] where 'id' is the unique identifier for each time series and 'cycle' is the index of the time series.
        - a single column 'wvf' if input_type is 'time', where 'wvf' is the single cardiac cycle that is being analyzed.
        - multiple columns if input_type is 'frequency', where each column represents a frequency component of the signal. The columns are the real and imaginary parts of the FFT. Note: excludes first imaginary part as it is always zero. 

        Parameters:
        -----------
        targets : pd.DataFrame
            The targets dataframe to be checked. It should be a pandas DataFrame.
        target_type : str
            The type of the target, either 'time' or 'frequency'. This determines the expected shape and columns of the targets dataframe.
        
        Raises:
        -------
        ValueError
            If the targets is not a pandas DataFrame.
            If the targets do not have a multiindex of ['id', 'cycle'].
            If the targets shape is incorrect for the given target_type.
            If the targets columns are incorrect for the given target_type.
        
        Returns:
        --------
        None
        """

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
        """
        Calculate the sample weights for the inputs.

        This function calculates the sample weights for each input based on the 
        frequency of the first level index values ('id'). The weight for each sample 
        is computed as the inverse of the count of its corresponding index value.

        Parameters:
        -----------
        inputs : pd.DataFrame
            The input data for training.  The first level of the 
            index is used to calculate the sample weights.

        Returns:
        --------
        pd.Series
            A pandas Series containing the sample weights for each input.
        """
        index_cnt = inputs.index.get_level_values(0).value_counts()
        sample_weights = inputs.apply(lambda x: 1/index_cnt.loc[x.name[0]], axis=1)
        return sample_weights

    def time_to_frequency_transform(self, time_series, num_modes, cols):
        """ 
        Convert the time series to frequency domain.
        
        This function takes a time series dataset and converts it to the frequency domain using the real 
        Fast Fourier Transform (rFFT). It extracts the cosine and sine components of the FFT up to a 
        specified number of modes and returns a DataFrame containing these components along with the 
        original length of each time series.
        
        Parameters:
        -----------
        time_series : pd.DataFrame
            A DataFrame where each row contains a time series to be transformed. 
            Each row should have a 'wvf' column containing the waveform data.
        num_modes : int
            The number of frequency modes to retain in the transformation.
        cols : list of str
            The column names for the resulting DataFrame, excluding the 'len' column 
            which will be added automatically.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame where each row contains the sine and cosine components of the FFT 
            for the corresponding time series, along with the original length of the time series.
        """
        
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
        """
        Convert the frequency series to time domain.
        
        This function takes a frequency domain representation of a signal and 
        converts it to the time domain using the inverse real Fourier transform (irfft).
        
        Parameters:
        -----------
        frequency_series : pd.DataFrame
            A DataFrame where each row represents a frequency series with columns 'len', 
            'cos_0', 'cos_1', ..., 'cos_n', 'sin_1', 'sin_2', ..., 'sin_n'. The 'len' column 
            represents the length of the original time series, and 'cos_i' and 'sin_i' columns 
            represent the cosine and sine components of the frequency series respectively.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame with the same index as the input, containing a single column 'wvf' 
            where each entry is the time domain representation of the corresponding frequency series.
        """
        
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
        """
        Save the entire model class in the save_path directory as a .pkl file.
        
        This method serializes the current instance of the model class and saves it 
        to a file in the specified directory. The filename is constructed using the 
        model's name with a .pkl extension.
        
        Parameters:
        -------
        self : FML
            The instance of the model class containing attributes such as 
            save_path (the directory where the file will be saved) and model_name 
            (the name of the model to be used in the filename).
        
        Returns:
        --------
        None
        """    
        # save the model
        with open(os.path.join(self.save_path, f"{self.model_name}.pkl"), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, name="FML"):
        """
        Load the entire class from the specified path.
        This method deserializes and loads a previously saved class instance from a 
        pickle file located at the given path.
        
        Parameters:
        -----------
        path : str
            The directory path where the pickle file is located.
        name : str, optional
            The base name of the pickle file (without extension). Defaults to "FML".
        
        Returns:
        --------
        object
            The loaded class instance.
        
        Raises:
        -------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If there is an error reading the file.
        pickle.UnpicklingError
            If there is an error unpickling the file.
        """
        # load the model
        with open(os.path.join(path, f"{name}.pkl"), 'rb') as f:
            return pickle.load(f)
        