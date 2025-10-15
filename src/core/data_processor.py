"""
Core data processing utilities for SkySense
Handles signal resampling, preprocessing, and ULog parsing
"""
import numpy as np
import pandas as pd
from pyulog import ULog
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal, interpolate
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Core data processing utilities for flight log analysis"""
    
    def __init__(self, target_freq: float = 20.0):
        """
        Initialize data processor
        
        Args:
            target_freq: Target frequency for resampling (Hz), default 20Hz (50ms)
        """
        self.target_freq = target_freq
        self.target_dt = 1.0 / target_freq
        
    def load_ulog(self, log_path: str) -> ULog:
        """Load ULog file and return ULog object"""
        try:
            return ULog(log_path)
        except Exception as e:
            raise ValueError(f"Failed to load ULog file {log_path}: {e}")
    
    def extract_dataset(self, ulog: ULog, topic_name: str) -> Optional[pd.DataFrame]:
        """
        Extract dataset from ULog for a specific topic
        
        Args:
            ulog: ULog object
            topic_name: Name of the topic to extract
            
        Returns:
            DataFrame with timestamp and data columns
        """
        try:
            dataset = ulog.get_dataset(topic_name)
            if dataset is None:
                return None
                
            # Convert to DataFrame
            data = {}
            data['timestamp'] = dataset.data['timestamp'] / 1e6  # Convert to seconds
            
            # Add all other fields
            for field_name in dataset.data.keys():
                if field_name != 'timestamp':
                    data[field_name] = dataset.data[field_name]
                    
            df = pd.DataFrame(data)
            return df.sort_values('timestamp')
            
        except Exception as e:
            print(f"Warning: Could not extract topic {topic_name}: {e}")
            return None
    
    def resample_to_frequency(self, df: pd.DataFrame, freq: float = None) -> pd.DataFrame:
        """
        Resample DataFrame to target frequency using interpolation
        
        Args:
            df: Input DataFrame with 'timestamp' column
            freq: Target frequency (Hz), uses instance default if None
            
        Returns:
            Resampled DataFrame
        """
        if freq is None:
            freq = self.target_freq
            
        if df is None or len(df) == 0:
            return df
            
        # Create uniform time grid
        t_start = df['timestamp'].iloc[0]
        t_end = df['timestamp'].iloc[-1]
        dt = 1.0 / freq
        
        # Generate uniform time points
        t_uniform = np.arange(t_start, t_end + dt, dt)
        
        # Interpolate all numeric columns
        result = {'timestamp': t_uniform}
        
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Remove NaN values for interpolation
            valid_mask = ~df[col].isna()
            if valid_mask.sum() < 2:
                continue
                
            t_valid = df['timestamp'][valid_mask]
            y_valid = df[col][valid_mask]
            
            # Interpolate
            try:
                interp_func = interpolate.interp1d(
                    t_valid, y_valid, 
                    kind='linear', 
                    fill_value='extrapolate',
                    bounds_error=False
                )
                result[col] = interp_func(t_uniform)
            except Exception as e:
                print(f"Warning: Could not interpolate column {col}: {e}")
                continue
                
        return pd.DataFrame(result)
    
    def compute_moving_stats(self, series: pd.Series, window_sec: float, 
                           stats: List[str] = ['mean', 'std', 'rms']) -> pd.DataFrame:
        """
        Compute moving window statistics
        
        Args:
            series: Input time series
            window_sec: Window size in seconds
            stats: List of statistics to compute ('mean', 'std', 'rms', 'max', 'min')
            
        Returns:
            DataFrame with computed statistics
        """
        # Convert window to samples
        window_samples = int(window_sec * self.target_freq)
        
        result = {}
        
        if 'mean' in stats:
            result['mean'] = series.rolling(window_samples, center=True).mean()
        if 'std' in stats:
            result['std'] = series.rolling(window_samples, center=True).std()
        if 'rms' in stats:
            result['rms'] = np.sqrt(series.rolling(window_samples, center=True).apply(lambda x: (x**2).mean()))
        if 'max' in stats:
            result['max'] = series.rolling(window_samples, center=True).max()
        if 'min' in stats:
            result['min'] = series.rolling(window_samples, center=True).min()
        if 'median' in stats:
            result['median'] = series.rolling(window_samples, center=True).median()
            
        return pd.DataFrame(result)
    
    def smooth_signal(self, series: pd.Series, method: str = 'median', window_sec: float = 0.5) -> pd.Series:
        """
        Smooth signal using various methods
        
        Args:
            series: Input signal
            method: Smoothing method ('median', 'mean', 'gaussian')
            window_sec: Window size in seconds
            
        Returns:
            Smoothed signal
        """
        window_samples = int(window_sec * self.target_freq)
        
        if method == 'median':
            return series.rolling(window_samples, center=True).median()
        elif method == 'mean':
            return series.rolling(window_samples, center=True).mean()
        elif method == 'gaussian':
            # Use scipy gaussian filter
            sigma = window_samples / 6  # 3-sigma window
            return pd.Series(
                signal.gaussian_filter1d(series.values, sigma, mode='nearest'),
                index=series.index
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def compute_attitude_errors(self, attitude_df: pd.DataFrame, setpoint_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute attitude tracking errors in degrees
        
        Args:
            attitude_df: DataFrame with attitude data (roll, pitch, yaw in radians)
            setpoint_df: DataFrame with attitude setpoints (roll_sp, pitch_sp, yaw_sp)
            
        Returns:
            DataFrame with error metrics
        """
        # Resample both to same frequency
        att_resampled = self.resample_to_frequency(attitude_df)
        sp_resampled = self.resample_to_frequency(setpoint_df)
        
        # Merge on timestamp
        merged = pd.merge_asof(att_resampled.sort_values('timestamp'), 
                              sp_resampled.sort_values('timestamp'), 
                              on='timestamp', suffixes=('', '_sp'))
        
        # Convert to degrees and compute errors
        errors = {}
        errors['timestamp'] = merged['timestamp']
        
        if 'roll' in merged.columns and 'roll_sp' in merged.columns:
            errors['e_roll'] = np.degrees(merged['roll'] - merged['roll_sp'])
        if 'pitch' in merged.columns and 'pitch_sp' in merged.columns:
            errors['e_pitch'] = np.degrees(merged['pitch'] - merged['pitch_sp'])
        if 'yaw' in merged.columns and 'yaw_sp' in merged.columns:
            # Handle yaw wrapping
            yaw_error = merged['yaw'] - merged['yaw_sp']
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            errors['e_yaw'] = np.degrees(yaw_error)
            
        # Compute tilt error (combined roll/pitch)
        if 'e_roll' in errors and 'e_pitch' in errors:
            errors['tilt_err'] = np.sqrt(errors['e_roll']**2 + errors['e_pitch']**2)
            
        return pd.DataFrame(errors)
    
    def compute_psd(self, signal_data: np.ndarray, fs: float, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch method
        
        Args:
            signal_data: Input signal
            fs: Sampling frequency
            nperseg: Length of each segment for Welch method
            
        Returns:
            Tuple of (frequencies, power_spectral_density)
        """
        if nperseg is None:
            nperseg = min(len(signal_data) // 8, int(5 * fs))  # 5 second segments
            
        frequencies, psd = signal.welch(
            signal_data, 
            fs=fs, 
            nperseg=nperseg,
            noverlap=nperseg//2,
            detrend='linear'
        )
        
        return frequencies, psd
    
    def detect_bursts(self, condition: pd.Series, min_duration_sec: float) -> List[Tuple[float, float]]:
        """
        Detect continuous bursts where condition is True for minimum duration
        
        Args:
            condition: Boolean series indicating condition
            min_duration_sec: Minimum duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples
        """
        min_samples = int(min_duration_sec * self.target_freq)
        
        # Find continuous regions where condition is True
        condition_diff = condition.astype(int).diff()
        starts = condition.index[condition_diff == 1]
        ends = condition.index[condition_diff == -1]
        
        # Handle edge cases
        if len(starts) == 0:
            return []
        
        if condition.iloc[0]:
            starts = [condition.index[0]] + starts.tolist()
        if condition.iloc[-1]:
            ends = ends.tolist() + [condition.index[-1]]
            
        # Filter by minimum duration
        bursts = []
        for start, end in zip(starts, ends):
            if end - start >= min_samples:
                # Convert to timestamps if available
                if hasattr(condition, 'timestamp'):
                    start_time = condition.timestamp.iloc[start]
                    end_time = condition.timestamp.iloc[end]
                else:
                    start_time = start / self.target_freq
                    end_time = end / self.target_freq
                bursts.append((start_time, end_time))
                
        return bursts
    
    def merge_nearby_bursts(self, bursts: List[Tuple[float, float]], 
                           max_gap_sec: float = 1.0) -> List[Tuple[float, float]]:
        """
        Merge bursts that are separated by less than max_gap_sec
        
        Args:
            bursts: List of (start_time, end_time) tuples
            max_gap_sec: Maximum gap to merge
            
        Returns:
            List of merged bursts
        """
        if len(bursts) <= 1:
            return bursts
            
        merged = []
        current_start, current_end = bursts[0]
        
        for start, end in bursts[1:]:
            if start - current_end <= max_gap_sec:
                # Merge with current burst
                current_end = end
            else:
                # Start new burst
                merged.append((current_start, current_end))
                current_start, current_end = start, end
                
        merged.append((current_start, current_end))
        return merged