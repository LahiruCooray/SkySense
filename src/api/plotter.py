"""
Insight plotter for generating focused visualizations
Creates plots for each insight type with relevant signals
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class InsightPlotter:
    """Generates focused plots for insights"""
    
    def __init__(self, plots_dir: str = "data/plots"):
        """
        Initialize plotter
        
        Args:
            plots_dir: Directory to save generated plots
        """
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Default plot settings
        self.figsize = (12, 8)
        self.dpi = 100
        
    def generate_plot(self, insight_data: Dict) -> str:
        """
        Generate plot for insight and return file path
        
        Args:
            insight_data: Insight data dictionary
            
        Returns:
            Path to generated plot file
        """
        insight_id = insight_data['id']
        insight_type = insight_data['type']
        
        # Check if plot already exists
        plot_file = self.plots_dir / f"{insight_id}.png"
        if plot_file.exists():
            return str(plot_file)
        
        # Generate plot based on type
        try:
            if insight_type == 'tracking_error':
                fig = self._plot_tracking_error(insight_data)
            elif insight_type == 'rate_saturation':
                fig = self._plot_saturation(insight_data)
            elif insight_type == 'motor_dropout':
                fig = self._plot_motor_dropout(insight_data)
            elif insight_type == 'battery_sag':
                fig = self._plot_battery_sag(insight_data)
            elif insight_type == 'ekf_spike':
                fig = self._plot_ekf_spike(insight_data)
            elif insight_type == 'vibration_peak':
                fig = self._plot_vibration(insight_data)
            elif insight_type == 'phase':
                fig = self._plot_phase(insight_data)
            elif insight_type == 'timeline':
                fig = self._plot_timeline(insight_data)
            else:
                fig = self._plot_generic(insight_data)
            
            # Save plot
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            return str(plot_file)
            
        except Exception as e:
            print(f"Error generating plot for {insight_id}: {e}")
            # Generate a simple error plot
            fig = self._plot_error(insight_data, str(e))
            fig.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            return str(plot_file)
    
    def _create_time_window(self, t_start: float, t_end: float, 
                           buffer_sec: float = 1.0) -> Tuple[float, float]:
        """Create time window with buffer around insight"""
        duration = t_end - t_start
        buffer = max(buffer_sec, duration * 0.1)  # At least 10% buffer
        return t_start - buffer, t_end + buffer
    
    def _plot_tracking_error(self, insight_data: Dict) -> plt.Figure:
        """Plot attitude tracking error signals"""
        
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        plot_start, plot_end = self._create_time_window(t_start, t_end)
        
        # Generate synthetic data for demo (replace with actual data loading)
        times = np.linspace(plot_start, plot_end, 1000)
        roll_err = 2 * np.sin(0.5 * times) + np.random.normal(0, 0.5, len(times))
        pitch_err = 1.5 * np.cos(0.3 * times) + np.random.normal(0, 0.3, len(times))
        tilt_err = np.sqrt(roll_err**2 + pitch_err**2)
        
        # Roll error
        axes[0].plot(times, roll_err, 'b-', label='Roll Error', linewidth=1)
        axes[0].axhspan(-5, 5, alpha=0.2, color='green', label='Normal')
        axes[0].axhspan(-8, -5, alpha=0.3, color='orange')
        axes[0].axhspan(5, 8, alpha=0.3, color='orange', label='Warning')
        axes[0].axhspan(-15, -8, alpha=0.4, color='red')
        axes[0].axhspan(8, 15, alpha=0.4, color='red', label='Critical')
        axes[0].set_ylabel('Roll Error (°)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Pitch error
        axes[1].plot(times, pitch_err, 'g-', label='Pitch Error', linewidth=1)
        axes[1].axhspan(-5, 5, alpha=0.2, color='green')
        axes[1].axhspan(-8, -5, alpha=0.3, color='orange')
        axes[1].axhspan(5, 8, alpha=0.3, color='orange')
        axes[1].axhspan(-15, -8, alpha=0.4, color='red')
        axes[1].axhspan(8, 15, alpha=0.4, color='red')
        axes[1].set_ylabel('Pitch Error (°)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Combined tilt error
        axes[2].plot(times, tilt_err, 'r-', label='Tilt Error', linewidth=2)
        axes[2].axhline(5, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        axes[2].axhline(8, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        axes[2].set_ylabel('Tilt Error (°)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Highlight insight period
        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.3, color='yellow', label='Insight Period')
        
        # Add title
        rms_deg = insight_data.get('metrics', {}).get('rms_deg', 0)
        max_deg = insight_data.get('metrics', {}).get('max_deg', 0)
        plt.suptitle(f'Attitude Tracking Error\nRMS: {rms_deg:.1f}°, Max: {max_deg:.1f}°', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_saturation(self, insight_data: Dict) -> plt.Figure:
        """Plot rate controller saturation"""
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        plot_start, plot_end = self._create_time_window(t_start, t_end)
        
        # Generate synthetic data
        times = np.linspace(plot_start, plot_end, 1000)
        saturation = 0.3 + 0.4 * np.exp(-((times - (t_start + t_end)/2)**2) / 2) + np.random.normal(0, 0.05, len(times))
        saturation = np.clip(saturation, 0, 1)
        thrust = 0.6 + 0.2 * np.sin(0.1 * times) + np.random.normal(0, 0.05, len(times))
        thrust = np.clip(thrust, 0, 1)
        
        # Saturation
        axes[0].plot(times, saturation, 'r-', linewidth=2, label='Saturation')
        axes[0].axhline(0.4, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        axes[0].axhline(0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        axes[0].set_ylabel('Saturation Ratio')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True)
        
        # Thrust
        axes[1].plot(times, thrust, 'b-', linewidth=2, label='Thrust %')
        axes[1].set_ylabel('Thrust %')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True)
        
        # Highlight insight period
        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.3, color='yellow', label='Saturation Period')
        
        # Add title
        mean_sat = insight_data.get('metrics', {}).get('mean_sat', 0)
        max_sat = insight_data.get('metrics', {}).get('max_sat', 0)
        plt.suptitle(f'Rate Controller Saturation\nMean: {mean_sat:.2f}, Max: {max_sat:.2f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_motor_dropout(self, insight_data: Dict) -> plt.Figure:
        """Plot motor dropout event"""
        
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        plot_start, plot_end = self._create_time_window(t_start, t_end, 2.0)
        
        # Generate synthetic data
        times = np.linspace(plot_start, plot_end, 1000)
        motor_count = 4
        motor_index = insight_data.get('motor_index', 0)
        
        # Motor PWM outputs
        motor_pwms = []
        for i in range(motor_count):
            if i == motor_index:
                # Dropped motor
                pwm = 0.7 * np.ones_like(times)
                dropout_mask = (times >= t_start) & (times <= t_end)
                pwm[dropout_mask] = 0.1  # Dropout
            else:
                # Normal motors
                pwm = 0.7 + 0.1 * np.sin(0.2 * times + i) + np.random.normal(0, 0.02, len(times))
                pwm = np.clip(pwm, 0.1, 1.0)
            motor_pwms.append(pwm)
        
        # Plot motor PWMs
        for i, pwm in enumerate(motor_pwms):
            color = 'red' if i == motor_index else 'blue'
            linewidth = 3 if i == motor_index else 1
            axes[0].plot(times, pwm, color=color, linewidth=linewidth, 
                        label=f'Motor {i}' + (' (DROPOUT)' if i == motor_index else ''))
        
        axes[0].axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Dropout Threshold')
        axes[0].set_ylabel('Motor PWM (normalized)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Battery voltage
        voltage = 14.8 - 0.5 * np.exp(-((times - (t_start + t_end)/2)**2) / 0.5) + np.random.normal(0, 0.1, len(times))
        axes[1].plot(times, voltage, 'g-', linewidth=2, label='Battery Voltage')
        axes[1].set_ylabel('Voltage (V)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Thrust command
        thrust = 0.6 + 0.2 * (times > t_start) + np.random.normal(0, 0.05, len(times))
        thrust = np.clip(thrust, 0, 1)
        axes[2].plot(times, thrust, 'm-', linewidth=2, label='Thrust Command')
        axes[2].set_ylabel('Thrust %')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Highlight dropout period
        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.3, color='red', label='Dropout Period')
        
        # Add title
        dV = insight_data.get('metrics', {}).get('dV', 0)
        plt.suptitle(f'Motor {motor_index} Dropout\nVoltage Drop: {dV:.1f}V', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_battery_sag(self, insight_data: Dict) -> plt.Figure:
        """Plot battery voltage sag"""
        
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        plot_start, plot_end = self._create_time_window(t_start, t_end)
        
        # Generate synthetic data
        times = np.linspace(plot_start, plot_end, 1000)
        
        # Voltage with sag
        baseline_voltage = 14.8
        voltage = baseline_voltage + np.random.normal(0, 0.1, len(times))
        sag_mask = (times >= t_start) & (times <= t_end)
        voltage[sag_mask] -= 1.2  # Voltage sag
        
        axes[0].plot(times, voltage, 'g-', linewidth=2, label='Battery Voltage')
        axes[0].axhline(baseline_voltage, color='blue', linestyle='--', alpha=0.7, label='Baseline')
        axes[0].set_ylabel('Voltage (V)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Current
        current = 15 + 10 * (times >= t_start) * (times <= t_end) + np.random.normal(0, 1, len(times))
        current = np.clip(current, 0, 50)
        axes[1].plot(times, current, 'orange', linewidth=2, label='Current')
        axes[1].set_ylabel('Current (A)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Thrust
        thrust = 0.4 + 0.3 * (times >= t_start) * (times <= t_end) + np.random.normal(0, 0.05, len(times))
        thrust = np.clip(thrust, 0, 1)
        axes[2].plot(times, thrust, 'purple', linewidth=2, label='Thrust Command')
        axes[2].axhline(0.6, color='red', linestyle='--', alpha=0.7, label='High Thrust Threshold')
        axes[2].set_ylabel('Thrust %')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Highlight sag period
        for ax in axes:
            ax.axvspan(t_start, t_end, alpha=0.3, color='yellow', label='Sag Period')
        
        # Add title
        dv_min = insight_data.get('metrics', {}).get('dv_min', 0)
        thrust_max = insight_data.get('metrics', {}).get('thrust_max', 0)
        plt.suptitle(f'Battery Voltage Sag\nDrop: {dv_min:.1f}V, Max Thrust: {thrust_max:.1%}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_ekf_spike(self, insight_data: Dict) -> plt.Figure:
        """Plot EKF innovation spike"""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        plot_start, plot_end = self._create_time_window(t_start, t_end)
        
        # Generate synthetic data
        times = np.linspace(plot_start, plot_end, 1000)
        
        # Different innovation channels
        vel_ratio = 1.0 + 2.0 * np.exp(-((times - (t_start + t_end)/2)**2) / 1.0) + np.random.normal(0, 0.1, len(times))
        pos_ratio = 0.8 + 1.5 * np.exp(-((times - (t_start + t_end)/2)**2) / 0.8) + np.random.normal(0, 0.1, len(times))
        mag_ratio = 0.5 + 0.5 * np.random.normal(0, 0.2, len(times))
        
        ax.plot(times, vel_ratio, 'b-', linewidth=2, label='Velocity')
        ax.plot(times, pos_ratio, 'g-', linewidth=2, label='Position') 
        ax.plot(times, mag_ratio, 'm-', linewidth=1, label='Magnetometer')
        
        ax.axhline(2.5, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(4.0, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        ax.set_ylabel('Innovation Test Ratio')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
        
        # Highlight spike period
        ax.axvspan(t_start, t_end, alpha=0.3, color='yellow', label='Spike Period')
        
        # Add title
        tst_max = insight_data.get('metrics', {}).get('tst_max', 0)
        channels = insight_data.get('metrics', {}).get('channels', [])
        channel_str = ','.join(channels) if channels else 'unknown'
        plt.title(f'EKF Innovation Spike\nMax Ratio: {tst_max:.1f}, Channels: {channel_str}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_vibration(self, insight_data: Dict) -> plt.Figure:
        """Plot vibration peak in frequency domain"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Generate synthetic PSD data
        freqs = np.linspace(10, 300, 1000)
        peak_freq = insight_data.get('metrics', {}).get('peak_hz', 120)
        peak_db = insight_data.get('metrics', {}).get('peak_db', 15)
        
        # Baseline noise + peak
        baseline = -20 - 0.02 * freqs + np.random.normal(0, 1, len(freqs))
        
        # Add peak
        peak_width = 10
        peak_mask = np.abs(freqs - peak_freq) < peak_width
        peak_signal = peak_db * np.exp(-((freqs - peak_freq)**2) / (2 * (peak_width/3)**2))
        psd_db = baseline + peak_signal
        
        # Plot PSD
        ax1.plot(freqs, psd_db, 'b-', linewidth=1, label='Gyro X PSD')
        ax1.axvline(peak_freq, color='red', linestyle='--', linewidth=2, 
                   label=f'Peak @ {peak_freq:.0f} Hz')
        ax1.axvspan(40, 250, alpha=0.2, color='green', label='Analysis Band')
        ax1.set_ylabel('PSD (dB)')
        ax1.set_xlim(10, 300)
        ax1.legend()
        ax1.grid(True)
        
        # Generate synthetic time series showing vibration
        times = np.linspace(0, 5, 2000)
        signal = (0.5 * np.sin(2 * np.pi * peak_freq * times) + 
                 0.1 * np.sin(2 * np.pi * 25 * times) +  # Motor frequency
                 np.random.normal(0, 0.1, len(times)))
        
        ax2.plot(times, signal, 'g-', linewidth=1, label='Gyro X (rad/s)')
        ax2.set_ylabel('Angular Rate (rad/s)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True)
        
        # Add title
        plt.suptitle(f'Vibration Peak Detection\n{peak_freq:.0f} Hz, {peak_db:.0f} dB above baseline', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_phase(self, insight_data: Dict) -> plt.Figure:
        """Plot flight phase"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        phase = insight_data.get('phase', 'Unknown')
        
        # Simple phase bar
        ax.barh(0, t_end - t_start, left=t_start, height=0.5, 
               label=phase, alpha=0.7)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time (s)')
        ax.set_yticks([])
        ax.legend()
        ax.grid(True, axis='x')
        
        plt.title(f'Flight Phase: {phase}\nDuration: {t_end - t_start:.1f}s', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_timeline(self, insight_data: Dict) -> plt.Figure:
        """Plot flight timeline"""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        entries = insight_data.get('entries', [])
        failsafes = insight_data.get('failsafes', [])
        
        # Plot mode changes
        for i, entry in enumerate(entries):
            ax.scatter(entry['t'], 1, s=100, c='blue', marker='o', alpha=0.7)
            ax.annotate(entry['event'], (entry['t'], 1), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, rotation=45)
        
        # Plot failsafes
        for i, failsafe in enumerate(failsafes):
            ax.scatter(failsafe['t'], 0, s=100, c='red', marker='x', alpha=0.7)
            ax.annotate(failsafe['flag'], (failsafe['t'], 0),
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=8, rotation=45)
        
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('Time (s)')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Failsafes', 'Mode Changes'])
        ax.grid(True, axis='x')
        
        plt.title(f'Flight Timeline\n{len(entries)} mode changes, {len(failsafes)} failsafes', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_generic(self, insight_data: Dict) -> plt.Figure:
        """Generic plot for unknown insight types"""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        t_start = insight_data['t_start']
        t_end = insight_data['t_end']
        
        ax.axvspan(t_start, t_end, alpha=0.3, color='blue', label='Insight Period')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal')
        ax.legend()
        ax.grid(True)
        
        plt.title(f'Insight: {insight_data.get("type", "Unknown")}\n{insight_data.get("text", "")}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_error(self, insight_data: Dict, error_msg: str) -> plt.Figure:
        """Plot showing error message"""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        ax.text(0.5, 0.5, f'Plot generation failed:\n{error_msg}', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.title(f'Error: {insight_data.get("id", "Unknown")}', 
                 fontsize=14, fontweight='bold', color='red')
        
        return fig