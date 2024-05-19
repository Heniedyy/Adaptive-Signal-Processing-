import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def lms_filter(y_noisy, mu):
    N = len(y_noisy)
    y_filtered = np.zeros(N)
    w = 0.1  # Initial filter weight
    error_signal = np.zeros(N)  # Initialize error signal
    for n in range(1, N):
        x = y_noisy[n - 1]  # Input for the filter
        d = y_noisy[n]  # Desired output
        y = w * x  # Filter output
        e = d - y  # Error signal
        error_signal[n] = e  # Store error signal
        w = w + mu * e * x  # Update filter weights
        y_filtered[n] = w * x  # Use updated filter to get filtered output
    return y_filtered, error_signal

def nlms_filter(y_noisy, mu):
    N = len(y_noisy)
    y_filtered = np.zeros(N)
    w = 0  # Initial filter weight
    epsilon = 1e-6  # Small constant to avoid division by zero
    error_signal = np.zeros(N)  # Initialize error signal
    for n in range(1, N):
        norm_factor = np.linalg.norm(y_noisy[n - 1]) ** 2 + epsilon  # Norm of input signal
        y_hat = w * y_noisy[n - 1]
        e = y_noisy[n] - y_hat
        error_signal[n] = e  # Store error signal
        w = w + (mu / norm_factor) * e * y_noisy[n - 1]
        y_filtered[n] = w * y_noisy[n - 1]  # Apply filter to get filtered output
    return y_filtered, error_signal

import numpy as np

def rls_filter(y_noisy, lambda_, delta=1e-6, overflow_threshold=1e10):
    N = len(y_noisy)
    y_filtered = np.zeros(N)
    w = np.zeros((1, 1))  # Filter weights as a column vector
    P = np.eye(1) / delta  # P is a 1x1 matrix
    error_signal = np.zeros(N)  # Initializing the error signal array

    for n in range(1, N):
        x = np.array([[y_noisy[n - 1]]])  # Ensure x is a column vector
        if not np.isfinite(x).all() or not np.isfinite(P).all():
            continue

        k_numerator = P @ x
        k_denominator = lambda_ + (x.T @ P @ x).item()  # Ensuring scalar
        if np.abs(k_numerator).max() > overflow_threshold or np.abs(k_denominator) > overflow_threshold:
            continue

        k = k_numerator / k_denominator  # Ensure k remains a column vector
        e = y_noisy[n] - (w.T @ x).item()  # Error calculation
        error_signal[n] = e
        w += k * e

        P_update = P - (k @ x.T @ P)
        if np.linalg.norm(P_update, ord=2) > overflow_threshold:
            continue

        P = (1 / lambda_) * P_update
        y_filtered[n] = (w.T @ x).item()  # Using .item() to extract scalar from array

    return y_filtered, error_signal


def hybrid_filter(y_noisy, step_size_nlms, forgetting_factor_rls, mix_param, switch_threshold):
    N = len(y_noisy)
    y_filtered = np.zeros(N)
    order = 2
    w_rls = np.zeros(order)
    w_nlms = np.zeros(order)
    P = np.eye(order) * 1000
    use_rls = True
    error_signal = np.zeros(N)

    for n in range(order, N):
        u = y_noisy[n - order:n]

        if use_rls:
            Pi_u = P @ u
            k = Pi_u / (forgetting_factor_rls + u.T @ Pi_u)
            e_rls = y_noisy[n] - w_rls @ u
            error_signal[n] = e_rls
            if np.abs(e_rls) < switch_threshold:
                use_rls = False
            else:
                w_rls += k * e_rls
                P_update = P - (k[:, None] @ Pi_u[None, :])
                P = (P_update / forgetting_factor_rls)
            y_filtered[n] = w_rls @ u
        else:
            norm_factor = np.dot(u, u) + np.finfo(float).eps
            e_nlms = y_noisy[n] - w_nlms @ u
            error_signal[n] = e_nlms
            w_nlms += (step_size_nlms / norm_factor) * e_nlms * u
            y_filtered[n] = w_nlms @ u

    return y_filtered, error_signal

def extract_mel_spectrogram(audio_data, sample_rate, n_mels=64, fmax=8000):
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_audio_directory(audio_dir, output_dir, n_mels=64, sample_rate=22050, fmax=8000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(audio_dir, filename)
            print(f"Processing {audio_path}")

            # Load audio
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Apply filters
            y_lms, _ = lms_filter(y, mu=0.3)
            y_nlms, _ = nlms_filter(y, mu=0.3)
            y_rls, _ = rls_filter(y, lambda_=0.3)
            y_hybrid, _ = hybrid_filter(y, step_size_nlms=0.3, forgetting_factor_rls=0.99,
                                        mix_param=0.5, switch_threshold=0.01)

            # Extract Mel spectrograms
            
            S_dB_lms = extract_mel_spectrogram(y_lms, sr, n_mels, fmax)
            S_dB_nlms = extract_mel_spectrogram(y_nlms, sr, n_mels, fmax)
            S_dB_rls = extract_mel_spectrogram(y_rls, sr, n_mels, fmax)
            S_dB_hybrid = extract_mel_spectrogram(y_hybrid, sr, n_mels, fmax)

            # Save Mel spectrograms
            output_filename = os.path.splitext(filename)[0]
            
            np.save(os.path.join(output_dir, f"{output_filename}_lms.npy"), S_dB_lms)
            np.save(os.path.join(output_dir, f"{output_filename}_nlms.npy"), S_dB_nlms)
            np.save(os.path.join(output_dir, f"{output_filename}_rls.npy"), S_dB_rls)
            np.save(os.path.join(output_dir, f"{output_filename}_hybrid.npy"), S_dB_hybrid)

if __name__ == "__main__":
    audio_dir = "dataset"
    output_dir = "model"
    process_audio_directory(audio_dir, output_dir)