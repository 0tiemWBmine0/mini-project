import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Library for plotting graphs
import librosa  # Library for audio signal processing
import os  # Library for interacting with the operating system

def pre_emphasis(audio_data, coeff=0.5):
    """Apply pre-emphasis filter to the audio data."""
    return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])

def compute_lsf(audio_data, sample_rate, order=10):
    """Compute Line Spectrum Frequencies (LSF) from LPC coefficients."""
    lpc_coeffs = librosa.lpc(audio_data, order=12)  # Compute LPC coefficients
    p = np.append(1, -lpc_coeffs)  # Prepare polynomial

    if np.any(np.isinf(p)) or np.any(np.isnan(p)):  # Check for infinite or NaN values
        print("Warning: LPC polynomial contains inf or NaN values.")
        return np.array([])

    roots = np.roots(p)  # Find roots of the polynomial

    if np.any(np.isinf(roots)) or np.any(np.isnan(roots)):  # Check for infinite or NaN roots
        print("Warning: LPC roots contain inf or NaN values.")
        return np.array([])

    stable_roots = roots[np.abs(roots) < 1]  # Select roots inside the unit circle
    if len(stable_roots) == 0:
        return np.array([])  # Return empty array if no stable roots
    lsf = np.angle(stable_roots)  # Compute LSF from roots
    return np.sort(lsf)[:, np.newaxis]  # Sort LSF and add new axis

def compute_mfcc(audio_data, sample_rate, n_mfcc=1):
    """Extract MFCC features from audio data."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)  # Compute MFCC
    return mfccs

def compute_lpc(audio_data):
    """Calculate LPC coefficients and reshape for visualization."""
    lpc_coeffs = librosa.lpc(audio_data, order=12)  # Compute LPC coefficients
    return lpc_coeffs[:, np.newaxis]  # Reshape for plotting

def compute_cqcc(audio_data, sample_rate, n_bands=6, n_coeffs=6, hop_length=2048):
    """Compute Constant-Q Cepstral Coefficients (CQCC) using librosa functions."""
    # Apply Constant Q Transform (CQT)
    cqt = librosa.cqt(audio_data, sr=sample_rate, n_bins=n_bands, hop_length=hop_length)
    magnitude = np.abs(cqt)  # Compute magnitude

    # Convert to log scale
    log_magnitude = np.log(magnitude + 1e-6)  # Add small constant to avoid log(0)

    # Use librosa's mfcc function on log-magnitude spectrum to get cepstral coefficients
    cqcc = librosa.feature.mfcc(S=log_magnitude, n_mfcc=n_coeffs)

    # Normalize CQCC features to ensure the plot is visible
    cqcc -= np.min(cqcc)
    cqcc /= np.max(cqcc)  # Normalize to [0, 1]

    return cqcc

def process_audio(file_path, hop_length=256, win_length=512):
    """Process the audio file and compute various features."""
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)  # Load audio file
    pre_emphasized_audio = pre_emphasis(audio_data)  # Apply pre-emphasis

    if np.any(np.isnan(pre_emphasized_audio)) or np.any(np.isinf(pre_emphasized_audio)):  # Check for NaN or inf values
        print(f"Warning: Pre-emphasized audio contains NaN or inf values for {file_path}.")
        return None

    lsf_result = compute_lsf(pre_emphasized_audio, sample_rate)  # Compute LSF
    if lsf_result.size == 0:  # Check if LSF computation failed
        print(f"Warning: LSF computation failed for {file_path}.")
        return None

    mfccs = compute_mfcc(pre_emphasized_audio, sample_rate)  # Compute MFCC
    lpc = compute_lpc(pre_emphasized_audio)  # Compute LPC
    cqcc_features = compute_cqcc(audio_data, sample_rate)  # Compute CQCC
    return lsf_result, mfccs, lpc, cqcc_features

def save_spectrograms(lsf_result, mfccs, lpc, cqcc_features, output_path):
    """Save the computed spectrograms to a file in a 2-column format."""
    plt.figure(figsize=(80, 40), dpi=50)  # Adjusted figure size for better visibility

    # Left column
    plt.subplot(1, 2, 1)
    plt.imshow(lsf_result.T, cmap='gray', aspect='auto')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lpc, cmap='gray', aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)  # Save the plot
    plt.close()

def turn(input_directory, output_directory):
    """Process all audio files in the input directory and save results."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create output directory if it doesn't exist

    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_directory, filename)
            results = process_audio(file_path)
            if results is None:
                print(f"Skipping {filename} due to errors in processing.")
                continue

            lsf_result, mfccs, lpc, cqcc = results
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_directory, output_filename)
            save_spectrograms(lsf_result, mfccs, lpc, cqcc, output_path)
            print(f'Saved spectrograms for {filename} to {output_path}')

# Usage example for running the code
turn('wav', 'wav-feature')