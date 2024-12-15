import numpy as np  # Import the numpy library for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import librosa  # Import librosa for audio processing
import librosa.display  # Import librosa display utilities
import os  # Import os for interacting with the operating system

def extract_audio_features(audio_data, sample_rate):
    """
    Extracts audio features using librosa.

    Parameters:
    audio_data (numpy.ndarray): The audio data.
    sample_rate (int): The sampling rate.

    Returns:
    tuple: Fbank, fundamental frequency, spectral centroid, and spectral contrast.
    """
    # Calculate Mel-spectrogram features (Mel frequency)
    fbank_features = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    # Calculate pitch
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
    pitch_indices = np.argmax(magnitudes, axis=0)
    fundamental_frequencies = pitches[pitch_indices, np.arange(pitches.shape[1])]
    
    # Calculate fundamental period, removing invalid values
    fundamental_period = 1 / fundamental_frequencies
    fundamental_period = fundamental_period[np.isfinite(fundamental_period)]  # Remove non-finite values
    fundamental_period = fundamental_period[fundamental_period > 0]  # Keep only positive values

    # Calculate spectral centroid and spectral contrast
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    
    return fbank_features, fundamental_period, spectral_centroid, spectral_contrast

def visualize_and_save_features(fbank_features, fundamental_period, spectral_centroid, spectral_contrast, output_path):
    """
    Visualizes and saves audio features.

    Parameters:
    fbank_features (numpy.ndarray): Fbank features.
    fundamental_period (numpy.ndarray): Fundamental frequency.
    spectral_centroid (numpy.ndarray): Spectral centroid.
    spectral_contrast (numpy.ndarray): Spectral contrast.
    output_path (str): Path to save the image.
    """
    plt.figure(figsize=(60, 60))
    
    # Histogram of fundamental period
    plt.subplot(3, 1, 1)
    if fundamental_period.size > 0:  # Check if there are valid fundamental periods
        plt.hist(fundamental_period, bins=100, color='blue', alpha=0.7)
    plt.axis('off')  # Remove axis

    # Histogram of spectral centroid
    plt.subplot(3, 1, 2)
    plt.hist(spectral_centroid.flatten(), bins=100, color='green', alpha=0.7)
    plt.axis('off')  # Remove axis

    # Histogram of spectral contrast
    plt.subplot(3, 1, 3)
    plt.hist(spectral_contrast.flatten(), bins=100, color='red', alpha=0.7)
    plt.axis('off')  # Remove axis

    plt.tight_layout()
    plt.savefig(output_path, dpi=50)
    plt.close()

def process_audio_directory(input_directory, output_directory):
    """
    Processes all audio files in a directory and saves their features.

    Parameters:
    input_directory (str): Directory containing audio files.
    output_directory (str): Directory to save feature images.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.wav'):
            try:
                file_path = os.path.join(input_directory, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
                fbank_features, fundamental_period, spectral_centroid, spectral_contrast = extract_audio_features(audio_data, sample_rate)
                
                output_filename = os.path.splitext(filename)[0] + '.jpg'
                output_path = os.path.join(output_directory, output_filename)
                visualize_and_save_features(fbank_features, fundamental_period, spectral_centroid, spectral_contrast, output_path)
                
                print(f"Successfully processed and saved feature image: {output_filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = 'wav'
    output_directory = 'wav_features'
    
    process_audio_directory(input_directory, output_directory)