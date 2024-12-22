import librosa #for audio processing
import soundfile as sf #read write audio files
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox #for gui
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import butter, lfilter

# Function to load audio file
def load_audio(audio_path):#Loads an audio file and returns the audio signal (audio) and its sampling rate (sr)
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr
    except FileNotFoundError:
        raise ValueError("Audio file not found. Please check the file path.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading the audio: {e}")

# Function to check the length of audio
def check_audio_length(audio, sr, min_length_sec=1): #Ensures the audio file is at least 1 second long otherwise error of audio being short
    duration = len(audio) / sr
    if duration < min_length_sec:
        raise ValueError(f"Audio is too short ({duration:.2f}s). Minimum length is {min_length_sec}s.")
    print(f"Audio duration: {duration:.2f}s")
    return True

# Plot the waveform of audio
def plot_waveform(audio, sr): #Plots the audio signal's waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# Compute Short-Time Fourier Transform (STFT)
def compute_stft(audio, sr, n_fft=2048, hop_length=512, window=np.hamming): # Computes the Short-Time Fourier Transform (STFT) of the audio signal, producing a spectrogram.
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window) #Allows configuration of the FFT size (n_fft), hop length, and windowing function.
    return np.abs(D)

# Plot the spectrogram
def plot_spectrogram(magnitude, sr, hop_length):#Plots a spectrogram of the audio signal in decibels for a visual representation of frequencies over time
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        magnitude_db, sr=sr, hop_length=hop_length, y_axis="log", x_axis="time", cmap="viridis"
    )
    plt.title("Magnitude Spectrogram (STFT)")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# Apply noise reduction by estimating a noise profile and performing spectral subtraction
def noise_reduction(audio, sr): #Reduces noise by subtracting a noise profile (estimated from the first second of audio) from the STFT of the audio signal.
    # Estimate a noise profile using the first few seconds
    noise_clip = audio[:sr]  # Take the first second as the noise profile
    noise_stft = compute_stft(noise_clip, sr)
    audio_stft = compute_stft(audio, sr)

    # Subtract noise profile from the audio STFT
    noise_magnitude = np.mean(noise_stft, axis=1)
    clean_stft = audio_stft - noise_magnitude[:, None]

    # Ensure no negative values
    clean_stft = np.maximum(clean_stft, 0)

    # Inverse STFT to reconstruct the clean audio
    cleaned_audio = librosa.istft(clean_stft)
    return cleaned_audio

# Apply frequency filtering (bandpass filter)
def bandpass_filter(audio, sr, low_cutoff=500, high_cutoff=3000, order=6): #Applies a bandpass filter to isolate frequencies between 500 Hz and 3000 Hz.
    nyquist = 0.5 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Design a bandpass filter
    b, a = butter(order, [low, high], btype="band")
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# Apply gain enhancement
def apply_gain(audio, gain_factor=2.0):#Enhances audio by multiplying its amplitude with a gain factor (default: 2.0).
    return audio * gain_factor

# Perform Griffin-Lim reconstruction
def griffin_lim_reconstruction(magnitude, sr, n_fft=2048, hop_length=512, iterations=32, tol=1e-6):#Reconstructs an audio signal from a modified magnitude spectrogram
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape)) 
    previous_audio = None
    for i in range(iterations):
        complex_spectrogram = magnitude * phase
        reconstructed_audio = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=n_fft)
        reconstructed_stft = librosa.stft(reconstructed_audio, n_fft=n_fft, hop_length=hop_length)
        phase = np.angle(reconstructed_stft)
        if previous_audio is not None and np.linalg.norm(reconstructed_audio - previous_audio) < tol:
            print(f"Converged at iteration {i}")
            break
        previous_audio = reconstructed_audio
    return reconstructed_audio

# Ask the user for modification choice
def choose_modification():
    print("Choose an audio modification:")
    print("1. Noise Reduction")
    print("2. Frequency Filtering")
    print("3. Gain Enhancement")
    choice = input("Enter the number of the modification you want to apply (1/2/3): ")

    if choice == "1":
        return "noise_reduction"
    elif choice == "2":
        return "frequency_filtering"
    elif choice == "3":
        return "gain_enhancement"
    else:
        print("Invalid choice. Defaulting to 'Gain Enhancement'.")
        return "gain_enhancement"

def modify_stft(magnitude, sr, modification_type, noise_clip=None, gain_factor=2.0, low_cutoff=500, high_cutoff=3000):
    if modification_type == "noise_reduction" and noise_clip is not None:
        # Compute noise profile
        noise_stft = compute_stft(noise_clip, sr)
        noise_magnitude = np.mean(noise_stft, axis=1)  # Average noise profile
        magnitude = np.maximum(magnitude - noise_magnitude[:, None], 0)

    elif modification_type == "frequency_filtering":
        # Apply bandpass filter in frequency domain
        nyquist = 0.5 * sr
        low_bin = int(low_cutoff / nyquist * magnitude.shape[0])
        high_bin = int(high_cutoff / nyquist * magnitude.shape[0])
        filtered_magnitude = np.zeros_like(magnitude)
        filtered_magnitude[low_bin:high_bin, :] = magnitude[low_bin:high_bin, :]
        magnitude = filtered_magnitude

    elif modification_type == "gain_enhancement":
        # Apply gain
        magnitude *= gain_factor

    return magnitude

# Modified process_audio function
def process_audio(file_path, modification_type):
    audio, sr = load_audio(file_path)
    check_audio_length(audio, sr)

    # Compute STFT
    n_fft = 2048
    hop_length = 512
    original_stft = compute_stft(audio, sr, n_fft=n_fft, hop_length=hop_length)

    # Optional: Extract a noise clip for noise reduction
    noise_clip = audio[:sr] if modification_type == "noise_reduction" else None

    # Apply modifications to the STFT magnitude
    modified_stft_magnitude = modify_stft(
        original_stft, sr, modification_type, noise_clip=noise_clip
    )

    # Reconstruct audio using Griffin-Lim with modified STFT magnitude
    reconstructed_audio = griffin_lim_reconstruction(
        modified_stft_magnitude, sr, n_fft=n_fft, hop_length=hop_length
    )

    # Plot original and reconstructed waveforms
    print("Original audio waveform:")
    plot_waveform(audio, sr)
    print("Reconstructed audio waveform after modification:")
    plot_waveform(reconstructed_audio, sr)

    # Plot spectrograms
    print("Original audio spectrogram:")
    plot_spectrogram(original_stft, sr, hop_length)
    print("Modified audio spectrogram:")
    plot_spectrogram(modified_stft_magnitude, sr, hop_length)

    # Save the processed and reconstructed audio
    output_file = "processed_and_reconstructed_audio.wav"
    sf.write(output_file, reconstructed_audio, sr)
    print(f"Processed and reconstructed audio saved to {output_file}")
    
    return output_file

# Upload and process audio file
def upload_and_process():
    file_path = filedialog.askopenfilename(
        title="Select an Audio File", filetypes=(("WAV Files", "*.wav"), ("All Files", "*.*"))
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    # Ask user for modification choice
    modification_type = choose_modification()

    try:
        output_file = process_audio(file_path, modification_type)
        messagebox.showinfo("Success", f"Audio processed successfully.\nProcessed file saved to: {output_file}")
    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Set up the Tkinter GUI
root = tk.Tk()
root.title("Audio File Processor")
root.geometry("300x150")

upload_button = tk.Button(root, text="Upload and Process Audio", command=upload_and_process, width=25, height=2)
upload_button.pack(pady=20)

root.mainloop()
