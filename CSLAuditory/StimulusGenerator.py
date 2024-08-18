import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


class StimulusGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_pure_tones(frequency, duration, sample_rate, amplitude=0.5, fade_duration=0.05):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        acoustic_signal = np.zeros_like(t)

        pure_tone = amplitude * np.sin(2 * np.pi * frequency * t)
        acoustic_signal += pure_tone

        acoustic_signal = StimulusGenerator.apply_fade(acoustic_signal, sample_rate, fade_duration)

        # Normalize the signal
        acoustic_signal = acoustic_signal / np.max(np.abs(acoustic_signal))

        return StimulusGenerator.zero_pad(acoustic_signal, int(duration * sample_rate))


    @staticmethod
    def generate_harmonic_series(fundamental_freq, stimulus_duration, signal_duration, sample_rate, num_harmonics=5,
                                 amplitude=0.5,
                                 fade_duration=0.05):
        t = np.linspace(0, stimulus_duration, int(sample_rate * stimulus_duration), endpoint=False)
        acoustic_signal = np.zeros_like(t)

        for n in range(1, num_harmonics + 1):
            harmonic = amplitude / n * np.sin(2 * np.pi * fundamental_freq * n * t)
            acoustic_signal += harmonic

        acoustic_signal = StimulusGenerator.apply_fade(acoustic_signal, sample_rate, fade_duration)

        # Normalize the signal
        acoustic_signal = acoustic_signal / np.max(np.abs(acoustic_signal))

        return StimulusGenerator.zero_pad(acoustic_signal, int(signal_duration * sample_rate))


    @staticmethod
    def generate_white_noise(stimulus_duration, signal_duration, sample_rate, fade_duration=0.05):
        t = np.linspace(0, stimulus_duration, int(sample_rate * stimulus_duration), endpoint=False)
        acoustic_signal = np.random.normal(0, 1, len(t))

        acoustic_signal = StimulusGenerator.apply_fade(acoustic_signal, sample_rate, fade_duration)

        # Normalize the signal
        acoustic_signal = acoustic_signal / np.max(np.abs(acoustic_signal))

        return StimulusGenerator.zero_pad(acoustic_signal, int(signal_duration * sample_rate))


    @staticmethod
    def generate_tremolo_signal(carrier_freq, modulator_freq, modulation_index, duration, sample_rate):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        tremolo_signal = (1 + modulation_index * modulator) * carrier
        tremolo_signal = tremolo_signal / np.max(np.abs(tremolo_signal))
        return tremolo_signal


    @staticmethod
    def generate_vibrato_signal(carrier_freq, modulator_freq, modulation_index, duration, sample_rate):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        vibrato_signal = np.sin(2 * np.pi * (carrier_freq + modulation_index * modulator) * t)
        vibrato_signal = vibrato_signal / np.max(np.abs(vibrato_signal))
        return vibrato_signal


    @staticmethod
    def generate_frequency_sweep_signal(start_freq, end_freq, duration, sample_rate):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sweep_signal = np.sin(2 * np.pi * ((start_freq + (end_freq - start_freq) * t / duration)) * t)
        sweep_signal = sweep_signal / np.max(np.abs(sweep_signal))
        return sweep_signal


    @staticmethod
    def zero_pad(signal, target_length):
        current_length = len(signal)

        if current_length >= target_length:
            raise ValueError("Target length must be greater than the current signal length.")

        # Calculate padding on each side
        total_padding = target_length - current_length
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        # Apply zero padding
        padded_signal = np.pad(signal, (pad_left, pad_right), 'constant', constant_values=(0, 0))

        return padded_signal


    @staticmethod
    def apply_fade(acoustic_signal, sample_rate, fade_duration):
        fade_in_samples = int(sample_rate * fade_duration)
        fade_out_samples = int(sample_rate * fade_duration)

        if fade_duration > 0:
            # Create the fade-in and fade-out envelopes
            fade_in = np.linspace(0, 1, fade_in_samples)
            fade_out = np.linspace(1, 0, fade_out_samples)

            acoustic_signal[:fade_in_samples] *= fade_in
            acoustic_signal[-fade_out_samples:] *= fade_out
        return acoustic_signal


    @staticmethod
    def get_spectrum(signal, sample_rate):
        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(spectrum), 1 / sample_rate)
        return freqs, spectrum


    @staticmethod
    def get_spectrogram(signal, sample_rate, window_duration_s=0.023, plot=False):
        n_fft = StimulusGenerator._next_power_of_2(int(window_duration_s * sample_rate))
        hop_length = n_fft // 4
        spectrogram = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

        if plot:
            librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate,
                                     hop_length=hop_length, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.show()

        return spectrogram


    @staticmethod
    def reverse_spectrogram(spectrogram, sample_rate, window_duration_s=0.023):
        n_fft = StimulusGenerator._next_power_of_2(int(window_duration_s * sample_rate))
        hop_length = n_fft // 4
        sig_reconstructed = librosa.griffinlim(spectrogram, hop_length=hop_length, n_fft=n_fft)
        return sig_reconstructed


    @staticmethod
    def save_signal_to_wav(signal, sample_rate, filename):
        sf.write(filename, signal, sample_rate)


    @staticmethod
    def _next_power_of_2(x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()


if __name__ == "__main__":
    sr = 44100
    stimulus_type = "frequency_sweep"

    if stimulus_type == "harmonic_series":
        sig = StimulusGenerator.generate_harmonic_series(440.0, 0.2, 5, sr, 5, 0.5, 0.05)
    elif stimulus_type == "white_noise":
        sig = StimulusGenerator.generate_white_noise(0.2, 5, sr, 0.05)
    elif stimulus_type == "tremolo":
        sig = StimulusGenerator.generate_tremolo_signal(440.0, 5.0, 0.5, 5, sr)
    elif stimulus_type == "vibrato":
        sig = StimulusGenerator.generate_vibrato_signal(440.0, 5.0, 0.5, 5, sr)
    elif stimulus_type == "frequency_sweep":
        sig = StimulusGenerator.generate_frequency_sweep_signal(440.0, 880.0, 5, sr)
    # sig = StimulusGenerator.generate_harmonic_series(440.0, 0.2, 5, sr, 5, 0.5, 0.05)

    spectrogram = StimulusGenerator.get_spectrogram(sig, 44100, plot=True)
    r_signal = StimulusGenerator.reverse_spectrogram(spectrogram, 44100)
    StimulusGenerator.save_signal_to_wav(r_signal, 44100, f"output/{stimulus_type}_reconstructed.wav")
    # plt.plot(r_signal)

    plt.plot(sig)
    # plt.plot(r_signal)
    # plt.show()
    StimulusGenerator.save_signal_to_wav(sig, 44100, f"output/{stimulus_type}.wav")

    spectrum = StimulusGenerator.get_spectrum(sig, 44100)
    plt.plot(spectrum[0], spectrum[1])
    plt.show()
