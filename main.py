import numpy as np
import pyaudio
import scipy.signal
import time


class VoiceRangeDetector:
    def __init__(self):
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.MIN_FREQUENCY = 50
        self.MAX_FREQUENCY = 1000

        self.p = pyaudio.PyAudio()
        self.min_pitch = float('inf')
        self.max_pitch = 0

    def get_pitch(self, data):
        # Convert bytes to numpy array
        signal = np.frombuffer(data, dtype=np.float32)

        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window

        # Compute FFT
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft), 1.0 / self.RATE)

        # Get magnitude spectrum
        magnitude = np.abs(fft)

        # Find peak frequency
        peak_freq_idx = np.argmax(magnitude[:len(magnitude) // 2])
        peak_frequency = frequencies[peak_freq_idx]

        # Only return frequency if magnitude is significant
        if magnitude[peak_freq_idx] > 1 and self.MIN_FREQUENCY <= peak_frequency <= self.MAX_FREQUENCY:
            return abs(peak_frequency)
        return 0

    def determine_voice_type(self):
        # Frequency ranges for male voice types (in Hz)
        ranges = {
            'Bass': (80, 330),
            'Baritone': (100, 400),
            'Tenor': (130, 500)
        }

        possible_types = []
        for voice_type, (min_freq, max_freq) in ranges.items():
            if self.min_pitch >= min_freq and self.max_pitch <= max_freq:
                possible_types.append(voice_type)

        return ' or '.join(possible_types) if possible_types else 'Unknown'

    def run(self):
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        print("Starting voice range detection... (Press Ctrl+C to stop)")
        print("Start singing different notes from low to high...")

        try:
            while True:
                data = stream.read(self.CHUNK)
                pitch = self.get_pitch(data)

                if pitch > 0:
                    self.min_pitch = min(self.min_pitch, pitch)
                    self.max_pitch = max(self.max_pitch, pitch)

                    print(f"\rCurrent pitch: {pitch:.1f} Hz | Range: {self.min_pitch:.1f} - {self.max_pitch:.1f} Hz",
                          end="")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nAnalysis complete!")
            print(f"Your vocal range: {self.min_pitch:.1f} Hz - {self.max_pitch:.1f} Hz")
            print(f"Possible voice type: {self.determine_voice_type()}")

        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()


if __name__ == "__main__":
    detector = VoiceRangeDetector()
    detector.run()