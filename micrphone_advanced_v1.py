import pyaudio
import serial
import numpy as np
import matplotlib.pyplot as plt
## define bins
bass_start =20
bass = 500
vocals = 4000
ser = serial.Serial('COM8', 9600)
if not ser.isOpen():
    ser.open()

## so basically when the base activity is higher than the vocals, then the led should be the brightness
## find out activity in bass and activity in vocals and if it bass is less than 50% of vocals, then led should be less bright and so on


# Configure audio input settings
audio_format = pyaudio.paInt16  # Audio format (16-bit)
channels = 1  # Mono
sample_rate = 44000  # Sample rate (Hz)
chunk_size = 1024  # Number of audio frames per buffer

# Create a PyAudio object
audio = pyaudio.PyAudio()

# Open an audio stream
stream = audio.open(
    format=audio_format,
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size
)

# Create a matplotlib figure and axis for live plotting
###plt.ion()
#fig= plt.figure()
#ax1=fig.add_subplot(121)
#ax2=fig.add_subplot(122)


#fig, ax = plt.subplots()
#x_data = np.arange(0, chunk_size)
#line, = ax1.plot(x_data, np.random.rand(chunk_size))
#ax1.set_ylim(0, 255)
#ax2.set_ylim(0, 255)
#audio_data = stream.read(chunk_size)
#audio_array = np.frombuffer(audio_data, dtype=np.int16)
#fft_result = np.fft.fft(audio_array)
#fft_magnitude = np.abs(fft_result)
#fft_freq = np.fft.fftfreq(len(fft_result), 1.0 / sample_rate)
#line2, = ax2.plot(fft_freq, fft_magnitude)
# Continuously read and plot audio data


count=0
try:
    while True:
        count=count+1
        audio_data = stream.read(chunk_size)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        fft_result = np.fft.fft(audio_array)
        fft_magnitude = np.abs(fft_result)
        fft_freq = np.fft.fftfreq(len(fft_result), 1.0 / sample_rate)
        bass_indices = np.where((fft_freq >bass_start ) &  (fft_freq  < bass ) )

        bass_intensity = np.sum(np.abs(fft_result[bass_indices]))

        vocals_indices = np.where((fft_freq >bass ) &  (fft_freq  < 1600 ) )

        vocal_intensity = np.sum(np.abs(fft_result[vocals_indices]))

        #print((bass_intensity/vocal_intensity)*100)
        if(((bass_intensity/vocal_intensity)*100) > 360):
            ser.write(b'1')
            count=0
        else:
            ser.write(b'0')

            '''
        line.set_ydata(audio_array)
        line2.set_ydata(fft_magnitude)
        line2.set_xdata(fft_freq)
        fig.canvas.flush_events()'''

except KeyboardInterrupt:
    pass

# Close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()
ser.close()