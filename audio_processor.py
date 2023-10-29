 
## LI8 v1


import pyaudio
import serial
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import struct
import scipy.fftpack
import argparse

# globals declarations 
'''
ser=serial.Serial('COM10', 9600)
if not ser.isOpen():
    ser.open()
'''
## defining colors red, green,yellow,orange,  lightblue, pink
colors_list=[[255,0,0],[0,255,0],[255,255,0],[255,165,0],[96,205,246],[238,71,153]]
IMIN=0
IMAX=3500
OMAX=100
OMIN=25

max_brightness =100
average_brightness=0
previous_non_zero_color=0
brightness_i=0
brightness_list = [max_brightness for i in range (5)]
color_index=0


## pyaudio needs me to configure some settings such as 
audio_format = pyaudio.paInt16  # Audio format (16-bit)
channels = 1  # Mono
sample_rate = 44000  # Sample rate (Hz) === ie., every 1/44100 = 22 micro seconds, one sample from the mic output is pulled out - each of these samples  = frame
chunk_size = 512  # Number of audio frames per buffer, since 1 sample = 1 audio frame ,it takes 1024*22 microseconds = 23 milliseconds 
#ie., it processes 1024 samples that are accumulated every 23 milliseconds 


## need to create a pyaudio object
audio = pyaudio.PyAudio()

class audio_characterstics():
    def __init__(self,sample_rate,chunk_size):
        self.format=pyaudio.paInt16
        self.channels=1
        self.rate=sample_rate
        self.input=True
        self.frames_per_buffer=chunk_size

def standardizer_brightness(low_freq_amp):
    #what percentage of output range ? 
    print(low_freq_amp)
    temp=((low_freq_amp-IMIN)/(IMAX-IMIN))*(OMAX-OMIN)
    print(temp)
    ##sometimes might shoot up more than 100%
    brightness= max(min(OMAX,temp),OMIN)
    return brightness


def find_average_brightness(brightness):
    ##makes sure we have the most recent brightnesses in memory 
    global brightness_i
    brightness_list[brightness_i]=brightness
    brightness_i+=1
    if(brightness_i==len(brightness_list)):
        brightness_i=0
    return int(np.mean(brightness_list))


def color_picker(average_brightness):
    ##pick randomly out of 
    global color_index
    color_pallete=colors_list[color_index]
    color_index1=(color_index+1)//5
    
    color_index=(color_index1)%len(colors_list)
    color_pallete=colors_list[color_index]
    r,g,b=color_pallete[0]*average_brightness,color_pallete[1]*average_brightness,color_pallete[2]*average_brightness


    return r,g,b
    
def analyser(sample_rate,chunk_size):

    crossover=400
    audio_obj= audio_characterstics(sample_rate,chunk_size)
    stream = audio.open(
    format= audio_obj.format,
    channels=audio_obj.channels,
    rate=audio_obj.rate,
    input=True,
    frames_per_buffer=audio_obj.frames_per_buffer
    )


    plt.ion()
    fig= plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    fig, ax = plt.subplots()
    x_data = np.arange(0, chunk_size)
    line, = ax1.plot(x_data, np.random.rand(chunk_size))
    ax1.set_ylim(0, 1000)
    ax2.set_ylim(0, 60000)
    ax2.set_xlim(0, 5000)


    audio_data = stream.read(chunk_size)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    fft_result = np.fft.fft(audio_array)
    fft_magnitude = np.abs(fft_result)
    fft_freq = np.fft.fftfreq(len(fft_result), 1.0 / sample_rate)
    line2, = ax2.plot(fft_freq, fft_magnitude)

    try:
        while True:

            data=stream.read(audio_obj.frames_per_buffer)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)


            # Calculate the FFT of the audio data
            fft_result = np.fft.fft(audio_array)
            fft_magnitude = np.abs(fft_result)
            frequencies = np.fft.fftfreq(audio_obj.frames_per_buffer, 1 /audio_obj.rate)

            #print(len(frequencies), len(fft_magnitude))
            max_magnitude = 0
            max_low_freq_mag = 0
            fundamental_freq = 0


            for i in range(1, len(frequencies)):

                mag=fft_magnitude[i]
                freq=frequencies[i]
                #find out fundamental frequency and corresponding magnitude


                if mag > max_magnitude:
                    fundamental_freq = freq
                    max_magnitude = mag


                #find out max magnitude among bass frequency and also its correspoinding frequency
                if (freq<crossover and mag> max_low_freq_mag):
                    max_low_freq_mag=mag


            #scaling the max bass amplitude 
            low_freq_amp = max_low_freq_mag * 2 / chunk_size
            #low_freq_amp = max_low_freq_mag
            
            #print(abs(fundamental_freq))
            #print(abs(low_freq_amp))

            ##call to scale the input and adjust the brightness 
            brightness= standardizer_brightness(low_freq_amp)
            print(brightness)
            global average_brightness
            average_brightness=find_average_brightness(brightness)


            ##we have the brightness, but for which colour
            r,g,b=color_picker(average_brightness)
            r_,g_,b_=str(r),str(g),str(b)
            
            information_color=r_+g_+b_+'\n'
            print(information_color)
            #ser.write(information_color.encode())

            line2.set_ydata(fft_magnitude)
            #line.set_ydata(audio_array)
            fig.canvas.flush_events()
            


    except KeyboardInterrupt:
        pass

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("sample_rate", default = 0, type = int)
    parser.add_argument("chunk_size", default = 0, type = int)
    args = parser.parse_args()

    analyser(args.sample_rate, args.chunk_size)
