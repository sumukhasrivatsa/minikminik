import serial
import matplotlib.pyplot as plt


x_data=[]
y_data=[]

#let number of data points be around 50 per graph
max_data_points=150

plt.ion()
plt.figure()
plt.title('Sensor Data Plot')
plt.xlabel('Time')
plt.ylabel('Sensor Value')
# Open a connection to the Arduino (adjust the port and baud rate as needed)
arduino = serial.Serial('COM9', 9600)  
if not arduino.isOpen():
    arduino.open()


try:
    while True:
        # Read a line of data from the Arduino
        data = arduino.readline().decode().strip()  # Read and decode the data
        analog_value = int(data)  # Convert the data to an integer
        
        x_data.append(len(x_data))
        y_data.append(analog_value)

        if len(x_data) > max_data_points:
            x_data.pop(0)
            y_data.pop(0)

        plt.clf()
        plt.plot(x_data, y_data)

        plt.draw()
        plt.pause(0.1)


except KeyboardInterrupt:
    print("Reading terminated by user")


