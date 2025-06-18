# Neural-Cursor-Decoder

## 🧠 Predicting Mouse Cursor Positions from Neural Data
This project uses neural data recorded from 32 channels to predict mouse cursor positions over time. The goal is to develop a neural network model capable of learning the mapping from neural signals to 2D cursor movements, contributing to research in brain-machine interfaces and motor control decoding.
The project was implemented in a Jupyter notebook environment and executed on Arizona State University's Sol supercomputer, utilizing its high-throughput GPU-enabled compute nodes for model training and data processing.

I use data from a .mat file containing:

-neural_data_pro: High-resolution neural activity from 32 channels
-mouse_position: X and Y cursor positions
-time_base: Time vector corresponding to each sample


## Mouse Cursor Positions Over Time
This plot shows how the X and Y cursor positions vary throughout the 60-second recording session. It also gives insight into the subject's movement patterns and helps verify data quality before training the model.
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/mouse_cursor_position.png)

## Neural Activity from Channels of Data
In this plot, each of the 32 channels of neural data is plotted with a vertical offset for clarity, showing raw neural dynamics over time.
It shows firing patterns and trends in the neural signals, which are inputs to the decoding model.
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/neural_activity_channels.png)

##Mouse Cursor Speed Over Time
By computing the derivative of position (velocity), we calculate and plot the speed of cursor movement.
This plot shows moments of rapid movement and can indicate behavioral responses or reaction times.
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/mouse_cursor_speed.png)
