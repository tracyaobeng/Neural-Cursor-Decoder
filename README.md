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
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/neural_input_window.png)

## Neural Network Model
I used a Long Short-Term Memory (LSTM) network to decode cursor position from temporal neural activity windows.

### Preprocessing Steps
Input Construction: Each input sample is a 100 ms (30-sample) sliding window from all 32 neural channels.
Output: The midpoint cursor position corresponding to each window.
#### Normalization:
Inputs: StandardScaler across all neural features.
Outputs: MinMaxScaler for x/y position values.
####  🏗️Model Architecture
Loss: Mean Squared Error
Metrics: Mean Absolute Error (MAE)
Callbacks: EarlyStopping and ReduceLROnPlateau
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/predicted_true_main.png)



## Model Comparison: LSTM Variants
To evaluate the impact of model complexity on decoding performance, I trained and compared four different LSTM architectures by varying the number and size of LSTM layers. All models were trained under similar conditions (same data splits, optimizer, and loss function) and evaluated using mean squared error and positional accuracy (percentage of test samples predicted within 2500 units of the ground truth).

### LSTM_64
Architecture: 1 LSTM layer with 64 units
Test Loss: 0.0173
Accuracy: 80.45%
Training Time: 65.17 sec

This baseline model with a single LSTM(64) layer achieves decent performance, capturing the general trajectory shape. However, the predictions show more noise and spatial spread, particularly near the trajectory edges, suggesting that a deeper or more expressive model could improve positional accuracy.
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/LSTM_64.png)

### LSTM_64_32
Architecture: 2 LSTM layers: 64 units → 32 units
Test Loss: 0.0198
Accuracy: 77.07%
Training Time: 77.91 sec

Adding a second LSTM layer did not improve performance in this case. The accuracy slightly dropped and predictions became noisier, likely due to overfitting or vanishing gradient issues in the smaller second layer. This highlights the importance of carefully balancing depth and unit size.
![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/LSTM_64_32.png)

### LSTM_128_64 (Best Performer)
Architecture: 2 LSTM layers: 128 units → 64 units
Test Loss: 0.0137
Accuracy: 85.73%
Training Time: 159.24 sec

This model achieved the highest accuracy by increasing the representational capacity of both LSTM layers. The predictions closely follow the true trajectory with reduced scatter and better endpoint convergence, demonstrating that increased depth and wider layers improve decoding performance from neural signals.

![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/LSTM_128_64.png)

### LSTM_128_64_32
Architecture: 3 LSTM layers: 128 → 64 → 32 units
Test Loss: 0.0201
Accuracy: 78.49%
Training Time: 276.92 sec

Despite its complexity, this deeper model underperformed compared to the two-layer variant. The additional LSTM layer may have introduced unnecessary complexity, leading to overfitting or diminished gradient flow. It highlights that deeper isn’t always better—model performance depends on tuning the architecture to match data complexity.

![image](https://github.com/tracyaobeng/Neural-Cursor-Decoder/blob/main/LSTM_128_64_32.png)
