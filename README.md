# MasterCard-Stock-Price-Prediction-Using-LSTM-and-GRU


## RNN
<p align="justify">Kaggle’s MasterCard stock dataset from May-25-2006 to Oct-11-2021 to train the LSTM and GRU models to forecast the stock price. I´m gonna show how to analyze data, preprocess the data to train it on advanced RNN models, and finally evaluate the results.

The project requires Pandas and Numpy for data manipulation, Matplotlib.pyplot for data visualization, scikit-learn for scaling and evaluation, and TensorFlow for modeling. We will also set seeds for reproducibility.</p>

<p align="justify">
The train_test_plot function takes three arguments: dataset, tstart, and tend and plots a simple line graph. The tstart and tend are time limits in years. We can change these arguments to analyze specific periods. The line plot is divided into two parts: train and test. This will allow us to decide the distribution of the test dataset.</p>
<p align="justify">
MasterCard stock prices have been on the rise since 2016. It had a dip in the first quarter of 2020 but it gained a stable position in the latter half of the year. Our test dataset consists of one year, from 2021 to 2022, and the rest of the dataset is used for training.</p>


### LSTM Model
<p align="justify">The model consists of a single hidden layer of LSTM and an output layer. You can experiment with the number of units, as more units will give you better results. For this experiment, we will set LSTM units to 125, tanh as activation, and set input size.Finally, we will compile the model with an RMSprop optimizer and mean square error as a loss function.</p>

<p align="justify">The model will train on 50 epochs with 32 batch sizes. You can change the hyperparameters to reduce training time or improve the results. The model training was successfully completed with the best possible loss.</p>

#### Results
<p align="justify">
We are going to repeat preprocessing and normalize the test set. First of all we will transform then split the dataset into samples, reshape it, predict, and inverse transform the predictions into standard form.The plot_predictions function will plot a real versus predicted line chart. This will help us visualize the difference between actual and predicted values.The return_rmse function takes in test and predicted arguments and prints out the root mean square error (rmse) metric.</p>

The results look promising as the model got 6.47 rmse on the test dataset.

### GRU Model
#### Results
GRU model got 5.95 rmse on the test dataset, which is an improvement from the LSTM model.
