import math
import numpy as np
##import pandas as pd

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_sample = len(data) - P - Q + 1
    x = np.zeros(shape = ((num_sample, P) + data.shape[1 :]))
    y = np.zeros(shape = ((num_sample, Q) + data.shape[1 :]))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def loadData(args):
    # Traffic
    Traffic = np.load(args.traffic_file)
    Traffic = Traffic[: -5]
    # train/val/test 
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]



    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding 
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
##    SE = np.zeros(shape = (75, 64), dtype = np.float32)
##    for i in range(75):
##        SE[i] = 1.0
        
    # temporal embedding
    num_day = math.ceil(Traffic.shape[0] / 24)
    num_week = math.ceil(num_day / 7)
    dayofweek = np.repeat(np.arange(7), repeats = 24, axis = 0)
    dayofweek = np.repeat(np.reshape(dayofweek, newshape = (1, -1)),
                          repeats = num_week, axis = 0)
    dayofweek = np.reshape(dayofweek, newshape = (-1, 1))
    dayofweek = dayofweek[: Traffic.shape[0]]
    timeofday = np.reshape(np.arange(24), newshape = (1, -1))
    timeofday = np.repeat(timeofday, repeats = num_day, axis = 0)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))
    timeofday = timeofday[: Traffic.shape[0]]
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    # Weather
    Weather = np.load(args.weather_file)
##    minimum, maximum = np.min(Weather, axis = (0, 1)), np.max(Weather, axis = (0, 1))
##    Weather = (Weather - minimum) / (maximum - minimum)
    mean1, std1 = np.mean(Weather, axis = (0, 1)), np.std(Weather, axis = (0, 1))
    Weather = (Weather - mean1) / std1
##    Weather1 = np.zeros(shape = (Weather.shape[0], Weather.shape[1], 3 * Weather.shape[2]))
##    for i in range(Weather.shape[0]):
##        if i == 0:
##            Weather1[i, :, : 6] = Weather[i]
##        else:
##            Weather1[i, :, : 6] = Weather[i - 1]
##        Weather1[i, :, 6 : 12] = Weather[i]
##        if i == Weather.shape[0] - 1:
##            Weather1[i, :, 12 :] = Weather[i]
##        else:
##            Weather1[i, :, 12 :] = Weather[i + 1]
##    Weather = Weather1
    # train/val/test
    train = Weather[: train_steps]
    val = Weather[train_steps : train_steps + val_steps]
    test = Weather[-test_steps :]    
    # shape = (num_sample, P + Q, N, 6)
    trainWeather = seq2instance(train, args.P, args.Q)
    trainWeather = np.concatenate(trainWeather, axis = 1).astype(np.float32)
    valWeather = seq2instance(val, args.P, args.Q)
    valWeather = np.concatenate(valWeather, axis = 1).astype(np.float32)
    testWeather = seq2instance(test, args.P, args.Q)
    testWeather = np.concatenate(testWeather, axis = 1).astype(np.float32)

    # dist
    dist = np.load(args.dist_file)
    std = np.std(dist)
    A = np.exp(- dist ** 2 / std ** 2)
    D = np.sum(A, axis = 1)
    D = np.diag(D ** -0.5)
    A = np.matmul(np.matmul(D, A), D)
    A = A.astype(np.float32)
    
    return (trainX, trainTE, trainWeather, trainY, valX, valTE, valWeather, valY,
            testX, testTE, testWeather, testY, SE, mean, std, A)
