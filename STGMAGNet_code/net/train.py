import math
import argparse
import utils, model
import time, datetime
import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type = int, default = 60,
                    help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = 24 * 1,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 24 * 1,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 8,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 8,
                    help = 'dims of each head attention outputs')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 100,
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = 10,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')
parser.add_argument('--traffic_file', default = 'x.npy',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = 'SE.txt',
                    help = 'spatial emebdding file')
parser.add_argument('--weather_file', default = 'Weather1.npy',
                    help = 'spatial emebdding file')
parser.add_argument('--dist_file', default = 'dist.npy',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'attention-score',
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = 'attention-score',
                    help = 'log file')
args = parser.parse_args()

start = time.time()

log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainWeather, trainY, valX, valTE, valWeather, valY,
 testX, testTE, testWeather, testY, SE, mean, std, A) = utils.loadData(args)
utils.log_string(log, 'trainX: %s\t\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

# train model
utils.log_string(log, 'compiling model...')
T = 24 * 60 // args.time_slot
num_train, _, N, dims = trainX.shape
X, TE, Weather, label, is_training = model.placeholder(args.P, args.Q, N, dims)
global_step = tf.Variable(0, trainable = False)
# bn层衰减
bn_momentum = tf.compat.v1.train.exponential_decay(
    0.5, global_step,
    decay_steps = args.decay_epoch * num_train // args.batch_size,
    decay_rate = 0.5, staircase = True)
bn_decay = tf.minimum(0.99, 1 - bn_momentum)
pred, spatial_attention_encoder, temporal_attention_encoder, transform_attention, spatial_attention_decoder, temporal_attention_decoder = model.GMAN(
    X, TE, Weather, SE, A, args.P, args.Q, T, args.L, args.K, args.d,
    bn = True, bn_decay = bn_decay, is_training = is_training)
pred = pred * std + mean
loss = model.mae_loss(pred, label)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
tf.compat.v1.add_to_collection('spatial_attention_encoder', spatial_attention_encoder)
tf.compat.v1.add_to_collection('temporal_attention_encoder', temporal_attention_encoder)
tf.compat.v1.add_to_collection('transform_attention', transform_attention)
tf.compat.v1.add_to_collection('spatial_attention_decoder', spatial_attention_decoder)
tf.compat.v1.add_to_collection('temporal_attention_decoder', temporal_attention_decoder)
learning_rate = tf.compat.v1.train.exponential_decay(
    args.learning_rate, global_step,
    decay_steps = args.decay_epoch * num_train // args.batch_size,
    decay_rate = 0.7, staircase = True)
learning_rate = tf.maximum(learning_rate, 1e-5)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step = global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])
utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainWeather = trainWeather[permutation]
    trainY = trainY[permutation]
    # train loss
    start_train = time.time()
    train_loss = 0
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
           X: trainX[start_idx : end_idx],
           TE: trainTE[start_idx : end_idx],
           Weather: trainWeather[start_idx : end_idx],
           label: trainY[start_idx : end_idx],
           is_training: True}
        _, loss_batch = sess.run([train_op, loss], feed_dict = feed_dict)
        train_loss += loss_batch * (end_idx - start_idx)
    train_loss /= num_train
    end_train = time.time()
    # val loss
    start_val = time.time()
    val_loss = 0
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
           X: valX[start_idx : end_idx],
           TE: valTE[start_idx : end_idx],
           Weather: valWeather[start_idx : end_idx],
           label: valY[start_idx : end_idx],
           is_training: False}
        loss_batch = sess.run(loss, feed_dict = feed_dict)
        val_loss += loss_batch * (end_idx - start_idx)
    val_loss /= num_val
    end_val = time.time()
    utils.log_string(
        log,
        '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
         args.max_epoch, end_train - start_train, end_val - start_val))
    utils.log_string(
        log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
           log,
           'val loss decrease from %.4f to %.4f, saving model to %s' %
           (val_loss_min, val_loss, args.model_file))
        wait = 0
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
    else:
        wait += 1
        
# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
num_test = testX.shape[0]
trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: trainX[start_idx : end_idx],
        TE: trainTE[start_idx : end_idx],
        Weather: trainWeather[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    trainPred.append(pred_batch)
trainPred = np.concatenate(trainPred, axis = 0)
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: valX[start_idx : end_idx],
        TE: valTE[start_idx : end_idx],
        Weather: valWeather[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    valPred.append(pred_batch)
valPred = np.concatenate(valPred, axis = 0)
testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        X: testX[start_idx : end_idx],
        TE: testTE[start_idx : end_idx],
        Weather: testWeather[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    testPred.append(pred_batch)
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)
train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
test_mae, test_rmse, test_mape = utils.metric(testPred, testY)

np.save('Test1010NTS.npy', testPred)
np.save('True1010NTS.npy', testY)
print('++++++++++++++++')
print(testPred.shape)
print(testY.shape)

test_mae1, test_rmse1, test_mape1 = utils.metric(testPred[..., 0], testY[..., 0])
test_mae2, test_rmse2, test_mape2 = utils.metric(testPred[..., 1], testY[..., 1])
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                 (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                 (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae, test_rmse, test_mape * 100))
utils.log_string(log, 'arr              %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae1, test_rmse1, test_mape1 * 100))
utils.log_string(log, 'dep              %.2f\t\t%.2f\t\t%.2f%%' %
                 (test_mae2, test_rmse2, test_mape2 * 100))
utils.log_string(log, 'performance in each prediction step')
MAE1, RMSE1, MAPE1 = [], [], []
MAE11, MAE12, MAE13 = [], [], []
MAE2, RMSE2, MAPE2 = [], [], []
MAE21, MAE22, MAE23 = [], [], []

type1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 26]
type2 = [17, 18, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53]
type3 = [51, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
for q in range(args.Q):
    mae1, rmse1, mape1 = utils.metric(testPred[:, q, :, 0], testY[:, q, :, 0])
    mae11, rmse11, mape11 = utils.metric(testPred[:, q, type1, 0], testY[:, q, type1, 0])
    mae12, rmse12, mape12 = utils.metric(testPred[:, q, type2, 0], testY[:, q, type2, 0])
    mae13, rmse13, mape13 = utils.metric(testPred[:, q, type3, 0], testY[:, q, type3, 0])
    mae2, rmse2, mape2 = utils.metric(testPred[:, q, :, 1], testY[:, q, :, 1])
    mae21, rmse21, mape21 = utils.metric(testPred[:, q, type1, 1], testY[:, q, type1, 1])
    mae22, rmse22, mape22 = utils.metric(testPred[:, q, type2, 1], testY[:, q, type2, 1])
    mae23, rmse23, mape23 = utils.metric(testPred[:, q, type3, 1], testY[:, q, type3, 1])
    MAE1.append(mae1)
    RMSE1.append(rmse1)
    MAPE1.append(mape1)
    MAE11.append(mae11)
    MAE12.append(mae12)
    MAE13.append(mae13)
    MAE2.append(mae2)
    MAE21.append(mae21)
    MAE22.append(mae22)
    MAE23.append(mae23)
    RMSE2.append(rmse2)
    MAPE2.append(mape2)
for q in range(args.Q):
    utils.log_string(log, 'arr: step: %02d         %.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
                     (q + 1, MAE11[q], MAE12[q], MAE13[q], MAE1[q]))
##    utils.log_string(log, 'arr: type1: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae11, rmse11, mape11 * 100))
##    utils.log_string(log, 'arr: type2: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae12, rmse12, mape12 * 100))
##    utils.log_string(log, 'arr: type3: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae13, rmse13, mape13 * 100))
for q in range(args.Q):
    utils.log_string(log, 'dep: step: %02d         %.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
                     (q + 1, MAE21[q], MAE22[q], MAE23[q], MAE2[q]))
##    utils.log_string(log, 'dep: type1: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae21, rmse21, mape21 * 100))
##    utils.log_string(log, 'dep: type2: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae22, rmse22, mape22 * 100))
##    utils.log_string(log, 'dep: type3: step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
##                     (q + 1, mae23, rmse23, mape23 * 100))
average_mae1 = np.mean(MAE1)
average_mae11 = np.mean(MAE11)
average_mae12 = np.mean(MAE12)
average_mae13 = np.mean(MAE13)
average_rmse1 = np.mean(RMSE1)
average_mape1 = np.mean(MAPE1)
utils.log_string(
    log, 'average arr:          %.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
    (average_mae11, average_mae12, average_mae13, average_mae1))
average_mae2 = np.mean(MAE2)
average_mae21 = np.mean(MAE21)
average_mae22 = np.mean(MAE22)
average_mae23 = np.mean(MAE23)
average_rmse2 = np.mean(RMSE2)
average_mape2 = np.mean(MAPE2)
utils.log_string(
    log, 'average dep:          %.2f\t\t%.2f\t\t%.2f\t\t%.2f' %
    (average_mae21, average_mae22, average_mae23, average_mae2))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
sess.close()
log.close()
