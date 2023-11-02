import tf_utils
import tensorflow as tf


def placeholder(P, Q, N, dims):
    X = tf.compat.v1.placeholder(
        shape = (None, P, N, dims), dtype = tf.float32, name = 'X')
    TE = tf.compat.v1.placeholder(
        shape = (None, P + Q, 2), dtype = tf.int32, name = 'TE')
    Weather = tf.compat.v1.placeholder(
        shape = (None, P + Q, N, 1), dtype = tf.float32, name = 'Weather')
    label = tf.compat.v1.placeholder(
        shape = (None, Q, N, dims), dtype = tf.float32, name = 'label')
    is_training = tf.compat.v1.placeholder(
        shape = (), dtype = tf.bool, name = 'is_training')
    return X, TE, Weather, label, is_training

def FC(x, units, activations, bn, bn_decay, is_training, use_bias = True, drop = None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = tf_utils.dropout(x, drop = drop, is_training = is_training)
        x = tf_utils.conv2d(
            x, output_dims = num_unit, kernel_size = [1, 1], stride = [1, 1],
            padding = 'VALID', use_bias = use_bias, activation = activation,
            bn = bn, bn_decay = bn_decay, is_training = is_training)
    return x

def GC2(x, A, bn, bn_decay, is_training):
    '''
    x: [None, P, N, d] 
    A: [N, N] 2d normalized laplacian matrix
    '''
    P = x.get_shape()[1].value
    N = x.get_shape()[2].value
    d = x.get_shape()[3].value
    z = tf.reshape(tf.transpose(x, perm = (2, 0, 1, 3)), shape = (N, -1))
    z = tf.reshape(tf.matmul(A, z), shape = (N, -1, P, d))
    z = tf.transpose(z, perm = (1, 2, 0, 3))
    z = FC(
        z, units = [d, d], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)  
    return z

def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0)
    SE = FC(
        SE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # temporal embedding
    dayofweek = tf.one_hot(TE[..., 0], depth = 7)
    timeofday = tf.one_hot(TE[..., 1], depth = T)
    TE = tf.concat((dayofweek, timeofday), axis = -1)
    TE = tf.expand_dims(TE, axis = 2)
    TE = FC(
        TE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)

##    N = Weather.get_shape()[2].value
##    TE = tf.tile(TE, multiples = (1, 1, N, 1))
##    W = FC(
##        Weather, units = [D, D], activations = [tf.nn.relu, None],
##        bn = bn, bn_decay = bn_decay, is_training = is_training)
##    TE = tf.concat((TE, W), axis = -1)
##    TE = FC(
##        TE, units = [D, D], activations = [tf.nn.relu, None],
##        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return SE + TE

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''

    D = K * d
    # [batch_size, num_step, N, 2D]
    X = tf.concat((X, STE), axis = -1)

    # [batch_size, num_step, N, K * d]
    # 每个节点有一个d维的隐藏状态
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # 每个节点有一个d维的状态表示
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # 每个节点有一个d维的value值
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [K * batch_size, num_step, N, d]
    # 根据最后一维，split成K个sub tensor,每一个sub tensor的最后一维变成D/K=d，然后再根据第一维把K个sub tensor concat
    # 第一维变成btach_size * K
    # 把K个头拆开
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # [K * batch_size, num_step, N, N]
    # 先做最后两维的矩阵的乘法，再在不同维度重复多次，transpose_b将第二个矩阵转置
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # attention 中包含了每个节点相应的来自于N个节点的attention value
    # attention是[K*batch_size, num_step, N, N]，value是[K*batch_size, num_step, N, d]
    # atttion[K*batch_size, num_step, i].dot(value[K*batch_size, num_step])，N个d维的向量加权求和得到一个d维的向量，共N个节点
    # 所以attention value * value 的维度应该是[K*batch_size, num_step, N, d]
    X = tf.matmul(attention, value)
    # 把K个头表示的向量拼接起来,得到# [batch_size, num_step, N, D]
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X, attention

def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''

    D = K * d
    X = tf.concat((X, STE), axis = -1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # 由于时间步存在先后顺序，mask的作用是屏蔽掉当前时间步后面的score，只对前面的socre进行softmax
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape = (num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
        mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype = tf.bool)
        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)
    # softmax   
    attention = tf.nn.softmax(attention, axis = -1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X, attention

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = False)
    XT = FC(
        HT, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return H

def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask = True):
    HS, spatial_attention = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT, temporal_attention = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H), spatial_attention, temporal_attention

def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        STE_P, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))    
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X, attention
    
def GMAN(X, TE, Weather, SE, A, P, Q, T, L, K, d, bn, bn_decay, is_training):
    '''
    GMAN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    D = K * d
    # input
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    W = FC(
        Weather, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    W = GC2(W, A, bn = bn, bn_decay = bn_decay, is_training = is_training)
    batch_size = tf.shape(W)[0]
    N = W.get_shape()[2]
    W = tf.transpose(W, perm = (0, 2, 1, 3))
    W = tf.reshape(W, shape = (batch_size * N, P + Q, D))
    cell = tf.keras.layers.GRUCell(units = D)
    W = tf.keras.layers.RNN(cell = cell, return_sequences = True)(W)
    W = tf.reshape(W, shape = (batch_size, N, P + Q, D))
    W = tf.transpose(W, perm = (0, 2, 1, 3))
    W = FC(
        W, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    W_P = W[:, : P]
    W_Q = W[:, P :]
    X = X - W_P
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)    
    # STE
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P :]
    # encoder
    for _ in range(L):
        X, spatial_attention_encoder, temporal_attention_encoder = STAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    # transAtt
    X, transform_attention = transformAttention(
        X, STE_P, STE_Q, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X, spatial_attention_decoder, temporal_attention_decoder = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training,
        use_bias = True, drop = 0.1)
    # Weather
    W_Q = FC(
        Weather[:, P :], units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    W_Q = GC2(W_Q, A, bn = bn, bn_decay = bn_decay, is_training = is_training)
    W_Q = tf.transpose(W_Q, perm = (0, 2, 1, 3))
    W_Q = tf.reshape(W_Q, shape = (batch_size * N, Q, D))
    cell_Q = tf.keras.layers.GRUCell(units = D)
    W_Q = tf.keras.layers.RNN(cell = cell_Q, return_sequences = True)(W_Q)
    W_Q = tf.reshape(W_Q, shape = (batch_size, N, Q, D))
    W_Q = tf.transpose(W_Q, perm = (0, 2, 1, 3))
    W_Q = FC(
        W_Q, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    X = X + W_Q
    X = FC(
        X, units = [D, 2], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training,
        use_bias = True, drop = 0.1)
    return X, spatial_attention_encoder, temporal_attention_encoder, transform_attention, spatial_attention_decoder, temporal_attention_decoder

def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss
