# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, LocallyConnected1D, \
    GaussianNoise

from datetime import datetime, timedelta
from keras.engine import Layer

import keras.backend as K
from keras import initializers
from keras.optimizers import SGD, RMSprop
from keras.utils import get_custom_objects
from config import tickets,output_path, use_today, today

# 导入数据
df = pd.read_csv(output_path + 'result' + '.csv')  # 读入股票数据
df = df.fillna(0)
data = df.loc[:, list(tickets.keys())]
x_date = df.loc[:, 'date']
normalize_data = data.values
# normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
# normalize_data = normalize_data[:, np.newaxis]  # 增加维度

# 生成训练集
# 设置常量
time_step = 40  # 时间步
rnn_unit = 128  # hidden layer units
input_size = 28  # 输入层维度
output_size = 28  # 输出层维度
time_window = 10
predict_time_interval = 100
empty_time = 10
print(data.head())

data_x, data_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - time_window):
    x = normalize_data[i:i + time_step]
    # 使用均值差作为预测标准
    y = np.mean(normalize_data[i + time_step: i + time_step + time_window], axis=0) - np.mean(
        normalize_data[i + time_step - time_window:i + time_step], axis=0)
    data_x.append(x)  # 将数组转化成列表
    data_y.append(y)

data_x = np.array(data_x)
data_y = np.array(data_y)
data_x = data_x.reshape((-1, time_step, input_size))
data_y = data_y.reshape((-1, output_size))
train_x = data_x[:]
train_y = data_y[:]

data_test_x, data_test_y = [], []
for i in range(len(normalize_data) - time_step):
    x = normalize_data[i:i + time_step]
    # 预测标准使用y真实值
    y = normalize_data[i + time_step]
    data_test_x.append(x)  # 将数组转化成列表
    data_test_y.append(y)

data_test_x = np.array(data_test_x)
data_test_y = np.array(data_test_y)

test_x = data_test_x[-predict_time_interval:]
test_y = data_test_y[-predict_time_interval:]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# 使用happynoom描述的网络模型
# https://github.com/happynoom/DeepTrade_keras
def risk_estimation(y_true, y_pred):
    return -100. * K.mean(y_true * y_pred)


def pairwise_logit(y_true, y_pred):
    loss_mat = K.log(1 + K.exp(K.sign(K.transpose(y_true) - y_true) * (y_pred - K.transpose(y_pred))))
    return K.mean(K.mean(loss_mat))


class ReLU(Layer):
    """Rectified Linear Unit."""

    def __init__(self, alpha=0.0, max_value=None, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = alpha
        self.max_value = max_value

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha, max_value=self.max_value)

    def get_config(self):
        config = {'alpha': self.alpha, 'max_value': self.max_value}
        base_config = super(ReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model:
    # 使用happynoom描述的网络模型
    # https://github.com/happynoom/DeepTrade_keras
    def __init__(self, input_shape=None, learning_rate=0.005, n_layers=2, n_hidden=8, rate_dropout=0.2,
                 loss=risk_estimation):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.rate_dropout = rate_dropout
        self.loss = loss
        self.model = None

    def lstmModel(self):
        self.model = Sequential()
        self.model.add(GaussianNoise(stddev=0.01, input_shape=self.input_shape))
        for i in range(0, self.n_layers - 1):
            self.model.add(LSTM(self.n_hidden * 4, return_sequences=True, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=self.rate_dropout, recurrent_dropout=self.rate_dropout))

        self.model.add(LSTM(self.n_hidden, return_sequences=False, activation='tanh',
                            recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros',
                            dropout=self.rate_dropout, recurrent_dropout=self.rate_dropout))
        self.model.add(Dense(output_size, kernel_initializer=initializers.glorot_uniform()))
        self.model.add(BatchNormalization(axis=-1, beta_initializer='ones'))
        self.model.add(ReLU(alpha=0.0, max_value=1.0))
        opt = RMSprop(lr=self.learning_rate)
        self.model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])
        self.model.summary()
        return self.model

    def train(self):
        # fit network
        history = self.model.fit(train_x, train_y, epochs=50, batch_size=64, verbose=1, shuffle=True)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.legend()
        plt.show()

    def save(self, file='lstm_28.h5'):
        self.model.save(file)

    def load(self, file='lstm_28.h5'):
        self.model = load_model(file, custom_objects={'risk_estimation': risk_estimation})

    def predict(self, test):
        predict = []
        for sample_index in range(test.shape[0]):
            test_data = test[sample_index].reshape(1, time_step, input_size)
            prev = self.model.predict(test_data)
            predict.append(prev)
        return np.array(predict)


get_custom_objects().update({'ReLU': ReLU})
model = Model(input_shape=(time_step, input_size), loss=risk_estimation)
net = model.lstmModel()

model.load()
# 训练模型
model.train()
# 储存模型
model.save()
# 读入模型
model.load()
# 预测
predict = model.predict(test_x)
predict = predict.reshape(-1, output_size)


# 画图使用
history = data_test_y[-predict_time_interval - empty_time:-predict_time_interval, :].reshape(-1, output_size)
val = data_test_y[-predict_time_interval:, :].reshape(-1, output_size)
eval_seq = np.vstack((history, val))

history_pad = 0.5 * np.ones((history.shape[0], history.shape[1]))
prev_seq = np.vstack((history_pad, predict))

dim_date = x_date[-predict_time_interval - empty_time:]
xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dim_date]
print("xs:", len(xs))
xs1 = [(datetime.strptime(d, '%Y-%m-%d') + timedelta(days=1)).date() for d in dim_date]

rcParams.update({'font.size': 16, 'font.family': 'serif'})

fig_num = 4
num = int(output_size/fig_num)
for i in range(fig_num):
    fig = plt.figure(figsize=(20, 40))
    for im_num in range(i*num, (i+1)*num):
        index = im_num
        ax1 = fig.add_subplot(num, 1, im_num - i*num + 1)
        ax1.set_ylabel(data.columns[index])
        l1 = ax1.plot(xs, eval_seq[:, index], color='red')[0]

        ax2 = ax1.twinx()
        l2 = ax2.plot(xs1, prev_seq[:, index], color='green')[0]
        ax2.set_ylabel('predict')
        fig.legend([l1, l2],['price predict', 'buy predict'], loc = 'upper right')
        plt.gcf().autofmt_xdate()
    plt.savefig(output_path + 'predict_{}.png'.format(i))
# plt.show()



