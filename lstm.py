import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, LocallyConnected1D, \
    GaussianNoise

from datetime import datetime
from keras.engine import Layer

import keras.backend as K
from keras import initializers
from keras.optimizers import SGD, RMSprop
from keras.utils import get_custom_objects

# 导入数据
df = pd.read_csv('result.csv')  # 读入股票数据
df = df.fillna(-1)
data = df.loc[:, ('medical', 'steel', 'public', 'transport', 'trade', 'computer', 'communication', 'bank')]

x_date = df.loc[:, 'date']

'''
data = df.loc[:, ('agriculture', 'excavation', 'chemical', 'steel',
                  'metals', 'electronic', 'electrical', 'food', 'cloths', 'lightIndustry',
                  'medical', 'public', 'transport', 'house', 'trade', 'service',
                  'integrated', 'building', 'decorating', 'electEquipment', 'war',
                  'computer', 'media', 'communication', 'bank', 'finance', 'automobile',
                  'mechanics')]
'''
normalize_data = data.values
# normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
# normalize_data = normalize_data[:, np.newaxis]  # 增加维度

# 生成训练集
# 设置常量
time_step = 40  # 时间步
rnn_unit = 128  # hidden layer units
input_size = 8  # 输入层维度
output_size = 8  # 输出层维度
lr = 0.01  # 学习率
lstm_layers = 2
time_window = 10
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

train_x = data_x[:-1]
train_y = data_y[:-1]
test_x = data_x[-10:]
test_y = data_y[-10:]

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
    def __init__(self, input_shape=None, learning_rate=0.001, n_layers=2, n_hidden=8, rate_dropout=0.2,
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
        history = self.model.fit(train_x, train_y, epochs=50, batch_size=64, verbose=1, shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.legend()
        plt.show()

    def save(self, file='D:/TensorFlow/stock_data/lstm.h5'):
        self.model.save(file)

    def load(self, file='D:/TensorFlow/stock_data/lstm.h5'):
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

#model.load()
#训练模型
model.train()
#储存模型
model.save()
#读入模型
model.load()
#预测
predict = model.predict(test_x)
predict = predict.reshape(-1, output_size)
print(predict)

# 画图使用
history = train_x[-20:-10, -1, :].reshape(-1, output_size)
val = test_x[:, -1, :].reshape(-1, output_size)
eval_seq = np.vstack((history, val))
prev_seq = np.vstack((history, predict))

dim_date = x_date[-20:]
xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dim_date]

for im_num in range(output_size):
    min = im_num * 1
    max = min + 1
    for index in range(min, max):
        fig, ax1 = plt.subplots(facecolor='white')
        ax1.set_ylabel(data.columns[index])
        ax1.plot(xs, eval_seq[:, index], color='red')
        ax2 = ax1.twinx()
        ax2.plot(xs, prev_seq[:, index], color='green')
        ax2.set_ylabel('predict')
        plt.gcf().autofmt_xdate()
    plt.show()
