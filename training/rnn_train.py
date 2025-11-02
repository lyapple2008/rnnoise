#!/usr/bin/python

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, Callback
from keras.optimizers import Adam
import numpy as np

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))


def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
main_input = Input(shape=(None, 42), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

# 定义学习率调度函数
def lr_schedule(epoch):
    """
    学习率调度策略：
    - 前30个epoch: 0.001
    - 30-60个epoch: 0.0005  
    - 60-90个epoch: 0.0001
    - 90+个epoch: 0.00005
    """
    if epoch < 30:
        return 0.001
    elif epoch < 60:
        return 0.0005
    elif epoch < 90:
        return 0.0001
    else:
        return 0.00005

# 创建优化器
optimizer = Adam(learning_rate=0.001)

model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer=optimizer, 
              loss_weights=[10, 0.5])


batch_size = 32

print('Loading data...')
with h5py.File('training_5000000.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')

window_size = 2000

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :42]
x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

all_data = 0
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

# 设置模型检查点回调，每个epoch保存模型
checkpoint = ModelCheckpoint(
    filepath='weights_5000000_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5',
    monitor='val_loss',
    save_best_only=False,  # 保存每个epoch的模型
    save_weights_only=False,  # 保存完整模型（包括结构）
    verbose=1,
    period=1  # 每个epoch保存一次
)

# 可选：同时保存最佳模型（基于验证损失）
best_checkpoint = ModelCheckpoint(
    filepath='weights_5000000_best.hdf5',
    monitor='val_loss',
    save_best_only=True,  # 只保存最佳模型
    save_weights_only=False,
    verbose=1
)

# 学习率调整策略1：基于验证损失的自适应调整
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # 学习率衰减因子
    patience=5,           # 5个epoch无改善则降低学习率
    min_lr=1e-6,          # 最小学习率
    verbose=1,
    mode='min'            # 监控指标越小越好
)

# 学习率调整策略2：预定义的学习率调度
lr_scheduler = LearningRateScheduler(
    schedule=lr_schedule,
    verbose=1
)

# 学习率调整策略3：余弦退火（可选）
def cosine_annealing(epoch, lr):
    """余弦退火学习率调度"""
    import math
    epochs = 120
    return 0.001 * (1 + math.cos(math.pi * epoch / epochs)) / 2

# cosine_scheduler = LearningRateScheduler(cosine_annealing, verbose=1)

# 学习率记录回调
class LearningRateLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        print(f'\nEpoch {epoch+1}: Learning rate = {lr:.6f}')
        
        # 可选：保存学习率到文件
        with open('learning_rate_log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}: {lr:.6f}\n')

lr_logger = LearningRateLogger()

print('Train...')
# 学习率调整策略配置说明：
# 1. reduce_lr: 基于验证损失自适应调整，当5个epoch无改善时学习率减半
# 2. lr_scheduler: 预定义的分段学习率调度
# 3. lr_logger: 记录每个epoch的学习率变化
# 4. cosine_scheduler: 余弦退火调度（可选，需要取消注释）

# 选择学习率调整策略（可以组合使用）
callbacks = [
    checkpoint, 
    best_checkpoint,
    reduce_lr,        # 自适应学习率调整
    lr_scheduler,     # 预定义学习率调度
    lr_logger         # 学习率记录
    # cosine_scheduler  # 可选：余弦退火
]

model.fit(x_train, [y_train, vad_train],
          batch_size=batch_size,
          epochs=120,
          validation_split=0.1,
          callbacks=callbacks)

# 保存最终模型
model.save("weights_5000000_final.hdf5")

# 构建并导出带GRU隐状态输入/输出端口的推理模型，便于外部管理隐状态
def build_export_model(training_model: Model) -> Model:
    """
    使用与训练模型相同的权重，构建一个具有显式GRU隐状态输入/输出端口的推理模型。
    输入：
      - features: (None, None, 42)
      - vad_gru_state: (None, 24)
      - noise_gru_state: (None, 48)
      - denoise_gru_state: (None, 96)
    输出：
      - denoise_output: (None, None, 22)
      - vad_output: (None, None, 1)
      - vad_gru_state (updated): (None, 24)
      - noise_gru_state (updated): (None, 48)
      - denoise_gru_state (updated): (None, 96)
    """
    # 新的推理输入（带状态）
    features_in = Input(shape=(None, 42), name='features')
    vad_state_in = Input(shape=(24,), name='vad_gru_state')
    noise_state_in = Input(shape=(48,), name='noise_gru_state')
    denoise_state_in = Input(shape=(96,), name='denoise_gru_state')

    # 复制训练模型的层配置并加载权重
    # 1) input_dense
    input_dense_src = training_model.get_layer('input_dense')
    input_dense = Dense(24, activation='tanh', name='input_dense_export',
                        kernel_constraint=input_dense_src.kernel_constraint,
                        bias_constraint=input_dense_src.bias_constraint)
    tmp_export = input_dense(features_in)
    input_dense.set_weights(input_dense_src.get_weights())

    # 2) vad_gru (return_sequences+return_state)
    vad_gru_src = training_model.get_layer('vad_gru')
    vad_gru_exp = GRU(24, activation='tanh', recurrent_activation='sigmoid',
                      return_sequences=True, return_state=True, name='vad_gru_export',
                      kernel_regularizer=vad_gru_src.kernel_regularizer,
                      recurrent_regularizer=vad_gru_src.recurrent_regularizer,
                      kernel_constraint=vad_gru_src.kernel_constraint,
                      recurrent_constraint=vad_gru_src.recurrent_constraint,
                      bias_constraint=vad_gru_src.bias_constraint)
    vad_seq, vad_state_out = vad_gru_exp(tmp_export, initial_state=vad_state_in)
    vad_gru_exp.set_weights(vad_gru_src.get_weights())

    # 3) vad_output
    vad_output_src = training_model.get_layer('vad_output')
    vad_output_exp_layer = Dense(1, activation='sigmoid', name='vad_output_export',
                                 kernel_constraint=vad_output_src.kernel_constraint,
                                 bias_constraint=vad_output_src.bias_constraint)
    vad_output_exp = vad_output_exp_layer(vad_seq)
    vad_output_exp_layer.set_weights(vad_output_src.get_weights())

    # 4) noise_gru 输入：concat([tmp_export, vad_seq, features_in])
    noise_in = concatenate([tmp_export, vad_seq, features_in], name='noise_concat_export')
    noise_gru_src = training_model.get_layer('noise_gru')
    noise_gru_exp = GRU(48, activation='relu', recurrent_activation='sigmoid',
                        return_sequences=True, return_state=True, name='noise_gru_export',
                        kernel_regularizer=noise_gru_src.kernel_regularizer,
                        recurrent_regularizer=noise_gru_src.recurrent_regularizer,
                        kernel_constraint=noise_gru_src.kernel_constraint,
                        recurrent_constraint=noise_gru_src.recurrent_constraint,
                        bias_constraint=noise_gru_src.bias_constraint)
    noise_seq, noise_state_out = noise_gru_exp(noise_in, initial_state=noise_state_in)
    noise_gru_exp.set_weights(noise_gru_src.get_weights())

    # 5) denoise_gru 输入：concat([vad_seq, noise_seq, features_in])
    denoise_in = concatenate([vad_seq, noise_seq, features_in], name='denoise_concat_export')
    denoise_gru_src = training_model.get_layer('denoise_gru')
    denoise_gru_exp = GRU(96, activation='tanh', recurrent_activation='sigmoid',
                          return_sequences=True, return_state=True, name='denoise_gru_export',
                          kernel_regularizer=denoise_gru_src.kernel_regularizer,
                          recurrent_regularizer=denoise_gru_src.recurrent_regularizer,
                          kernel_constraint=denoise_gru_src.kernel_constraint,
                          recurrent_constraint=denoise_gru_src.recurrent_constraint,
                          bias_constraint=denoise_gru_src.bias_constraint)
    denoise_seq, denoise_state_out = denoise_gru_exp(denoise_in, initial_state=denoise_state_in)
    denoise_gru_exp.set_weights(denoise_gru_src.get_weights())

    # 6) denoise_output
    denoise_output_src = training_model.get_layer('denoise_output')
    denoise_output_exp_layer = Dense(22, activation='sigmoid', name='denoise_output_export',
                                     kernel_constraint=denoise_output_src.kernel_constraint,
                                     bias_constraint=denoise_output_src.bias_constraint)
    denoise_output_exp = denoise_output_exp_layer(denoise_seq)
    denoise_output_exp_layer.set_weights(denoise_output_src.get_weights())

    export_model = Model(
        inputs=[features_in, vad_state_in, noise_state_in, denoise_state_in],
        outputs=[denoise_output_exp, vad_output_exp, vad_state_out, noise_state_out, denoise_state_out],
        name='rnnoise_export_with_states'
    )
    return export_model

# 训练结束后，构建并保存带状态端口的推理模型
export_model = build_export_model(model)
export_model.save("weights_5000000_with_states.hdf5")
