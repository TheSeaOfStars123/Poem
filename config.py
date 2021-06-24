# @Time : 2021/5/27 10:31 PM 
# @Author : zyc
# @File : config.py 
# @Title :
# @Description :

import warnings
import os

class DefaultConfig(object):
    print('DefaultConfig:', os.getcwd())
    model = 'LSTM'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    train_data_root = os.getcwd()+'/tang.npz'
    # load_model_path = os.getcwd() + '/checkpoints/lr0.00001_ResNet34_0523_11:18:15.pth'  # 加载预训练的模型的路径，为None代表不加载
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 16  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 1  # how many workers for loading data
    print_freq = 500  # print info every N batch

    seq_len = 48
    embedding_num = 128
    num_hiddens = 256
    LSTM_layers = 2

    training_log = os.getcwd() + '/result/training_log'
    tensorboard_path = os.getcwd() + '/tmp/tensorboard'
    loss_file = os.getcwd() + '/result/result_loss'
    #
    max_epoch = 10
    lr = 1e-3  # initial learning rate
    # lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()