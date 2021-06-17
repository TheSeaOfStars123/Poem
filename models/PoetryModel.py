# @Time : 2021/5/28 10:42 PM 
# @Author : zyc
# @File : PoetryModel.py 
# @Title :
# @Description :

from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F


class PoetryModel(BasicModule):
    def __init__(self, vocab_size, embedding_num, num_hiddens, num_layers):
        super(PoetryModel, self).__init__()
        self.vocab_size = vocab_size
        self.state = None
        self.num_hiddens = num_hiddens  # 隐层个数
        self.num_layers = num_layers
        # 词向量层，词表大小 * 向量维度，embeddings會生成一個隨機的詞向量
        self.embeddings = nn.Embedding(vocab_size, embedding_num)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_num, self.num_hiddens, self.num_layers, batch_first=True, dropout=0,  # input為(batch, seq, input_size)
                            bidirectional=False)
        # 进行分类
        self.dense = nn.Linear(self.num_hiddens, vocab_size)

    def forward(self, inputs, state=None):
        # 假如每個詞是100維的向量，每個句子含有24個單詞，一次訓練10個句子。
        # 那麼batch_size=10,seq=24,input_size=100。(seq指的是句子的長度)
        batch_size, seq_len = inputs.size()
        X = self.embeddings(inputs)  # (batch_size, seq_len序列长度) -> (batch_size, seq_len, embedding_num)

        if state is None:
            # h_0: (num_layers*bidirectional, batch_size, num_hiddens)
            # c_0: (num_layers*bidirectional, batch_size, num_hiddens)
            h_0 = inputs.data.new(self.num_layers * 1, batch_size, self.num_hiddens).fill_(0).float()
            c_0 = inputs.data.new(self.num_layers * 1, batch_size, self.num_hiddens).fill_(0).float()
        else:
            h_0, c_0 = state

        # X = batch_size * seq_len * embedding_num
        # Y = batch_size * seq_len * bidirectional*num_hiddens
        Y, self.state = self.lstm(X, (h_0, c_0))

        # 全连接层会首先将Y的形状变成(seq_len*batch_size, num_hiddens)
        # 它的输出形状为(seq_len*batch_size, vocab_size)
        output = self.dense(Y.contiguous().view(batch_size*seq_len, -1))
        return output, self.state
