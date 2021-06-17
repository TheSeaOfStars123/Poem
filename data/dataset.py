# @Time : 2021/5/27 10:50 PM 
# @Author : zyc
# @File : dataset.py 
# @Title :
# @Description :

from torch.utils import data
import os
import numpy as np
import torch


class PoemDataset(data.Dataset):
    # seq_len 默认是48
    def __init__(self, poem_path, seq_len=48):
        self.poem_path = poem_path
        self.seq_len = seq_len
        self.poem_data, self.ix2word, self.word2ix = self.get_data_from_npz()  # (57580, 125)=7197500  vocab_size:8293
        # 去除多余空格
        self.no_space_data = self.delete_blank()  # 3132489

    def get_data_from_npz(self):
        """
        data是一個57580 * 125的numpy陣列，
        即總共有57580首詩歌，
        每首詩歌長度為125個字元（不足125補空格，超過125的丟棄）
        :return:
        """
        poem_data = np.load(self.poem_path, allow_pickle=True)
        data = poem_data['data']
        # {0: '憁', 1: '耀', 2: '枅', 3: '涉', 4: '談',...,, 8290: '<EOP>', 8291: '<START>', 8292: '</s>'}
        # 关于item()操作：
        # 不加item(): <class 'numpy.ndarray'>
        # 加上item(): <class 'dict'>
        # 裡字典的儲存還是以numpy陣列格式儲存的，所以需要使用.item()把字典取出來
        ix2word = poem_data['ix2word'].item()

        word2ix = poem_data['word2ix'].item()
        return data, ix2word, word2ix

    def delete_blank(self):
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292):
                no_space_data.append(i)
        return no_space_data

    # 根据sep_len进行分句
    # sep_len选取为48，因为唐诗一半是五言绝句和七言绝句，各自加上一个标点符号是6和8，选择一个公约数48
    # 刚好凑够8句五言或者6句七言
    def __getitem__(self, idx):
        txt = self.no_space_data[idx * self.seq_len: (idx + 1) * self.seq_len]
        label = self.no_space_data[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]  # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt, label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)


# ix2word.npy：汉字的字典索引 dict结构 大小为8293
# word2ix.npy：汉字的字典索引 dict结构
# data.npy：data部分是唐诗数据的总共包含57580首唐诗数据
#           其中每一首都被格式化成125个字符
#           唐诗开始用’<START>‘标志，结束用’<EOP>‘标志,空余的用’</s>‘标志
def view_data(poem_path):
    poem_data = np.load(poem_path, allow_pickle=True)
    data = poem_data['data']
    ix2word = poem_data['ix2word'].item()
    word2ix = poem_data['word2ix'].item()
    # word_data[0]存储随机选出的一个数据
    # word_data的大小是1row*125cols
    word_data = np.zeros((1, data.shape[1]), dtype=np.str)
    # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = np.random.randint(data.shape[0])
    for col in range(data.shape[1]):
        # 将data中的每一个字符保存到word_data[0]中
        word_data[0, col] = ix2word[data[row, col]]
    print(data.shape)
    print(word_data)





