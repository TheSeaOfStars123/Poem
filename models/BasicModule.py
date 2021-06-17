# @Time : 2021/5/9 4:06 PM 
# @Author : zyc
# @File : BasicModule.py 
# @Title : 对 nn.Module 的简易封装
# @Description : 提供快速加载和保存模型的接口,主要提供 save 和 load 两个方法

import time
import torch as t

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path):
        """
        加载指定路径的模型
        :param name:模型地址
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，使用"模型名称+时间"作为文件名
        如 AlexNet_0710_23:57:29.pth
        :param name:模型名称
        :return:
        """
        if name is None:
            prefix = 'checkpoints/' + self.module_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


