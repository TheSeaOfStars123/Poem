# @Time : 2021/6/14 4:09 PM 
# @Author : zyc
# @File : AverageMeter.py 
# @Title :
# @Description :


class AvgrageMeter(object):
    """
    在之前的手写数字识别的准确率的计算和画图以日志的打印比较简单，
    在这更新为topk准确率以及使用tensorboard来画曲线。并且使用tqdm进度条来实时的打印日志。
    专门建立一个类来保存和更新准确率的结果，使用类来让代码更加的规范化
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# topk的准确率计算

# torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
# 返回某一维度前k个的索引
# input：一个tensor数据
# k：指明是得到前k个数据以及其index
# dim： 指定在哪个维度上排序， 默认是最后一个维度
# largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
# sorted：返回的结果按照顺序返回
# out：可缺省，不要

def accuracy(output, label, topk=(1,)):
    """

    :param output: 预测标签序列
    :param label: 正确标签序列
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn


