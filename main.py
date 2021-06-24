# @Time : 2021/5/27 10:30 PM 
# @Author : zyc
# @File : main.py 
# @Title :
# @Description :

from config import opt
import torch as t
import matplotlib.pyplot as plt
from models import PoetryModel
from data.dataset import view_data, PoemDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.AverageMeter import accuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(**kwargs, ):
    '''
    训练
    :return:
    '''
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    # 创建数据迭代器
    train_dataloader = DataLoader(poem_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # step2: 模型
    model = PoetryModel(len(word2ix),
                        embedding_num=opt.embedding_num,
                        num_hiddens=opt.num_hiddens,
                        num_layers=opt.LSTM_layers)
    print(model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        print('Cuda is available!')
        model.cuda()

    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)
    
    # step4:统计指标
    # AverageValueMeter能够计算所有数的平均值和标准差，这里用来统计一个epoch中损失的平均值。
    # confusionmeter用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标。
    loss_meter = meter.AverageValueMeter()
    top1 = meter.AverageValueMeter()

    # step5: 训练前准备工作
    previous_loss = 1e20
    # 绘制曲线需要
    period = []
    count = 0
    training_loss = []
    train_accs = []
    # validation_accs = []
    """
    保存训练日志
    路径:/tmp/training_log_0523_23:57:29.txt
    """
    prefix = opt.training_log + "_"
    path = time.strftime(prefix + '%m%d_%H:%M:%S.txt')
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.isfile(path):
        open(path, 'w')

    # step6: 训练
    for epoch in range(opt.max_epoch):
        f = open(path, 'a')
        start = time.time()
        loss_meter.reset()
        # 在tqdm读条的前后显示需要的信息
        # set_description()设定的是前缀
        # set_postfix()设定的是后缀
        train_dataloader = tqdm(train_dataloader)
        train_dataloader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch+1, opt.max_epoch, 'lr:', lr))
        for ii, (data, label) in enumerate(train_dataloader):  # 65260/16 = 4078
            # 如果模型没有设置 batch_first=True(将批次维度放到第一位)的话，需要将0维和1维互换位置。
            # transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置：
            # data = data.long().transpose(1,0).contiguous()
            # input, target = data[:-1, :], data[1:, :]
            # 训练模型
            input = Variable(data)  # [16, 48] 16是batch_size,48是seq_len
            target = Variable(label)  # [16, 48]
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score, hidden = model(input)
            target = target.view(-1)  # 变成一维数组 768=16*48
            loss = criterion(score, target)
            # _, pred = score.topK(1)
            prec1, prec2 = accuracy(score, target, topk=(1, 2))
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            top1.add(prec1.item())  # 准确率
            loss_meter.add(loss.item())   # 总loss
            period.append(epoch*len(train_dataloader)+ii)  # 一次epoch有4079次
            period.append(count)  # 一次epoch有4079次
            training_loss.append(loss_meter.value()[0])  # 记录每一个batch_size训练过程中的train_loss
            train_accs.append(top1.value()[0])

            postfix = {'train_loss': '%.6f' % (loss_meter.value()[0]), 'train_acc': '%.6f' % (top1.value()[0])}
            train_dataloader.set_postfix_str(postfix)
            # 使用tensorboard进行曲线绘制
            if not os.path.exists(opt.tensorboard_path):
                os.mkdir(opt.tensorboard_path)
            count = count + 1
            writer = SummaryWriter(opt.tensorboard_path)
            writer.add_scalar('Train/Loss', loss_meter.value()[0], count)
            writer.add_scalar('Train/Accuracy', top1.value()[0], count)
            writer.flush()

            if ii % opt.print_freq == opt.print_freq - 1:
                print('[%d,%5d] train_loss :%.3f' %
                      (epoch + 1, ii + 1, loss_meter.value()[0]))
                f.write('\n[%d,%5d] train_loss :%.3f' %
                        (epoch + 1, ii + 1, loss_meter.value()[0]))
                # 对目前模型情况进行测试
                # https://tw511.com/a/01/27245.html
                # https://blog.csdn.net/weixin_39845112/article/details/80045091
                # https://blog.csdn.net/Gaowahaha/article/details/113148697
                # https://bravey.github.io/2020-05-05-%E8%8C%83%E9%97%B2%E5%86%99%E8%AF%97%E5%99%A8%E4%B9%8B%E7%94%A8LSTM+Pytorch%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%86%99%E8%AF%97.html
                # 视觉化一次
                print(str(ii) + ":" + generate(model, '床前明月光', ix2word, word2ix))



        # 当一个epoch结束之后开始打印信息
        print('epoch %d, lr %.4f, train_loss %.4f, train_acc %.3f %%, time %.1f sec' %
              (epoch + 1, lr, loss_meter.value()[0], top1.value()[0], time.time() - start))
        f.write('\nepoch %d, lr %.4f, train_loss %.4f, train_acc %.3f %%, time %.1f sec' %
                (epoch + 1, lr, loss_meter.value()[0], top1.value()[0], time.time() - start))
        f.close()

        # 如果损失不下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

    model.save()
    plt.plot(period, training_loss)
    plt.plot(period, train_accs)
    # 使用plt保存loss图像
    loss_prefix = opt.loss_file + "_"
    loss_path = time.strftime(loss_prefix + "%m%d_%H:%M:%S.png")
    print('loss_path:', loss_path)
    plt.savefig(loss_path)
    plt.show()


# 给定几个词，根据这个词生成完成的诗歌
def generate(model, start_words, ix2word, word2ix):
    result = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    input = Variable(t.Tensor([word2ix['<START>']]).view(1, 1).long())

    # 最开始的隐状态初始为0矩阵
    hidden = None
    # hidden = t.zeros((2, opt.LSTM_layers, 1, opt.num_hiddens), dtype=t.float)
    if opt.use_gpu:
        input = input.cuda()
        # hidden = hidden.cuda()
        model = model.cuda()
    model.eval()
    with t.no_grad():
        for i in range(opt.seq_len):  # 诗的长度
            output, hidden = model(input, hidden)
            # 如果在给定的句首中，input为句首中的下一个字
            if i < start_words_len:
                w = result[i]
                input = Variable(input.data.new([word2ix[w]])).view(1, 1)
            # 否则将output作为下一个input输入
            else:
                top_index = output.data[0].topk(1)[1][0]  # 得分最高词的index
                w = ix2word[top_index.item()]
                result.append(w)
                input = Variable(input.data.new([top_index])).view(1, 1)
            if w == '<EOP>':  # 输出了结束标志就退出
                break
    # 把模型恢复为训练模式
    model.train()
    return ''.join(result)


def test(**kwargs):
    opt.parse(kwargs)
    model = PoetryModel(len(word2ix),
                        embedding_num=opt.embedding_num,
                        num_hiddens=opt.num_hiddens,
                        num_layers=opt.LSTM_layers)
    # 模型
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    model.eval()
    txt = generate(model, '床前明月光', ix2word, word2ix)
    print(txt)
    # txt = gen_acrostic(modle, '機器學習', ix2word, word2ix)
    # print(txt)



if __name__ == '__main__':
    rtn = []
    batch_size = 768
    correct = t.zeros([2, 768], dtype=t.bool)
    test = correct[:1].view(-1)
    for k in (1, 2):
        correct_k = correct[:k].view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    # 因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
    import shutil
    test_path = os.path.join(os.getcwd(), opt.tensorboard_path)
    if os.path.exists(os.path.join(os.getcwd(), opt.tensorboard_path)):
        shutil.rmtree(opt.tensorboard_path)
        os.mkdir(opt.tensorboard_path)

    # step1: 数据
    view_data(opt.train_data_root)
    # 载入数据集
    poem_ds = PoemDataset(opt.train_data_root, opt.seq_len)
    # 查看数据集详情
    print('len(poem_ds):', len(poem_ds))  # 65260=3132489/48
    # 通过iter()函数获取这些可迭代对象的迭代器。
    # 然后我们可以对获取到的迭代器不断使⽤next()函数来获取下⼀条数据
    data, label = next(iter(poem_ds))
    print('len(data):', len(data))  # 48
    print('*' * 15 + 'inputs' + '*' * 15)
    print([[poem_ds.ix2word[d.item()]] for d in data])
    print('*' * 15 + 'labels' + '*' * 15)
    print([[poem_ds.ix2word[d.item()]] for d in label])
    # 载入ix2word和word2ix
    ix2word = poem_ds.ix2word  # 8293
    word2ix = poem_ds.word2ix  # 8293
    train()
    # test()
