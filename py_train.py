# encoding: utf-8

from __future__ import print_function
from __future__ import division
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
#from warpctc_pytorch import CTCLoss
import os
import utils as utils1
import dataset
import hypy_alphabet
import models.crnn as crnn


parser = argparse.ArgumentParser()

parser.add_argument('--trainRoot', default = r'./train', help='path to dataset')
parser.add_argument('--valRoot', default = r'./test', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of demo_data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=300, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to sxmv for')   #迭代次数
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda',default=True, action='store_true', help='enables cuda')    #有GPU时--cuda，另加了default=True
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
# parser.add_argument('--pretrained', default=r'', help="path to pretrained model (to continue training)")
parser.add_argument('--pretrained', default=r'./exprpy/netCRNN_188_73.pth', help="path to pretrained model (to continue training)")
# parser.add_argument('--alphabet', type=list)  alphabet词典 ,default='0123456789'，可以放到list里 用上面语句
# parser.add_argument('--alphabet', type=str, default='-1234abcdefghijklmnopqrstuvwxyz')
# parser.add_argument('--alphabet', type=str, default='0123456789')
parser.add_argument('--alphabet', type=str, default='abcdefghijklmnopqrstuvwxyzāáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ')
parser.add_argument('--expr_dir', default='exprpy', help='Where to store samples and models')   #输出模型路径
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=119, help='Interval to be displayed')  #源代码为14,即14次验证一次,此处241是根据训练时显示的数改的，实现一轮保留一次模型
parser.add_argument('--saveInterval', type=int, default=119, help='Interval to be displayed')  #设置多少次迭代保存一次模型，源代码14次
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# 以下为两个优化器 ，可以选  ，其中下一行设置default=True，等于默认执行文件时加 --adam
parser.add_argument('--adam', default=True, help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', default=True, help='whether to sample the dataset with random sampler')

#以上参数可以通过opt.名字访问
opt = parser.parse_args()
#输出各参数内容
print(opt)
# opt.alphabet = hypy_alphabet.alphabet()
# print(opt.alphabet)
if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset

#设置随机采样
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

#加载训练及测试集
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((opt.imgW, 32)))

#分类含空格+Mathorcuo_final
nclass = len(opt.alphabet) + 1
nc = 1

converter = utils1.strLabelConverter(opt.alphabet)
criterion = torch.nn.CTCLoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
#加载预训练
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    # crnn.load_state_dict(torch.load(opt.pretrained))  #预训练模型出现参数报错，将本行改为下列代码执行成功
    crnn.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.pretrained).items()})
# print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)
if opt.cuda:
    crnn.cuda()
    # 迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
# print(image)
text = Variable(text)
# print(text)
length = Variable(length)
# print(length)
# loss averager
loss_avg = utils1.averager()

# 两种优化器参数设置setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils1.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils1.loadData(image, cpu_images)
        t,l = converter.encode(cpu_texts)
        utils1.loadData(text, t)
        utils1.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # print(preds.size())
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils1.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils1.loadData(text, t)
    utils1.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#    print(preds.shape)
#    print(text.shape)
#    print(text.demo_data, length.demo_data)
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))


