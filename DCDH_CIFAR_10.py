import pickle
import os
import argparse
import logging
import torch
import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.dcdh_loss as al
import utils.cnn_model as cnn_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

parser = argparse.ArgumentParser(description="DCDH demo")
parser.add_argument('--bits', default='48', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--arch', default='alexnet', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=160, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--num-samples', default=3000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--lamda', default=2, type=float)
parser.add_argument('--mu', default=1, type=float)

def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return

def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingNUS_WIDE(
        'data/CIFAR-10', 'database_img.txt', 'database_label_z.txt', transformations
    )
    dset_test = dp.DatasetProcessingNUS_WIDE(
        'data/CIFAR-10', 'test_img.txt', 'test_label_z.txt', transformations
    )
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        label_filepath = os.path.join(DATA_DIR, filename)
        label = np.loadtxt(label_filepath, dtype=np.int64)
        return torch.from_numpy(label)

    databaselabels = load_label('database_label_z.txt', 'data/CIFAR-10')
    testlabels = load_label('test_label_z.txt', 'data/CIFAR-10')

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

def calc_sim(train_label):
    S = (train_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(V, U, S, code_length, select_index, gamma):
    num_database = V.shape[0]
    square_loss = (U.dot(U.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss

def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 30, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def DCDH_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma

    record['param']['topk'] = 5000
    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    DCDH_loss = al.DCDHLoss(gamma, code_length, num_database)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    labelNet = cnn_model.MLP(code_length, 10)
    labelNet.cuda()
    label_loss = al.DCDHLoss(gamma, code_length, num_database)
    optimizer2 = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    product_model = cnn_model.ConcatMLP(code_length, 10)
    product_model.cuda()
    product_loss = al.ProductLoss(gamma, code_length, num_samples)
    optimizer3 = optim.Adam(product_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    model.train()
    labelNet.train()
    product_model.train()

    for iter in range(max_iter):
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label)

        U = np.zeros((num_samples, code_length), dtype=np.float)
        L = np.zeros((num_samples, code_length), dtype=np.float)
        I = np.zeros((num_samples, code_length), dtype=np.float)

        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                outputL = labelNet(train_label.type(torch.FloatTensor).cuda())
                S = calc_sim(train_label)
                U[u_ind, :] = output.cpu().data.numpy()
                L[u_ind, :] = outputL.cpu().data.numpy()
                #
                semanCode = outputL.clone().detach().requires_grad_(True)
                imgCode = output.clone().detach().requires_grad_(True)
                product = torch.einsum('bi,bj->bij', semanCode, imgCode)
                product = product.reshape(batch_size_, code_length * code_length)

                hashcode, classify = product_model(product.cuda())
                I[u_ind, :] = hashcode.cpu().data.numpy()

                model.zero_grad()
                labelNet.zero_grad()
                product_model.zero_grad()

                loss3 = product_loss(hashcode, V, S, V[batch_ind.cpu().numpy(), :], classify, train_label, imgCode,
                                     semanCode)
                loss2 = label_loss(output, V, S, V[batch_ind.cpu().numpy(), :])

                loss = DCDH_loss(output, V, S, V[batch_ind.cpu().numpy(), :]) + opt.lamda * loss2 + opt.mu * loss3

                loss.backward()
                optimizer.step()
                optimizer2.step()
                optimizer3.step()

        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''
        Q = -2 * code_length * Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * U

        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            V_ = V_[select_index, :]
            Uk = U[:, k]
            U_ = U[:, sel_ind]

            V[select_index, k] = -np.sign(
                (Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk))) + opt.lamda * L[:,k] + opt.mu * I[:,k])

        iter_time = time.time() - iter_time
        loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma)
        logger.info('[Iteration: %3d/%3d][Train Loss: %.4f]', iter, max_iter, loss_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

        '''
        training procedure finishes, evaluation
        '''
        if iter % 10 == 9:
            model.eval()
            testloader = DataLoader(dset_test, batch_size=1,
                                     shuffle=False,
                                     num_workers=4)
            qB = encode(model, testloader, num_test, code_length)
            rB = V
            map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
            topkmap = calc_hr.calc_topMap(qB, rB, test_labels.numpy(), database_labels.numpy(), record['param']['topk'])
            logger.info('[Evaluation: mAP: %.4f, top-%d mAP: %.4f]', map, record['param']['topk'], topkmap)
            record['rB'] = rB
            record['qB'] = qB
            record['map'] = map
            record['topkmap'] = topkmap
            filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

            _save_record(record, filename)


if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/log-DCDH-Flickr', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        DCDH_algo(bit)
