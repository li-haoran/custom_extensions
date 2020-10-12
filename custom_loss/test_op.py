import time
import numpy as np
import torch
from torch.autograd import gradcheck,grad

from emd import emdModule,emd_function


def test_emd():
    x1 = torch.rand(512, 729, 2).cuda()
    x2 = torch.rand(512, 729, 2).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 0.05, 3000)
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis = 1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())


def check_emd():

    x1 = torch.rand(2, 729, 2).cuda()
    x2 = torch.rand(2, 729, 2).cuda()
    x1.requires_grad = True
    x1.retain_grad()
    x2.requires_grad = True
    x2.retain_grad()
    print('emd: ',
          gradcheck(emd_function, (x1,x2),
                    ))#eps=1e-3, atol=1e-4, rtol=1e-2

if __name__=='main__':
    test_emd()
    check_emd()