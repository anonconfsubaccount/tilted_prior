import torch
import torch.nn as nn
from torch.autograd import Variable
import util

class Loss(nn.Module):
    def __init__(self, loss_type, tilt, nz):
        super(Loss, self).__init__()
        if tilt != None:
            print('optimizing for min kld')
            self.mu_star = util.kld_min(tilt, nz)
            print('mu_star: {:.3f}'.format(self.mu_star))
        else:
            self.mu_star = None

        self.nz = nz
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type 
        if loss_type != 'l2' and loss_type != 'cross_entropy':
            raise ValueError('{} is not a valid loss type, choose either l2 or cross_entropy')

    def forward(self, x, x_out, mu, logvar, ood=False):
        # recon loss options
        if self.loss_type == 'l2':
            recon = torch.sum(torch.square(x - x_out), dim=(1,2,3))
            if not ood: # batch support for aucroc testing
                recon = torch.mean(recon) 
        elif self.loss_type == 'cross_entropy':    
            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()
            out = x_out.contiguous().view(-1,256)
            recon = self.ce_loss(out, target)
            recon = torch.sum(recon) / b
            if ood: # batch support for aucroc testing
                print('not implimented yet')
                import sys
                sys.exit()

        # kld loss options
        if self.mu_star == None:
            kld = -1/2*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), (1))
        else:
            mu_norm = torch.linalg.norm(mu, dim=1)
            kld = 1/2*torch.square(mu_norm - self.mu_star)
        if not ood:
            kld = torch.mean(kld)

        return recon, kld
