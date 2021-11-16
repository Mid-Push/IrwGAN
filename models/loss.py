import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_gan_loss(gan_type):
    if gan_type == 'lsgan':
        return IRW_MS_LSGANLoss
    elif gan_type == 'logistic':
        return IRW_MS_LogisticLoss
    elif gan_type == 'wgangp':
        return IRW_MS_WGANGPLoss
    else:
        raise ValueError('gan_type: %s is not supported...' % gan_type )

class IRW_MS_WGANGPLoss:
    def __init__(self, dis, threshold, wgan_lambda=10, wgan_target=1.0,epsilon=1e-3):
        self.wgan_lambda = wgan_lambda
        self.wgan_target = wgan_target
        self.eps = epsilon
        self.dis = dis
        self.threshold = threshold

    def gen_loss(self,fake, beta_fake, **kwargs):
        dis_fake = self.dis(fake, **kwargs)
        loss = 0
        for dis_f in dis_fake:
            loss += (-torch.mean(beta_fake * dis_f))
        return loss

    def dis_loss(self, real, beta_real, fake, beta_fake,**kwargs):
        dis_real = self.dis(real, **kwargs)
        dis_fake = self.dis(fake, **kwargs)
        beta_real = torch.nn.functional.threshold(beta_real, self.threshold, 0.0)
        loss = 0
        i = 0
        for dis_r, dis_f in zip(dis_real, dis_fake):
            loss += (dis_f - beta_real * dis_r)
            # gradient penalty
            size = len(real)
            eps = torch.rand(size,1,1,1).to(real.device)
            x_hat = eps * real.data + (1-eps) * fake.data
            x_hat.requires_grad = True
            dis_hat = self.dis(x_hat, **kwargs)[i]
            size = np.prod(dis_hat.size()[1:])
            grad_x_hat = torch.autograd.grad(dis_hat.sum()/size, inputs=x_hat, create_graph=True)[0]
            grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0),-1).norm(2,dim=1)-self.wgan_target)**2)
            grad_penalty = self.eps * grad_penalty/(self.wgan_target**2)
            # additional epsilon penalty by NVIDIA
            epsilon_penalty = self.eps * (dis_r**2)
            loss = torch.mean(loss + grad_penalty+epsilon_penalty)
            i += 1
        return loss

class IRW_MS_LogisticLoss:
    def __init__(self, dis, threshold, G_saturate=False,D_gp='r1', gamma=10):
        self.sat = G_saturate
        self.gp = D_gp
        self.gamma = gamma
        self.dis = dis
        self.threshold = threshold

    def gen_loss(self, fake, beta_fake, **kwargs):
        dis_fake = self.dis(fake, **kwargs)
        loss_fake = 0
        for dis_f in dis_fake:
            if self.sat:
                loss_fake += beta_fake * (-F.softplus(dis_f))
            else:
                loss_fake += beta_fake * (F.softplus(-dis_f))
        return torch.mean(loss_fake)

    def dis_loss(self,real, beta_real, fake, beta_fake, **kwargs):
        real.requires_grad = True
        dis_fake = self.dis(fake, **kwargs)
        dis_real = self.dis(real, **kwargs)
        loss = 0
        beta_real = torch.nn.functional.threshold(beta_real, self.threshold, 0.0)
        for dis_r, dis_f in zip(dis_real, dis_fake):
            loss += F.softplus(dis_f)
            loss += beta_real * F.softplus(-dis_r)
            if self.gp in ['r1', 'r2']:
                if self.gp == 'r1':
                    size = np.prod(dis_r.size()[1:])
                    grad = torch.autograd.grad(outputs=dis_r.sum()/size, inputs=real, create_graph=True)[0]
                elif self.gp == 'r2':
                    size = np.prod(dis_f.size()[1:])
                    grad = torch.autograd.grad(outputs=dis_f.sum()/size, inputs=fake, create_graph=True)[0]
                grad_penalty = (grad.view(grad.size(0), -1).norm(2,dim=1)**2)
                loss += grad_penalty * self.gamma/2.0
        return torch.mean(loss)

class IRW_MS_LSGANLoss:
    def __init__(self, dis, threshold):
        self.dis = dis
        self.threshold = threshold # if beta<threshold, we set it 0, other betas are different

    def gen_loss(self, fake, beta_fake, **kwargs):
        """
        We ignore beta_fake whose values are below the threshold
        as we think the noise images should not be changed.
        """
        dis_fake = self.dis(fake, **kwargs)
        loss_fake = 0
        loss_unweight = 0
        beta_fake = torch.nn.functional.threshold(beta_fake, self.threshold, 0.0)
        for dis_f in dis_fake:
            assert len(beta_fake) == len(dis_f)
            loss = torch.mean((1-dis_f)**2)
            loss_fake += beta_fake * loss
            loss_unweight += loss.detach()
        return loss_fake, loss_unweight

    def dis_loss(self, real, beta_real, fake, beta_fake, **kwargs):
        """
        We ignore beta_fake as we wish discriminator can distinguish fake samples even they are from noise
        We use beta_real to reweight target domain as we want it to be clean domain.
        """
        dis_fake = self.dis(fake, **kwargs)
        dis_real = self.dis(real, **kwargs)
        loss_real = 0
        loss_fake = 0
        for dis_r, dis_f in zip(dis_real, dis_fake):
            beta_real = torch.nn.functional.threshold(beta_real, self.threshold, 0.0)
            assert len(beta_real) == len(dis_r)
            loss_real += torch.mean(beta_real * ((1-dis_r)**2))
            loss_fake += torch.mean((dis_f**2))
        return loss_real + loss_fake

class IRW_L1_Loss(nn.Module):
    def __init__(self, threshold):
        super(IRW_L1_Loss, self).__init__()
        self.threshold = threshold

    def forward(self, x, y, beta):
        beta = beta.view(len(x), 1, 1, 1)
        beta = torch.nn.functional.threshold(beta, self.threshold, 0.0)
        assert len(beta) == len(x)
        loss = torch.mean(torch.abs(beta*x-beta*y))
        return loss


def irw_cycle_loss(x,y, beta):
    beta = beta.view(len(beta),1,1,1)
    loss = F.l1_loss(beta*x, beta*y)
    return loss


def irw_l1_loss(x,y, beta):
    beta = beta.view(len(beta),1,1,1)
    loss = F.l1_loss(beta*x, beta*y)
    return loss