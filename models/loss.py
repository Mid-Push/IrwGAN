import torch
import torch.nn as nn
import torch.nn.functional as F

class IRW_MS_LSGANLoss:
	def __init__(self, dis):
		self.dis = dis

	def gen_loss(self, fake, beta_fake, **kwargs):
		dis_fake = self.dis(fake, **kwargs)
		loss_fake = 0
		for dis_f in dis_fake:
			loss_fake += torch.mean(beta_fake * ((1.0-dis_f)**2))
		return loss_fake

	def dis_loss(self, real, beta_real, fake, beta_fake, **kwargs):
		dis_fake = self.dis(fake, **kwargs)
		dis_real = self.dis(real, **kwargs)
		loss_real = 0
		loss_fake = 0
		for dis_r, dis_f in zip(dis_real, dis_fake):
			loss_real += torch.mean(beta_real * ((1-dis_r)**2))
			loss_fake += torch.mean(beta_fake * (dis_f**2))
		return loss_real + loss_fake

def l2norm_loss(x):
	return torch.norm(x)


class IRW_L1_Loss(nn.Module):
	def __init__(self):
		super(IRW_L1_Loss, self).__init__()
	def forward(self, x, y, beta):
		beta = beta.view(len(beta), 1, 1, 1)
		loss = F.l1_loss(beta * x, beta * y)
		return loss


def irw_cycle_loss(x,y, beta):
	beta = beta.view(len(beta),1,1,1)
	loss = F.l1_loss(beta*x, beta*y)
	return loss


def irw_l1_loss(x,y, beta):
	beta = beta.view(len(beta),1,1,1)
	loss = F.l1_loss(beta*x, beta*y)
	return loss