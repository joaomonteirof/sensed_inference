import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm


class TrainLoop(object):

	def __init__(self, generator, encoder, optimizer, train_loader, checkpoint_path=None, checkpoint_epoch=None, nadir_factor=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'e_checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = encoder
		self.generator = generator
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.history = {'loss': [], 'loss_minibatch': []}
		self.total_iters = 0
		self.cur_epoch = 0

		self.pool = torch.nn.MaxPool2d(2)

		self.A = 0

		'''
		A = torch.normal(torch.normal(means=torch.zeros(64*64, 64*64), std=torch.ones(64*64, 64*64)/(64*64)))

		if self.cuda_mode:
			A = A.cuda()

		self.A = Variable(A)
		'''

		if checkpoint_epoch is not None:
			self.load_checkpoint(checkpoint_epoch)

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			#self.scheduler.step()
			train_iter = tqdm(enumerate(self.train_loader))
			loss=0.0
			for t, batch in train_iter:
				new_loss = self.train_step(batch)
				loss+=new_loss
				self.total_iters += 1
				self.history['loss_minibatch'].append(new_loss)

			self.history['loss'].append(loss/(t+1))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		x, _ = batch

		if self.cuda_mode:
			x = x.cuda()

		y = self.pool(Variable(x))

		z = self.model.forward(y).unsqueeze(-1).unsqueeze(-1)

		out = self.generator.forward(z)

		#loss_e = self.compute_loss(out, x, z.squeeze())
		loss_e = F.mse_loss(self.pool(out), self.pool(x)) + 0.1*z.squeeze().norm(2)/z.numel()

		self.optimizer.zero_grad()
		loss_e.backward()
		self.optimizer.step()

		return loss_e.data[0]

	def compute_loss(self, out_, x_, z_):
		total_loss = 0
		for i in range(out_.size(0)):
			total_loss += ( torch.mm(self.A, out_[i].view(-1,1)) - x_[i].view(-1, 1) ).norm(2) + 0.1*z_[i].norm(2)
		return total_loss/(i+1)

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'A': self.A,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, epoch):

		ckpt = self.save_epoch_fmt_gen.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.A = ckpt['A']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!')
