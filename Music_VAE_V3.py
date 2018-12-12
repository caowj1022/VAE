import sys
import os
sys.path.append('/usr/local/lib/python2.7/site-packages')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils import data

import numpy as np

from pypianoroll import Multitrack, Track
import pypianoroll

class VAE(nn.Module):
	def __init__(self, input_size, hidden_size_1, hidden_size_2, z_dimension):
		super(VAE, self).__init__()

		self.hidden_size_1 = hidden_size_1
		self.hidden_size_2 = hidden_size_2
		self.z_dimension = z_dimension

		self.lstm1 = nn.LSTM(input_size, hidden_size_1, num_layers = 2, batch_first = True, bidirectional = True)

		self.fc1_1 = nn.Linear(2*self.hidden_size_1, z_dimension)
		self.fc1_2 = nn.Linear(2*self.hidden_size_1, z_dimension)

		self.fc2_1 = nn.Linear(self.z_dimension, 2*self.hidden_size_2)
		self.fc2_2 = nn.Linear(self.z_dimension, 2*self.hidden_size_2)

		self.lstm2 = nn.LSTM(input_size, self.hidden_size_2, num_layers = 2, batch_first = True)

		self.fc3_1 = nn.Linear(self.hidden_size_2, input_size)
		self.hidden1 = self.init_hidden()

		self.tanh = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim = 2)

	def init_hidden(self):
		return torch.zeros(2*2, 1, self.hidden_size_1), torch.zeros(2*2, 1, self.hidden_size_1)


	def reparameterize(self, mu, log_var):
		if self.training:
			std = torch.exp(0.5*log_var)
			e = torch.randn_like(std)
			return e.mul(std).add_(mu)

		else:
			return mu

	def encode(self, x):
		out, self.hidden1 = self.lstm1(x, self.hidden1)
		out = torch.cat((out[:, -1, :self.hidden_size_1], out[:, 0, -self.hidden_size_1:]), 1)
		mu = self.fc1_1(out)
		log_var = self.fc1_2(out)
		return mu, log_var

	def decode(self, z, x):
		recon = []
		h = self.tanh(self.fc2_1(z))
		c = self.tanh(self.fc2_2(z))
		h = h.view(-1, 2, self.hidden_size_2)
		c = c.view(-1, 2, self.hidden_size_2)
		h = h.permute(1, 0, 2)
		c = c.permute(1, 0, 2)
		self.hidden2 = h, c
		out, self.hidden2 = self.lstm2(x, self.hidden2)
		recon = self.fc3_1(out)

		return self.sigmoid(recon)

	def forward(self, x):
		mu, log_var = self.encode(x)
		z = self.reparameterize(mu, log_var)
		recon = self.decode(z, x)

		return recon, mu, log_var

class NotesDataset(data.Dataset):
	def __init__(self, folder_path):
		self.folder_path = folder_path
		filenames = os.listdir(folder_path)
		full_filenames = map(lambda filename: os.path.join(folder_path, filename), filenames)
		self.full_filenames = full_filenames

	def __len__(self):
		return len(self.full_filenames)

	def __getitem__(self, index):
		full_filename = self.full_filenames[index]

		multitrack = Multitrack(full_filename)

		piano_roll = multitrack.get_merged_pianoroll(mode = 'any')[::16, :]
		return torch.FloatTensor(piano_roll.astype(float))

trainset = NotesDataset('Piano-midi.de/train/')
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)


def loss_function(y, x, mu, log_var):
	BCE = F.binary_cross_entropy(y, x, reduction = 'sum')
	KL = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
	print 'BCE: ', BCE, '	KL: ', KL

	return BCE + KL


def train(epoch):
	model.train()
	train_loss = 0

	for batch_index, data in enumerate(train_loader):
		optimizer.zero_grad()
		recon, mu, log_var = model(data)
		loss = loss_function(recon, data, mu, log_var)
		loss.backward(retain_graph = True)
#		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_index % 1 == 0:
			print  'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index*len(data), len(train_loader.dataset), 100.*batch_index/len(train_loader), loss.item()/len(data))
			track = Track(pianoroll = recon.view(recon.shape[1], -1).detach().numpy())
			track.binarize(0.2)
			track = pypianoroll.Multitrack(tracks = [track], tempo = 30)
			track.write('sample/sample_%d.mid' % (batch_index))
	print '-----------------------------'
	print 'Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset))


if __name__ == '__main__':

	model = VAE(128, 2048, 1024, 512)
	optimizer = optim.Adam(model.parameters(), lr = 1e-3)
	for epoch in range(1, 2):
		train(epoch)







