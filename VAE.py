import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

batch_size = 32
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()), batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()), batch_size = batch_size, shuffle=True)

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(784, 400)
		self.fc2_1 = nn.Linear(400, 50)
		self.fc2_2 = nn.Linear(400, 50)

		self.fc3 = nn.Linear(50, 400)
		self.fc4 = nn.Linear(400, 784)

	def encode(self, x):

		x = F.relu(self.fc1(x))
		return self.fc2_1(x), self.fc2_2(x)

	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		x = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(x))

	def forward(self, x):
		mu, log_var = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, log_var)
		return self.decode(z), mu, log_var

def loss_function(y, x, mu, log_var):
	BCE = F.binary_cross_entropy(y, x.view(-1, 784), reduction = 'sum')
	KL = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

	return BCE + KL

def train(epoch):
	model.train()
	train_loss = 0

	for batch_index, (data, _) in enumerate(train_loader):
		optimizer.zero_grad()
		recon, mu, log_var = model(data)
		loss = loss_function(recon, data, mu, log_var)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		if batch_index % 10 == 0:
			print  ('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index*len(data), len(train_loader.dataset), 100.*batch_index/len(train_loader), loss.item()/len(data)))
	print ('-----------------------------')
	print ('Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss/len(train_loader.dataset)))

def test(epoch):
	model.eval()
	test_loss = 0

	with torch.no_grad():
		for batch_index, (data, _) in enumerate(test_loader):
			recon, mu, log_var = model(data)
			loss = loss_function(recon, data, mu, log_var)
			test_loss += loss.item()
			if batch_index == 0:
				img = torch.cat([data, recon.view(batch_size, 1, 28, 28)])
				save_image(img, 'test_img/epoch_'+str(epoch)+'.png', nrow = batch_size)

		test_loss /= len(test_loader.dataset)
		print ('-----------------------------')
		print ('Epoch: {} Test Loss: {:.4f}'.format(epoch, test_loss))
	
if __name__ == '__main__':
	model = VAE()
	optimizer = optim.Adam(model.parameters(), lr = 2e-4, betas = (0.5, 0.999))
	for epoch in range(1, 21):
		train(epoch)
		test(epoch)
		with torch.no_grad():
			sample = torch.randn(batch_size, 50)
			sample = model.decode(sample)
			save_image(sample.view(batch_size, 1, 28, 28), 'sample_img/epoch_'+str(epoch)+'.png')
	torch.save(model.state_dict(), 'model/VAE_z_50')

