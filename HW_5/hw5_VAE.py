import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
import torch.nn.functional as Func
import matplotlib.pyplot as plt


num_epochs = 10
batch_size = 64


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
        ])
    output_address = './dataset'
    return datasets.MNIST(root=output_address, train=True, transform=compose, download=True)

# Load data
data = mnist_data()
# Minibatch setting
data_holder = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_mini_batches = len(data_holder)


# Convert images to  vectors
def img2vec(images):
    return images.view(images.size(0), -1)

# Convert vectors to images
def vec2img(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


class VAE(nn.Module):
    # Define Our NN
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_sig = nn.Linear(400, 20)

        self.fc2 = nn.Linear(20, 400)
        self.fc3 = nn.Linear(400, 784)
    # Define the Encoder
    def encoder(self, x):
        h1 = Func.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_sig(h1)

    def reparameterizer(self, mu, st_d):
        epsilon = torch.randn_like(st_d)
        return mu + st_d * epsilon
    # Define Decoder
    def decode(self, z):
        x = self.fc2(z)
        x = Func.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, st_d = self.encoder(x.view(-1, 784))
        z = self.reparameterizer(mu, st_d)
        return self.decode(z), mu, st_d


bce_loss = nn.BCELoss(reduction='sum')


def loss_function(reconstruct_x, x, mu, st_d):

    BCE = bce_loss(reconstruct_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + torch.log(st_d.pow(2)) - mu.pow(2) - st_d.pow(2))

    return BCE + KLD


vae = VAE()
Adam_optimizer = optim.Adam(vae.parameters(), lr=2e-4)

VAE_final_loss = [0] * num_epochs

for epoch in range(num_epochs):
    Final_loss_sum = 0
    for n_batch, (images, _) in enumerate(data_holder):
        N = images.size(0)
        images = Variable(img2vec(images))
        out, mu, st_d = vae(images)
        Final_Loss = loss_function(out, images, mu, st_d)
        # Total loss
        Final_loss_sum += Final_Loss.item()
        Adam_optimizer.zero_grad()
        Final_Loss.backward()
        Adam_optimizer.step()

        if (n_batch+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, num_epochs, n_batch+1, len(data)//batch_size, Final_Loss.item()))
    VAE_final_loss[epoch] = Final_loss_sum / len(data) / 784
    # mean loss

# generate random gaussian noise
Gaussian_signal_noise = torch.randn((16, 20))

path_vae = "./model/hw5_VAE.pth"
torch.save(vae, path_vae)

reloaded_VAE = torch.load(path_vae)
reloaded_VAE.eval()

x_hat = reloaded_VAE.decode(Gaussian_signal_noise)

x_hat = vec2img(x_hat)

utils.save_image(x_hat, "VAE.png", nrow=4)

plt.plot(VAE_final_loss, label='VAE_loss')
plt.legend()
plt.title('VAE Loss')
plt.show()
