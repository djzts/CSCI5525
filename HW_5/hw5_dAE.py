import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
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
# setting the Minibatch
data_holder = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
# count the number of batches
num_mini_batches = len(data_holder)

# Convert images to  vectors
def img2vec(images):
    return images.view(images.size(0), -1)

# Convert vectors to images
def vec2img(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


ten_imgs = data.data[:10]
data_iter = iter(data_holder)
X_tensor, _ = next(data_iter)
# get ten digits for test
X_tensor = X_tensor[:5]

noise = torch.randn(X_tensor.size(0), 1, 28, 28)
# add noise to test samples
X_tensor += noise

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
            )


    def forward(self, x):
        output_value = self.encoder(x)
        output_value = self.decoder(output_value)
        return output_value


Auto_Encoder = Autoencoder()

# we are going to use BCE loss and Adam Adam_optimizer to get the result
loss = nn.BCELoss()
loss_sum = nn.BCELoss(reduction='sum')
Adam_optimizer = optim.Adam(Auto_Encoder.parameters(), lr=2e-4)

loss_epoch = [0] * num_epochs

for epoch in range(num_epochs):
    Final_loss_sum = 0
    for n_batch, (images, _) in enumerate(data_holder):
        N = images.size(0)
        images = Variable(images.view(images.size(0), -1))
        output_value = Auto_Encoder(images)
        Final_Loss = loss(output_value, images)
        # Total loss
        Final_loss_sum += loss_sum(output_value, images)

        Adam_optimizer.zero_grad()
        Final_Loss.backward()
        Adam_optimizer.step()

        if (n_batch+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.5f'
                % (epoch+1, num_epochs, n_batch, len(data)//batch_size, Final_Loss.item()))
    # Average loss
    loss_epoch[epoch] = Final_loss_sum / len(data) / 784

path = "./model/hw5_dAE.pth"
torch.save(Auto_Encoder, path)
reloaded_ae = torch.load(path)


recoverd = reloaded_ae(img2vec(X_tensor))
d = torch.cat((X_tensor, vec2img(recoverd)))
utils.save_image(d, "dAE.png", nrow=5)

plt.plot(loss_epoch)
plt.xlabel("epoch")
plt.ylabel("dAE_loss")
plt.show()
