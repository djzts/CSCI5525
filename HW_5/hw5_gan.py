import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
import torch.nn.functional as Func
import matplotlib.pyplot as plt


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
data_holder = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_holder)

# Convert images to single tensor
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# Convert vectors to images
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# create var vectors 0
def target_make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data

# create var vectors 1
def target_make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data

# Gen random white noise
def noise(size):
    noise_signal = Variable(torch.randn(size, 128))
    return noise_signal

# Define Discriminator
class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = Func.leaky_relu(x, 0.2, inplace=True)
        x = self.fc2(x)
        x = Func.leaky_relu(x, 0.2, inplace=True)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

# Define Generator
class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)

    def forward(self, x):
        x = self.fc1(x)
        x = Func.leaky_relu(x, 0.2, inplace=True)
        x = self.fc2(x)
        x = Func.leaky_relu(x, 0.2, inplace=True)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

# Train Discriminator
def train_discriminator(optimizer, real_data, Generator_data):
    shape_n = real_data.size(0)
    optimizer.zero_grad()

    # Train on real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, target_make_ones(shape_n))
    error_real.backward()

    # Train on generate Data
    prediction_gen = discriminator(Generator_data)
    # Calculate error and backpropagate
    error_gen = loss(prediction_gen, target_make_zeros(shape_n))
    error_gen.backward()
    optimizer.step()

    return error_real + error_gen

# Train Generator
def train_generator(optimizer, Generator_data):
    shape_n = Generator_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate data
    prediction = discriminator(Generator_data)
    # Calculate error and backpropagate
    error = loss(prediction, target_make_ones(shape_n))
    error.backward()
    optimizer.step()
    return error


discriminator = DiscriminatorNet()
generator = GeneratorNet()

# Use adam optimizer
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)


loss = nn.BCELoss(reduction='sum')

# generate 16 test examples
num_test_samples = 16
test_noise = noise(num_test_samples)


#Train all epochs in train data
train_times = 50
Gen_loss_final = [0] * train_times
Dis_loss_final = [0] * train_times
for epoch in range(train_times):
    Generator_loss_sum = 0
    Discriminator_loss_sum = 0
    for n_batch, (images, _) in enumerate(data_holder):
        shape_n = images.size(0)
        # Train Discriminator
        real_data = Variable(images_to_vectors(images))
        # Generate GN and data then detach
        Generator_data = generator(noise(shape_n)).detach()
        Discriminator_loss = train_discriminator(d_optimizer, real_data, Generator_data)

        Discriminator_loss_sum += Discriminator_loss

        # Train Generator
        Generator_data = generator(noise(shape_n))  # Generate data
        Generator_loss = train_generator(g_optimizer, Generator_data)

        Generator_loss_sum += Generator_loss

    Gen_loss_final[epoch] = -1 * Generator_loss_sum / len(data)  # get average loss and get take minus sign off
    Dis_loss_final[epoch] = -1 * Discriminator_loss_sum / len(data)

    print("{} epochs. Generator loss: {}. Discriminator loss: {} "
          .format(epoch + 1, Generator_loss, Discriminator_loss))
    if (epoch+1) % 10 == 0:
        test_images = vectors_to_images(generator(test_noise))
        test_images = test_images.data

        utils.save_image(test_images, "reconst%s.png" % epoch, nrow=4)


path_gen = "./model/hw5_gan_gen.pth"
path_dis = "./model/hw5_gan_dis.pth"
torch.save(generator, path_gen)
torch.save(discriminator, path_dis)

plt.plot(Dis_loss_final, label='Discriminator_loss')
plt.plot(Gen_loss_final, label='Generator_loss')
plt.legend()
plt.title('Loss of GAN')
plt.show()
