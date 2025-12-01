import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



class ResidualBlock(nn.Module):
    def __init__(self, input_height, input_width, in_channels, out_channels, kernel_1, kernel_2):
        super().__init__()
        self.in_height = input_height
        self.in_width = input_width
        self.kernel_size = kernel_1
        self.kernel_size_2 = kernel_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding1=(self.kernel_size - 1)//2
        self.padding2=(self.kernel_size_2 -1)//2
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding1, dilation=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size_2, stride=1, padding=self.padding2, dilation=1, bias=True)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
         identity = self.shortcut(x)
         out = F.gelu(self.bn1(self.conv1(x)))
         out = self.bn2(self.conv2(out))
         out += identity
         return F.gelu(out)

    def get_output_shape(self):
        return self.in_height, self.in_width, self.out_channels

class EncoderLayer(nn.Module):

    def __init__(self, input_height, input_width, in_channels, out_channels, kernel, padding, stride):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size = self.kernel, padding=self.padding, stride=self.stride)

    def forward(self, x):
        x = F.gelu(self.conv(x))
        return x

    def get_output_shape(self):
        output_height = (self.input_height - self.kernel + 2 * self.padding) // self.stride + 1
        output_width = (self.input_width - self.kernel + 2 * self.padding) // self.stride + 1
        return output_height, output_width, self.out_channels

class DecoderLayer(nn.Module):

    def __init__(self, input_height, input_width, in_channels, out_channels, kernel, padding, out_padding, stride, tanh_activation=None):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.padding = padding
        self.out_padding = out_padding
        self.stride = stride
        self.tanh_activation = tanh_activation
        self.inv_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size = self.kernel, padding=self.padding, output_padding=self.out_padding, stride=self.stride)

    def forward(self, x):
        if self.tanh_activation:
            x = torch.tanh(self.inv_conv(x))
        else:
            x = F.gelu(self.inv_conv(x))
        return x

    def get_output_shape(self): # για transposed conv2d ο υπολογισμός των διαστάσεων της εξόδου γίνεται βάσει διαφορετικών τύπων
        output_height = (self.input_height - 1) * self.stride - 2 * self.padding + self.kernel + self.out_padding
        output_width = (self.input_width - 1) * self.stride - 2 * self.padding + self.kernel + self.out_padding
        #out​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        #Wout​=(Win​−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1
        return output_height, output_width, self.out_channels

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        return x

    def get_output_shape(self):
        return self.output_size


class BatchNormLayer(nn.Module):
    def __init__(self, input_height, input_width, channels):
        super().__init__()
        self.channels = channels
        self.input_height = input_height
        self.input_width = input_width
        self.bn = nn.BatchNorm2d(self.channels)

    def forward(self, x):
        x = self.bn(x)
        return x

    def get_output_shape(self):
        return self.input_height, self.input_width, self.channels

# autoencoder = input_height, input_width, input_channels, encoder_layers, fully_connected_layers, decoder_layers
# encoder_layers / decoder_layers = [[out_channels1, kernel1, padding1, stride, 'encoder_layer'], ['batchnorm'], [out_channels2, kernel1, kernel2, 'res_layer'], ['batchnorm'], ...]
# fully_connected_layers = [[output_size1], [output_size2], ...]
# decoder_layer = [out_channels, kernel, padding, out_padding, stride, tanh_activation]

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_height, input_width, input_channels, encoder_layers, fully_connected_layers, decoder_layers):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.encoder_layers = encoder_layers
        self.fully_connected_layers = fully_connected_layers
        self.decoder_layers = decoder_layers
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.fully_connected_section = nn.ModuleList()
        self.dimensions_memory = []
        self.train_loss_memory = []
        self.val_loss_memory = []
        for i in range(len(self.encoder_layers)): # κατασκευή του encoder
            if self.encoder_layers[i][-1] == 'encoder_layer':
              encoder_layer = EncoderLayer(self.input_height, self.input_width, self.input_channels, self.encoder_layers[i][0], self.encoder_layers[i][1], self.encoder_layers[i][2], self.encoder_layers[i][3])
              self.encoder.append(encoder_layer)
              self.input_height, self.input_width, self.input_channels = encoder_layer.get_output_shape()
              self.dimensions_memory.append([self.input_height, self.input_width, self.input_channels])
            elif self.encoder_layers[i][-1] == 'batchnorm':
              batchnorm_layer = BatchNormLayer(self.input_height, self.input_width, self.input_channels)
              self.encoder.append(batchnorm_layer)
              self.input_height, self.input_width, self.input_channels = batchnorm_layer.get_output_shape()
              self.dimensions_memory.append([self.input_height, self.input_width, self.input_channels])
            else:
              res_layer = ResidualBlock(self.input_height, self.input_width, self.input_channels, self.encoder_layers[i][0], self.encoder_layers[i][1], self.encoder_layers[i][2])
              self.encoder.append(res_layer)
              self.input_height, self.input_width, self.input_channels = res_layer.get_output_shape()
              self.dimensions_memory.append([self.input_height, self.input_width, self.input_channels])

        self.fully_connected_input_size = self.input_height * self.input_width * self.input_channels
        self.latent_size = self.fully_connected_input_size  #αρχικά η συγκεκριμένη μεταβλητή τίθεται ίση με self.fully_connected_input_size
        self.latent_height = self.input_height
        self.latent_width = self.input_width
        self.latent_channels = self.input_channels

        if len(self.fully_connected_layers) > 0:
            for i in range(len(self.fully_connected_layers)):
              fully_connected_layer = FullyConnectedLayer(self.fully_connected_input_size, self.fully_connected_layers[i][0])
              self.fully_connected_section.append(fully_connected_layer)
              self.fully_connected_input_size = fully_connected_layer.get_output_shape() # fully_connected_input_size μεταβάλλεται, latent_size μένει σταθερό
            self.fully_connected_section.append(FullyConnectedLayer(self.fully_connected_input_size, self.latent_size)) # project back στο latent size



        for i in range(len(self.decoder_layers)): # κατασκευή του decoder
            if self.decoder_layers[i][-1] == 'decoder_layer':
              decoder_layer = DecoderLayer(self.input_height, self.input_width, self.input_channels, self.decoder_layers[i][0], self.decoder_layers[i][1], self.decoder_layers[i][2], self.decoder_layers[i][3], self.decoder_layers[i][4], self.decoder_layers[i][5])
              self.decoder.append(decoder_layer)
              self.input_height, self.input_width, self.input_channels = decoder_layer.get_output_shape()
              self.dimensions_memory.append([self.input_height, self.input_width, self.input_channels])
            elif self.decoder_layers[i][-1] == 'batchnorm':
              batchnorm_layer = BatchNormLayer(self.input_height, self.input_width, self.input_channels)
              self.decoder.append(batchnorm_layer)
              self.input_height, self.input_width, self.input_channels = batchnorm_layer.get_output_shape()
              self.dimensions_memory.append([self.input_height, self.input_width, self.input_channels])

        self.encoder_sequence = nn.Sequential(*self.encoder)
        self.fc_sequence = nn.Sequential(*self.fully_connected_section)
        self.decoder_sequence = nn.Sequential(*self.decoder)



    def encode(self, x):
        x = self.encoder_sequence(x)
        return torch.flatten(x, 1)

    def fc_forward(self, x):
        x = self.fc_sequence(x)
        return x

    def decode(self, x):
        b = x.size(0)
        x = x.view(b, self.latent_channels, self.latent_height, self.latent_width)
        x = self.decoder_sequence(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.fc_forward(x)
        x = self.decode(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_out = self(x)
        loss = 0.5 * (x_out - x).pow(2).sum() / x.size(0)
        self.train_loss_memory.append(loss.item())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_out = self(x)
        loss = 0.5 * (x_out - x).pow(2).sum() / x.size(0)
        self.val_loss_memory.append(loss.item())
        self.log('val_loss', loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_latent_space_dimensions(self):
        return self.latent_height, self.latent_width, self.latent_channels

    def get_dimensions_memory(self):
        return self.dimensions_memory

def visualize_reconstructions(model, test_loader, n=10):
    model.eval()
    batch = next(iter(test_loader))
    x, _ = batch
    x = x[:n].to(model.device)
    with torch.no_grad():
        x_hat = model(x)

    mean = torch.tensor([0.5, 0.5, 0.5], device=model.device).view(1,3,1,1)
    std = torch.tensor([0.5, 0.5, 0.5], device=model.device).view(1,3,1,1)
    x = x * std + mean
    x_hat = x_hat * std + mean

    fig, axes = plt.subplots(n, 2, figsize=(2, n))
    for i in range(n):
        axes[i, 0].imshow(x[i].permute(1, 2, 0).cpu().clamp(0,1).numpy())
        axes[0, 0].set_title('x')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(x_hat[i].permute(1, 2, 0).cpu().clamp(0,1).numpy())
        axes[0, 1].set_title('D(E(x))')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()



class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples, channels, height, width):
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.channels, self.height, self.width) * 0.5
        return x, 0  # dummy label

def latent_search(model, test_loader, train_loader, n=20, k=3, metric='euclidean'):
    from sklearn.metrics.pairwise import cosine_distances
    model.eval()
    batch = next(iter(test_loader))
    x_test, _ = batch
    x_test = x_test[:n].to(model.device)

    with torch.no_grad():
        z_test = model.encode(x_test).cpu()
        train_latents = []
        train_images = []
        for xb, _ in train_loader:
            xb = xb.to(model.device)
            zb = model.encode(xb).cpu()
            train_latents.append(zb)
            train_images.append(xb.cpu())
        train_latents = torch.cat(train_latents)
        train_images = torch.cat(train_images)

        fig, axes = plt.subplots(n, k + 1, figsize=((k+1), n))
        for i in range(n):
            if metric == 'cosine':
                z_norm = z_test[i] / (z_test[i].norm(p=2) + 1e-8)
                train_norm = train_latents / (train_latents.norm(p=2, dim=1, keepdim=True) + 1e-8)
                dists = 1 - torch.mm(train_norm, z_norm.unsqueeze(1)).squeeze()
            elif metric == 'euclidean' or metric == 'l2':
                dists = torch.sqrt(torch.sum((train_latents - z_test[i]) ** 2, dim=1))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            nearest_idx = torch.topk(-dists, k).indices

            sample = x_test[i] * 0.5 + 0.5
            axes[i, 0].imshow(sample.permute(1, 2, 0).clamp(0,1).numpy())
            axes[0, 0].set_title('Test')
            axes[i, 0].axis('off')

            for j in range(k):
                neighbor = train_images[nearest_idx[j]] * 0.5 + 0.5
                axes[i, j+1].imshow(neighbor.permute(1, 2, 0).clamp(0,1).numpy())
                axes[0, j+1].set_title(f'k: {j+1}')
                axes[i, j+1].axis('off')
        plt.tight_layout()
        plt.show()
