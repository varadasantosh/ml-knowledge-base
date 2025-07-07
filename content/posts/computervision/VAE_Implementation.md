
---
title: "Variational AutoEncoder Implementation"
---

<!--more-->

# Import Libraries


```python
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

from torchvision import transforms
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader

```

# Device Info


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```




    device(type='cpu')



# Loading Dataset


```python
train_data = datasets.MNIST(root='./data', train=True, download=True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, download=True, transform = transforms.ToTensor())
```

    100%|██████████| 9.91M/9.91M [00:00<00:00, 16.2MB/s]
    100%|██████████| 28.9k/28.9k [00:00<00:00, 484kB/s]
    100%|██████████| 1.65M/1.65M [00:00<00:00, 4.49MB/s]
    100%|██████████| 4.54k/4.54k [00:00<00:00, 7.89MB/s]


# Linear VAE Model (W/O using Conv Layers)


```python
class LinearVAE(nn.Module):

  def __init__(self,input_dim, latent_dim):

    super().__init__()
    self.latent_dim = latent_dim
    self.encoder = nn.Sequential(

              nn.Flatten(),
              nn.Linear(input_dim , 156),
              nn.Tanh(),
              nn.Linear(156, 48),
              nn.Tanh()
    )

    self.mean_fc = nn.Sequential(

             nn.Linear(48, 16),
             nn.Tanh(),
             nn.Linear(16,latent_dim)
    )

    self.logvar_fc = nn.Sequential(

             nn.Linear(48, 16),
             nn.Tanh(),
             nn.Linear(16,latent_dim)
    )

    self.decoder = nn.Sequential(

              nn.Linear(latent_dim, 16),
              nn.Tanh(),
              nn.Linear(16, 48),
              nn.Tanh(),
              nn.Linear(48, 156),
              nn.Tanh(),
              nn.Linear(156, input_dim),
    )

    self._initialize_weights()

  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0) # Initialize biases to zero

  def encode(self, X):

      X = self.encoder(X)

      mu = self.mean_fc(X)
      sigma = self.logvar_fc(X)
      return mu, sigma

  def reparamterize(self, mu, sigma):

      epsilon = torch.rand_like(sigma)
      logvar = torch.exp(0.5* sigma)
      latent_sample = mu + epsilon * logvar
      return latent_sample


  def decode(self,latent_sample):

      return self.decoder(latent_sample)

  def forward(self, X):
      batch_size = X.shape[0]
      mu, sigma = self.encode(X)
      latent_sample = self.reparamterize(mu, sigma)
      X_reconstructed = self.decode(latent_sample)
      X_reconstructed = X_reconstructed.view(batch_size, 1, 28, 28)
      return mu, sigma, X_reconstructed

  def fit(self, epochs=10):

    train_loader = DataLoader(batch_size = 32, shuffle=True, dataset=train_data)
    test_loader = DataLoader(batch_size=32, shuffle=True, dataset=test_data)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)


    for epoch in range(epochs):
        reconstruct_loss_epoch= 0
        kl_loss_epoch = 0
        total_loss_epoch =0
        for batch, (X,y) in enumerate(train_loader):
            X = X.to(device)
            mu, sigma, X_reconstructed = self(X)
            reconstruction_loss = loss_fn(X_reconstructed, X)
            kl_loss = torch.mean(0.5* torch.sum(torch.exp(sigma) + mu**2 - 1 -sigma, dim=-1))
            loss = reconstruction_loss + 0.000001 * kl_loss
            reconstruct_loss_epoch = reconstruct_loss_epoch + reconstruction_loss.item()
            kl_loss_epoch = kl_loss_epoch + kl_loss.item()
            total_loss_epoch = total_loss_epoch + reconstruction_loss.item() + kl_loss.item()
            # reconstruct_losses.append(reconstruction_loss.item())
            # kl_losses.append(kl_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss_epoch = total_loss_epoch/len(train_loader)
        reconstruct_loss_epoch = reconstruct_loss_epoch/len(train_loader)
        kl_loss_epoch = kl_loss_epoch/len(train_loader)

        print(f'Epoch:- {epoch},Total Loss:-{total_loss_epoch} , Reconstrucion Loss after Epoch:-{reconstruct_loss_epoch}, KL Loss after Epoch:-{kl_loss_epoch} ')


  def save_model(self):

       checkpoint = {
                    'model_state_dict': vae_model.state_dict()
                    }
       torch.save(checkpoint,'linear_vae.pth')




```

# Trainining Loop


```python
vae_model = LinearVAE(784, 2)
vae_model = vae_model.to(device)

vae_model.fit(10)
```

    Epoch:- 0,Total Loss:-1.620043286939462 , Reconstrucion Loss after Epoch:-0.0916821592926979, KL Loss after Epoch:-1.5283611276467641 
    Epoch:- 1,Total Loss:-2.4466339523136615 , Reconstrucion Loss after Epoch:-0.05852550930778185, KL Loss after Epoch:-2.38810844300588 
    Epoch:- 2,Total Loss:-3.3052036174257595 , Reconstrucion Loss after Epoch:-0.05529772346417109, KL Loss after Epoch:-3.2499058939615884 
    Epoch:- 3,Total Loss:-4.257120461044709 , Reconstrucion Loss after Epoch:-0.05370230486591657, KL Loss after Epoch:-4.203418156178793 
    Epoch:- 4,Total Loss:-5.246161178302765 , Reconstrucion Loss after Epoch:-0.05191369269688924, KL Loss after Epoch:-5.194247485605875 
    Epoch:- 5,Total Loss:-6.198763153584798 , Reconstrucion Loss after Epoch:-0.05033046340942383, KL Loss after Epoch:-6.148432690175374 
    Epoch:- 6,Total Loss:-7.065699731987714 , Reconstrucion Loss after Epoch:-0.04910271024902662, KL Loss after Epoch:-7.016597021738688 
    Epoch:- 7,Total Loss:-7.844319595227639 , Reconstrucion Loss after Epoch:-0.048096689242124555, KL Loss after Epoch:-7.796222905985514 
    Epoch:- 8,Total Loss:-8.491182029853265 , Reconstrucion Loss after Epoch:-0.0473443729420503, KL Loss after Epoch:-8.443837656911214 
    Epoch:- 9,Total Loss:-8.97091471409003 , Reconstrucion Loss after Epoch:-0.04683623305161794, KL Loss after Epoch:-8.92407848103841 


# Original Image


```python
ori_img = test_data[223][0]
ori_img = ori_img.squeeze(0).numpy()

plt.imshow(ori_img)
```




    <matplotlib.image.AxesImage at 0x781be1e880d0>




    
![png](VAE_Implementation_files/VAE_Implementation_11_1.png)
    


# Reconstructed Image => Inference with Encoder + Latent + Decoder


```python
import matplotlib.pyplot as plt
img = test_data[223][0]
vae_model.eval()
mu, sigma,X_reconstructed = vae_model(img.unsqueeze(0).to(device))
X_reconstructed.shape

X_reconstructed = X_reconstructed.squeeze(0,1).detach().numpy()

plt.imshow(X_reconstructed)
```




    <matplotlib.image.AxesImage at 0x781be1d393d0>




    
![png](VAE_Implementation_files/VAE_Implementation_13_1.png)
    


# Images Generated from Gaussian Distribution => Inference with Decoder


```python
vae_model.eval()

num_samples = 16

latent_samples = torch.randn(num_samples, 2).to(device)

# Generate images using the decoder
generated_images = vae_model.decoder(latent_samples)

# Reshape the generated images to (batch_size, channels, height, width)
# Assuming grayscale images (1 channel) and 28x28 size
generated_images = generated_images.view(num_samples, 1, 28, 28)

# Create a grid of images
grid = torchvision.utils.make_grid(generated_images, nrow=4, padding=2) # Adjust nrow as needed

# Convert the grid tensor to a PIL Image and then to a NumPy array for displaying
grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()

# Display the grid of images
plt.imshow(grid_np, cmap='gray') # Use cmap='gray' for grayscale images
plt.axis('off') # Hide axes
plt.show()
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-1.1588097..1.2557548].



    
![png](VAE_Implementation_files/VAE_Implementation_15_1.png)
    

