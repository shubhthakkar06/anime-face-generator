import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Generator model (same architecture as used in training)
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Load Generator
@st.cache_resource
def load_generator(weights_path):
    netG = Generator()
    netG.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    netG.eval()
    return netG

# Generate image
def generate_image(model, nz=100):
    noise = torch.randn(1, nz, 1, 1)
    with torch.no_grad():
        fake_img = model(noise).detach().cpu()
    return fake_img

# Streamlit UI
st.title("Anime Face Generator 🎨")

model_path = "models/netG_epoch_20.pth"
netG = load_generator(model_path)

if st.button("Generate New Face"):
    image_tensor = generate_image(netG)
    grid = make_grid(image_tensor, normalize=True).permute(1, 2, 0).numpy()
    
    st.image(grid, caption="Generated Anime Face", use_column_width=True)
