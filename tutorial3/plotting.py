import torch
import numpy as np
import pandas as pd
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import utils
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})
#from .optimization import *

def plot_conv(image, filter):
    """Plot convs with matplotlib."""
    with torch.no_grad():
        d = filter.shape[-1]
        conv = torch.nn.Conv2d(1, 1, kernel_size=(d, d), padding=1)
        conv.weight[:] = filter
        fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
        #print(image.shape)
        ax1.imshow(image, cmap='gray')
        ax1.axis('off')
        ax1.set_title("Original")
        #print(image[None, None, :].shape)
        x = image[None, None, :].detach().squeeze(0)
        #print(x.shape)
        x_permute = x.permute(3,0,1,2)
        #print(conv(x_permute)[0].squeeze(0).shape)
        ax2.imshow(conv(x_permute)[1].squeeze(0), cmap='gray')  
        ax2.set_title("Filtered")
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig("plot_conv")
    
def plot_convs(image, conv_layer, axis=False):
    """Plot convs with matplotlib. Sorry for this lazy code :D"""
    with torch.no_grad():
        x= image[None, :]
        x_permute = x.permute(3,0,1,2)

        filtered_image = conv_layer(x_permute)
        n = filtered_image.shape[1]
        if n == 1:
            fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2)
            ax1.imshow(image, cmap='gray')
            ax1.set_title("Original")
            ax2.imshow(filtered_image[0].squeeze(0), cmap='gray')  
            ax2.set_title("Filter 1")
            ax1.grid(False)
            ax2.grid(False)
            if not axis:
                ax1.axis(False)
                ax2.axis(False)
            plt.tight_layout()
            plt.savefig("plot_convs")
        elif n == 2:
            filtered_image_1 = filtered_image[:,0,:,:]
            filtered_image_2 = filtered_image[:,1,:,:]
            fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 4), ncols=3)
            ax1.imshow(image, cmap='gray')
            ax1.set_title("Original")
            ax2.imshow(filtered_image_1[0].squeeze(0), cmap='gray')  
            ax2.set_title("Filter 1")
            ax3.imshow(filtered_image_2[0].squeeze(0), cmap='gray')  
            ax3.set_title("Filter 2")
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            if not axis:
                ax1.axis(False)
                ax2.axis(False)
                ax3.axis(False)
            plt.tight_layout()
            plt.savefig("plot_convs")
        elif n == 3:
            filtered_image_1 = filtered_image[:,0,:,:]
            filtered_image_2 = filtered_image[:,1,:,:]
            filtered_image_3 = filtered_image[:,2,:,:]
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(12, 4), ncols=4)
            ax1.imshow(image, cmap='gray')
            ax1.set_title("Original")
            ax2.imshow(filtered_image_1[0].squeeze(0), cmap='gray')  
            ax2.set_title("Filter 1")
            ax3.imshow(filtered_image_2[0].squeeze(0), cmap='gray')  
            ax3.set_title("Filter 2")
            ax4.imshow(filtered_image_3[0].squeeze(0), cmap='gray')  
            ax4.set_title("Filter 3")
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax4.grid(False)
            if not axis:
                ax1.axis(False)
                ax2.axis(False)
                ax3.axis(False)
                ax4.axis(False)
            plt.tight_layout()
            plt.savefig("plot_convs")