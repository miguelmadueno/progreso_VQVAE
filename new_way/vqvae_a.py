# Code modified and commented by Iqbal Chaudhry Mora (UC3M-TSC-GTS)
# The orginal code was written and commented by Rodrigo Oliver Coimbra (UC3M-TSC-GTS).
# The Quantizer class was originally written by
# Diego Quevedo Herrero (UC3M-NYU) and modified by
# Rodrigo Oliver Coimbra.


import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from collections import OrderedDict


class Quantizer(nn.Module):

    """
    The Quantizer module implements the vector quantization layer used in VQ-VAE.
    It maintains a codebook of embeddings and maps input vectors to the nearest embedding.

    Attributes:
        embed_dim (int): Dimensionality of each embedding vector.
        num_embed (int): Number of embeddings in the codebook.
        decay (float): Decay rate for the exponential moving average updates.
        threshold (float): Threshold for embedding updates.
        eps (float): Small value to prevent division by zero.
        embed (torch.Tensor): The codebook embeddings.
        cluster_size (torch.Tensor): Counts of how many times each embedding is selected.
        embed_mean (torch.Tensor): Sum of input vectors assigned to each embedding.
    """

    def __init__(self, embed_dim, num_embed, decay, threshold, eps=1e-5):
        
        """
        Initializes the Quantizer module with the given parameters.

        Args:
            embed_dim (int): Dimensionality of each embedding vector
            num_embed (int): Number of embeddings in the codebook.
            decay (float): Decay rate for the exponential moving average updates.
            threshold (float): Threshold for embedding updates.
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-5.
        """

        super().__init__()

        # Initialize quantizer parameters
        self.embed_dim = embed_dim
        self.num_embed = num_embed
        self.decay = decay
        self.threshold = threshold
        self.eps = eps

        # Initialize embeddings properties with random values
        embed = torch.randn(embed_dim, num_embed)

        # Register buffers for embeddings and their statistics.
        # These buffers are not updadted via backpropagation.
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embed))
        self.register_buffer("embed_mean", embed.clone())

    def forward(self, input: torch.Tensor, return_info=False):

        """
        Forward pass of the Quantizer. Maps input vectors to the nearest embeddings.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, embed_dim, sequence_length).

        Returns:
            quantize (torch.Tensor): Quantized tensor with embeddings replacing input vectors.
            diff (torch.Tensor): Average squared distance betwe–en quantized vectors and inputs.
            embed_ind (torch.Tensor): Indices of the selected embeddings for each input vector.
            embedding_info (dict): Dictionary containing ranking and distance information of the embeddings.
        """

        # Flatten batch inputs (encoder outputs) to shape (batch_size * sequence_length, embed_dim)
        flatten = input.reshape(-1, self.embed_dim)

        # Compute squared Euclidian distance between input vectors and embeddings
        dist = (
            flatten.pow(2).sum(1, keepdim = True) # ||x||^2
            -2 * flatten @ self.embed # -2*x.e
            + self.embed.pow(2).sum(0, keepdim = True) # ||e||^2
        )

        # Get the indices of the nearest embeddings
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embed).type(flatten.dtype)

        # Quantize the input by replacing with nearest embeddings
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        embedding_info = None
        if return_info:
    
            # Compute the Euclidian distances
            euclidian_dist = torch.sqrt(dist + self.eps)
            # Compute normalized distances (pseudo-probabilities) using Softmax
            pseudo_probs = torch.softmax(-euclidian_dist, dim=1)

            # Find indices of the nearest embeddings
            sorted_distances, sorted_indices = torch.sort(euclidian_dist, dim=1)
            sorted_probs = torch.gather(pseudo_probs, 1, sorted_indices)

            embedding_info = {
                "embed_id": sorted_indices.view(*input.shape[:-1], self.num_embed),
                "eu_dist": sorted_distances.view(*input.shape[:-1], self.num_embed),
                "pseudo_probs": sorted_probs.view(*input.shape[:-1], self.num_embed)
            }

        # Update embeddings using Exponential Moving Average (EMA) during training
        if self.training:
            # Compute the number of times each embedding is selected
            embed_onehot_sum = embed_onehot.sum(0)

            # Compute the sum of inputs assigned to each embedding
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # Update moving averages for cluster size and embed_mean
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_mean.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Normalize cluster sizes to prevent division by zero
            # Compute stable total count
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embed * self.eps) * n
            )

            # Compute representation mean per embedding
            pool = torch.tile(flatten, (int(np.ceil(self.num_embed / flatten.size(0))), 1))
            
            rand_indices = torch.randperm(pool.size(0), device=flatten.device)[:self.num_embed]
            rand_embed = pool[rand_indices, :].transpose(0, 1)

            usage = (self.cluster_size >= self.threshold).float()
            embed_normalized = self.embed_mean / cluster_size.unsqueeze(0)
            embed = usage * embed_normalized + (1 - usage) * rand_embed
            self.embed.data.copy_(embed)

        # Compute the average distance between quantization and input
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach() # Straight-through estimator

        return quantize, diff, embed_ind, embedding_info

class Encoder(nn.Module):

    """
    The Encoder module transforms input signals into a latent representation suitable for quantization.

    Attributes:
        num_layers (int): Number of convolutional layers.
        mask_flag (int): Flag indicating the use of masking.
        input_dim (int): Number of input channels/features.
        relu (nn.ReLU): ReLU activation function.
        dropout (nn.Dropout1d): Dropout layer for regularization.
        pre_mask (nn.Sequential): Pre-processing layers for mask input (if applicable).
        conv1, conv2, conv3, conv4, conv5, conv6 (nn.Conv1d): Convolutional layers.
        bn1, bn2, bn3, bn4, bn5, bn6 (nn.BatchNorm1d): Batch normalization layers.
        conv_layers (nn.Sequential): Sequential container of convolutional layers.
    """
    
    def __init__(self, input_dim, output_dim, num_layers,
                conv_dims, kernel_sizes, strides, p, mask_flag):
        
        """
        Initializes the Encoder module with the given parameters.

        Args:
            input_dim (int): Number of input channels/features.
            output_dim (int): Output dimensionality after encoding.
            num_layers (int): Number of convolutional layers.
            conv_dims (list of int): List specifying the number of channels for each conv layer.
            kernel_sizes (list of int): List specifying the kernel size for each conv layer.
            strides (list of int): List specifying the stride for each conv layer.
            p (float): Dropout probability.
            mask_flag (int): Flag indicating the use of masking.
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mask_flag = mask_flag

        #self.num_layers = num_layers
        self.conv_dims = conv_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p)

        
        # Handle a base case for the Encoder

        # if (len(self.conv_dims) == 1) & (self.num_layers > 1):
        #     self.conv_dims *= (self.num_layers - 1 ) # Last layer is defined by output_dim
        
        # if (len(self.kernel_sizes) == 1) & (self.num_layers > 1):
        #     self.kernel_sizes *= self.num_layers
        
        # if (len(self.strides) == 1) & (self.num_layers > 1):
        #     self.strides *= self.num_layers
    

        # If masking is enabled, initialize pre-mask convolutional layers
        if self.mask_flag in [1, 2]:
            self.pre_mask = nn.Sequential(
                OrderedDict({
                    "intra_mask_1": nn.Conv1d(
                        in_channels=self.input_dim, out_channels=self.input_dim,
                        kernel_size=3, stride=1, padding=1),
                    "bn_intra_mask_1": nn.BatchNorm1d(self.input_dim),
                    "relu_intra_mask_1": self.relu,

                    "intra_mask_2": nn.Conv1d(
                        in_channels=self.input_dim, out_channels=self.input_dim,
                        kernel_size=3, stride=1, padding=1),

                    "bn_intra_mask_2": nn.BatchNorm1d(self.input_dim),
                    "relu_intra_mask_2": self.relu,
                })
                
            )

        # Define the  convolutional layers, adjusting input channels based on masking
        layers = OrderedDict()
        in_channels = self.input_dim
        idx = 0

        if self.mask_flag in [1, 2]:
            
            layers["conv1"] = nn.Conv1d(
                in_channels=self.input_dim * 2, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            layers["bn1"] = nn.BatchNorm1d(self.input_dim)
            layers["relu1"] = self.relu

            for i in range(len(self.conv_dims)): # 
                idx = i + 2
                
                layers[f"conv{idx}"] = nn.Conv1d(
                    in_channels=in_channels, out_channels=self.conv_dims[i],
                    kernel_size=kernel_sizes[i],stride=self.strides[i],padding=kernel_sizes[i] // 2
                )
                layers[f"bn{idx}"] = nn.BatchNorm1d(self.conv_dims[i])
                layers[f"relu{idx}"] = self.relu

                in_channels = self.conv_dims[i]

            layers[f"conv{idx + 1}"] = nn.Conv1d(
                in_channels=in_channels, out_channels=self.output_dim,
                kernel_size=kernel_sizes[-1],stride=self.strides[-1],padding=kernel_sizes[-1] // 2
            )
            layers[f"bn{idx + 1}"] = nn.BatchNorm1d(self.output_dim)
            layers[f"relu{idx + 1}"] = self.relu

        else:
            for i in range(len(self.conv_dims)):
                idx = i + 1
                
                layers[f"conv{idx}"] = nn.Conv1d(
                    in_channels=in_channels, out_channels=self.conv_dims[i],
                    kernel_size=kernel_sizes[i],stride=self.strides[i],padding=kernel_sizes[i] // 2
                )
                layers[f"bn{idx}"] = nn.BatchNorm1d(self.conv_dims[i])
                layers[f"relu{idx}"] = self.relu

                in_channels = self.conv_dims[i]

            layers[f"conv{idx + 1}"] = nn.Conv1d(
                in_channels=in_channels, out_channels=self.output_dim,
                kernel_size=kernel_sizes[-1],stride=self.strides[-1],padding=kernel_sizes[-1] // 2
            )
            layers[f"bn{idx + 1}"] = nn.BatchNorm1d(self.output_dim)
            layers[f"relu{idx + 1}"] = self.relu

        self.conv_layers = nn.Sequential(layers)
    
    def forward(self, input, mask=None):

        """
        Forward pass of the Encoder. Processes the input signal and optional mask.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim, sequence_length).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, input_dim, sequence_length).
                                           Required if mask_flag is 1 or 2.

        Returns:
            torch.Tensor: Encoded latent representation.
        """

        # Ensure mask is provided if required
        if self.mask_flag in [1, 2]:
            assert mask is not None, "Mask must be provided when mask_flag is 1 or 2."
        else:
            assert mask is None, "Mask should not be provided when mask_flag is not 1 or 2."

        # Pre-process mask if applicable
        if self.mask_flag in [1, 2]:
            mask = self.pre_mask(mask)

        # Concatenate input and mask if masking is enabled
        if mask is not None and self.mask_flag in [1, 2]:
            input = torch.cat([input, mask], dim=1)

        # Pass through convolutional layers
        return self.conv_layers(input)

class Decoder(nn.Module):

    """
    The Decoder module reconstructs the input signal from the quantized latent representation.

    Attributes:
        num_layers (int): Number of deconvolutional layers.
        mask_flag (int): Flag indicating the use of masking.
        input_dim (int): Number of input channels/features.
        relu (nn.ReLU): ReLU activation function.
        id (nn.Identity): Identity layer for optional activation.
        dropout (nn.Dropout1d): Dropout layer for regularization.
        deconv1, deconv2, deconv3, deconv4, deconv5 (nn.ConvTranspose1d): Transposed convolutional layers.
        bn1, bn2, bn3, bn4, bn5 (nn.BatchNorm1d): Batch normalization layers.
        deconv_layers (nn.Sequential): Sequential container of deconvolutional layers.
        pre_mask (nn.Sequential): Pre-processing layers for mask input (if mask_flag == 2).
        fine_tune_layers (nn.Sequential): Fine-tuning layers for incorporating mask information (if mask_flag == 2).
    """
    
    def __init__(self, input_dim, output_dim, num_layers,
                conv_dims, kernel_sizes, strides, p,
                mask_flag):
        
        """
        Initializes the Decoder module with the given parameters.

        Args:
            input_dim (int): Number of input channels/features.
            output_dim (int): Output dimensionality after decoding.
            num_layers (int): Number of deconvolutional layers.
            conv_dims (list of int): List specifying the number of channels for each deconv layer.
            kernel_sizes (list of int): List specifying the kernel size for each deconv layer.
            strides (list of int): List specifying the stride for each deconv layer.
            p (float): Dropout probability.
            mask_flag (int): Flag indicating the use of masking.
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mask_flag = mask_flag
        
        #self.num_layers = num_layers
        self.conv_dims = conv_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.relu = nn.ReLU()
        self.id = nn.Identity()
        self.dropout = nn.Dropout1d(p)

        # Define the  transposed convolutional layers, adjusting input channels based on masking
        layers = OrderedDict()
        in_channels = self.output_dim
        idx = 0

        for i in range(len(self.conv_dims)):
            idx = i + 1
            
            layers[f"deconv{idx}"] = nn.ConvTranspose1d(
                in_channels=in_channels, out_channels=self.conv_dims[i],
                kernel_size=kernel_sizes[i],stride=self.strides[i],padding=kernel_sizes[i] // 2
            )
            layers[f"bn{idx}"] = nn.BatchNorm1d(self.conv_dims[i])
            layers[f"relu{idx}"] = self.relu

            in_channels = self.conv_dims[i]

        layers[f"deconv{idx + 1}"] = nn.ConvTranspose1d(
            in_channels=in_channels, out_channels=self.input_dim,
            kernel_size=kernel_sizes[-1],stride=self.strides[-1],padding=kernel_sizes[-1] // 2
        )
        layers[f"bn{idx + 1}"] = nn.BatchNorm1d(self.input_dim)

        if self.mask_flag in [0, 1]:
            layers[f"id{idx + 1}"] = self.id

        elif self.mask_flag == 2:
            layers[f"relu{idx + 1}"] = self.relu            
            
        self.deconv_layers = nn.Sequential(layers)

        # If mask_flag == 2, initialize pre-mask layers
        if self.mask_flag == 2:

            self.intra_mask_1 = nn.Conv1d(
                in_channels=self.input_dim, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_intra_mask_1 = nn.BatchNorm1d(self.input_dim)

            self.intra_mask_2 = nn.Conv1d(
                in_channels=self.input_dim, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_intra_mask_2 = nn.BatchNorm1d(self.input_dim)

            self.pre_mask = nn.Sequential(
                self.intra_mask_1, self.bn_intra_mask_1, self.relu,
                self.intra_mask_2, self.bn_intra_mask_2, self.relu,
            )

            # Define fine-tuning layers to incorporate mask information
            self.fine1 = nn.Conv1d(
                in_channels=self.input_dim * 2, out_channels=self.input_dim * 2,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_fine1 = BatchNorm1d(self.input_dim * 2)

            self.fine2 = nn.Conv1d(
                in_channels=self.input_dim * 2, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_fine2 = BatchNorm1d(self.input_dim)

            self.fine3 = nn.Conv1d(
                in_channels=self.input_dim, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_fine3 = BatchNorm1d(self.input_dim)

            self.fine4 = nn.Conv1d(
                in_channels=self.input_dim, out_channels=self.input_dim,
                kernel_size=3, stride=1, padding=1
            )
            self.bn_fine4 = BatchNorm1d(self.input_dim)

            self.fine_tune_layers = nn.Sequential(
                self.fine1, self.bn_fine1, self.relu,
                self.fine2, self.bn_fine2, self.relu,
                self.fine3, self.bn_fine3, self.relu,
                self.fine4, self.bn_fine4, self.id
            )

    def forward(self, input, mask=None):

        """
        Forward pass of the Decoder. Reconstructs the input signal from the quantized latent representation.

        Args:
            input (torch.Tensor): Quantized latent tensor of shape (batch_size, input_dim, sequence_length).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, input_dim, sequence_length).
                                           Required if mask_flag == 2.

        Returns:
            torch.Tensor: Reconstructed signal tensor.
        """

        # Ensure mask is provided if required
        if self.mask_flag == 2:
            assert mask is not None, "Mask must be provided when mask_flag == 2."
        else:
            assert mask is None, "Mask should not be provided when mask_flag != 2."

        # Pass through deconvolutional layers
        input = self.deconv_layers(input)

        # If mask_flag == 2, process and incorporate mask information
        if self.mask_flag == 2:
            mask = self.pre_mask(mask)

        if mask is not None and self.mask_flag == 2:
            input = torch.cat([input, mask], dim=1)

        # If mask_flag == 2 apply finetuning layers
        # to incorporate the mask information.
        if self.mask_flag == 2:
            input = self.fine_tune_layers(input)

        return input
    
class VQVAE(nn.Module):

    """
    The VQVAE class encapsulates the entire Vector Quantized Variational Autoencoder model,
    including the encoder, decoder, and quantizer components.

    Attributes:
        mask_flag (int): Flag indicating the use of masking.
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        quantizer (Quantizer): The quantizer module.
    """
    
    def __init__(self, num_features, embed_dim, num_embed, num_layers, 
                 conv_dims, conv_kernel_sizes, conv_strides,
                 p, decay, threshold, mask_flag,
                 deconv_dims = None, deconv_kernel_sizes = None, deconv_strides = None):
        
        """
        Initializes the VQVAE model with the given parameters.

        Args:
            num_features (int): Number of input features.
            embed_dim (int): Dimensionality of the embeddings.
            num_embed (int): Number of embeddings in the codebook.
            num_layers (int): Number of layers in encoder and decoder.
            conv_dims (list of int): List specifying the number of channels for each conv layer.
            kernel_sizes (list of int): List specifying the kernel size for each conv layer.
            strides (list of int): List specifying the stride for each conv layer.
            p (float): Dropout probability.
            decay (float): Decay rate for EMA updates in quantizer.
            threshold (float): Threshold for embedding updates in quantizer.
            mask_flag (int): Flag indicating the use of masking.
        """

        if deconv_dims == None:

            deconv_dims = conv_dims[::-1]
            deconv_kernel_sizes = conv_kernel_sizes[::-1]
            deconv_strides = conv_strides[::-1]

        super().__init__()

        self.mask_flag = mask_flag

        # Initialize encoder, decoder, and quantizer
        self.encoder = Encoder(num_features, embed_dim, num_layers, 
                               conv_dims, conv_kernel_sizes, conv_strides, p, mask_flag)

        self.decoder = Decoder(num_features, embed_dim, num_layers, 
                              deconv_dims, deconv_kernel_sizes, deconv_strides, p, mask_flag)
        
        self.quantizer = Quantizer(embed_dim, num_embed, decay, threshold)

    def forward(self, input, mask=None, return_info= False):

        """
        Forward pass of the VQVAE model. Encodes the input, quantizes the latent representation,
        and decodes it to reconstruct the input.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, num_features, sequence_length).
                                           Required if mask_flag is 1 or 2.

        Returns:
            tuple:
                decoded (torch.Tensor): Reconstructed input tensor.
                diff (torch.Tensor): Average squared distance between quantized vectors and inputs.
                indices (torch.Tensor): Indices of the selected embeddings.
                embedding_info (np.ndarray): Structured array containing embedding ranking and distance info.
        """
        
        # Encode the input signal
        encoded = self.encoder(input, mask if self.mask_flag in [1, 2] else None)

        # Quantize the encoded representation
        quantized, diff, indices, embedding_info = self.quantizer(encoded.permute(0, 2, 1), return_info)
        quantized = quantized.permute(0, 2, 1)

        # Decode the quantized representation to reconstruct the input
        decoded = self.decoder(quantized, mask if self.mask_flag == 2 else None)

        return decoded, diff, indices, embedding_info
