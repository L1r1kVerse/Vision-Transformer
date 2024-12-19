import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """
    Convert an input image into a 1D vector of learnable embeddings.

    Parameters:
        in_channels (int): The number of color channels in the input image. Default is 3.
        patch_size (int): The size of each patch to extract from the input image. Default is 16.
        embedding_dim (int): The size of the embedding dimension. Default is 768.

    Returns:
        torch.Tensor: A tensor representing the image as a sequence of learnable embeddings.
    """
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = embedding_dim, 
                              kernel_size = patch_size,
                              stride = patch_size,
                              padding = 0)
        
    def forward(self, x: torch.Tensor):
        # Apply convolution to extract patches
        x = self.conv(x)

        # Get the shape for view
        bs, c, h, w = x.shape
        #print(f"Shape after convolution: {x.shape} [batch_size, channels, height, width]") # check shape for debugging purposes

        # Rearrange shape to get [batch_size, height, width, embedding_dim]
        x = x.permute(0, 2, 3, 1)

        # Reshape to [batch_size, num_patches, embedding_dim] where num_patches = h * w
        x = x.view(bs, h * w, c)  # Shape will be (batch_size, num_patches, embedding_dim)
        
        return x
    
class VisionTransformer(nn.Module):
    """
    Creates the Vision Transformer architecture with hyperparameters for ViT base from the paper
    """
    def __init__(self,
                 img_size: int = 224, # Input images dimensions
                 in_channels: int = 3, # Input number of channels
                 patch_size: int = 16, # Patch size
                 num_transformer_layers: int = 12, # How many times input is run through equations 2 and 3
                 embedding_dim: int = 768, # Size of the embedding dimension
                 mlp_size: int = 3072, # Size of the input of the second MLP layer 
                 num_heads: int = 12, # Number of head for Multi-Head Self-Attention 
                 attn_dropout: float = 0 , # Dropout for attention
                 mlp_dropout: float = 0.1, # Dropout for linear layers (MLP)
                 embedding_dropout: float = 0.1, # Dropout for patch and position embeddings
                 num_classes: int = 1000): # Number of outputs for the Classifier head
                 
        super().__init__()
        # Check if image size is divisible by patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # Calculate number of patches
        self.num_patches = (img_size * img_size) // (patch_size * patch_size)

        # Create the learnable class embedding
        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim),
                                            requires_grad = True)
        # Create the learnable position embedding
        self.position_embedding = nn.Parameter(data = torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad = True)
        # Create embedding dropout
        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        # Create the patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels = in_channels,
                                              patch_size = patch_size,
                                              embedding_dim = embedding_dim)
        
        # Create temporary encoder layer for the constructor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embedding_dim, # Embedding dimension
            nhead = num_heads,       # Number of attention heads
            dim_feedforward = mlp_size, # Feedforward network size
            dropout = mlp_dropout,      # Dropout in the MLP layer
            activation = "gelu",        # GELU Non-linearity 
            batch_first = True        
        )

        # Create the Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = num_transformer_layers,
            
        )

        # Create the Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = embedding_dim),
            nn.Linear(in_features = embedding_dim,
                      out_features = embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features = embedding_dim,
                      out_features = num_classes)

        )

    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]

        # Create the class token and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # Create the patch embedding
        x = self.patch_embedding(x)

        # Prepend the class token embedding
        x = torch.cat((class_token, x), dim = 1)

        # Add the position embedding
        x = self.position_embedding + x

        # Run the embedding dropout
        x = self.embedding_dropout(x)

        # Pass the input batch with prepended class tokens and added 
        # position embeddings through the Transformer Encoder
        x = self.transformer_encoder(x)

        # Run the 0th index element through the classifier hear
        x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

        return x
    