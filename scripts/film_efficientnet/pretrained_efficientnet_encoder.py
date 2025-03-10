import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Resize
from film_efficientnet.film_conditioning_layer import FilmConditioning
from film_efficientnet.film_efficientnet_encoder import EfficientNet


# Dictionary for model variants to corresponding torchvision model functions
_MODELS = {
    'b3': 'efficientnet-b3',
}

_SIZES = {
    'b3': 300,
}

class EfficientNetEncoder(nn.Module):
    """Applies a pretrained EfficientNet-based encoder with optional FiLM conditioning."""
    
    def __init__(self, model_variant='b3', freeze=False, early_film=True, weights='imagenet',
                 include_top=False, pooling=True):
        super(EfficientNetEncoder, self).__init__()
        if model_variant not in _MODELS:
            raise ValueError(f'Unknown variant {model_variant}')
        self.model_variant = model_variant
        self.early_film = early_film
        self.freeze = freeze
        
        self.conv1x1 = nn.Conv2d(
            in_channels=1536,  # EfficientNet-B3 feature size before pooling
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        self.net = EfficientNet.from_pretrained(_MODELS[model_variant], 
                                                # weights_path='rt1-pytorch/efficientnet-b3-5fb5a3c3.pth',
                                                weights_path='/xxx/xxx/Anonymous/Robotics/RT-1-X/Support_Files/efficientnet-b3-5fb5a3c3.pth',
                                                include_film=early_film,
                                                include_top=include_top)
        self.film_layer = FilmConditioning(in_dim=512, num_channels=512)  # Assume implementation
        self.pooling = pooling
        
        # if not include_top:
        #     self.net._global_params.include_top = False
        #     self.net._dropout = nn.Identity()
        #     self.net._fc = nn.Identity()
        
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False
                
    def _prepare_image(self, image):
        size = _SIZES[self.model_variant]
        if len(image.shape) != 4 or image.shape[1] != 3:
            raise ValueError('Provided image should have shape (b, 3, h, w).')
        size = _SIZES[self.model_variant]
        if image.shape[2] < size / 4 or image.shape[3] < size / 4:
            raise ValueError('Provided image is too small.')
        if image.shape[2] > size * 4 or image.shape[3] > size * 4:
            raise ValueError('Provided image is too large.')
        resize = Resize((size, size))
        assert image.min() >= 0.
        assert image.max() <= 1.
        
        # PyTorch models expect normalized images
        # Normalize with EfficientNet-B3's expected stats could be added here
        return image
    
    def _encode(self, image, context=None):
        image = self._prepare_image(image)
        if self.early_film and context is not None:
            return self.net(image, context_input=context)
        return self.net(image)
    
    def forward(self, image, context=None):
        features = self._encode(image, context)
        if context is not None:
            features = self.conv1x1(features) # out_channel: 512
            features = self.film_layer(features, context)
        
        # import pdb
        # pdb.set_trace()

        if not self.pooling:
            return features
        
        # Global average pooling
        return torch.mean(features, dim=[2, 3])

