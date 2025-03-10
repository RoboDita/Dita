import torch
import torchvision.transforms as T
from typing import Optional

CROP_SIZE = 472

def convert_dtype_and_crop_images(images,
                                  crop_size: int = CROP_SIZE,
                                  training: bool = True,
                                  pad_then_crop: bool = False,
                                  convert_dtype: bool = True,
                                  seed: Optional[torch.Tensor] = None):
    """Convert uint8 [B, H, W, 3] images to float32 and square crop in PyTorch.

    Args:
        images: [B, C, H, W] uint8 tensor of images (Note the channel location difference from TensorFlow).
        crop_size: Width of the square crop.
        training: If we are in training (random crop) or not-training (center crop).
        pad_then_crop: If True, pads image and then crops to the original image size.
            This allows full field of view to be extracted.
        convert_dtype: whether or not to convert the image to float32 in the range of (0, 1).
        seed: Optional seed for random operations. In PyTorch, set torch.manual_seed(seed).

    Returns:
        [B, C, crop_size, crop_size] images of dtype float32.
    """
    if seed is None:
        seed = torch.randint(0, 2**30, (2,), dtype=torch.int32)
    
    torch.manual_seed(seed[0])

    if convert_dtype:
        images = images.to(torch.float32) / 255.0  # Convert images to float32 and scale to [0, 1].

    image_height, image_width = images.shape[2], images.shape[3]

    if pad_then_crop:
        # Padding then cropping logic.
        if training:
            if image_height == 512:
                ud_pad, lr_pad = 40, 100
            elif image_height == 256:
                ud_pad, lr_pad = 20, 50
            else:
                raise ValueError('Only supports image height 512 or 256 for padding then cropping.')
            padding = (lr_pad, ud_pad, lr_pad, ud_pad)  # left, top, right, bottom
            images = T.functional.pad(images, padding, padding_mode='constant', fill=0)

            # For random crop, we adjust the starting point of crop.
            i, j, h, w = T.RandomCrop.get_params(
                images, output_size=(image_height, image_width))
            images = T.functional.crop(images, i, j, h, w)
        else:
            # Not covered due to the specific logic of pad_then_crop during training only.
            pass
    else:
        # Standard cropping logic.
        if training:
            images = T.RandomCrop((crop_size, crop_size))(images)
        else:
            images = T.CenterCrop((crop_size, crop_size))(images)

    return images
