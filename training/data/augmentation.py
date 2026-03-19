from typing import Optional, Dict
from torchvision import transforms
import torch

def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None


class SequenceAugmentation:
    """
    Augmentation class that supports both independent and co-jittering modes.
    
    Co-jittering applies the same augmentation parameters to all frames in a sequence,
    which is useful for maintaining temporal consistency in video data.
    """
    
    def __init__(
        self,
        color_jitter: Optional[Dict[str, float]] = None,
        gray_scale: bool = True,
        gau_blur: bool = False,
        cojitter: bool = True,
        cojitter_ratio: float = 0.5
    ):
        """
        Initialize sequence augmentation.
        
        Args:
            color_jitter: Color jitter parameters
            gray_scale: Whether to apply random grayscale
            gau_blur: Whether to apply gaussian blur
            cojitter: Whether to use co-jittering mode
            cojitter_ratio: Probability of using co-jittering vs independent jittering
        """
        self.cojitter = cojitter
        self.cojitter_ratio = cojitter_ratio
        self.image_aug = get_image_augmentation(color_jitter, gray_scale, gau_blur)
        
    def __call__(self, images, apply_augmentation=True):
        """
        Apply augmentation to a sequence of images.
        
        Args:
            images: Tensor of shape (B, C, H, W) containing sequence of images
            apply_augmentation: Whether to apply augmentation (useful for validation)
            
        Returns:
            Augmented images tensor of the same shape
        """
        if not apply_augmentation or self.image_aug is None:
            return images
            
        import random
        
        if self.cojitter and random.random() > self.cojitter_ratio:
            # Apply the same augmentation to all frames
            return self.image_aug(images)
        else:
            # Apply different augmentation to each frame
            augmented_frames = []
            for i in range(len(images)):
                augmented_frames.append(self.image_aug(images[i]))
            return torch.stack(augmented_frames, dim=0)


def create_augmentation_config(
    brightness: float = 0.5,
    contrast: float = 0.5,
    saturation: float = 0.5,
    hue: float = 0.1,
    color_jitter_prob: float = 0.9,
    gray_scale: bool = True,
    gaussian_blur: bool = False,
    co_jitter: bool = True,
    co_jitter_ratio: float = 0.5
) -> Dict:
    """
    Create a standardized augmentation configuration dictionary.
    
    Args:
        brightness: Brightness jitter strength
        contrast: Contrast jitter strength
        saturation: Saturation jitter strength
        hue: Hue jitter strength
        color_jitter_prob: Probability of applying color jitter
        gray_scale: Whether to apply random grayscale
        gaussian_blur: Whether to apply gaussian blur
        co_jitter: Whether to use co-jittering
        co_jitter_ratio: Probability of co-jittering vs independent jittering
        
    Returns:
        Dictionary containing augmentation configuration
    """
    return {
        "color_jitter": {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "p": color_jitter_prob
        },
        "gray_scale": gray_scale,
        "gau_blur": gaussian_blur,
        "cojitter": co_jitter,
        "cojitter_ratio": co_jitter_ratio
    }


def get_training_augmentations() -> SequenceAugmentation:
    """Get default training augmentations with strong data augmentation."""
    config = create_augmentation_config(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1,
        color_jitter_prob=0.9,
        gray_scale=True,
        gaussian_blur=False,
        co_jitter=True,
        co_jitter_ratio=0.5
    )
    return SequenceAugmentation(**config)


def get_validation_augmentations() -> SequenceAugmentation:
    """Get validation augmentations with minimal augmentation."""
    config = create_augmentation_config(
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        color_jitter_prob=0.0,
        gray_scale=False,
        gaussian_blur=False,
        co_jitter=False,
        co_jitter_ratio=0.0
    )
    return SequenceAugmentation(**config) 