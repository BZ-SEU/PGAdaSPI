"""Grayscale image loading; no resize for native templates."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_image(path: str | Path, size: int=256, native: bool=False) -> torch.Tensor:
    with Image.open(path) as src:
        image=src.convert('L')
        if native and image.size != (size,size):
            raise ValueError(f'{path}: native image {image.size} does not match profile {size}')
        if image.size == (size,size):
            return torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255)
        side=min(image.size); left=(image.width-side)//2; top=(image.height-side)//2
        image=image.crop((left,top,left+side,top+side)).resize((size,size),Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255)

def load_image_template(path: str | Path, height: int, width: int,
                        rotate_portrait: bool = False) -> tuple[torch.Tensor, dict]:
    """Load an explicit rectangular Y/gray template without padding or resize."""
    with Image.open(path) as src:
        image=src.convert('L')
        original=(image.height,image.width)
        rotated=False
        if rotate_portrait and image.height > image.width:
            # Match the comparison protocol's cv2.ROTATE_90_CLOCKWISE.
            image=image.rotate(-90,expand=True); rotated=True
        if image.size != (width,height):
            raise ValueError(f'{path}: template expects {(height,width)}, got {(image.height,image.width)}')
        arr=np.asarray(image,dtype=np.float32)/255.0
    return torch.from_numpy(arr.copy()), dict(top=0,left=0,height=height,width=width,
        original_height=original[0],original_width=original[1],canvas_height=height,
        canvas_width=width,rotated_clockwise_90=rotated,roi_is_full_canvas=True)
