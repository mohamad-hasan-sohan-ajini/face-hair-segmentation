import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_ce import UNet
from model_fl import UNet
from utils import squrify, val_transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ckpt-path',
    default='lightning_logs/version_0/checkpoints/last.ckpt',
)
parser.add_argument('--image-path', default='ca.jpg')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

ckpt_path = Path(args.ckpt_path)
saving_path = ckpt_path.with_suffix('.pt')

device = torch.device(args.device)

model = UNet.load_from_checkpoint(
    args.ckpt_path,
    in_channels=3,
    out_channels=3,
    init_features=32,
    map_location=device,
)
model.to_torchscript().save(saving_path)

image = Image.open(args.image_path).convert('RGB')
squared_size = max(image.size)
pad_width, pad_height, padded_image = squrify(image, squared_size)
padded_224_image_np = np.array(padded_image.resize((224, 224)))
padded_224_image_pt = val_transforms(image=padded_224_image_np)['image']
padded_224_image_pt = padded_224_image_pt.unsqueeze(0).to(device)

with torch.inference_mode():
    pred_224 = (
        model(padded_224_image_pt)
        .softmax(1)
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
        * 255
    )

pred_224_image = Image.fromarray(pred_224.astype(np.uint8))
pred_224_image.save('/tmp/ztmp.png')
