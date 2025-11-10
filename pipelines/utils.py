import logging
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import torch

def validate_inputs(image_path: str, mask_path: str, motion_path: str) -> None:
    for p in [image_path, mask_path, motion_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file not found: {p}")

def compute_hw_from_area(
    image_height: int,
    image_width: int,
    max_area: int,
    mod_value: int,
) -> Tuple[int, int]:
    """Compute (height, width) with same math and rounding as original."""
    aspect_ratio = image_height / image_width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(height), int(width)


def load_video_to_tensor(video_path):
    """Returns a video tensor from a video file. shape [1, T, C, H, W], [0, 1] range."""
        # load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # Convert frames to tensor, shape [T, H, W, C], [0, 1] range
    frames = np.array(frames)

    video_tensor = torch.tensor(frames)
    video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video_tensor
