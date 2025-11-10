try:
    import argparse
    import os
    import torch
    from pipelines.svd_pipeline import StableVideoDiffusionTTMPipeline
    from diffusers.utils import export_to_video, load_image
    from PIL import Image
    import json
    from pathlib import Path
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/Stability-AI/generative-models")


MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
DTYPE = torch.float16

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--input-path", type=str, default="./examples/cutdrag_cog_Monkey", help="Path to input image")
    parser.add_argument("--output-path", type=str, default="./outputs/output_cog_Monkey.mp4", help="Path to save output video")
    parser.add_argument("--tweak-index", type=int, default=16, help="t weak timestep index- when to start denoising")
    parser.add_argument("--tstrong-index", type=int, default=21, help="t strong timestep index- when to start denoising within the mask")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num-frames", type=int, default=21, help="Number of frames to generate")
    parser.add_argument("--motion_bucket_id", type=int, default=17, help="Amount of motion to condition on")
    return parser.parse_args()


args = parse_args()

image_path = os.path.join(args.input_path, "first_frame.png")
motion_signal_mask_path = os.path.join(args.input_path, "mask.mp4")
motion_signal_video_path = os.path.join(args.input_path, "motion_signal.mp4")

num_inference_steps = args.num_inference_steps
seed = args.seed
tweak_index = args.tweak_index
tstrong_index = args.tstrong_index
num_frames = args.num_frames
motion_bucket_id = args.motion_bucket_id
output_path = args.output_path


# make sure output directory exists
Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

# -----------------------
# Setup Pipeline
# -----------------------
def setup_svd_pipeline(model_id: str, dtype: torch.dtype):
    pipe = StableVideoDiffusionTTMPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="balanced",      # keep this
        variant="fp16"
    )
    return pipe


def main():
    pipe = setup_svd_pipeline(MODEL_ID, DTYPE)
    image = load_image(image_path)
    result = pipe(
                [image],
                generator=torch.Generator().manual_seed(seed),
                motion_bucket_id=motion_bucket_id,
                num_inference_steps=num_inference_steps,
                motion_signal_video_path=motion_signal_video_path,
                motion_signal_mask_path=motion_signal_mask_path,
                tweak_index=tweak_index,
                tstrong_index=tstrong_index,
                num_frames=num_frames
                )

    frames = result.frames[0]
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    export_to_video(frames, output_path, fps=7)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()

