try:
    import os
    from pathlib import Path
    import torch
    from diffusers.utils import export_to_video, load_image
    from pipelines.wan_pipeline import WanImageToVideoTTMPipeline
    from pipelines.utils import (
        validate_inputs,
        compute_hw_from_area,
    )
    import argparse
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/Wan-Video/Wan2.2")

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DTYPE = torch.bfloat16

# -----------------------
# Argument Parser
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--input-path", type=str, default="./examples/wan_monkey", help="Path to input image")
    parser.add_argument("--output-path", type=str, default="./outputs/output_wan_monkey.mp4", help="Path to save output video")
    parser.add_argument("--negative-prompt", type=str, default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
        "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ), help="Default negative prompt in Wan2.2")
    parser.add_argument("--tweak-index", type=int, default=3, help="t weak timestep index- when to start denoising")
    parser.add_argument("--tstrong-index", type=int, default=6, help="t strong timestep index- when to start denoising within the mask")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-area", type=int, default=480 * 832, help="Maximum area for resizing")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale for generation")
    return parser.parse_args()


args = parse_args()
image_path = os.path.join(args.input_path, "first_frame.png")
motion_signal_mask_path = os.path.join(args.input_path, "mask.mp4")
motion_signal_video_path = os.path.join(args.input_path, "motion_signal.mp4")
prompt_path = os.path.join(args.input_path, "prompt.txt")

output_path = args.output_path
negative_prompt = args.negative_prompt
tweak_index = args.tweak_index
tstrong_index = args.tstrong_index
num_inference_steps = args.num_inference_steps
seed = args.seed
device = args.device
max_area = args.max_area
num_frames = args.num_frames
guidance_scale = args.guidance_scale

# make sure output directory exists
Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)

# -----------------------
# Setup Pipeline
# -----------------------
def setup_wan_pipeline(model_id: str, dtype: torch.dtype, device: str):
    pipe = WanImageToVideoTTMPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    return pipe


# -----------------------
# Main (same functional steps)
# -----------------------
def main():
    validate_inputs(image_path, motion_signal_mask_path, motion_signal_video_path)
    pipe = setup_wan_pipeline(MODEL_ID, DTYPE, device)

    # Load and resize image (unchanged logic)
    image = load_image(image_path)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height, width = compute_hw_from_area(image.height, image.width, max_area, mod_value)
    image = image.resize((width, height))

    # Load prompt (unchanged)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    prompt = (prompt)

    # Generator / seed (unchanged)
    gen_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            motion_signal_video_path=motion_signal_video_path,
            motion_signal_mask_path=motion_signal_mask_path,
            tweak_index=tweak_index,
            tstrong_index=tstrong_index,
        )

    frames = result.frames[0]
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    export_to_video(frames, output_path, fps=16)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
