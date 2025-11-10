# Copyright 2025 Noam Rotstein
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Hugging Face Diffusers (Apache-2.0):
#   https://github.com/huggingface/diffusers/blob/8abc7aeb715c0149ee0a9982b2d608ce97f55215/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L147

try:
    import inspect
    from dataclasses import dataclass
    from typing import Callable, Dict, List, Optional, Union
    import numpy as np
    import PIL.Image
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
    from diffusers.schedulers import EulerDiscreteScheduler
    from diffusers.utils import BaseOutput, is_torch_xla_available, logging, replace_example_docstring
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.video_processor import VideoProcessor
    import torch.nn.functional as F
    from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
    from pipelines.utils import load_video_to_tensor

except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see:https://github.com/Stability-AI/generative-models")

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
"""


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionTTMPipeline(StableVideoDiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion combined with Time to Move (TTM).
    This model inherits from [`StableVideoDiffusionPipeline`].
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)


    def encode_frames(self, frames: torch.Tensor, num_frames: int, encode_chunk_size: int = 14):
        """
        Args:
            frames: [B, C, T, H, W] tensor, preprocessed to VAE's expected range (e.g., [-1, 1]).
            num_frames: T (used for reshaping back).
            encode_chunk_size: process at most this many frames at a time to avoid OOM.

        Returns:
            latents: [B, T, C_latent, h, w], multiplied by self.vae.config.scaling_factor.

        Notes:
            - Stochastic: samples from posterior (latent_dist.sample()).
            - If the VAE's compiled module hides the signature, we inspect the original .forward
            and pass num_frames only if it's accepted (same pattern as decode).
        """
        if frames.dim() != 5:
            raise ValueError(f"Expected frames with shape [B, C, T, H, W], got {list(frames.shape)}")
        B, C, T, H, W = frames.shape

        # [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
        frames_bt = frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        # Use the *encode* signature (decoder may accept num_frames, encoder usually doesn't)
        encode_fn = self.vae._orig_mod.encode if hasattr(self.vae, "_orig_mod") else self.vae.encode
        try:
            accepts_num_frames = ("num_frames" in inspect.signature(encode_fn).parameters)
        except (TypeError, ValueError):
            # Signature might be obscured by wrappers/compilation; be conservative
            accepts_num_frames = False

        latents_chunks = []
        for i in range(0, frames_bt.shape[0], encode_chunk_size):
            chunk = frames_bt[i : i + encode_chunk_size]

            # match VAE device/dtype to avoid implicit casts
            chunk = chunk.to(device=self.vae.device, dtype=self.vae.dtype)

            encode_kwargs = {}
            if accepts_num_frames:
                # This will normally be False for AutoencoderKLTemporalDecoder.encode()
                encode_kwargs["num_frames"] = chunk.shape[0]

            # Be robust to unexpected wrappers hiding the signature
            try:
                enc_out = self.vae.encode(chunk, **encode_kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'num_frames'" in str(e):
                    enc_out = self.vae.encode(chunk)
                else:
                    raise

            posterior = enc_out.latent_dist  # DiagonalGaussianDistribution
            latents_chunks.append(posterior.sample())

        latents = torch.cat(latents_chunks, dim=0)  # [B*T, C_lat, h, w]
        latents = latents * self.vae.config.scaling_factor

        # [B*T, C_lat, h, w] -> [B, T, C_lat, h, w]
        latents = latents.reshape(B, num_frames, *latents.shape[1:])

        return latents

    def convert_rgb_mask_to_latent_mask(self, mask: torch.Tensor, first_different=True) -> torch.Tensor:
        """
        Args:
            mask: [T, 1, H, W] tensor (0/1 or any float in [0,1]).
        Returns:
            latent_mask: [1, T_latent, 1, H, W], where
                T_latent = ceil(T / self.vae_scale_factor_temporal)
                For CogVideoX-style VAE (k=4), groups are [0], [1-4], [5-8], ..., achieved by
                pre-padding zeros at the start before max-pooling with stride=k.
        """
        T, _, H, W = mask.shape

        k = self.vae_scale_factor_temporal
        # Pre-pad zeros along time so that the first pooled window corresponds to frame 0 alone
        if first_different:
            num_pad = (k - (T % k)) % k
            pad = torch.zeros((num_pad, 1, H, W), device=mask.device, dtype=mask.dtype)
            mask = torch.cat([pad, mask], dim=0)
        

        # [T,1,H,W] -> [1,1,T,H,W]
        x = mask.permute(1, 0, 2, 3).unsqueeze(0)
        if k > 1:
            # Max-pool over time with kernel=stride=k (no spatial pooling)
            pooled = F.max_pool3d(x, kernel_size=(k, 1, 1), stride=(k, 1, 1))
        else:
            pooled = x

        # Up-sample spatially to match latent spatial resolution
        s = self.vae_scale_factor_spatial
        H_latent = pooled.shape[-2] // s
        W_latent = pooled.shape[-1] // s
        pooled = F.interpolate(pooled, size=(pooled.shape[2], H_latent, W_latent), mode="nearest")

        # Back to [1, T_latent, 1, H, W]
        latent_mask = pooled.permute(0, 2, 1, 3, 4)

        return latent_mask
    

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        motion_signal_video_path: Optional[str] = None,
        motion_signal_mask_path: Optional[str] = None,
        tweak_index: int = 0,
        tstrong_index: int = 0
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
                1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
                `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
                For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            motion_signal_video_path (`str`):
                Path to the video file containing the motion signal to guide the motion of the generated video.
                It should be a crude version of the reference video, with pixels with motion dragged to their target.
            motion_signal_mask_path (`str`):
                Path to the mask video file containing the motion mask of TTM.
                The mask should be a binary with the conditioning motion pixels being 1 and the rest being 0.
            tweak_index (`int`):
                The index of the tweak, from which the denoising process starts.
            tstrong_index (`int`):
                The index of the tweak, from which the denoising process starts in the motion conditioned region.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
                returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        if motion_signal_mask_path is None:
            raise ValueError("`motion_signal_mask_path` is required for TTM.")
        if motion_signal_video_path is None:
            raise ValueError("`motion_signal_video_path` is required for TTM.")
        
        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # ---- Sanity checks for TTM indices (0 ≤ tstrong < tweak < num_steps) ----
        if not (0 <= tstrong_index < num_inference_steps):
            raise ValueError(f"tstrong_index must be in [0, {num_inference_steps-1}], got {tstrong_index}.")
        if not (0 <= tweak_index < num_inference_steps):
            raise ValueError(f"tweak_index must be in [0, {num_inference_steps-1}], got {tweak_index}.")
        if not (tstrong_index > tweak_index):
            raise ValueError(f"Require tweak_index < tstrong_index, got {tweak_index} >= {tstrong_index}.")

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Initialize for TTM
        ref_vid = load_video_to_tensor(motion_signal_video_path).to(device=device) # shape [1, C, T, H, W]
        refB, refC, refT, refH, refW = ref_vid.shape

        ref_vid = F.interpolate(
            ref_vid.permute(0, 2, 1, 3, 4).reshape(refB*refT, refC, refH, refW),
            size=(height, width), mode="bicubic", align_corners=True, 
        ).reshape(refB, refT, refC, height, width).permute(0, 2, 1, 3, 4)

        ref_vid = self.video_processor.normalize(ref_vid.to(dtype=self.vae.dtype)) # Normalize and convert dtype for VAE encoding 
        
        if num_frames < refT:
            logger.warning(f"num_frames ({num_frames}) < input frames ({refT}); trimming reference video.")
            ref_vid = ref_vid[:, :, :num_frames]
        elif num_frames > refT:
            raise ValueError(f"num_frames ({num_frames}) is greater than input frames ({refT}). This is not supported.")
        
        ref_latents = self.encode_frames(ref_vid, num_frames, decode_chunk_size).detach()
        ref_latents = ref_latents.to(dtype=latents.dtype, device=device)
        
        if not hasattr(self, "vae_scale_factor_temporal"): # encode ref video to latents
            if hasattr(self.vae, "scale_factor_temporal"):
                self.vae_scale_factor_temporal = self.vae.scale_factor_temporal
            else:
                if ref_latents.shape[1] == num_frames:
                    self.vae_scale_factor_temporal = 1
                else:
                    raise ValueError("Please configure the temporal scale factor of the VAE.")
        
        self.vae_scale_factor_spatial = self.vae_scale_factor
    
        ref_mask = load_video_to_tensor(motion_signal_mask_path).to(device=device) # shape [1, C, T, H, W]

        mB, mC, mT, mH, mW = ref_mask.shape # do resizing with nearest neighbor to avoid interpolation artifacts
        ref_mask = F.interpolate(
            ref_mask.permute(0, 2, 1, 3, 4).reshape(mB*mT, mC, mH, mW),
            size=(height, width), mode="nearest", 
        ).reshape(mB, mT, mC, height, width).permute(0, 2, 1, 3, 4)
        ref_mask = ref_mask[0].permute(1, 0, 2, 3).contiguous() # (1, C, T, H, W) -> (T, H, W, 1)
        if ref_mask.shape[0] > num_frames:
            print(f"Warning: num_frames ({num_frames}) is less than input mask frames ({mT}). Trimming to {num_frames}.")
            ref_mask = ref_mask[:num_frames]
        elif ref_mask.shape[0] < num_frames:
            raise ValueError(f"num_frames ({num_frames}) is greater than input mask frames ({mT}). This is not supported.")
        ref_mask = (ref_mask > 0.5).float().max(dim=1, keepdim=True)[0] # [T, 1, H, W]
        motion_mask = self.convert_rgb_mask_to_latent_mask(ref_mask, False)  # [1, T, 1, H, W]
        motion_mask = motion_mask.to(dtype=latents.dtype)
        background_mask = 1.0 - motion_mask

        if tweak_index >= 0:
            tweak = self.scheduler.timesteps[tweak_index]
            tweak = torch.tensor([tweak], device=device)
            fixed_noise = randn_tensor(ref_latents.shape,
                                       generator=generator,
                                       device=ref_latents.device,
                                       dtype=ref_latents.dtype)
            noisy_latents = self.scheduler.add_noise(ref_latents, fixed_noise, tweak)
            latents = noisy_latents.to(dtype=latents.dtype, device=latents.device)
        else:
            tweak = torch.tensor(-1)
            fixed_noise = randn_tensor(ref_latents.shape,
                                       generator=generator,
                                       device=ref_latents.device,
                                       dtype=ref_latents.dtype)
            tweak_index = 0


        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=len(timesteps) - tweak_index) as progress_bar:
            for i, t in enumerate(timesteps[tweak_index:]):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # In between tweak and tstrong, replace mask with noisy reference latents
                in_between_tweak_tstrong = (i+tweak_index) < tstrong_index

                if in_between_tweak_tstrong:
                    if i+tweak_index+1 < len(timesteps):
                        prev_t = torch.tensor([timesteps[i+tweak_index+1]], device=device)
                        noisy_latents = self.scheduler.add_noise(ref_latents, fixed_noise, prev_t).to(dtype=latents.dtype, device=latents.device)
                        latents = latents * background_mask + noisy_latents * motion_mask
                    elif i+tweak_index+1  == len(timesteps):
                        latents = latents * background_mask + ref_latents * motion_mask
                    else:
                        raise ValueError(f"Unexpected timestep index {i+tweak_index+1} >= {len(timesteps)}")
                    
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(
            frames=frames)
