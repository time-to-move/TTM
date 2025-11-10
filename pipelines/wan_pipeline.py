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
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py


try:
    import html
    from typing import Any, Callable, Dict, List, Optional, Union
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
    from diffusers.image_processor import PipelineImageInput
    from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.video_processor import VideoProcessor
    from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
    from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents, WanImageToVideoPipeline

    import torch.nn.functional as F
    from pipelines.utils import load_video_to_tensor

    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm

        XLA_AVAILABLE = True
    else:
        XLA_AVAILABLE = False
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/Wan-Video/Wan2.2")


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# after logger/is_ftfy_available
_ftfy = None
if is_ftfy_available():
    import ftfy as _ftfy


EXAMPLE_DOC_STRING = """
"""

    
class WanImageToVideoTTMPipeline(WanImageToVideoPipeline):
    r"""
    Pipeline for image-to-video generation using Wan with Time-To-Move (TTM) conditioning.
    This model inherits from [`WanImageToVideoPipeline`].
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            image_processor=image_processor,
            image_encoder=image_encoder,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    
    def convert_rgb_mask_to_latent_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a per-frame mask [T, 1, H, W] to latent resolution [1, T_latent, 1, H', W'].
        T_latent groups frames by the temporal VAE downsample factor k = vae_scale_factor_temporal:
        [0], [1..k], [k+1..2k], ...
        """

        k = self.vae_scale_factor_temporal    
        mask0 = mask[0:1]  # [1,1,H,W]
        mask1 = mask[1::k]  # [T'-1,1,H,W]
        sampled = torch.cat([mask0, mask1], dim=0)  # [T',1,H,W]
        pooled = sampled.permute(1, 0, 2, 3).unsqueeze(0)

        # Up-sample spatially to match latent spatial resolution
        spatial_downsample = self.vae_scale_factor_spatial
        H_latent = pooled.shape[-2] // spatial_downsample
        W_latent = pooled.shape[-1] // spatial_downsample
        pooled = F.interpolate(pooled, size=(pooled.shape[2], H_latent, W_latent), mode="nearest")

        # Back to [1, T_latent, 1, H, W]
        latent_mask = pooled.permute(0, 2, 1, 3, 4)

        return latent_mask
        

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        motion_signal_video_path: Optional[str] = None,
        motion_signal_mask_path: Optional[str] = None,
        tweak_index: int = 0,
        tstrong_index: int = 0
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.
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
        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if motion_signal_video_path is None:
            raise ValueError("`motion_signal_video_path` must be provided for TTM.")
        if motion_signal_mask_path is None:
            raise ValueError("`motion_signal_mask_path` must be provided for TTM.")

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        tweak_index = int(tweak_index)
        tstrong_index = int(tstrong_index)

        if tweak_index < -1:
            raise ValueError(f"`tweak_index` ({tweak_index}) must be >= -1.")
        if tweak_index >= len(timesteps):
            raise ValueError(f"`tweak_index` ({tweak_index}) must be < {len(timesteps)}.")

        if tstrong_index < 0:
            raise ValueError(f"`tstrong_index` ({tstrong_index}) must be >= 0.")
        if tstrong_index >= len(timesteps):
            raise ValueError(f"`tstrong_index` ({tstrong_index}) must be < {len(timesteps)}.")
        if tstrong_index < max(0, tweak_index):
            raise ValueError(f"`tstrong_index` ({tstrong_index}) must be >= `tweak_index` ({tweak_index}).")
                            
        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        if self.config.expand_timesteps:
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

         # 6. Initialize for TTM
        ref_vid = load_video_to_tensor(motion_signal_video_path).to(device=device) # shape [1, C, T, H, W]
        refB, refC, refT, refH, refW = ref_vid.shape

        ref_vid = F.interpolate(
            ref_vid.permute(0, 2, 1, 3, 4).reshape(refB*refT, refC, refH, refW),
            size=(height, width), mode="bicubic", align_corners=True, 
        ).reshape(refB, refT, refC, height, width).permute(0, 2, 1, 3, 4)

        ref_vid = self.video_processor.normalize(ref_vid.to(dtype=self.vae.dtype))         # [1, C, T, H, W]
        ref_latents = retrieve_latents(self.vae.encode(ref_vid), sample_mode="argmax")     # [1, z, T', H', W']
        latents_mean = torch.tensor(self.vae.config.latents_mean)\
            .view(1, self.vae.config.z_dim, 1, 1, 1).to(ref_latents.device, ref_latents.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)\
            .view(1, self.vae.config.z_dim, 1, 1, 1).to(ref_latents.device, ref_latents.dtype)
        ref_latents = (ref_latents - latents_mean) * latents_std      

        
        ref_mask = load_video_to_tensor(motion_signal_mask_path).to(device=device) # shape [1, C, T, H, W]
        mB, mC, mT, mH, mW = ref_mask.shape
        ref_mask = F.interpolate(
            ref_mask.permute(0, 2, 1, 3, 4).reshape(mB*mT, mC, mH, mW),
            size=(height, width), mode="nearest", 
        ).reshape(mB, mT, mC, height, width).permute(0, 2, 1, 3, 4) # [1, C, T, H, W] -> [T, C, H, W]
        mask_tc_hw = ref_mask[0].permute(1, 0, 2, 3).contiguous()

        if mask_tc_hw.shape[0] > num_frames: # Align time dimension to num_frames
            logger.warning("Mask has %d frames but num_frames=%d; trimming.", mask_tc_hw.shape[0], num_frames)
            mask_tc_hw = mask_tc_hw[:num_frames]
        elif mask_tc_hw.shape[0] < num_frames:
            raise ValueError(f"num_frames ({num_frames}) is greater than mask frames ({mask_tc_hw.shape[0]}). "
                            "Please pad/extend your mask or lower num_frames.")
        
        if mask_tc_hw.shape[1] > 1: # Reduce channels if needed -> [T,1,H,W], binarize once
            mask_t1_hw = (mask_tc_hw > 0.5).any(dim=1, keepdim=True).float()
        else:
            mask_t1_hw = (mask_tc_hw > 0.5).float()

        motion_mask = self.convert_rgb_mask_to_latent_mask(mask_t1_hw).permute(0, 2, 1, 3, 4).contiguous()
        background_mask = 1.0 - motion_mask

        if tweak_index >= 0:
            tweak = timesteps[tweak_index]
            fixed_noise = randn_tensor(
                ref_latents.shape,
                generator=generator,
                device=ref_latents.device,
                dtype=ref_latents.dtype,
            )
            tweak = torch.as_tensor(tweak, device=ref_latents.device, dtype=torch.long).view(1)
            noisy_latents = self.scheduler.add_noise(ref_latents, fixed_noise, tweak.long())
            latents = noisy_latents.to(dtype=latents.dtype, device=latents.device)
        else:
            tweak = torch.tensor(-1)
            fixed_noise = randn_tensor(
                ref_latents.shape,
                generator=generator,
                device=ref_latents.device,
                dtype=ref_latents.dtype,
            )
            tweak_index = 0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=len(timesteps) - tweak_index) as progress_bar:
            for i, t in enumerate(timesteps[tweak_index:]):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                if self.config.expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # In between tweak and tstrong, replace mask with noisy reference latents
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                in_between_tweak_tstrong = (i+tweak_index) < tstrong_index

                if in_between_tweak_tstrong:
                    if i+tweak_index+1 < len(timesteps):
                        prev_t = timesteps[i+tweak_index+1]
                        prev_t = torch.as_tensor(prev_t, device=ref_latents.device, dtype=torch.long).view(1)
                        noisy_latents = self.scheduler.add_noise(ref_latents, fixed_noise, prev_t.long()).to(dtype=latents.dtype, device=latents.device)
                        latents = latents * background_mask + noisy_latents * motion_mask
                    else:
                        latents = latents * background_mask + ref_latents.to(dtype=latents.dtype, device=latents.device) * motion_mask
                    

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - tweak_index - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if self.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]

            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)