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
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py

try:
    from dataclasses import dataclass
    import math
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
    from diffusers.image_processor import PipelineImageInput
    from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
    from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
    from diffusers.utils import (
        is_torch_xla_available,
        logging,
        replace_example_docstring,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.video_processor import VideoProcessor
    from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
    from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps
    from diffusers import CogVideoXImageToVideoPipeline

    import torch.nn.functional as F
    from pipelines.utils import load_video_to_tensor

except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/zai-org/CogVideo")


try:
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm
        XLA_AVAILABLE = True
    else:
        XLA_AVAILABLE = False
except ImportError:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
"""


class CogVideoXImageToVideoTTMPipeline(CogVideoXImageToVideoPipeline):
    r"""
    Pipeline for image-to-video generation using CogVideoX combined with Time to Move (TTM).
    This model inherits from [`CogVideoXImageToVideoPipeline`].
    """
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scaling_factor_image = self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.7
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor, vae_scale_factor: float = None) -> torch.Tensor:
        """Encode video frames into latent space with shape  (B, F, C, H, W). Input shape (B, C, F, H, W), expected range [-1, 1]."""
        latents = self.vae.encode(frames)[0].sample()
        # latents = self.vae.encode(frames)[0].mode()
        vae_scale_factor = vae_scale_factor or self.vae_scaling_factor_image
        latents = latents * vae_scale_factor
        return latents.permute(0, 2, 1, 3, 4).contiguous() # shape (B, C, F, H, W) -> (B, F, C, H, W)


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
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        motion_signal_video_path: Optional[str] = None,
        motion_signal_mask_path: Optional[str] = None,
        tweak_index: int = 0,
        tstrong_index: int = 0
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

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
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.
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
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if motion_signal_mask_path is None:
            raise ValueError("`motion_signal_mask_path` is required for TTM.")
        if motion_signal_video_path is None:
            raise ValueError("`motion_signal_video_path` is required for TTM.")

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 9. Initialize for TTM
        ref_vid = load_video_to_tensor(motion_signal_video_path).to(device=device) # shape [1, C, T, H, W]
        refB, refC, refT, refH, refW = ref_vid.shape
        ref_vid = F.interpolate(
            ref_vid.permute(0, 2, 1, 3, 4).reshape(refB*refT, refC, refH, refW),
            size=(height, width), mode="bicubic", align_corners=True, 
        ).reshape(refB, refT, refC, height, width).permute(0, 2, 1, 3, 4)

        ref_vid = self.video_processor.normalize(ref_vid.to(dtype=self.vae.dtype)) # Normalize and convert dtype for VAE encoding 
        ref_latents = self.encode_frames(ref_vid).float().detach() # shape [1, T, C, H, W]

        ref_mask = load_video_to_tensor(motion_signal_mask_path).to(device=device) # shape [1, C, T, H, W]
        mB, mC, mT, mH, mW = ref_mask.shape

        ref_mask = F.interpolate(
                ref_mask.permute(0, 2, 1, 3, 4).reshape(mB*mT, mC, mH, mW),
                size=(height, width), mode="nearest", 
            ).reshape(mB, mT, mC, height, width).permute(0, 2, 1, 3, 4)
        ref_mask = ref_mask[0].permute(1, 0, 2, 3).contiguous() # (1, C, T, H, W) -> (T, H, W, 1)

        if len(ref_mask.shape) == 4:
            ref_mask = ref_mask.unsqueeze(0)

        ref_mask = ref_mask[0,:,:1].contiguous() # (1, T, C, H, W) -> (T, 1, H, W)
        ref_mask = (ref_mask > 0.5).float().max(dim=1, keepdim=True)[0] # [T, 1, H, W]
        motion_mask = self.convert_rgb_mask_to_latent_mask(ref_mask)  # [1, T, 1, H, W]
        background_mask = 1.0 - motion_mask

        if tweak_index >= 0:
            tweak = self.scheduler.timesteps[tweak_index]
            fixed_noise = randn_tensor(
                ref_latents.shape,
                generator=generator,
                device=ref_latents.device,
                dtype=ref_latents.dtype,
            )
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

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # logging
        # ------------------------------------------------------------------
        with self.progress_bar(total=len(timesteps) - tweak_index) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps[tweak_index:]):
                if self.interrupt:
                    continue
                
                self._current_timestep = t
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents, old_pred_original_sample = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )

                # In between tweak and tstrong, replace mask with noisy reference latents
                in_between_tweak_tstrong = (i+tweak_index) < tstrong_index
                
                if in_between_tweak_tstrong:
                    if i+tweak_index+1 < len(timesteps):
                        prev_t = timesteps[i+tweak_index+1]
                        noisy_latents = self.scheduler.add_noise(ref_latents, fixed_noise, prev_t.long()).to(dtype=latents.dtype, device=latents.device)
                        latents = latents * background_mask + noisy_latents * motion_mask
                    else:
                        latents = latents * background_mask + ref_latents * motion_mask
                    
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            frames = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(
            frames=video,
            )