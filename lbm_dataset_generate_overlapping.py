import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from datasets.sl_data import MaskedRelightDataset
import pandas as pd

from accelerate import Accelerator
from accelerate.utils import set_seed

def main():

    accelerator = Accelerator()
    device = accelerator.device
    # 'stablediffusionapi/realistic-vision-v51'
    # 'runwayml/stable-diffusion-v1-5'
    sd15_name = 'stablediffusionapi/realistic-vision-v51'
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    
    

    # Change UNet

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward


    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


    unet.forward = hooked_unet_forward

    # Load

    model_path = './models/iclight_sd15_fbc.safetensors'

    if not os.path.exists(model_path):
        download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    keys = sd_origin.keys()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, keys

    # Device

    # device = torch.device('cuda')
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    rmbg = rmbg.to(device=device, dtype=torch.float32)

    # SDP

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())


    

    # Samplers

    dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )

    # Pipelines

    t2i_pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=dpmpp_2m_sde_karras_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )

    t2i_pipe.to(device)

    i2i_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=dpmpp_2m_sde_karras_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
    i2i_pipe.to(device)

    text_encoder, vae, unet, rmbg = accelerator.prepare(text_encoder, vae, unet, rmbg)


    @torch.inference_mode()
    def encode_prompt_inner(txt: str):
        max_length = tokenizer.model_max_length
        chunk_length = tokenizer.model_max_length - 2
        id_start = tokenizer.bos_token_id
        id_end = tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
        conds = text_encoder(token_ids).last_hidden_state

        return conds
    # @torch.inference_mode()
    # def encode_prompt_inner(prompts):
    #     if isinstance(prompts, str):
    #         prompts = [prompts]

    #     all_conds = []

    #     for txt in prompts:
    #         max_length = tokenizer.model_max_length
    #         chunk_length = max_length - 2
    #         id_start = tokenizer.bos_token_id
    #         id_end = tokenizer.eos_token_id
    #         id_pad = id_end

    #         def pad(x, p, i):
    #             return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    #         tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    #         chunks = [[id_start] + tokens[i:i+chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    #         chunks = [pad(chunk, id_pad, max_length) for chunk in chunks]

    #         token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    #         cond = text_encoder(token_ids).last_hidden_state
    #         all_conds.append(cond)

    #     return all_conds



    # @torch.inference_mode()
    # def encode_prompt_pair(positive_prompt, negative_prompt):
    #     c = encode_prompt_inner(positive_prompt)
    #     uc = encode_prompt_inner(negative_prompt)

    #     c_len = float(len(c))
    #     uc_len = float(len(uc))
    #     max_count = max(c_len, uc_len)
    #     c_repeat = int(math.ceil(max_count / c_len))
    #     uc_repeat = int(math.ceil(max_count / uc_len))
    #     max_chunk = max(len(c), len(uc))

    #     c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    #     uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    #     c = torch.cat([p[None, ...] for p in c], dim=1)
    #     uc = torch.cat([p[None, ...] for p in uc], dim=1)

    #     return c, uc

    @torch.inference_mode()
    def encode_prompt_pair(positive_prompts, negative_prompts):
        assert len(positive_prompts) == len(negative_prompts)

        all_c, all_uc = [], []

        for pos, neg in zip(positive_prompts, negative_prompts):
            c = encode_prompt_inner(pos)
            uc = encode_prompt_inner(neg)

            c_len = float(len(c))
            uc_len = float(len(uc))
            max_count = max(c_len, uc_len)
            c_repeat = int(math.ceil(max_count / c_len))
            uc_repeat = int(math.ceil(max_count / uc_len))
            max_chunk = max(len(c), len(uc))

            c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
            uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]
            
            c = torch.cat([p[None, ...] for p in c], dim=1)
            uc = torch.cat([p[None, ...] for p in uc], dim=1)
            
            all_c.append(c.squeeze(0))
            all_uc.append(uc.squeeze(0))

        # Stack into (B, T, D)
        return torch.stack(all_c, dim=0), torch.stack(all_uc, dim=0)



    @torch.inference_mode()
    def pytorch2numpy(imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)

            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

            results.append(y)
        return results


    @torch.inference_mode()
    def numpy2pytorch(imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
        h = h.movedim(-1, 1)
        return h

    @torch.inference_mode()
    def numpy2pytorch_fullnp(imgs):
        h = torch.from_numpy(imgs).float() / 127.0 - 1.0
        h = np.transpose(h, (0, 3, 1, 2))
        return h


    def resize_and_center_crop(image, target_width, target_height):
        out_images = []
        if len(image.shape) == 4:
            loop = image.shape[0]
            batch_img = True
        else:
            loop = 1
            batch_img = False
        for i in range(loop):
            if batch_img:
                image_i = image[i]
            else:
                image_i = image
            pil_image = Image.fromarray(image_i)
            original_width, original_height = pil_image.size
            scale_factor = max(target_width / original_width, target_height / original_height)
            resized_width = int(round(original_width * scale_factor))
            resized_height = int(round(original_height * scale_factor))
            resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
            left = (resized_width - target_width) / 2
            top = (resized_height - target_height) / 2
            right = (resized_width + target_width) / 2
            bottom = (resized_height + target_height) / 2
            cropped_image = resized_image.crop((left, top, right, bottom))
            cropped_image = np.array(cropped_image)
            out_images.append(cropped_image)
        
        return np.array(out_images)


    def resize_without_crop(image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)



    @torch.inference_mode()
    def process_batch(
        input_fgs, input_bgs, prompts,
        image_width, image_height,
        num_samples, seeds, steps,
        a_prompt, n_prompt, cfg,
        highres_scale, highres_denoise,
        bg_source
    ):
        N = len(input_fgs)
        fgs = numpy2pytorch_fullnp(input_fgs)
        processed_bgs = numpy2pytorch_fullnp(input_bgs)
        concat_inputs = torch.cat([fgs, processed_bgs], dim=0)  # shape: [2B, 3, H, W]
        concat_inputs = concat_inputs.to(device, dtype=vae.module.dtype)
        concat_latents = vae.module.encode(concat_inputs).latent_dist.mode() * vae.module.config.scaling_factor

        concat_conds = torch.cat([
            torch.cat([concat_latents[i][None], concat_latents[i + N][None]], dim=1)
            for i in range(N)
        ], dim=0)

        conds, unconds = encode_prompt_pair(
            positive_prompts=[f"{p}, {a_prompt}" for p in prompts],
            negative_prompts=[n_prompt] * N
        )
        

        generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=generators,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds}
        ).images.to(vae.module.dtype) / vae.module.config.scaling_factor

        # Decode and upscale
        decoded = vae.module.decode(latents).sample
        upscaled = [
            resize_without_crop(p, int(round(image_width * highres_scale / 64.0) * 64),
                                int(round(image_height * highres_scale / 64.0) * 64))
            for p in pytorch2numpy(decoded)
        ]
        upscaled_tensor = numpy2pytorch(upscaled).to(device, dtype=vae.module.dtype)
        latents = vae.module.encode(upscaled_tensor).latent_dist.mode() * vae.module.config.scaling_factor

        # Update size
        new_height, new_width = latents.shape[2] * 8, latents.shape[3] * 8

        # Recompute conditioning
        fgs = resize_and_center_crop(input_fgs, new_height, new_width)
        processed_bgs = resize_and_center_crop(input_bgs, new_height, new_width)
        fgs =  numpy2pytorch_fullnp(fgs)
        processed_bgs = numpy2pytorch_fullnp(processed_bgs)
        all_inputs = torch.cat([fgs, processed_bgs], dim=0)  # shape: [2B, 3, H, W]
        all_inputs = all_inputs.to(vae.module.device, dtype=vae.module.dtype)
        encoded = vae.module.encode(all_inputs).latent_dist.mode() * vae.module.config.scaling_factor

        concat_conds = torch.cat([
            torch.cat([encoded[i][None], encoded[i + N][None]], dim=1)
            for i in range(N)
        ], dim=0)

        latents = i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=new_width,
            height=new_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=generators,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds}
        ).images.to(vae.module.dtype) / vae.module.config.scaling_factor

        final_images = pytorch2numpy(vae.module.decode(latents).sample, quant=False)
        return final_images, [input_fgs, processed_bgs]

    @torch.inference_mode()
    def process_relight_batch(
        input_fgs, input_bgs, prompts,
        image_width, image_height,
        num_samples, seeds, steps,
        a_prompt, n_prompt, cfg,
        highres_scale, highres_denoise,
        bg_source
    ):
        # Assume images are already resized/cropped
        # input_fgs_cleaned, mattings = zip(*(run_rmbg(fg) for fg in input_fgs))

        results, extra_images = process_batch(
            input_fgs,
            input_bgs,
            prompts,
            image_width,
            image_height,
            num_samples,
            seeds,
            steps,
            a_prompt,
            n_prompt,
            cfg,
            highres_scale,
            highres_denoise,
            bg_source
        )

        results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
        return results, extra_images


    quick_prompts = [
        'beautiful woman',
        'handsome man',
        'beautiful woman, cinematic lighting',
        'handsome man, cinematic lighting',
        'beautiful woman, natural lighting',
        'handsome man, natural lighting',
        'beautiful woman, neo punk lighting, cyberpunk',
        'handsome man, neo punk lighting, cyberpunk',
    ]
    quick_prompts = [[x] for x in quick_prompts]


    class BGSource(Enum):
        UPLOAD = "Use Background Image"
        UPLOAD_FLIP = "Use Flipped Background Image"
        LEFT = "Left Light"
        RIGHT = "Right Light"
        TOP = "Top Light"
        BOTTOM = "Bottom Light"
        GREY = "Ambient"




    def load_image_and_mask_paths_from_metadata(
        metadata_csv,
        image_folder,
        mask_folder,
        partition='train',
        good_only=True
    ):
        df = pd.read_csv(metadata_csv)

        # Filter based on partition and good flag
        if partition:
            df = df[df['partition'] == partition]
        if good_only:
            df = df[df['good'] == True]

        # Construct full paths
        image_paths = [os.path.join(image_folder, fname) for fname in df['image_path']]
        mask_paths = [os.path.join(mask_folder, fname) for fname in df['mask_path']]

        return image_paths, mask_paths

    metadata_csv = "/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/record_v1.csv"
    image_folder = "/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images"
    mask_folder = "/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images"
    source_folder = "/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images_source2/"

    image_paths, mask_paths = load_image_and_mask_paths_from_metadata(
        metadata_csv=metadata_csv,
        image_folder=image_folder,
        mask_folder=mask_folder,
        partition='train',
        good_only=True
    )


   
    def collate_fn(batch):
        keys = batch[0].keys()
        collated = {k: [d[k] for d in batch] for k in keys}

        for k in ['input_fg', 'input_bg']:
            collated[k] = np.stack(collated[k])  # stack into (B, H, W, 3)

        # for k in ['seed']:  # force to list of ints
        #     collated[k] = [int(s) for s in collated[k]]
        for k in ['prompt']:
            collated['prompt'] = [str(prompt) for prompt in collated['prompt']]
        
        
        for k in ['path']:
            collated['path'] = [str(path) for path in collated["path"]]

        return collated


    def collate_fn_torch(batch):
        keys = batch[0].keys()
        collated = {k: [d[k] for d in batch] for k in keys}

        # Stack torch tensors
        for k in ['input_fg', 'input_bg']:
            collated[k] = torch.stack(collated[k], dim=0)  # shape: (B, 3, H, W)

        # Optionally convert 'seed' to list of ints
        if 'seed' in collated:
            collated['seed'] = [int(s) for s in collated['seed']]

        if 'path' in collated:
            collated['path'] = [str(path) for path in collated["path"]]

        return collated

    

    # Fixed hyperparameters
    image_width = 840
    image_height = 840
    num_samples = 1
    steps = 20
    a_prompt = "best quality"
    n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    cfg = 7.0
    highres_scale = 1.5
    highres_denoise = 0.5
    bg_source = "UPLOAD"

     # --- Your background samples and prompts ---
    background_samples = db_examples.bg_samples  
    prompts = [sample[0] for sample in quick_prompts]

    dataset = MaskedRelightDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        bg_paths=background_samples,       # your background image list
        prompts=prompts,           # your text prompt list
        image_width=image_width,
        image_height = image_height,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    loader = accelerator.prepare(loader)


    os.makedirs(source_folder, exist_ok=True)

    
    def is_valid_image(path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False
    # Collect all results
    all_results = []

    for batch in loader:
        input_fgs = batch["input_fg"]          # [B, 3, H, W]
        input_bgs = batch["input_bg"]          # [B, 3, H, W]
        prompts = batch["prompt"]              # list of str
        paths = batch["path"]
        seeds = [random.randint(0, 999999) for _ in range(len(prompts))]

        # Determine which items to process
        to_process_indices = []
        result_paths = []
        bg_paths = []

        for i, original_path in enumerate(paths):
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)

            result_path = os.path.join(source_folder, filename)
            bg_filename = f"{name}_bg{ext}"
            bg_path = os.path.join(source_folder, bg_filename)

            result_paths.append(result_path)
            bg_paths.append(bg_path)

            if is_valid_image(result_path) and is_valid_image(bg_path):
                print(f"Skipping existing and valid result+bg: {filename}")
                continue

            to_process_indices.append(i)

        # If all items are already processed, skip this batch
        if not to_process_indices:
            continue

        # Filter items that need processing
        filtered_fgs = input_fgs[to_process_indices]
        filtered_bgs = input_bgs[to_process_indices]
        filtered_prompts = [prompts[i] for i in to_process_indices]
        filtered_seeds = [seeds[i] for i in to_process_indices]
        filtered_paths = [paths[i] for i in to_process_indices]

        results, extras = process_relight_batch(
            input_fgs=filtered_fgs,
            input_bgs=filtered_bgs,
            prompts=filtered_prompts,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seeds=filtered_seeds,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            bg_source=bg_source
        )

        for result_img, bg_img, original_path in zip(results, filtered_bgs, filtered_paths):
            # filename = os.path.basename(original_path)
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)
            # save_path = os.path.join(source_folder, filename)
             # Save generated result
            result_path = os.path.join(source_folder, filename)
            Image.fromarray(result_img).save(result_path)

            # Save input background (already uint8)
            bg_filename = f"{name}_bg{ext}"
            bg_path = os.path.join(source_folder, bg_filename)
            Image.fromarray(bg_img).save(bg_path)
        accelerator.wait_for_everyone()

    
if __name__ == "__main__":
    main()