from diffusers import StableDiffusionPipeline
from peft import LoraConfig
import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import yaml
import math
import datasets
import shutil
import logging
import transformers
import argparse
from tqdm import tqdm
from packaging import version
import torch.nn.functional as F
import torchvision.transforms as transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

sys.path.append("/home/chengtianle/MotionLora")
from data.datasets import Modified

check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Super Params for MotionLora")
    parser.add_argument("--config", type=str, required=True, help="config scripts")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        # default=500,
        default=5,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def genLoRACompatibleLinear(module, config):
    bias = module.bias is not None
    state_dict = module.state_dict()
    # module = LoRACompatibleLinear(module.in_features, module.out_features, bias, state_dict = module.state_dict())
    module = LoRACompatibleLinear(module.in_features, module.out_features, bias)
    module.load_state_dict(state_dict)
    # print(module.state_dict())
    # print(state_dict)
    # assert(False)
    module.set_lora_layer(
        LoRALinearLayer(
            in_features=module.in_features, out_features=module.out_features, rank=config["lora_config"]["r"]
        )
    )
    return module

def main():
    # 加载配置文件
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(config["lora_weights_path"], args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=config["lora_weights_path"], logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # 提取目录路径
        save_directory = os.path.dirname(config["lora_weights_path"])
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(config["lora_weights_path"]).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # 从配置文件中读取路径和超参数
    model_path = config["model_path"]
    lora_weights_path = config["lora_weights_path"]
    data_path = config["data_path"]
    num_epochs = config["num_epochs"]

    # 从本地路径加载模型，训练过程把safety_checker关了，感觉这样不会返回黑色图片，或许可以提高训练效果
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder = "unet").to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder = "vae").to(accelerator.device)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder = "text_encoder").to(accelerator.device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder = "tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder = "scheduler")

    # referred to https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q = genLoRACompatibleLinear(attn_module.to_q, config).to(accelerator.device)
        attn_module.to_k = genLoRACompatibleLinear(attn_module.to_k, config).to(accelerator.device)
        attn_module.to_v = genLoRACompatibleLinear(attn_module.to_v, config).to(accelerator.device)
        attn_module.to_out[0] = genLoRACompatibleLinear(attn_module.to_out[0], config).to(accelerator.device)

    # Freeze vae, text_encoder and unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_params = []
    for name, parameter in unet.named_parameters():
        if "lora" in name: 
            parameter.requires_grad = True
            lora_params.append(parameter)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # 加速模块
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")  

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 学习率缩放
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * config["dataloader_config"]["batch_size"] * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # 训练 LoRA 模型
    optimizer = optimizer_cls(
        lora_params,
        lr=float(config["optimizer_config"]["lr"]),
        betas=(config["optimizer_config"]["adam_beta1"], config["optimizer_config"]["adam_beta2"]),
        weight_decay=float(config["optimizer_config"]["adam_weight_decay"]),
        eps=float(config["optimizer_config"]["adam_epsilon"]),
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(examples, args.resolution)]
        )
        images = torch.tensor(original_images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        original_images = preprocessed_images.reshape(-1, 3, args.resolution, args.resolution)
        return original_images

    # 加载数据
    data = Modified(data_path)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            data = data.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = []
        for item in data:
            train_dataset.append(preprocess_train(item))

    def collate_fn(examples):
        original_pixel_values = torch.stack([example for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        return original_pixel_values

    dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config["dataloader_config"]["batch_size"], shuffle=config["dataloader_config"]["shuffle"], num_workers=config["dataloader_config"]["num_workers"])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # 计算每个epoch中需要进行多少次参数更新。这是通过将数据集的大小除以梯度累积步数，并向上取整来实现的
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, lora_params, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, lora_params, optimizer, dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Motion_Lora", config=vars(args))

    # Train!
    total_batch_size = config["dataloader_config"]["batch_size"] * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f'  Instantaneous batch size per device = {config["dataloader_config"]["batch_size"]}')
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Get the text embedding for conditioning.
    encoder_hidden_states = text_encoder(tokenize_captions(config["prompt"]).to(accelerator.device))[0]
    best_loss = float('inf')

    for epoch in range(first_epoch, num_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Skip steps until we reach the resumed step
            # 用于训练的中断恢复机制
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                latents = vae.encode(batch[0].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                # model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config["dataloader_config"]["batch_size"])).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            # 同步梯度
            if accelerator.sync_gradients:
                # EMA更新
                if args.use_ema:
                    ema_unet.step(lora_params)
                # 更新进度条和日志记录
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # 检查点保存
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # 保存最低损失的检查点
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            save_path = config["lora_weights_path"]
                            print("Save!")

                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            
                            # accelerator.save_state(save_path)
                            unet.save_attn_procs(save_path)
                            logger.info(f"Saved best state to {save_path} with loss {best_loss}")

            # 更新进度条的后缀信息:
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # 终止条件
            if global_step >= args.max_train_steps:
                break

if __name__ == "__main__":
    main()
