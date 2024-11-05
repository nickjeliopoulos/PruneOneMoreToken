import torch
import torchvision.transforms as xform
from torchvision.datasets import ImageFolder
import os

### TIMM Imports
import timm
from timm.data.loader import (
    fast_collate,
    MultiEpochsDataLoader,
    PrefetchLoader,
    partial,
    _worker_init,
)
from timm.data.transforms_factory import create_transform


### Constants
IMAGENET1K_INPUT_SIZE = (3, 224, 224)

###
### Modified TIMM Create Loader - overrides for transform and sampler
###
def create_loader(
    dataset,
    input_size,
    batch_size,
    ### CUSTOM
    ### ====================
    transform_override=None,
    sampler_override=None,
    force_shuffle=False,
    ### ====================
    is_training=False,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_split=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    num_aug_repeats=0,
    num_aug_splits=0,
    interpolation="bilinear",
    mean=timm.data.constants.IMAGENET_DEFAULT_MEAN,
    std=timm.data.constants.IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    crop_pct=None,
    crop_mode=None,
    collate_fn=None,
    pin_memory=False,
    fp16=False,  # deprecated, use img_dtype
    img_dtype=torch.float32,
    device=torch.device("cuda"),
    tf_preprocessing=False,
    use_multi_epochs_loader=False,
    persistent_workers=True,
    worker_seeding="all",
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2

    ### CUSTOM
    ### Check if user supplied transform is there
    if transform_override is not None:
        dataset.transform = transform_override
    else:
        dataset.transform = create_transform(
            input_size,
            is_training=is_training,
            use_prefetcher=use_prefetcher,
            no_aug=no_aug,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
            tf_preprocessing=tf_preprocessing,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=num_aug_splits > 0,
        )

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None

    ### Check user supplied sampler
    if sampler_override is None:
        if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
            if is_training:
                if num_aug_repeats:
                    sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
                else:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                # This will add extra duplicate entries to result in equal num
                # of samples per-process, will slightly alter validation results
                sampler = OrderedDistributedSampler(dataset)
        else:
            assert (
                num_aug_repeats == 0
            ), "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    else:
        sampler = sampler_override

    if collate_fn is None:
        collate_fn = (
            fast_collate
            if use_prefetcher
            else torch.utils.data.dataloader.default_collate
        )

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    if force_shuffle:
        do_shuffle = True
    else:
        do_shuffle = (
            not isinstance(dataset, torch.utils.data.IterableDataset)
            and sampler is None
            and is_training
        )

    loader_args = dict(
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.0
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,  # deprecated, use img_dtype
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
        )

    return loader


###
### Treat ImageNet1K as a just an ImageFolder
### Need to organize "val" folder with a script so it has the same form as the "train" folder
### See README.md
###
def create_imagenet1k_dataset(
    root: str,
    is_training: bool = False,
    *args,
    **kwargs,
) -> torch.utils.data.Dataset:

    ### Get the dataset as a ImageFolder
    imagenet1k_dataset = ImageFolder(
        root=os.path.join(root, "val" if not is_training else "train")
    )

    return imagenet1k_dataset


###
### DinoV2 Dataloader
###
def create_im1k_dinov2_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    is_training: bool,
) -> torch.utils.data.DataLoader:
    ### TIMM ImageNet1K Mean, STD
    ### Used by dinov2/data/transforms.py
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    ### Transforms (stolen from dinov2/data/transforms.py - see that on their github page
    dinov2_evaluation_transform = xform.Compose(
        [
            xform.Resize(256, interpolation=xform.InterpolationMode.BICUBIC),
            xform.CenterCrop(224),
            xform.ToTensor(),
            xform.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    ### Get TIMM Dataloader
    imagenet1k_dataloader = create_loader(
        dataset,
        input_size=IMAGENET1K_INPUT_SIZE,
        transform_override=dinov2_evaluation_transform,
        sampler_override=None,
        force_shuffle=False,
        is_training=False,
        use_prefetcher=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return imagenet1k_dataloader


###
### TIMM Dataloader
###
def create_im1k_timm_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    is_training: bool,
) -> torch.utils.data.DataLoader:
    ### Get TIMM Dataloader
    ### NOTE: the context of is_training here is for controlling the type of transform
    ### We just use the standard evaluation transform even if we are using the training split
    imagenet1k_dataloader = create_loader(
        dataset,
        input_size=IMAGENET1K_INPUT_SIZE,
        sampler_override=None,
        force_shuffle=False,
        is_training=False,
        use_prefetcher=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return imagenet1k_dataloader
