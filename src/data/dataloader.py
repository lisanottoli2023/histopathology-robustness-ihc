import torch
from torch.utils.data import DataLoader

from .dataset import IHCDataset
from .transforms import get_train_transforms, get_val_transforms


def get_dataloaders(cfg):
	"""Build train/val/test dataloaders from the OmegaConf config."""
	train_dataset = IHCDataset(
		root_dir=cfg.data.dataset_path_train,
		transform=get_train_transforms(cfg.data.image_size),
		stains_filter=cfg.data.stains_filter,
	)
	val_dataset = IHCDataset(
		root_dir=cfg.data.dataset_path_val,
		transform=get_val_transforms(cfg.data.image_size),
		stains_filter=cfg.data.stains_filter,
	)
	test_dataset = IHCDataset(
		root_dir=cfg.data.dataset_path_test,
		transform=get_val_transforms(cfg.data.image_size),
		stains_filter=cfg.data.stains_filter,
	)

	common_loader_kwargs = {
		"batch_size": cfg.training.batch_size,
		"num_workers": cfg.data.num_workers,
		"pin_memory": torch.cuda.is_available(),
		"persistent_workers": cfg.data.num_workers > 0,
	}

	train_loader = DataLoader(
		train_dataset,
		shuffle=True,
		**common_loader_kwargs,
	)
	val_loader = DataLoader(
		val_dataset,
		shuffle=False,
		**common_loader_kwargs,
	)
	test_loader = DataLoader(
		test_dataset,
		shuffle=False,
		**common_loader_kwargs,
	)

	return train_loader, val_loader, test_loader

