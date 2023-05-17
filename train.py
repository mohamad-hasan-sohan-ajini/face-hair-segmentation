from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import SegmentationDataModule, train_transforms, val_transforms
from model_ce import UNet
from model_fl import UNet

# data module
datamodule = SegmentationDataModule(
    dataset_path_regex='data/lfw-funneled/lfw_funneled/*/*.jpg',
    labels_base_path='data/parts_lfw_funneled_gt_images',
    batch_size=32,
    num_workers=8,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
)
datamodule.setup()

# model
model = UNet(in_channels=3, out_channels=3, init_features=32)

# callbacks
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor='val_accuracy',
    mode='max',
    save_last=True,
    every_n_train_steps=100,
)

# trainer and start training
trainer = Trainer(
    devices=1,
    max_epochs=100,
    callbacks=[checkpoint_callback],
    # precision=16,
)
trainer.fit(model, datamodule)
