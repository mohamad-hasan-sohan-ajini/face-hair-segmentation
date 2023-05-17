from collections import OrderedDict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Accuracy


class UNet(LightningModule):

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            init_features: int = 32,
    ) -> None:
        super(UNet, self).__init__()
        self.save_hyperparameters()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name='enc1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name='enc2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name='enc3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name='enc4')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(
            features * 8,
            features * 16,
            name='bottleneck',
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16,
            features * 8,
            kernel_size=2,
            stride=2,
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2,
            features * 8,
            name='dec4',
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8,
            features * 4,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2,
            features * 4,
            name='dec3',
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4,
            features * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2,
            features * 2,
            name='dec2',
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2,
            features,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = UNet._block(features * 2, features, name='dec1')

        self.conv = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1,
        )

        class_weights = torch.FloatTensor([.43, .36, .21])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.metric = Accuracy(task='multiclass', num_classes=3)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + 'conv1',
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + 'norm1', nn.BatchNorm2d(num_features=features)),
                    (name + 'lrelu1', nn.LeakyReLU(inplace=True)),
                    (
                        name + 'conv2',
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + 'norm2', nn.BatchNorm2d(num_features=features)),
                    (name + 'lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
        return {'optimizer': optimizer}

    def _process_target(self, target_: Tensor) -> Tensor:
        target = target_.clone()
        # offset blue channel to be selected in black areas
        target[..., 2] = torch.maximum(target[..., 2], target[..., 2] + 2)
        target_indices = target.argmax(dim=3)
        target_onehot = nn.functional.one_hot(target_indices).float()
        return target_indices, target_onehot

    def _step(self, batch: tuple[Tensor]) -> Tensor:
        x, target = batch
        target_indices, _ = self._process_target(target)
        prediction = self(x)
        loss = self.criterion(prediction, target_indices)
        return prediction, loss

    def training_step(self, batch: tuple[Tensor], _: int) -> dict[str, Tensor]:
        _, loss = self._step(batch)
        self.log('train_loss', loss.detach().cpu().item())
        return {'loss': loss}

    def validation_step(
            self,
            batch: tuple[Tensor],
            _: int,
    ) -> dict[str, Tensor]:
        pred, loss = self._step(batch)
        self.log('val_loss', loss.detach().cpu().item())
        _, target = batch
        target_indices, _ = self._process_target(target)
        pred = pred.softmax(dim=1)
        accuracy = self.metric(pred, target_indices)
        self.log('val_accuracy', accuracy)
        return {'loss': loss, 'val_accuracy': accuracy}
