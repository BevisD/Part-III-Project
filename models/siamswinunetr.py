import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool3d, Conv3d, Linear, Sigmoid, LeakyReLU, Sequential
from monai.networks.nets import SwinUNETR
from typing import Sequence


class SiamSwinUNETR(nn.Module):
    def __init__(self,
                 img_size: Sequence[int] | int,
                 in_channels: int,
                 out_channels: int,
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24),
                 feature_size: int = 24,
                 norm_name: tuple | str = "instance",
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 dropout_path_rate: float = 0.0,
                 normalize: bool = True,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3,
                 conv_activation=LeakyReLU(),
                 linear_activation=Sigmoid()
                 ) -> None:
        super().__init__()

        self.swin_unetr = CachedSwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        self.conv_1 = ConvBlock(
            in_channels=192,
            out_channels=32,
            kernel_size=3,
            activation=conv_activation
        )

        self.conv_2 = ConvBlock(
            in_channels=768,
            out_channels=32,
            kernel_size=3,
            activation=conv_activation
        )

        self.conv_3 = ConvBlock(
            in_channels=192,
            out_channels=32,
            kernel_size=3,
            activation=conv_activation
        )

        self.linear = Sequential(
            Linear(96, 2),
            linear_activation
        )

    def forward(self, x):
        img_1, img_2 = x  # B 1 H W D

        seg_1, (feat1_1, feat1_2, feat1_3) = self.swin_unetr(img_1)
        seg_2, (feat2_1, feat2_2, feat2_3) = self.swin_unetr(img_2)

        diff_1 = feat1_1 - feat2_1  # B L1 H/? W/? D/?
        diff_2 = feat1_2 - feat2_2  # B L2 H/? W/? D/?
        diff_3 = feat1_3 - feat2_3  # B L3 H/? W/? D/?

        diff_1 = self.conv_1(diff_1)  # B 32 H/? W/? D/?
        diff_2 = self.conv_2(diff_2)  # B 32 H/? W/? D/?
        diff_3 = self.conv_3(diff_3)  # B 32 H/? W/? D/?

        avg_1 = AdaptiveAvgPool3d((1, 1, 1))(diff_1)  # B 32 1 1 1
        avg_2 = AdaptiveAvgPool3d((1, 1, 1))(diff_2)  # B 32 1 1 1
        avg_3 = AdaptiveAvgPool3d((1, 1, 1))(diff_3)  # B 32 1 1 1

        linear_input = torch.concat((avg_1, avg_2, avg_3), dim=1).squeeze()  # B 96
        label = self.linear(linear_input)  # B 2

        return (seg_1, seg_2), label


class CachedSwinUNETR(SwinUNETR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x_in):
        # B = Batch, H = Height, W = Width, D = Depth, F = Features
        hidden_states_out = self.swinViT(x_in, self.normalize)
        # B, 48*(2**i), (H, W, D)/(2**i)
        enc0 = self.encoder1(x_in)  # B 48 H W D
        enc1 = self.encoder2(hidden_states_out[0])  # B 48 H/2 W/2 D/2
        enc2 = self.encoder3(hidden_states_out[1])  # B 96 H/4 W/4 D/4
        enc3 = self.encoder4(hidden_states_out[2])  # B 192 H/8 W/8 D/8
        dec4 = self.encoder10(hidden_states_out[4])  # B 768 H/32 W/32 D/32
        dec3 = self.decoder5(dec4, hidden_states_out[3])  # B 384 H/16 W/316 D/16
        dec2 = self.decoder4(dec3, enc3)  # B 192 H/8 W/8 D/8
        dec1 = self.decoder3(dec2, enc2)  # B 96 H/4 W/4 D/4
        dec0 = self.decoder2(dec1, enc1)  # B 48 H/2 W/2 D/2
        out = self.decoder1(dec0, enc0)  # B 48 H W D
        logits = self.out(out)  # B F H W D

        return_layers = [
            hidden_states_out[2],  # B 192 H/8 W/8 D/8
            dec4,                  # B 768 H/32 W/32 D/32
            dec2                   # B 192 H/8 W/8 D/8
        ]
        return logits, return_layers


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding="same",
                 activation=LeakyReLU()
                 ) -> None:
        super().__init__()
        self.conv = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


def main():
    model = SiamSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )


if __name__ == '__main__':
    main()
