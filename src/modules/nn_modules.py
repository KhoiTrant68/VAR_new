import torch
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def group_norm(num_channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
    )


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.op = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
            if with_conv
            else nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            if with_conv
            else nn.Identity()
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.2
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1 if not conv_shortcut else 3,
                stride=1,
                padding=0 if not conv_shortcut else 1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.block = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = group_norm(in_channels)
        self.qkv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        b, c, h, w = q.shape

        q = q.reshape(b, c, -1).transpose(1, 2)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1)

        attn = torch.bmm(q, k) * (c**-0.5)
        attn = attn.softmax(dim=2)
        h = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)

        return x + self.proj_out(h)


def make_attn(in_channels, use_attn=True):
    return AttnBlock(in_channels) if use_attn else nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        z_channels=64,
        double_z=False,
        use_attn=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            if i != len(ch_mult) - 1:
                self.down_blocks.append(Downsample(in_ch))

        self.mid_block = nn.Sequential(
            ResnetBlock(in_ch, in_ch, dropout=dropout),
            make_attn(in_ch, use_attn),
            ResnetBlock(in_ch, in_ch, dropout=dropout),
        )

        self.norm_out = group_norm(in_ch)
        self.conv_out = nn.Conv2d(
            in_ch,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        return self.conv_out(nn.SiLU()(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        out_channels=3,
        z_channels=64,
        use_attn=True,
    ):
        super().__init__()
        in_ch = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, in_ch, kernel_size=3, stride=1, padding=1)

        self.mid_block = nn.Sequential(
            ResnetBlock(in_ch, in_ch, dropout=dropout),
            make_attn(in_ch, use_attn),
            ResnetBlock(in_ch, in_ch, dropout=dropout),
        )

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResnetBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            if i != 0:
                self.up_blocks.append(Upsample(in_ch))

        self.norm_out = group_norm(in_ch)
        self.conv_out = nn.Conv2d(
            in_ch, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        return self.conv_out(nn.SiLU()(self.norm_out(h)))
