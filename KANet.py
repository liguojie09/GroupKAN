import torch
import torch.nn.functional as F

from utils import *
# from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from kan import KANLinear

__all__ = ['GroupKANLayer', 'GroupKANBlock', 'PWDWConv', 'PatchEmbed', 'ConvLayer', 'D_ConvLayer', 'KANet']

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# to_2tuple
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

# trunc_normal_
def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor

class GroupKANChannelBlock(nn.Module):
    def __init__(self, in_channels, group=16, grid_size=5, spline_order=3):
        super(GroupKANChannelBlock, self).__init__()
        assert in_channels % group == 0, "in_channels must be divisible by group"

        self.in_channels = in_channels
        self.group = group
        self.ch_per_group = in_channels // group
        # init group-wise KANLinear modules
        self.group_kan = nn.ModuleList([
            KANLinear(self.ch_per_group, self.ch_per_group, grid_size=grid_size, spline_order=spline_order)
            for _ in range(group)
        ])

    def forward(self, x):
        """
        Args:
            x: tensor of shape [B, N, C]
        Returns:
            x_out: tensor of shape [B, N, C], after group-wise KAN mapping
        """
        B, N, C = x.shape
        group_outputs = []
        for g in range(self.group):
            ch_start = g * self.ch_per_group
            ch_end = (g + 1) * self.ch_per_group
            x_g = x[:, :, ch_start:ch_end]  # [B, N, ch_per_group]
            x_g = x_g.reshape(B * N, self.ch_per_group)  # [B*N, ch_per_group]
            kan_out = self.group_kan[g](x_g)  # [B*N, ch_per_group]
            kan_out = kan_out.view(B, N, ch_end - ch_start)  # [B, N, ch_per_group]
            group_outputs.append(kan_out)
        x_out = torch.cat(group_outputs, dim=2)  # [B, N, C]
        return x_out

class GroupKANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False,
                 active_kan_group=16, group_kan_num=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.active_kan_group = active_kan_group
        self.group_kan_num = group_kan_num
        self.dim = in_features
        self.group_kan = nn.ModuleList([
            KANLinear(1, 1, grid_size=5, spline_order=3) for _ in range(self.active_kan_group)
        ])
        if not no_kan:
            self.fc1 = GroupKANChannelBlock(in_features, group=group_kan_num)
            self.fc2 = GroupKANChannelBlock(in_features, group=group_kan_num)
            self.fc3 = GroupKANChannelBlock(in_features, group=group_kan_num)
            # self.fc4 = GroupKANChannelBlock(in_features, group=group_kan_num)
            # self.fc5 = GroupKANChannelBlock(in_features, group=group_kan_num)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.pwdwconv_1 = PWDWConv(hidden_features)
        self.pwdwconv_2 = PWDWConv(hidden_features)
        self.pwdwconv_3 = PWDWConv(hidden_features)
        # self.pwdwconv_4 = PWDWConv(hidden_features)
        # self.pwdwconv_5 = PWDWConv(hidden_features)

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        # pdb.set_trace()
        # KAN activation
        B, N, C = x.shape
        group = self.active_kan_group
        ch_per_group = C // group
        group_outputs = []
        for g in range(group):
            ch_start = g * ch_per_group
            ch_end = (g + 1) * ch_per_group if g < group - 1 else C
            x_g = x[:, :, ch_start:ch_end]
            x_g = x_g.reshape(-1, 1)
            kan_out = self.group_kan[g](x_g)
            group_outputs.append(kan_out.view(B, N, ch_per_group))
        x_grouped = torch.cat(group_outputs, dim=2)  # [B, N, groups]

        x = self.fc1(x_grouped)
        x = x.reshape(B,N,C).contiguous()
        x = self.pwdwconv_1(x, H, W)
        x = self.fc2(x)
        x = x.reshape(B,N,C).contiguous()
        x = self.pwdwconv_2(x, H, W)
        x = self.fc3(x)
        x = x.reshape(B,N,C).contiguous()
        x = self.pwdwconv_3(x, H, W)
        return x

class GroupKANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False,
                 active_kan_group=16, group_kan_num=16):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = GroupKANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan,
                                   active_kan_group=active_kan_group, group_kan_num=group_kan_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x

class PWDWConv(nn.Module):
    def __init__(self, dim=768, expansion=1):
        super(PWDWConv, self).__init__()
        hidden_dim = dim * expansion

        # 1x1 Conv
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 DW Conv
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # x = self.pwconv2(x)
        # x = self.bn3(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class KANet(nn.Module):
    def __init__(self, num_classes, input_channels=3,  img_size=224, embed_dims=[128, 256, 320], no_kan=False,
    drop_rate=0., drop_path_rate=0., active_kan_group=16, group_kan_num=16, norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(input_channels, kan_input_dim//8)
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([GroupKANBlock(
            dim=embed_dims[1],no_kan=no_kan,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            active_kan_group=active_kan_group, group_kan_num=group_kan_num
            )])

        self.block2 = nn.ModuleList([GroupKANBlock(
            dim=embed_dims[2],no_kan=no_kan,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            active_kan_group=active_kan_group, group_kan_num=group_kan_num
            )])

        self.dblock1 = nn.ModuleList([GroupKANBlock(
            dim=embed_dims[1], no_kan=no_kan,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            active_kan_group=active_kan_group, group_kan_num=group_kan_num
            )])

        self.dblock2 = nn.ModuleList([GroupKANBlock(
            dim=embed_dims[0],no_kan=no_kan,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            active_kan_group=active_kan_group, group_kan_num=group_kan_num
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4)
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

if __name__ == "__main__":
    model = KANet(input_channels=3, num_classes=1, embed_dims=[128, 160, 256], active_kan_group=16, group_kan_num=16)
    print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)
