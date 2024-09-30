# from re import I
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_grouper, furthest_point_sample, three_interpolation
from utils.cutils import knn_edge_maxpooling
import copy
from einops import repeat

class DirectionalEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_direction = 32
        self.num_head = 1
        self.dir_vectors = nn.Parameter(torch.randn(self.num_direction, 3))
        self.linear = nn.Sequential(
            nn.Conv1d(self.num_direction, out_channels//2, 1, bias=False, groups=self.num_head),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU(),
            nn.Conv1d(out_channels//2, out_channels, 1,),
        )

    def forward(self, dp, f0, idx):
        B, C, N, K = dp.shape
        dp_norm = nn.functional.normalize(dp) # [b c n k]
        vec_norm = nn.functional.normalize(self.dir_vectors)
        vec_norm = repeat(vec_norm, 'm d -> b m d', b=B)
        theta = torch.bmm(vec_norm, dp_norm.view(B, C, -1)).view(B, self.num_direction, N, K) # [b m n k]
        theta_max = theta.max(dim=-1)[0] # [b m n]
        f = self.linear(theta_max)
        return f
                

class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_ratio=2):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, hid_dim, 1, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
            nn.Conv1d(hid_dim, in_dim, 1, bias=False),
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class LPFMA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, out_dim, 1)
        self.linear2 = nn.Conv1d(in_dim, out_dim, 1)
        self.linear3 = nn.Conv1d(in_dim, out_dim, 1)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, f, idx):
        B, C, N = f.shape
        f1 = self.linear1(f)
        f2 = self.linear2(f)
        f3 = self.linear3(f)
        f_agg = knn_edge_maxpooling(f1.transpose(1, 2).contiguous(), idx.long(), self.training).transpose(1, 2).contiguous()
        f_agg2 = knn_edge_maxpooling(f2.transpose(1, 2).contiguous(), idx.long(), self.training).transpose(1, 2).contiguous() + f2
        f_agg = self.bn(f_agg + f_agg2 + f3)
        return f_agg


class SetAbstraction(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 use_res=False,
                 is_head=False,
                 bn_after=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        if self.use_res:
            self.skipconv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels) if bn_after else nn.Identity(),
                ) if in_channels != out_channels else nn.Identity()
        if self.all_aggr:
            out_channels = 1024 * 2
            self.convs1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
        elif not self.is_head:
            self.convs1 = nn.Conv1d(in_channels, out_channels, 1)
            self.linear_after = nn.BatchNorm1d(out_channels)
            self.dir_encoder = DirectionalEncoding(3, out_channels)
        else:
            self.convs1 = LPFMA(in_channels, out_channels)
            self.dir_encoder = DirectionalEncoding(3, out_channels)
        self.grouper = create_grouper(group_args)
        self.sample_fn = furthest_point_sample

    def forward(self, pf_pe):
        p, f, pe_prev = pf_pe
        if self.all_aggr:
            f = (self.convs1(f)).max(dim=-1)[0]
        elif self.is_head:
            dp, fj, qidx = self.grouper(p, p, f)
            f = self.convs1(f, qidx)
            pe = self.dir_encoder(dp, fj, None)
            f = f + pe
        else:
            idx = self.sample_fn(p, p.shape[1] // self.stride).long()
            new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            if self.use_res:
                fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                identity = self.skipconv(fi)
            # preconv
            f = self.convs1(f)
            dp, fj, qidx = self.grouper(new_p, p, f)
            pe = self.dir_encoder(dp, f, qidx)
            f = self.linear_after(fj.max(dim=-1)[0])
            f = f + pe
            f = f + identity
            p = new_p
        return p, f, None


class DEBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 bn_after=False,
                 ):
        super().__init__()
        self.dir_encoder = DirectionalEncoding(3, in_channels)
        self.agg = LPFMA(in_channels, in_channels)
        self.mlp = Mlp(in_channels, 2)
        self.bn0 = nn.BatchNorm1d(in_channels) if bn_after else nn.Identity()
        self.bn1 = nn.BatchNorm1d(in_channels) if bn_after else nn.Identity()

    def forward(self, pf_pe0_pe_qidx):
        p, f, dp, pe_prev, qidx = pf_pe0_pe_qidx
        pe = self.dir_encoder(dp, f, qidx)
        f = f + self.bn0(self.agg(f, qidx) + pe)
        f = f + self.bn1(self.mlp(f))
        return [p, f, dp, pe_prev, qidx]

class FeaturePropogation(nn.Module):
    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'gelu'},
                 linear=False,
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=None,
                                                ))
            self.convs = nn.Sequential(*convs)
        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


@MODELS.register_module()
class DENetEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[DEBlock] = 'DEBlock',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 pe_dim: int = 32,
                 bn_after: bool=False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.bn_after = bn_after
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')
        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            group_args.scale = radius_scaling
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1,
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append(create_grouper(group_args))
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1, pe_dim=pe_dim
                ))
                sgroup_args = copy.deepcopy(group_args)
                if sgroup_args.radius is not None:
                    sgroup_args.radius = sgroup_args.radius * sgroup_args.scale
                pe_grouper.append(create_grouper(sgroup_args))
        self.encoder = nn.Sequential(*encoder)
        self.pe_grouper = pe_grouper
        self.out_channels = channels[-1]
        self.out_channels = 1024 * 2
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False, pe_dim=32):
        if blocks > 1:
            return nn.ModuleList()
        else:
            return nn.ModuleList()

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]

        layers.append(SetAbstraction(self.in_channels, channels,
                                     stride,
                                     group_args=group_args,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, bn_after=self.bn_after,
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i] 
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels, bn_after=self.bn_after))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        pe = None
        for i in range(0, len(self.encoder)):
            p0, f0, pe = self.encoder[i][0]([p0, f0, pe])
            if self.blocks[i] > 1:
                dp, _, qidx = self.pe_grouper[i](p0, p0, None)
                pe_defined = dp
                p0, f0, _, pe, _ = self.encoder[i][1:]([p0, f0, pe_defined, pe, qidx])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        pe = None
        for i in range(0, len(self.encoder)):
            if False:
            #if i == 0:
                _p, _f, pe = self.encoder[i]([p[-1], f[-1], pe])
            else:
                _p, _f, pe = self.encoder[i][0]([p[-1], f[-1], pe])
                if self.blocks[i] > 1:
                    # grouping
                    dp, _, qidx = self.pe_grouper[i](_p, _p, None)
                    pe_defined = dp
                    _p, _f, _, pe, _ = self.encoder[i][1:]([_p, _f, pe_defined, pe, qidx])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)



@MODELS.register_module()
class DENetDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        fp_channels[-n_decoder_stages] = 256
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            linear = True if i == -n_decoder_stages else False
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], linear=linear)
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, linear=False):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        # mlp = [self.in_channels] + \
        #      [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, linear=linear))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]


@MODELS.register_module()
class DENetPartDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_blocks: List[int] = [1, 1, 1, 1],
                 decoder_strides: List[int] = [4, 4, 4, 4],
                 act_args: str = 'gelu',
                 cls_map='pointnet2',
                 num_classes: int = 16,
                 cls2partembed=None,
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]
        n_decoder_stages = len(fp_channels)
        # the following is for decoder blocks
        self.conv_args = kwargs.get('conv_args', None)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)
        block = kwargs.get('block', 'DEBlock')
        if isinstance(block, str):
            block = eval(block)
        self.blocks = decoder_blocks
        self.strides = decoder_strides
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.expansion = kwargs.get('expansion', 4)
        radius = kwargs.get('radius', 0.1)
        nsample = kwargs.get('nsample', 16)
        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        self.cls_map = cls_map
        self.num_classes = num_classes
        self.use_res = kwargs.get('use_res', True)
        group_args = kwargs.get('group_args', {'NAME': 'ballquery'})
        self.aggr_args = kwargs.get('aggr_args', 
                                    {'feature_type': 'dp_fj', "reduction": 'max'}
                                    )  
        if self.cls_map == 'custom':
            self.convc = nn.Sequential(
                nn.Conv1d(16, 64, 1, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, skip_channels[0], 1, bias=False),
            )

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i], group_args=group_args, block=block, blocks=self.blocks[i])

        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels, group_args=None, block=None, blocks=1):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp, act_args=self.act_args))
        self.in_channels = fp_channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def forward(self, p, f, cls_label):
        B, N = p[0].shape[0:2]
        if self.cls_map == 'custom':
            cls_one_hot = torch.zeros((B, self.num_classes), device=p[0].device)
            cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
            cls_one_hot = self.convc(cls_one_hot)

        for i in range(-1, -len(self.decoder), -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i-1], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

        f[-len(self.decoder) - 1] = self.decoder[0][1:](
            [p[1], self.decoder[0][0]([p[1], f[1] + cls_one_hot], [p[2], f[2]])])[1]
        return f[-len(self.decoder) - 1]
