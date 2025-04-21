import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#EFATSR
class EFATSR(nn.Module):
    def __init__(self, up_scale=4, dim=60, groups=5,num=4):
        super(EFATSR, self).__init__()
        self.init = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.num = num
        self.body = nn.ModuleList()
        self.groups = groups
        for i in range(groups):
            self.body.append(GroupSR(dim,num=num,wsize=16))
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3 * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
    def forward(self, x0):
        x = self.init(x0)
        for i in range(self.groups):
            x  = self.body[i](x)
        x = self.up(x) + F.interpolate(x0, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        return x
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('up') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('up') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# Residual Group
class GroupSR(nn.Module):
    def __init__(self, dim, num=6,wsize=16):
        super().__init__()
        self.num = num
        self.body = nn.ModuleList()
        for i in range(num):
            self.body.append(BasicBlockSR(dim,spatial=(i%2==0),channel=(i%2==1),window_sizes = wsize ,shift=((i+2)%4==0)))
        self.conv = nn.Conv2d(dim,dim,1)
    def forward(self,x0):
        x=x0
        for i in range(self.num):
            x = self.body[i](x)
        x = self.conv(x)
        return x+x0

#FSAB or FCSB
class BasicBlockSR(nn.Module):
    def __init__(self, dim, spatial = False, channel = False, window_sizes = 8,shift = False):
        super().__init__()
        self.spatial = SelfAttention(dim,heads=2,wsize=window_sizes,shift=shift) if spatial else None
        self.channel = DWBlcok(dim,wsize=8) if channel else None
        self.window_sizes = window_sizes
        self.MLP = MLP(dim,ratio=2)
    def check_image_size2(self, x, wsize):
        _, _, h, w = x.size()
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, x):
        b,c,h,w=x.shape
        if self.spatial:
            x = self.check_image_size2(x,self.window_sizes)
            x = self.spatial(x)[:,:,:h,:w]
        if self.channel:
            x = self.check_image_size2(x,8)
            x = self.channel(x)[:,:,:h,:w]
        x = self.MLP(x)
        return x

#FCSB (excluding GDFN)    
class DWBlcok(nn.Module):
    def __init__(self,dim,wsize = 8):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.projin = nn.Conv2d(dim,dim*2,1)
        self.local = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.q = nn.Conv2d(dim,dim,1)
        self.wsize = wsize
        self.proj = nn.Conv2d(dim,dim,1)
        self.filter = nn.Parameter(torch.randn(dim, wsize, wsize // 2 + 1, 2)*0.02)
    def forward(self,x0):
        b,c,h,w = x0.shape
        x = self.norm(x0)
        x1,x2 = self.projin(x).chunk(2,dim=1)
        x1 = rearrange(x1,'b c (h dh) (w dw)->(b h w) c dh dw',dh=self.wsize,dw =self.wsize)
        x1 = torch.fft.rfft2(x1,norm='ortho')
        q_r = self.q(x1.real)
        q_i = self.q(x1.imag)
        x1 = torch.complex(q_r,q_i) * torch.view_as_complex(self.filter)
        x1 = torch.fft.irfft2(x1,norm='ortho')
        x1 = rearrange(x1,'(b h w) c dh dw->b c (h dh) (w dw)',b=b,h=h//self.wsize,w = w//self.wsize)
        x = x1 * self.local(x2)
        x = self.proj(x)        
        return x + x0

#FSAB (excluding GDFN)  
class SelfAttention(nn.Module):
    def __init__(self,dim,heads=1,wsize = 8,shift =False):
        super().__init__()
        adim = dim//2
        self.norm = LayerNorm2d(dim)
        self.adim=adim
        self.qkv = nn.Conv2d(dim,3*adim,1)
        self.wsize = wsize
        self.scale = (adim//heads) ** -0.5
        self.shift = shift
        self.softmax = nn.Softmax(dim=-1)
        self.heads = heads
        self.proj = nn.Conv2d(adim,dim,1)
        self.filter = nn.Parameter(torch.randn(adim, wsize//4, wsize // 8 + 1, 2)*0.02)
        self.fusion = GatedFusion(adim)
        self.local = nn.Sequential(nn.Conv2d(dim,adim,1),nn.Conv2d(adim,adim,3,1,1,groups=adim))
    def forward(self,x0):
        b,c,h,w = x0.shape
        x = self.norm(x0)
        qkv = self.qkv(x)
        v1 = qkv[:,-self.adim:,:,:]
        local = rearrange(v1,'b c (h dh) (w dw)->(b h w) c dh dw',dh=self.wsize//4,dw =self.wsize//4)
        local = torch.fft.rfft2(local,norm='ortho')
        weight = torch.view_as_complex(self.filter)
        local = local * weight
        local = torch.fft.irfft2(local,norm='ortho')
        local = rearrange(local,'(b h w) c dh dw->b c (h dh) (w dw)',b=b,h=h//self.wsize*4,w = w//self.wsize*4)
        local = local * self.local(x)
        if self.shift:
            qkv = torch.roll(qkv,shifts=(-self.wsize//2, -self.wsize//2), dims=(2,3))
        q,k,v = qkv.chunk(3,dim=1)
        q = rearrange(q,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        k = rearrange(k,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        v = rearrange(v,'b (hed c) (h dh) (w dw)->(b h w) hed (dh dw) c',dh=self.wsize,dw =self.wsize,hed=self.heads)
        atn = torch.matmul(q,k.transpose(-1,-2)) * self.scale 
        atn = self.softmax(atn)
        y = torch.matmul(atn,v)
        y = rearrange(y,'(b h w) hed (dh dw) c->b (hed c) (h dh) (w dw)',h = h//self.wsize,w=w//self.wsize,dh=self.wsize,dw=self.wsize)
        if self.shift:
            y =  torch.roll(y, shifts=(self.wsize//2, self.wsize//2), dims=(2, 3))
        y = self.proj(self.fusion(y,local))
        return y + x0

#ECF
class GatedFusion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.ecal = effCA(dim)
        self.ecah = effCA(dim)
    def forward(self,xl0,xh0):
        x = xl0 + xh0
        cal = self.ecal(x)
        car = self.ecah(x)
        CAs = torch.stack([cal,car],dim=1).softmax(dim=1)
        x = CAs[:,0,:,:,:]*xl0 + CAs[:,1,:,:,:]*xh0
        return x

#GDFN            
class MLP(nn.Module):
    def __init__(self, dim,ratio=2):
        super(MLP,self).__init__()
        self.layernorm1 = LayerNorm2d(dim)
        expandim = int(dim*ratio)
        self.proj1 = nn.Conv2d(dim,expandim,1)
        self.conv = nn.Conv2d(expandim,expandim,3,1,1,groups=expandim)
        self.projout = nn.Conv2d(dim,dim,1)
    def forward(self,x0):
        x = self.layernorm1(x0)
        x = self.proj1(x)
        x1,x2 = self.conv(x).chunk(2,dim=1)
        x = F.gelu(x1) * x2
        x = self.projout(x)
        return x + x0

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        return to_4d(x, h, w)
    
class effCA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(effCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,h,w = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).permute(0,2,1)
        y = self.conv1(y)
        y = self.sigmoid(y.permute(0,2,1).unsqueeze(-1))
        return y

if __name__ == "__main__":
    net = EFATSR(up_scale=4,  dim=60, groups=5,num=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('Number of params: %.2fK' % (total / 1e3))
