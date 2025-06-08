import torch
import torch.nn as nn
import torch.nn.functional as F
from decoders import UGC
from .maxvit import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from .maxvit import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class LightweightReconstructionModel(nn.Module):
    def __init__(self):
        super(LightweightReconstructionModel, self).__init__()
        self.conv1 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_transpose2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1, x2 = x
        x = F.interpolate(x2, size=(x1.shape[-2:]), mode='bilinear')
        x = x + x1
        x = F.relu(self.conv1(x))
        x = self.upsample1(x)
        x = F.relu(self.conv_transpose2(x))
        return x

class MMC_NET(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), k=11, padding=5, conv='mr', 
                 gcb_act='gelu', activation='relu', interpolation='bilinear', **kwargs):
        super(MMC_NET, self).__init__()
        
        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.recon = LightweightReconstructionModel()
        
        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
        
        state_dict1 = torch.load('pretrained_pth/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')        
        self.backbone1.load_state_dict(state_dict1, strict=False)
        
        state_dict2 = torch.load('pretrained_pth/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')        
        self.backbone2.load_state_dict(state_dict2, strict=False)
        
        
        self.channels = [768, 384, 192, 96]
        
        # decoder initialization
        self.decoder1 = UGC(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        self.decoder2 = UGC(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder1.parameters()])))
        print('Model %s created, param count: %d' %
                ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder2.parameters()])))
                
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
            
        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        recon1 = f1[0]
        f1 = [f1[0], f1[1], f1[2], f1[3]]
        
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder1(f1[3], [f1[2], f1[1], f1[0]])

        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)
        

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        
        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)        
        p14_in = F.interpolate(p14_in, size=(x.size(2), x.size(3)), mode=self.interpolation)
        x_in = x * p14_in
                
        f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))
        recon2 = f2[0]
        f2 = [f2[0], f2[1], f2[2], f2[3]]
                    
        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)

        
        x21_o, x22_o, x23_o, x24_o = self.decoder2(f2[3]+skip1_3, [f2[2]+skip1_2, f2[1]+skip1_1, f2[0]+skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)

        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)
        
        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        
        recon_out = self.recon([recon1, recon2]) 
        
        
        return [p1, p2, p3, p4] , recon_out

if __name__ == '__main__':
    model = MERIT_GCASCADE()
    input_tensor = torch.randn(1, 3, 224, 224)

    p1, p2, p3, p4 = model(input_tensor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params}")
    print(p1.size(), p2.size(), p3.size(), p4.size())