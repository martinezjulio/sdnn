import torch
import numpy

def init_mask(dim):
    #return torch.randn(1,dim,1,1)
    return torch.ones(dim)
    #return torch.FloatTensor(1, dim, 1, 1).uniform_(0, 1).round()

class VGG16LotteryTicket(torch.nn.Module):
    def __init__(self, model=None):
        super(VGG16LotteryTicket, self).__init__()
        
        # features
        W0 = model.module.features[0].weight.cpu().detach().numpy()
        b0 = model.module.features[0].bias.cpu().detach().numpy()
        self.W0 = torch.nn.Parameter(torch.from_numpy(W0), requires_grad=False)
        self.b0 = torch.nn.Parameter(torch.from_numpy(b0), requires_grad=False)
        self.m0 = torch.nn.Parameter(init_mask(W0.shape), requires_grad=True)
        

        W2 = model.module.features[2].weight.cpu().detach().numpy()
        b2 = model.module.features[2].bias.cpu().detach().numpy()
        self.W2 = torch.nn.Parameter(torch.from_numpy(W2), requires_grad=False)
        self.b2 = torch.nn.Parameter(torch.from_numpy(b2), requires_grad=False)
        self.m2 = torch.nn.Parameter(init_mask(W2.shape), requires_grad=True)
        

        W5 = model.module.features[5].weight.cpu().detach().numpy()
        b5 = model.module.features[5].bias.cpu().detach().numpy()
        self.W5 = torch.nn.Parameter(torch.from_numpy(W5), requires_grad=False)
        self.b5 = torch.nn.Parameter(torch.from_numpy(b5), requires_grad=False)
        self.m5 = torch.nn.Parameter(init_mask(W5.shape[1]), requires_grad=True)

        W7 = model.module.features[7].weight.cpu().detach().numpy()
        b7 = model.module.features[7].bias.cpu().detach().numpy()
        self.W7 = torch.nn.Parameter(torch.from_numpy(W7), requires_grad=False)
        self.b7 = torch.nn.Parameter(torch.from_numpy(b7), requires_grad=False)
        self.m7 = torch.nn.Parameter(init_mask(W7.shape[1]), requires_grad=True)

        W10 = model.module.features[10].weight.cpu().detach().numpy()
        b10 = model.module.features[10].bias.cpu().detach().numpy()
        self.W10 = torch.nn.Parameter(torch.from_numpy(W10), requires_grad=False)
        self.b10 = torch.nn.Parameter(torch.from_numpy(b10), requires_grad=False)
        self.m10 = torch.nn.Parameter(init_mask(W10.shape[1]), requires_grad=True)

        W12 = model.module.features[12].weight.cpu().detach().numpy()
        b12 = model.module.features[12].bias.cpu().detach().numpy()
        self.W12 = torch.nn.Parameter(torch.from_numpy(W12), requires_grad=False)
        self.b12 = torch.nn.Parameter(torch.from_numpy(b12), requires_grad=False)
        self.m12 = torch.nn.Parameter(init_mask(W12.shape[1]), requires_grad=True)

        W14 = model.module.features[14].weight.cpu().detach().numpy()
        b14 = model.module.features[14].bias.cpu().detach().numpy()
        self.W14 = torch.nn.Parameter(torch.from_numpy(W14), requires_grad=False)
        self.b14 = torch.nn.Parameter(torch.from_numpy(b14), requires_grad=False)
        self.m14 = torch.nn.Parameter(init_mask(W14.shape[1]), requires_grad=True)

        W17 = model.module.features[17].weight.cpu().detach().numpy()
        b17 = model.module.features[17].bias.cpu().detach().numpy()
        self.W17 = torch.nn.Parameter(torch.from_numpy(W17), requires_grad=False)
        self.b17 = torch.nn.Parameter(torch.from_numpy(b17), requires_grad=False)
        self.m17 = torch.nn.Parameter(init_mask(W17.shape[1]), requires_grad=True)

        W19 = model.module.features[19].weight.cpu().detach().numpy()
        b19 = model.module.features[19].bias.cpu().detach().numpy()
        self.W19 = torch.nn.Parameter(torch.from_numpy(W19), requires_grad=False)
        self.b19 = torch.nn.Parameter(torch.from_numpy(b19), requires_grad=False)
        self.m19 = torch.nn.Parameter(init_mask(W19.shape[1]), requires_grad=True)

        W21 = model.module.features[21].weight.cpu().detach().numpy()
        b21 = model.module.features[21].bias.cpu().detach().numpy()
        self.W21 = torch.nn.Parameter(torch.from_numpy(W21), requires_grad=False)
        self.b21 = torch.nn.Parameter(torch.from_numpy(b21), requires_grad=False)
        self.m21 = torch.nn.Parameter(init_mask(W21.shape[1]), requires_grad=True)

        W24 = model.module.features[24].weight.cpu().detach().numpy()
        b24 = model.module.features[24].bias.cpu().detach().numpy()
        self.W24 = torch.nn.Parameter(torch.from_numpy(W24), requires_grad=False)
        self.b24 = torch.nn.Parameter(torch.from_numpy(b24), requires_grad=False)
        self.m24 = torch.nn.Parameter(init_mask(W24.shape[1]), requires_grad=True)

        W26 = model.module.features[26].weight.cpu().detach().numpy()
        b26 = model.module.features[26].bias.cpu().detach().numpy()
        self.W26 = torch.nn.Parameter(torch.from_numpy(W26), requires_grad=False)
        self.b26 = torch.nn.Parameter(torch.from_numpy(b26), requires_grad=False)
        self.m26 = torch.nn.Parameter(init_mask(W26.shape[1]), requires_grad=True)

        W28 = model.module.features[28].weight.cpu().detach().numpy()
        b28 = model.module.features[28].bias.cpu().detach().numpy()
        self.W28 = torch.nn.Parameter(torch.from_numpy(W28), requires_grad=False)
        self.b28 = torch.nn.Parameter(torch.from_numpy(b28), requires_grad=False)
        self.m28 = torch.nn.Parameter(init_mask(W28.shape[1]), requires_grad=True)

        # classifier
        W31 = model.module.classifier[0].weight.cpu().detach().numpy()
        b31 = model.module.classifier[0].bias.cpu().detach().numpy()
        self.W31 = torch.nn.Parameter(torch.from_numpy(W31), requires_grad=False)
        self.b31 = torch.nn.Parameter(torch.from_numpy(b31), requires_grad=False)
        self.m31 = torch.nn.Parameter(init_mask(W31.shape[1]), requires_grad=True)

        W34 = model.module.classifier[3].weight.cpu().detach().numpy()
        b34 = model.module.classifier[3].bias.cpu().detach().numpy()
        self.W34 = torch.nn.Parameter(torch.from_numpy(W34), requires_grad=False)
        self.b34 = torch.nn.Parameter(torch.from_numpy(b34), requires_grad=False)
        self.m34 = torch.nn.Parameter(init_mask(W34.shape[1]), requires_grad=True)

        W37 = model.module.classifier[6].weight.cpu().detach().numpy()
        b37 = model.module.classifier[6].bias.cpu().detach().numpy()
        self.W37 = torch.nn.Parameter(torch.from_numpy(W37), requires_grad=False)
        self.b37 = torch.nn.Parameter(torch.from_numpy(b37), requires_grad=False)
        self.m37 = torch.nn.Parameter(init_mask(W37.shape[1]), requires_grad=True)  

    def forward(self, x, activations=False, gradients=False):
        conv_layers    = [0,2,5,7,10,12,14,17,19,21,24,26,28]
        relu_layers    = [1,3,6,8,11,13,15,18,20,22,25,27,29, 32, 35]
        pool_layers    = [4,9,16,23,30]
        linear_layers  = [31,34,37]
        dropout_layers = [33,36]
        
        for i in range(38):
            if (i in conv_layers):
                weight = getattr(self, 'W' + str(i))
                bias   = getattr(self, 'b' + str(i))
                mask   = getattr(self, 't' + str(i))
                 
                x = torch.nn.functional.conv2d(input=x, weight=trace_weight, bias=bias, padding=1)
                x = torch.mul(mask, x)
            
            elif (i in linear_layers):
                weight = getattr(self, 'W' + str(i))
                bias   = getattr(self, 'b' + str(i))
                mask   = getattr(self, 't' + str(i))
                
                x = torch.nn.functional.linear(input=x, weight=weight, bias=bias)
                if i != 37: # skip mask for final layer (maybe consider zeroing out all non face units)
                    x = torch.mul(mask, x)
            
            elif (i in pool_layers):
                
                x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            elif (i in relu_layers):
                
                x = torch.nn.functional.relu(x)
                
            if (i == 30):
                x = x.view(x.size(0), -1)
        return x
    
    def getLayerMapping(self):
        conv_layers    = [0,2,5,7,10,12,14,17,19,21,24,26,28]
        relu_layers    = [1,3,6,8,11,13,15,18,20,22,25,27,29, 32, 35]
        pool_layers    = [4,9,16,23,30]
        linear_layers  = [31,34,37]
        dropout_layers = [33,36]
        subnet2net = []
     
        for i in range(38):
            if (i in conv_layers):
                subnet2net.append(i)
            elif (i in linear_layers):
                subnet2net.append(i)
            elif (i in pool_layers):
                subnet2net.append(i)
            elif (i in relu_layers):
                subnet2net.append(i)
        return subnet2net