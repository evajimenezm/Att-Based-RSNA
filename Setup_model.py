import torch
import re

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2) -> None:
        super(ConvBlock, self).__init__()
        self.net = torch.nn.Sequential(
                                        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
                                        torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                       )
    
    def forward(self, x):
        return self.net(x)


class CNNMIL(torch.nn.Module):
    def __init__(self, input_size=(3, 512, 512), cnn_name='cnn_2_0.0', network = 'Simple') -> None:
        super().__init__()

        # cnn_name format: cnn_<num_conv_blocks>_<p_dropout>

        r = re.compile('cnn_.*_.*')

        valid_name = r.match(cnn_name) is not None

        if valid_name:
            num_blocks = int(cnn_name.split('_')[1])
            p_dropout = float(cnn_name.split('_')[2])

            if network == 'Compleja':
                module_list = [ConvBlock(input_size[0], 16, 15, 1, 2), ConvBlock(16, 64, 10, 1, 2), ConvBlock(64, 128, 10, 1, 2), ConvBlock(128, 128*2, 5, 1, 2), ConvBlock(128*2, 64, 5, 1, 2), ConvBlock(64, 32, 3, 1, 0)]
            elif network == 'Simple':
                module_list = [ConvBlock(input_size[0], 16, 5, 1, 2), ConvBlock(16, 32, 3, 1, 0)]
            for _ in range(num_blocks-2):
                module_list.append(ConvBlock(32, 32, 3, 1, 0))
                if p_dropout > 0.0:
                    dropout = torch.nn.Dropout(p=p_dropout)
                    module_list.append(dropout)
            module_list.append(torch.nn.Flatten())
        else:
            raise NotImplementedError
        
        self.net = torch.nn.Sequential(*module_list)
        self.output_size = self._get_output_size(self.net, input_size)
    
    def forward(self, Xin):
        """
        input:
            Xin: tensor (batch_size, bag_size, C, H, W)
        output:
            Xout: tensor (batch_size, bag_size, 32*(H//64-2)*(W//64-2))
        """
        #return torch.stack([self.net(Xin[i,:,:,:,:]) for i in range(Xin.shape[0])])
        return self.net(Xin.view(-1, *Xin.shape[2:])).view(Xin.shape[0], Xin.shape[1], -1)

    def _get_output_size(self, module, input_size):
        """
        input:
            input_size: tuple (C, H, W)
        output:
            output_size: int
        """
        x = torch.randn(1, *input_size)
        return module(x).shape[-1]

class AttentionModule(torch.nn.Module):
    def __init__(self, D_dim, L_dim=50):
        super(AttentionModule, self).__init__()
        self.D_dim = D_dim
        self.L_dim = L_dim

        self.fc1 = torch.nn.Linear(D_dim, L_dim)
        self.fc2 = torch.nn.Linear(L_dim, 1, bias=False)      
    
    def forward(self, X):
        """
        input:
            X: tensor (batch_size, bag_size, D)
            L: tensor (batch_size, bag_size, bag_size)
        output:
            Z: tensor (batch_size, D)
            S: tensor (bag_size, 1)
        """

        batch_size = X.shape[0]
        bag_size = X.shape[1]
        D = X.shape[2]

        H = self.fc1(X.reshape(-1, D)).view(batch_size, bag_size, -1) # (batch_size, bag_size, L_dim)
        f = self.fc2(H.reshape(-1, self.L_dim)).view(batch_size, bag_size, -1) # (batch_size, bag_size, 1)

        s = torch.nn.functional.softmax(f, dim=1) # (batch_size, bag_size, 1) # Esto es la función que se encarga de calcular los pesos
        z = torch.matmul(X.transpose(1,2), s).squeeze(dim=2) # (batch_size, D)
        return z, s.squeeze(dim=2)

class AttCNNMILModule(torch.nn.Module):
    def __init__(self, cnn, att, fc):
        super(AttCNNMILModule, self).__init__()
        self.cnn = cnn
        self.att = att
        self.fc = fc
    
    def forward(self, Xhat):
        """
        input:
            Xhat: tensor(batch_size, bag_size, 3, H, W)
            L: tensor(batch_size, bag_size, bag_size)
        output: 
            T_logits: (batch_size,)
            S: (batch_size, bag_size)
        """
        X = self.cnn(Xhat) # (batch_size, bag_size, D)
        Z, S = self.att(X) # (batch_size, D), (batch_size, bag_size)
        T_logits = self.fc(Z).squeeze(dim=1) # (batch_size,)
        return T_logits, S
    

class MeanModule(torch.nn.Module):
    def __init__(self, D_dim, L_dim=50, device = 'cuda'):
        super(MeanModule, self).__init__()   
        self.device = device  
    
    def forward(self, X):
        """
        input:
            X: tensor (batch_size, bag_size, D)
            L: tensor (batch_size, bag_size, bag_size)
        output:
            Z: tensor (batch_size, D)
            S: tensor (bag_size, 1)
        """

        bag_size = X.shape[1]
        
        s = torch.ones((1,)).new_full((bag_size, 1), (1./bag_size)).to(self.device) # (batch_size, bag_size, 1) # Esto es la función que se encarga de calcular los pesos
        z = torch.matmul(X.transpose(1,2), s).squeeze(dim=2) # (batch_size, D)
        
        return z , s.squeeze(dim=1)
    


class MaxModule(torch.nn.Module):
    def __init__(self, D_dim, L_dim=50):
        super(MaxModule, self).__init__()     
    
    def forward(self, X):
        """
        input:
            X: tensor (batch_size, bag_size, D)
            L: tensor (batch_size, bag_size, bag_size)
        output:
            Z: tensor (batch_size, D)
            S: tensor (bag_size, 1)
        """

        bag_size = X.shape[1]
        
        s = torch.ones((1,)).new_full((bag_size, 1), bag_size) # (batch_size, bag_size, 1) # Esto es la función que se encarga de calcular los pesos
        z = torch.max(X,1)[0] #.squeeze(dim=2) # (batch_size, D)

        return z , s.squeeze(dim=1)