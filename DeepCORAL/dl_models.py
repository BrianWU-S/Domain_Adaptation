import torch.nn as nn
from torchvision import models
import torchvision
import torch
from loss_func import MMD_loss, CORAL
from torchsummary import summary


class TransferModel(nn.Module):
    def __init__(self,
                 base_model: str = 'resnet50',
                 pretrain: bool = True,
                 n_class: int = 65):
        super(TransferModel, self).__init__()
        self.base_model = base_model
        self.pretrain = pretrain
        self.n_class = n_class
        if self.base_model == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            n_features = self.model.fc.in_features
            fc = torch.nn.Linear(n_features, n_class)  # fc: fully connected block
            self.model.fc = fc  # add one layer to classify
        else:
            # Use other models you like, such as vgg or alexnet
            pass
        self.model.fc.weight.data.normal_(0,
                                          0.005)  # Note: need to initialize fc layer, in tf this can be integrated into keras.layers.Dense()
        self.model.fc.bias.data.fill_(0.1)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.forward(x)


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)  # Instancing a pre-trained model will download its
        # weights to a cache directory.
        # This directory can be set using the TORCH_MODEL_ZOO environment variable.
        self.conv1 = model_resnet50.conv1  # conv
        self.bn1 = model_resnet50.bn1  # batch norm
        self.relu = model_resnet50.relu  # activation
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # resize (batch_size,-1)
        return x
    
    def output_num(self):
        return self.__in_features


class TransferNet(nn.Module):
    def __init__(self,
                 num_class,
                 base_net='resnet50',
                 transfer_loss='mmd',
                 use_bottleneck=True,
                 bottleneck_width=128,
                 width=1024):
        super(TransferNet, self).__init__()
        if base_net == 'resnet50':
            self.base_network = ResNet50Fc()
        else:
            self.base_network = base_fc()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(
        ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)
    
    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        return source_clf, transfer_loss
    
    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf
    
    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss


class base_fc(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512):
        super(base_fc, self).__init__()
        self.input_size = input_size  # 2048
        self.hidden_size = hidden_size  # 512
        self.layer1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2))  # 256
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        return x
    
    def predict(self, x):
        return self.forward(x)
    
    def output_num(self):
        return int(self.hidden_size/2)


class FinetuneModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=256, class_num=65):
        super(FinetuneModel, self).__init__()
        self.base_network = base_fc(input_size=input_size, hidden_size=hidden_size, class_num=class_num)
        self.classify_layer = nn.Linear(in_features=int(hidden_size / 2), out_features=class_num)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.base_network(x)
        x = self.classify_layer(x)
        x = self.softmax(x)
        return x
    
    def predict(self, x):
        return self.forward(x)


if __name__ == '__main__':
    res_model = ResNet50Fc().cuda()
    print(res_model)
    summary(res_model, input_size=(3, 224, 224), batch_size=32, device='cuda')
    # finetune_model=TransferModel().cuda()
    # summary(finetune_model, input_size=(3, 224, 224), batch_size=32, device='cuda')
    # transfer_model = TransferNet(num_class=65).cuda()
    # summary(transfer_model, input_size=[(3, 224, 224), (3, 224, 224)], batch_size=32, device='cuda')
    
    # base_model = base_fc().cuda()
    # summary(base_model, input_size=(2048,), batch_size=32, device='cuda')
