import torch
import torch.nn as nn
import torchvision


def print_requires_grad(module):
    for i, child in enumerate(module.children()):
        print(i)
        for param in child.parameters():
            print(param.requires_grad)


class Alexnet(nn.Module):
    def __init__(self, num_classes):
        super(Alexnet, self).__init__()
        original_model = torchvision.models.alexnet(pretrained=True)
        self.features = original_model.features.children()
        self.features = torch.nn.Sequential(*self.features)
        for child in list(self.features.children()):
            for param in child.parameters():
                param.requires_grad = True
        for child in list(self.features.children())[:-7]:
            for param in child.parameters():
                param.requires_grad = False
        #print_requires_grad(self.features)

        num_features = original_model.classifier[6].in_features
        self.classifier = list(original_model.classifier.children())[:-1]
        self.classifier.extend([torch.nn.Linear(num_features, num_classes)])
        self.classifier = torch.nn.Sequential(*self.classifier)
        for child in list(self.classifier.children()):
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        original_model = torchvision.models.vgg16(pretrained=True)
        self.features = original_model.features.children()
        self.features = torch.nn.Sequential(*self.features)
        for child in list(self.features.children()):
            for param in child.parameters():
                param.requires_grad = True
        for child in list(self.features.children())[:-21]:
            for param in child.parameters():
                param.requires_grad = False
        #print_requires_grad(self.features)

        num_features = original_model.classifier[6].in_features
        self.classifier = list(original_model.classifier.children())[:-1]
        self.classifier.extend([torch.nn.Linear(num_features, num_classes)])
        self.classifier = torch.nn.Sequential(*self.classifier)
        for child in list(self.classifier.children()):
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Linear(num_ftrs, num_classes)
        ct = 0
        for child in self.resnet50.children():
            ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.resnet50(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.inceptionv3 = torchvision.models.inception_v3(pretrained=True)
        self.inceptionv3.aux_logits = True
        num_ftrs = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Freeze all the layers till "Conv2d_4a_3*3"
        for name, child in self.inceptionv3.named_children():
                for params in child.parameters():
                    params.requires_grad = False
        ct = []
        for name, child in self.inceptionv3.named_children():
            if "Conv2d_4a_3x3" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)
        #print_requires_grad(self.inceptionv3)

    def forward(self, x):
        x = self.inceptionv3(x)
        return x
