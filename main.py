import cv2
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image 
from pathlib import Path


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().requires_grad_(True).view(-1, 1, 1)
        self.std = std.clone().detach().requires_grad_(True).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def image_loader(img_name, imsize):
    img_path = Path.cwd() / "data" / img_name
    return Image.open(img_path).resize((imsize, imsize), Image.ANTIALIAS)


def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)


def get_model_and_losses(cnn,
                        style_img,
                        content_img,
                        content_layers,
                        style_layers,
                        normalization_mean,
                        normalization_std):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, 
                                  normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss{i}", content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)


    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses


if __name__ == "__main__":

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = 512 if torch.cuda.is_available() else 128

    # settings
    content_img_name = "content.jpg"
    style_img_name = "style.jpg"
    content_layers_default = ["conv_4"]
    style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    num_steps = 300
    style_weight = 1000000
    content_weight = 1

    # normalization used by VGG19
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # load content and style images onto memory
    content_img = image_loader(content_img_name, imsize)
    style_img = image_loader(style_img_name, imsize)

    # resize (again) and transform to tensors
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])

    # transform the images
    content_img = loader(content_img).unsqueeze(0).to(device, torch.float)
    style_img = loader(style_img).unsqueeze(0).to(device, torch.float)
    
    # load VGG19
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=0.05, max_iter=1)

    # build model and losses
    model, style_losses, content_losses = get_model_and_losses(cnn,
                                                            style_img,
                                                            content_img,
                                                            content_layers_default,
                                                            style_layers_default,
                                                            cnn_normalization_mean,
                                                            cnn_normalization_std)


    # train model and capture intermediate images
    img_over_time = list()
    run = [0]
    while run[0] <= num_steps:
        def closure():  
            
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss 
            
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            img_over_time.append(input_img.data.clamp(0, 1).cpu().detach().squeeze(0))
            
            return style_score + content_score
        
        optimizer.step(closure)
    
    # final result
    output = input_img.data.clamp(0, 1)

    # create the video of the image being stylized
    unloader = transforms.ToPILImage()
    imgs_for_video = [cv2.cvtColor(np.array(unloader(i)), cv2.COLOR_RGB2BGR) for i in img_over_time]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("project.mp4", fourcc, 45, (128, 128))
    for i in range(len(imgs_for_video)):
        out.write(imgs_for_video[i])
    out.release()