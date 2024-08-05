import torch
import torchvision.transforms as transforms
# from main import device
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

fixed_size = (512, 512)
def image_loader(image_name, size=None):
    image = Image.open(image_name).convert('RGB')
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./data/galaxy.jpg", fixed_size)
style_img_1 = image_loader("./data/picasso.jpg", fixed_size)
content_img = image_loader("./data/dancing.jpg", fixed_size)

print(style_img.size(), content_img.size())

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer