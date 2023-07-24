import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from VGG.model import VGG16
from Grad_CAM.utils import visualize_cam, Normalize
from Grad_CAM.gradcam import GradCAM, GradCAMpp

#img_dir = 'D:/Data RYM/Data Kuliah/S2/Semester 1/Thesis/Percobaan/Dataset/ExDark/all/images/test/'
#img_name = '2015_00409.jpg'
#img_name = '2015_01203.jpg'
#img_name = '2015_02378.jpg'

img_dir = 'D:/Data RYM/Data Kuliah/S2/Semester 1/Thesis/Percobaan/Dataset/COCO Normal/all/images/test/'
#img_name = '000000047571.jpg'
img_name = '000000181421.jpg'



img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
normed_torch_img = normalizer(torch_img)

vgg = VGG16(num_classes=2)
vgg.eval(), vgg.cuda()
# load weights
vgg.load_state_dict(torch.load("./VGG/results/vgg16-32-95.pt"))

cam_dict = dict()

vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='layer13_2', input_size=(227, 227))
vgg_gradcam = GradCAM(vgg_model_dict, True)
vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img, 1)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img, 1)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    
images = make_grid(torch.cat(images, 0), nrow=5)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_name = f"vgg16_{img_name}"
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
PIL.Image.open(output_path)