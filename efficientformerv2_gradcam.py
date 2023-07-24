import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from timm.models import create_model

from EfficientFormer.models.efficientformer_v2 import efficientformerv2_l, efficientformerv2_s2, EfficientFormerV2
from Grad_CAM.utils import visualize_cam, Normalize
from Grad_CAM.gradcam import GradCAM, GradCAMpp

import pdb

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

#img_dir = 'D:/Data RYM/Data Kuliah/S2/Semester 1/Thesis/Percobaan/Dataset/ExDark/all/images/test/'
#img_name = '2015_00409.jpg'
#img_name = '2015_01203.jpg'

img_dir = 'D:/Data RYM/Data Kuliah/S2/Semester 1/Thesis/Percobaan/Dataset/COCO Normal/all/images/test/'
#img_name = '000000047571.jpg'
img_name = '000000181421.jpg'


img_path = os.path.join(img_dir, img_name)

pil_img = PIL.Image.open(img_path)

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
normed_torch_img = normalizer(torch_img)

efficientformerv2 = create_model(
        "efficientformerv2_l",
        num_classes=2,
        distillation=(False),
        pretrained=True,
        fuse=True,
)

pdb.set_trace()

# load weights
checkpoint = torch.load("./EfficientFormer/efficientformerv2_l_exdark_coco/best.pth", map_location='cpu')
efficientformerv2.load_state_dict(checkpoint['model'], strict=True)

efficientformerv2.to("cuda:0")
efficientformerv2.eval()

cam_dict = dict()

efficientformerv2_model_dict = dict(type='efficientformerv2', arch=efficientformerv2, layer_name='efficientformerv2_l', input_size=(224, 224))
efficientformerv2_gradcam = GradCAM(efficientformerv2_model_dict, True)
efficientformerv2_gradcampp = GradCAMpp(efficientformerv2_model_dict, True)
cam_dict['efficientformerv2'] = [efficientformerv2_gradcam, efficientformerv2_gradcampp]

images = []
for gradcam, gradcam_pp in cam_dict.values():
    mask, _ = gradcam(normed_torch_img, 0)
    #mask = reshape_transform(mask)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img, 0)
    #mask_pp = reshape_transform(mask_pp)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
    
    images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
    
images = make_grid(torch.cat(images, 0), nrow=5)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_name = f"efficientformerv2_l_{img_name}"
output_path = os.path.join(output_dir, output_name)

save_image(images, output_path)
PIL.Image.open(output_path)