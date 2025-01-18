import torch

img = torch.load('img_backbone_ResNet50.pth', map_location='cpu')
pts = torch.load('TransFusionL_Noise.pth', map_location='cpu')
new_model = {"state_dict": pts["state_dict"]}
for k,v in img["state_dict"].items():
    if 'backbone' in k or 'neck' in k:
        new_model["state_dict"]['img_'+k] = v
        print('img_'+k)
torch.save(new_model, "fusion_model_Noise.pth")
