from mambavision import create_model
import torch

image = torch.rand(1, 3, 512, 224).cuda() # place image on cuda

model = create_model('mamba_vision_T', pretrained=False, model_path="/tmp/mambavision_tiny_1k.pth.tar")
model = model.cuda() # place model on cuda
output, features = model(image) # output logit size is [1, 1000]

print(output.shape)

for f in features:
    print(f.shape)


