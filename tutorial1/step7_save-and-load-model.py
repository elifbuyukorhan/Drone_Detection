import torch
import torchvision.models as models

#Saving and Loading Model Weights
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

#Saving and Loading Models with Shapes
torch.save(model, 'model.pth')
model = torch.load('model.pth')