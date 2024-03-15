
"""
Module for model creation, savings and many more. Just create the function if 
you need it.

Available Functions:
  create_convnext_tiny,


"""

from torch import nn
import torchvision
from torchvision import transforms


def create_convnext_tiny(num_classes:int=43):
  """
  Creates a ConvNeXt Tiny model with default weights.

  Keyword Arguments:
    :arg num_classes: Number of classes that data have. Default 43.
    :type num_classes: int

  Example Usage:
    model, train_transforms, test_transforms = create_convnext_tiny(num_classes=43)
  """
  weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
  model = torchvision.models.convnext_tiny(weights=weights) 
  
  test_transforms = weights.transforms()
  train_transforms = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    test_transforms
  ])

  for param in model.features.parameters():
    param.requires_grad = False

  model.heads  = nn.Sequential(
    nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(in_features=768, out_features=num_classes, bias=True)
    )

  
  return model, train_transforms, test_transforms
