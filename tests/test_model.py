import pytest
import torch
import hydra
import omegaconf
import pyrootutils

def test_layers_yolov1():
    """
    Test YOLOv1.
    """
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "yolo.yaml")
    _ = hydra.utils.instantiate(cfg)

    input = torch.rand(1, 3, 448, 448)

    x = _.net[0](input) 
    assert x.shape == torch.Size([1, 64, 224, 224])
    x = _.net[1](x)
    assert x.shape == torch.Size([1, 64, 112, 112])
    x = _.net[2](x)
    assert x.shape == torch.Size([1, 192, 112, 112])
    x = _.net[3](x)
    assert x.shape == torch.Size([1, 192, 56, 56])
    x = _.net[4](x)
    assert x.shape == torch.Size([1, 128, 56, 56])
    x = _.net[5](x)
    assert x.shape == torch.Size([1, 256, 56, 56])
    x = _.net[6](x)
    assert x.shape == torch.Size([1, 256, 56, 56])
    x = _.net[7](x)
    assert x.shape == torch.Size([1, 512, 56, 56])
    x = _.net[8](x)
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[9](x)
    assert x.shape == torch.Size([1, 256, 28, 28])
    x = _.net[10](x) 
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[11](x)
    assert x.shape == torch.Size([1, 256, 28, 28])
    x = _.net[12](x) 
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[13](x)
    assert x.shape == torch.Size([1, 256, 28, 28])
    x = _.net[14](x) 
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[15](x)
    assert x.shape == torch.Size([1, 256, 28, 28])
    x = _.net[16](x) 
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[17](x)
    assert x.shape == torch.Size([1, 512, 28, 28])
    x = _.net[18](x)
    assert x.shape == torch.Size([1, 1024, 28, 28])
    x = _.net[19](x)
    assert x.shape == torch.Size([1, 1024, 14, 14])
    x = _.net[20](x)
    assert x.shape == torch.Size([1, 512, 14, 14])
    x = _.net[21](x)
    assert x.shape == torch.Size([1, 1024, 14, 14])
    x = _.net[22](x)
    assert x.shape == torch.Size([1, 512, 14, 14])
    x = _.net[23](x)
    assert x.shape == torch.Size([1, 1024, 14, 14])
    x = _.net[24](x)
    assert x.shape == torch.Size([1, 1024, 14, 14])
    x = _.net[25](x)
    assert x.shape == torch.Size([1, 1024, 7, 7])
    x = _.net[26](x)
    assert x.shape == torch.Size([1, 1024, 7, 7])
    x = _.net[27](x)
    assert x.shape == torch.Size([1, 1024, 7, 7])

def test_yolov1():
    """
    Test YOLOv1.
    """
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "yolo.yaml")
    yolo = hydra.utils.instantiate(cfg)

    input = torch.rand(2, 3, 448, 448)
    assert yolo.net(input).shape == torch.Size([2, 1024, 7, 7])
    assert yolo(input).shape == torch.Size([2, 1470])