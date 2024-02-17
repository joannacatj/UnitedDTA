import torch
import torchvision




def get_support_visual_model_names():
    """
    获取支持的视觉模型名称列表。

    Returns:
        List[str]: 支持的视觉模型名称列表。
    """
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", feature_dim=128):
    """
    加载指定名称的预训练视觉模型，并修改模型的全连接层以适应指定的类别数量和特征维度。

    Args:
        modelname (str): 视觉模型的名称，默认为 "ResNet18"。
        feature_dim (int): 特征维度，默认为 128。

    Returns:
        torch.nn.Module: 加载并修改后的视觉模型。
    """
    assert modelname in get_support_visual_model_names()

    # 根据模型名称选择相应的预训练视觉模型
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
    if modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
    if modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
    if modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
    if modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)

    # 替换全连接层为自定义的特征层
    model.fc = torch.nn.Linear(model.fc.in_features, feature_dim)

    return model
