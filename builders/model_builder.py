from model.WFDCNet import WFDCNet

def build_model(model_name, num_classes):
    if model_name == 'WFDCNet':
        return WFDCNet(classes=num_classes)
   