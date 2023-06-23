import unet.architectures as archi

def define_model(model_opt, shape_input, output_channels):
    if model_opt["type"] == "unet":
        return archi.unet_maker(shape_input, output_channels, model_opt["layers"], model_opt["filters"]) 
    elif model_opt["type"] == "resunet":
        return archi.ResUNet_maker(shape_input, output_channels, model_opt["layers"], model_opt["filters"]) 
    elif model_opt["type"] == "uresnet":
        return archi.UResNet_maker(shape_input, output_channels, model_opt["layers"], model_opt["filters"]) 
    else: 
        raise NotImplementedError
    

def load_weights(path_opt, model):
    if path_opt["weights"] is not None:
        model.load_weights(path_opt["weights"], by_name=False)