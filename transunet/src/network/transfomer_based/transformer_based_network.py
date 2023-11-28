from .transUnet.transunet import TransUnet


def get_transformer_based_model(model_name: str, img_size: int, num_classes: int, in_ch: int):

    if model_name == "TransUnet":
        model = TransUnet(img_ch=in_ch, output_ch=num_classes)
    else:
        print("model err")
        exit(0)
    return model
