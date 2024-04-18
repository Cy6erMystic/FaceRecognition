from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200, iresnet1024, iresnet2060
from .mobilefacenet import get_mbf, get_mbf_large

__all__ = ["get_model"]

def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r1024":
        return iresnet1024(False, **kwargs)
    elif name == "r2060":
        return iresnet2060(False, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)
    elif name == "mbf_large":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)