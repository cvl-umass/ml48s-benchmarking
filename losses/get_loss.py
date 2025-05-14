from losses.AssumeNegativeLoss import AssumeNegativeLoss
from losses.LabelSmoothingLoss import LabelSmoothingLoss
from losses.WeakNegativesLoss import WeakNegativesLoss
from losses.IgnoreLargeLoss import IgnoreLargeLoss
from losses.EntropyMaximizationLoss import EntropyMaximizationLoss
from losses.ROLELoss import ROLELoss

def get_loss_func(loss_name, args, model=None):
    if loss_name == "BCE":
        return AssumeNegativeLoss()
    elif loss_name == "EML":
        return EntropyMaximizationLoss(alpha=args.eml_alpha)
    elif loss_name == "LL":
        return IgnoreLargeLoss(mode=args.ll_mode, delta=args.ll_delta)
    elif loss_name == "LSL":
        return LabelSmoothingLoss(eps=args.lsl_eps)
    elif loss_name == "ROLE":
        return ROLELoss(lamb=args.epl_lamb, k=args.epl_k, L=args.num_classes)
    elif loss_name == "WNL":
        if args.wnl_gamma is None:
            args.wnl_gamma = 1 / (args.num_classes - 1)
        return WeakNegativesLoss(gamma=args.wnl_gamma)
    else:
        assert loss_name in ["BCE", "LSL", "WNL", "LL", "EML", "ROLE"]
