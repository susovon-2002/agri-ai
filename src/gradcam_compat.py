from pytorch_grad_cam import GradCAM as _RealCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAM(_RealCAM):
    pass
