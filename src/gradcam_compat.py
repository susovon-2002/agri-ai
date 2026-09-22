import numpy as np

from pytorch_grad_cam import CAM as _RealCAM


class ClassifierOutputTarget:
    def __init__(self, category_index):
        self.category_index = int(category_index)

    def __int__(self):
        return self.category_index

    def __repr__(self):
        return f"ClassifierOutputTarget(category_index={self.category_index})"


class GradCAM:
    def __init__(self, model, target_layers, use_cuda=False):
        layer = target_layers
        if isinstance(target_layers, (list, tuple)):
            layer = target_layers[0] if target_layers else None
        if layer is None:
            raise ValueError("GradCAM requires at least one target layer.")
        self._cam = _RealCAM(model=model, target_layer=layer, use_cuda=use_cuda)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, input_tensor, targets=None, method="gradcam", target_category=None, **kwargs):
        if targets is not None:
            if isinstance(targets, (list, tuple)) and targets:
                first = targets[0]
                if hasattr(first, "category_index"):
                    target_category = int(first.category_index)
                elif isinstance(first, (int, float)):
                    target_category = int(first)
            elif hasattr(targets, "category_index"):
                target_category = int(targets.category_index)

        if target_category is None:
            target_category = 0

        heatmap = self._cam(
            input_tensor=input_tensor,
            method=method,
            target_category=int(target_category),
            **kwargs,
        )
        return np.asarray([heatmap], dtype=np.float32)
