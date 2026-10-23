"""ComfyUI nodes of Vidlamp Stereo (the maths is in stereo_core.py)."""
import torch
import torch.nn.functional as F

from . import stereo_core as core

try:
    import comfy.model_management as mm
    import comfy.utils
except ImportError:            # outside ComfyUI (tests)
    mm = None
    comfy = None


def _device():
    if mm is not None:
        return mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _depth_batch(depth, n, h, w):
    """IMAGE (B, h, w, C) -> (n, h, w) in [0, 1] at the size of the frames; a single map is reused for every frame."""
    d = depth[..., :3].float().mean(-1) if depth.shape[-1] >= 3 else depth[..., 0].float()
    if d.shape[1:] != (h, w):
        d = F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    if d.shape[0] == 1 and n > 1:
        d = d.expand(n, h, w)
    if d.shape[0] < n:
        raise ValueError(f"depth has {d.shape[0]} frames, the images {n}")
    return d[:n]


class VidlampStereo3D:
    """Turns images (a video's frames) and their depth into stereoscopic 3D."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "depth": ("IMAGE", {"tooltip": "Depth map(s), bright = near (Video Depth Anything, MoGe, Depth Anything)."}),
            "output_format": (core.FORMATS, {"default": "side_by_side"}),
            "divergence": ("FLOAT", {"default": 3.0, "min": 0.0, "max": core.MAX_DIVERGENCE, "step": 0.1,
                                     "tooltip": "How much 3D, in % of the width (3 recommended)."}),
            "convergence": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Depth on the screen plane: lower = more comes out of the screen."}),
            "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Share of the shift on the left eye (0.5 = half each)."}),
            "depth_gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
            "edge_expand": ("INT", {"default": 3, "min": 0, "max": 16,
                                    "tooltip": "Pixels the foreground is grown by, so outlines move with the subject."}),
            "smooth": ("INT", {"default": 1, "min": 0, "max": 16}),
            "normalize": (["whole_clip", "per_frame", "none"], {"default": "whole_clip",
                          "tooltip": "whole_clip: one depth range for all frames (steady 3D)."}),
            "swap_eyes": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "panorama_360": ("BOOLEAN", {"default": False,
                             "tooltip": "Equirectangular 360 video: the 3D wraps around the borders and fades towards "
                                        "the poles, the far horizon stays still (convergence and balance unused). "
                                        "Use side_by_side or over_under_full."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stereo", "depth_used")
    FUNCTION = "run"
    CATEGORY = "Vidlamp/Stereo"

    def run(self, images, depth, output_format, divergence, convergence, balance, depth_gamma, edge_expand, smooth,
            normalize, swap_eyes, panorama_360=False):
        n, h, w, _ = images.shape
        dev = _device()
        d_all = _depth_batch(depth, n, h, w)
        if normalize == "whole_clip":
            d_all = core.normalize_clip(d_all.to(dev)).cpu()
        oh, ow = core.output_size(output_format, h, w)
        out = torch.empty((n, oh, ow, 3), dtype=torch.float32)
        used = torch.empty((n, h, w, 3), dtype=torch.float32)
        bar = comfy.utils.ProgressBar(n) if comfy is not None else None
        for k in range(n):
            img = images[k, ..., :3].to(dev).float()
            d = d_all[k].to(dev)
            if normalize == "per_frame":
                d = core.normalize_clip(d[None])[0]
            d = core.clean_depth(d, depth_gamma, edge_expand, smooth)
            if panorama_360:
                left, right = core.stereo_pair_360(img, d, divergence)
            else:
                left, right = core.stereo_pair(img, d, divergence, convergence, balance)
            if swap_eyes:
                left, right = right, left
            out[k] = core.compose(left, right, output_format).clamp(0, 1).cpu()
            used[k] = d[..., None].expand(h, w, 3).cpu()
            if bar is not None:
                bar.update(1)
        return (out, used)


class VidlampDepthDetailMerge:
    """Large shapes from a steady video depth (Video Depth Anything), fine detail from a sharper per-frame depth
    (MoGe-2): the relief of faces without the flicker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_depth": ("IMAGE", {"tooltip": "Steady depth (e.g. Video Depth Anything), bright = near."}),
            "detail_depth": ("IMAGE", {"tooltip": "Detailed depth (e.g. MoGe-2), bright = near."}),
            "detail_size": ("FLOAT", {"default": 0.03, "min": 0.005, "max": 0.2, "step": 0.005,
                                      "tooltip": "What counts as 'fine detail', as a fraction of the width."}),
            "temporal_window": ("INT", {"default": 7, "min": 1, "max": 61, "step": 2}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "run"
    CATEGORY = "Vidlamp/Stereo"

    def run(self, base_depth, detail_depth, detail_size, temporal_window):
        n, h, w, _ = detail_depth.shape
        dev = _device()
        base = _depth_batch(base_depth, n, h, w).to(dev)
        detail = _depth_batch(detail_depth, n, h, w).to(dev)
        merged = core.merge_detail(base, detail, detail_size, temporal_window)
        return (merged[..., None].expand(n, h, w, 3).float().cpu(),)


class VidlampProjectTo360:
    """Puts ordinary frames (a pose, depth or edges video) at the centre of a black equirectangular 360 canvas, with
    the same field of view used to make the 360 first frame: the guide of a 360 video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 2}),
            "height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 2}),
            "field_of_view": ("FLOAT", {"default": 75.0, "min": 10.0, "max": 170.0, "step": 1.0,
                                        "tooltip": "Horizontal degrees the frames cover in the 360."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Vidlamp/Stereo"

    def run(self, images, width, height, field_of_view):
        dev = _device()
        out = torch.empty((images.shape[0], height, width, images.shape[-1]), dtype=torch.float32)
        for k in range(0, images.shape[0], 16):
            out[k:k + 16] = core.project_to_360(images[k:k + 16].to(dev).float(), width, height,
                                                field_of_view).cpu()
        return (out,)


class VidlampRoll360:
    """Half a turn of an equirectangular 360 video: the left/right seam goes to the middle (to repaint it), and the
    same node turns it back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Vidlamp/Stereo"

    def run(self, images):
        return (core.roll_half(images),)


NODE_CLASS_MAPPINGS = {"VidlampStereo3D": VidlampStereo3D, "VidlampDepthDetailMerge": VidlampDepthDetailMerge,
                       "VidlampProjectTo360": VidlampProjectTo360, "VidlampRoll360": VidlampRoll360}
NODE_DISPLAY_NAME_MAPPINGS = {"VidlampStereo3D": "Vidlamp Stereo 3D",
                              "VidlampDepthDetailMerge": "Vidlamp Depth Detail Merge",
                              "VidlampProjectTo360": "Vidlamp Project to 360",
                              "VidlampRoll360": "Vidlamp Roll 360 (half turn)"}

# The node names before the Vidlamp rebrand (18/10/2026): workflows saved with them keep loading.
_LEGACY = {"MVBStereo3D": "VidlampStereo3D", "MVBDepthDetailMerge": "VidlampDepthDetailMerge",
           "MVBProjectTo360": "VidlampProjectTo360", "MVBRoll360": "VidlampRoll360"}
for _old, _new in _LEGACY.items():
    NODE_CLASS_MAPPINGS[_old] = NODE_CLASS_MAPPINGS[_new]
    NODE_DISPLAY_NAME_MAPPINGS[_old] = NODE_DISPLAY_NAME_MAPPINGS[_new] + " (old name)"
