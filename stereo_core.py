"""Stereo 3D from an image/video and its depth map: pure torch, no ComfyUI imports (testable on its own).

Conventions: depth in [0, 1] with 1 = near (Video Depth Anything / MoGe "depth" renders: bright = near).
The convergence is the depth value that sits on the screen plane: nearer comes out of the screen, farther goes behind.
"""
import math

import torch
import torch.nn.functional as F

MAX_DIVERGENCE = 6.0          # % of the width: more than this tires the eyes

# Dubois red/cyan anaglyph (on linear RGB): fewer ghosts and truer colours than "red from the left"
_DUBOIS_L = [[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]]
_DUBOIS_R = [[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]]
_LUMA = [0.2126, 0.7152, 0.0722]

FORMATS = ["side_by_side", "side_by_side_half", "over_under", "over_under_full", "cross_eyed", "anaglyph_red_cyan",
           "anaglyph_red_blue", "left_only", "right_only"]


def to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def normalize_clip(depth, low=1.0, high=99.0):
    """depth (B, H, W) -> [0, 1] with ONE range for the whole clip (percentiles): the 3D does not "breathe" when one
    frame has something nearer than the others."""
    sample = depth[:, ::4, ::4].reshape(-1).float()
    if sample.numel() > 4_000_000:
        sample = sample[torch.randperm(sample.numel(), device=sample.device)[:4_000_000]]
    lo, hi = torch.quantile(sample, low / 100), torch.quantile(sample, high / 100)
    return ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def clean_depth(d, gamma=1.0, edge_expand=3, smooth=1):
    """d (H, W): curve, then the foreground "grown" by a few pixels (the mixed pixels on a person's outline move with
    the person instead of smearing onto the background), then a light blur against steps."""
    d = d.clamp(0, 1) ** gamma
    if edge_expand > 0:
        k = 2 * int(edge_expand) + 1
        d = F.max_pool2d(d[None, None], k, stride=1, padding=k // 2)[0, 0]
    if smooth > 0:
        k = 2 * int(smooth) + 1
        d = F.avg_pool2d(d[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]
    return d


def warp_view(img, d, shift):
    """One eye: every pixel moves to x + shift (pixels). Two pixels on the same spot: the nearer wins (z-buffer).
    Holes (uncovered background) are filled from the FARTHER side, i.e. from the background, never from the person.
    img (H, W, 3), d (H, W), shift (H, W). Returns the view and the mask of the filled holes."""
    h, w, _ = img.shape
    dev = img.device
    xs = torch.arange(w, device=dev).expand(h, w)
    ys = torch.arange(h, device=dev)[:, None].expand(h, w)
    xt = torch.round(xs + shift).long()
    ok = (xt >= 0) & (xt < w)
    idx = (ys * w + xt)[ok]
    key = d[ok]
    best = torch.full((h * w,), -1.0, device=dev).scatter_reduce(0, idx, key, "amax")
    win = key >= best[idx]
    colors = img.reshape(-1, 3)[(ys * w + xs)[ok][win]]
    out = torch.zeros(h * w, 3, device=dev, dtype=img.dtype)
    dep = torch.full((h * w,), -1.0, device=dev)
    out[idx[win]] = colors
    dep[idx[win]] = key[win]
    full = (dep >= 0).reshape(h, w)
    out, dep = out.reshape(h, w, 3), dep.reshape(h, w)
    left = torch.cummax(torch.where(full, xs, torch.full_like(xs, -1)), dim=1).values
    right = torch.flip(torch.cummin(torch.flip(torch.where(full, xs, torch.full_like(xs, w)), [1]), dim=1).values, [1])
    d_left = torch.where(left >= 0, dep.gather(1, left.clamp(0, w - 1)), torch.full_like(dep, 9))
    d_right = torch.where(right < w, dep.gather(1, right.clamp(0, w - 1)), torch.full_like(dep, 9))
    source = torch.where(d_left <= d_right, left, right).clamp(0, w - 1)
    filled = out.gather(1, source[..., None].expand(h, w, 3))
    return torch.where(full[..., None], out, filled), ~full


def stereo_pair(img, d, divergence, convergence, balance=0.5):
    """The left and right views. divergence in % of the width (capped), balance = share of the shift on the left eye."""
    w = img.shape[1]
    disp = min(float(divergence), MAX_DIVERGENCE) / 100 * w * (d - convergence)     # positive = in front of the screen
    left, _ = warp_view(img, d, disp * balance)               # left eye: near things move right
    right, _ = warp_view(img, d, -disp * (1 - balance))       # right eye: near things move left
    return left, right


def compose(left, right, fmt):
    h, w, _ = left.shape
    luma = torch.tensor(_LUMA, device=left.device)
    if fmt == "side_by_side":
        return torch.cat([left, right], 1)
    if fmt == "cross_eyed":
        return torch.cat([right, left], 1)
    if fmt == "side_by_side_half":
        both = torch.cat([left, right], 1).permute(2, 0, 1)[None]
        return F.interpolate(both, size=(h, w), mode="area")[0].permute(1, 2, 0)
    if fmt == "over_under_full":
        return torch.cat([left, right], 0)
    if fmt == "over_under":
        both = torch.cat([left, right], 0).permute(2, 0, 1)[None]
        return F.interpolate(both, size=(h, w), mode="area")[0].permute(1, 2, 0)
    if fmt == "anaglyph_red_cyan":
        ml, mr = torch.tensor(_DUBOIS_L, device=left.device), torch.tensor(_DUBOIS_R, device=left.device)
        return to_srgb(to_linear(left) @ ml.T + to_linear(right) @ mr.T)
    if fmt == "anaglyph_red_blue":
        # half colour (tested with red/blue glasses): red = the left eye's brightness, green and blue = the right view
        red = to_srgb(0.85 * (to_linear(left) @ luma))
        return torch.stack([red, right[..., 1], right[..., 2]], -1)
    if fmt == "left_only":
        return left
    if fmt == "right_only":
        return right
    raise ValueError(f"unknown format: {fmt}")


def output_size(fmt, h, w):
    if fmt == "over_under_full":
        return (2 * h, w)
    return (h, 2 * w) if fmt in ("side_by_side", "cross_eyed") else (h, w)


# ------------------------------------------------------------------ 360 (equirectangular)
def stereo_pair_360(img, d, divergence):
    """The two eyes of an equirectangular 360 frame (simplified omnidirectional stereo): the shift goes to zero at
    infinity (no convergence plane: the far horizon stays still) and towards the poles (x cos of the latitude), and
    the left and right borders are the same place, so the warp wraps around. img (H, W, 3), d (H, W)."""
    h, w, _ = img.shape
    lat = (0.5 - (torch.arange(h, device=img.device, dtype=torch.float32) + 0.5) / h) * math.pi
    disp = min(float(divergence), MAX_DIVERGENCE) / 100 * w * d * torch.cos(lat)[:, None]
    m = min(w // 2, int(math.ceil(float(disp.max()))) + 8)

    def eye(shift):
        v, _ = warp_view(torch.cat([img[:, -m:], img, img[:, :m]], 1), torch.cat([d[:, -m:], d, d[:, :m]], 1),
                         torch.cat([shift[:, -m:], shift, shift[:, :m]], 1))
        return v[:, m:-m]
    return eye(disp * 0.5), eye(-disp * 0.5)


def project_to_360(images, width, height, fov=75.0):
    """Ordinary frames (B, H, W, C) at the centre of a BLACK equirectangular canvas (B, height, width, C), horizontal
    field of view `fov` degrees: a pose/depth/edges video becomes the guide of a 360 video whose first frame was made
    with the same field (Vidlamp panorama360.tela)."""
    b, fh, fw, c = images.shape
    dev = images.device
    f = (fw / 2) / math.tan(math.radians(fov) / 2)
    lon = ((torch.arange(width, device=dev, dtype=torch.float32) + 0.5) / width - 0.5) * 2 * math.pi
    lat = (0.5 - (torch.arange(height, device=dev, dtype=torch.float32) + 0.5) / height) * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    x, y, z = torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)
    zz = z.clamp_min(1e-6)
    px = f * x / zz + fw / 2 - 0.5
    py = -f * y / zz + fh / 2 - 0.5
    inside = (z > 1e-6) & (px >= 0) & (px <= fw - 1) & (py >= 0) & (py <= fh - 1)
    grid = torch.stack([px / max(fw - 1, 1) * 2 - 1, py / max(fh - 1, 1) * 2 - 1], -1).clamp(-1, 1)
    out = torch.empty((b, height, width, c), dtype=images.dtype, device=dev)
    for k in range(b):            # one frame at a time: a long video never needs one huge batch in memory
        v = F.grid_sample(images[k:k + 1].permute(0, 3, 1, 2).float(), grid[None], mode="bilinear",
                          padding_mode="border", align_corners=True)[0].permute(1, 2, 0)
        out[k] = torch.where(inside[..., None], v, torch.zeros_like(v)).to(images.dtype)
    return out


def roll_half(images):
    """Half a turn of an equirectangular video (B, H, W, C): the left/right seam goes to the middle, where a model can
    repaint it; the same call turns it back (even widths)."""
    return torch.roll(images, shifts=images.shape[2] // 2, dims=2)


# ------------------------------------------------------------------ depth detail merge (Video Depth Anything + MoGe)
def box_blur(x, k):
    """three box blurs in a row ~ a gaussian; borders replicated. x (H, W)."""
    t = x[None, None]
    k = max(3, int(k) | 1)
    for _ in range(3):
        t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, stride=1)
    return t[0, 0]


def fit_curves(base, detail, window=7):
    """For every frame, the quadratic that brings `detail` onto the scale of `base` (robust least squares on a
    subsample), then a temporal median of the coefficients. base, detail (B, H, W). Returns (B, 3)."""
    coeffs = []
    for b, m in zip(base, detail):
        v = b[::4, ::4].reshape(-1).double()
        x = m[::4, ::4].reshape(-1).double()
        A = torch.stack([x * x, x, torch.ones_like(x)], 1)
        c = torch.linalg.lstsq(A, v[:, None]).solution[:, 0]
        for _ in range(2):
            res = (A @ c - v).abs()
            wgt = 1 / torch.maximum(res, res.median())
            c = torch.linalg.lstsq(A * wgt[:, None], (v * wgt)[:, None]).solution[:, 0]
        coeffs.append(c)
    coeffs = torch.stack(coeffs)
    r = window // 2
    return torch.stack([coeffs[max(0, k - r):k + r + 1].median(0).values for k in range(len(coeffs))]).float()


def merge_detail(base, detail, detail_size=0.03, window=7):
    """Large shapes from `base` (steady in time, e.g. Video Depth Anything), fine detail from `detail` (e.g. MoGe-2,
    which has the real relief of a face but is normalised frame by frame): the detail's flicker can only touch the
    fine detail. base, detail (B, H, W) in [0, 1], same size."""
    coeffs = fit_curves(base, detail, window)
    k = detail_size * base.shape[-1]
    out = []
    for b, m, c in zip(base, detail, coeffs):
        aligned = c[0] * m * m + c[1] * m + c[2]
        out.append((box_blur(b, k) + (aligned - box_blur(aligned, k))).clamp(0, 1))
    return torch.stack(out)
