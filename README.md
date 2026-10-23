# Vidlamp Stereo

Stereoscopic 3D for ComfyUI from any video (or image) and its depth map. Pure torch, no extra dependencies.

## Nodes

**Vidlamp Stereo 3D** — `images` + `depth` (bright = near) → stereo frames.
- `output_format`: side by side, side by side half, over/under, cross-eyed, red/cyan anaglyph (Dubois), red/blue
  anaglyph (half colour), left only, right only.
- `divergence` (% of the width, capped at 6), `convergence` (depth on the screen plane), `balance`, `depth_gamma`,
  `edge_expand`, `smooth`, `normalize` (one depth range for the whole clip by default), `swap_eyes`.
- Occlusions with a z-buffer (the nearer pixel wins), uncovered areas filled from the background side.

**Vidlamp Depth Detail Merge** — `base_depth` (steady, e.g. Video Depth Anything) + `detail_depth` (sharp,
e.g. MoGe-2) → the large shapes of the first with the fine relief of the second, without its frame-to-frame flicker.

**360 (equirectangular)** — `panorama_360` on Vidlamp Stereo 3D: the shift wraps around the left/right
borders, fades towards the poles and is zero at infinity (use `side_by_side` or `over_under_full`, the full-height
over/under). **Vidlamp Project to 360** puts ordinary frames (a pose, depth or edges video) at the centre of a
black equirectangular canvas with a given horizontal field of view: the guide of a 360 video. **Vidlamp Roll 360**
turns a 360 video by half a turn (the seam goes to the middle to be repainted; the same node turns it back).

## Typical graph

Load Video → Get Video Components → Video Depth Anything (depth, gray) → Vidlamp Stereo 3D → Create Video
(with the original audio) → Save Video.
