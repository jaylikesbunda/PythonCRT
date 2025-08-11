# PythonCRT
Apply CRT-style effects to video with a fast GUI and CLI.

### Requirements

- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

### Quick start (GUI)

- Run: `python crt_filter.py` (or `python crt_filter.py --gui`)
- Open a video, adjust sliders, and click Render to export
- Tabs:
  - Effects: scanline, triad, aberration, noise, bloom, vignette, persistence
  - Motion: scanline speed/period, glitch
  - Advanced: brightness, contrast, gamma, saturation, temperature, bloom threshold, flicker, grain size, scanline angle/thickness, warp
  - Text: overlay text (with font family or custom .ttf/.otf), size, color, position, draw after effects; includes Save/Load text presets
  - Output: hardware encode toggle, encoder (Auto/NVIDIA/AMD/CPU), decoder (Auto/NVIDIA/AMD/Intel/CPU), CRF/CQ, NVENC preset

Notes:
- Toolbar has Play/Pause, Render, HW Encode, HW Decode (preview), and Fast Bloom toggles.
- Side panel width is fixed so the preview scales predictably.

### Quick start (CLI)

Basic render:
```bash
python crt_filter.py --input input.mp4 --output output.mp4 --gpu
```

Hardware encode/decode selection:
```bash
# Encoder: auto|nvidia|amd|cpu  |  Decoder: auto|nvidia|amd|intel|cpu
python crt_filter.py --input in.mp4 --output out.mp4 --gpu \
  --encoder nvidia --decoder nvidia --crf 18 --scanline-strength 0.6
```

Text overlay (can use font family or a .ttf/.otf file path):
```bash
python crt_filter.py --input in.mp4 --output out.mp4 \
  --text "Retro Night" --text-font "C:\\Windows\\Fonts\\segoeui.ttf" \
  --text-size 64 --text-color "#FFD700" --text-x 64 --text-y 64 --text-after
```

Advanced effects example:
```bash
python crt_filter.py --input in.mp4 --output out.mp4 \
  --bloom-strength 0.35 --bloom-sigma 1.5 --bloom-threshold 0.7 \
  --brightness 0.05 --contrast 1.15 --gamma 1.0 --saturation 1.1 --temperature 0.1 \
  --scanline-angle 3.0 --scanline-thickness 1.2 --warp-strength 0.15 \
  --flicker-strength 0.25 --flicker-hz 60 --grain-size 2
```

Defaults: if `--output` is omitted it writes `<input>_crt.mp4`; width/height/fps stay unchanged when 0. `--gpu` enables hardware encode; the encoder/decoder fall back automatically if not available.

### Features
- Real-time preview with optional hardware decode (toolbar HW Decode)
- Hardware encode: NVIDIA NVENC or AMD AMF, with auto fallback to CPU x264
- CRT effects: scanlines, triad mask, chromatic aberration, bloom (fast/gaussian), noise, vignette, persistence, glitch
- Advanced color/geometry: brightness, contrast, gamma, saturation, temperature, bloom threshold, flicker, grain, scanline angle/thickness, warp
- Text overlay: live preview, custom font file or system family, position/size/color, before/after effects; save/load presets
- Preset management: save/load full effect presets in Output; save/load text presets in Text


