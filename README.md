# PythonCRT
a python script to add crt effects to video

### Requirements

- Python 3.9+
- Install deps: `pip install -r requirements.txt`

### Quick start (GUI)

- Run: `python crt_filter.py` (or `python crt_filter.py --gui`)
- Open a video, adjust sliders, click Export to render

### Quick start (CLI)

```bash
python crt_filter.py --input input.mp4 --output output.mp4 --gpu
```

Defaults: if `--output` is omitted it writes `<input>_crt.mp4`; width/height/fps stay unchanged when 0; `--gpu` uses NVENC when available, otherwise CPU x264.

