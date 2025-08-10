from __future__ import annotations
import argparse
import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import threading


def ensure_deps() -> None:
    try:
        import importlib.util as _iu
        need = ["numpy", "PIL", "moviepy", "PySide6"]
        for name in need:
            if _iu.find_spec(name) is None:
                raise RuntimeError("missing")
        return
    except Exception:
        req = Path(__file__).parent / "requirements.txt"
        if req.exists():
            cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", str(req)]
        else:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                "moviepy==1.0.3",
                "Pillow>=9.0.0",
                "numpy>=1.21.0",
                "imageio-ffmpeg>=0.4.8",
                "PySide6>=6.5.0",
            ]
        subprocess.run(cmd, check=True)
        importlib.invalidate_caches()


ensure_deps()

from PIL import Image, ImageFilter
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import imageio_ffmpeg as iio_ffmpeg


def shift_channel(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return arr
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


def make_scanline_mask_dynamic(h: int, strength: float, period_px: float, phase_px: float) -> np.ndarray:
    y = np.arange(h, dtype=np.float32)
    s = 0.5 * (1.0 + np.sin((2.0 * np.pi / max(1e-6, period_px)) * (y + phase_px)))
    line = 1.0 - strength * s
    return line


def make_triad_mask(h: int, w: int, strength: float) -> np.ndarray:
    x = np.arange(w)[None, :]
    m0 = (x % 3 == 0).astype(np.float32)
    m1 = (x % 3 == 1).astype(np.float32)
    m2 = (x % 3 == 2).astype(np.float32)
    base = 1.0 - strength
    r = base + strength * m0
    g = base + strength * m1
    b = base + strength * m2
    mask = np.stack([r, g, b], axis=2)
    mask = np.repeat(mask, h, axis=0)
    return mask


def make_vignette(h: int, w: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rx = max(1.0, w / 2.0)
    ry = max(1.0, h / 2.0)
    nx = (xx - cx) / rx
    ny = (yy - cy) / ry
    r2 = nx * nx + ny * ny
    v = 1.0 - strength * np.clip(r2, 0.0, 1.0)
    return v


def apply_crt_effect(
    frame: np.ndarray,
    scanline_strength: float,
    triad_mask: Optional[np.ndarray],
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    noise_strength: float,
    vignette_mask: Optional[np.ndarray],
    persistence: float,
    state_prev: Optional[np.ndarray],
    scanline_period_px: float,
    scanline_phase_px: float,
    fast_bloom: bool,
    pixel_size: int,
    glitch_amp_px: int = 0,
    glitch_height_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[0], frame.shape[1]
    img = frame.astype(np.float32) / 255.0
    if aberration_px != 0:
        r = shift_channel(img[:, :, 0], aberration_px, 0)
        g = img[:, :, 1]
        b = shift_channel(img[:, :, 2], -aberration_px, 0)
        img = np.stack([r, g, b], axis=2)
    if pixel_size > 1:
        im_px = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        sw = max(1, w // int(pixel_size))
        sh = max(1, h // int(pixel_size))
        small = im_px.resize((sw, sh), Image.NEAREST)
        big = small.resize((w, h), Image.NEAREST)
        img = np.asarray(big).astype(np.float32) / 255.0
    if bloom_strength > 0.0 and (bloom_sigma > 0.0 or fast_bloom):
        if fast_bloom:
            im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            ds = im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
            us = ds.resize((w, h), Image.BILINEAR)
            blurf = np.asarray(us).astype(np.float32) / 255.0
        else:
            im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            blur = im.filter(ImageFilter.GaussianBlur(radius=bloom_sigma))
            blurf = np.asarray(blur).astype(np.float32) / 255.0
        img = np.clip(img + bloom_strength * blurf, 0.0, 1.0)
    if triad_mask is not None:
        img = np.clip(img * triad_mask, 0.0, 1.0)
    if scanline_strength > 0.0:
        sl = make_scanline_mask_dynamic(h, scanline_strength, scanline_period_px, scanline_phase_px)
        img = np.clip(img * sl[:, None], 0.0, 1.0)
    if vignette_mask is not None:
        img = np.clip(img * vignette_mask[:, :, None], 0.0, 1.0)
    if noise_strength > 0.0:
        noise = np.random.standard_normal(size=img.shape[:2]).astype(np.float32)
        noise = noise * noise_strength / 255.0
        img = np.clip(img + noise[:, :, None], 0.0, 1.0)
    if glitch_amp_px > 0 and glitch_height_frac > 0.0:
        h2, w2 = img.shape[0], img.shape[1]
        y0 = max(0, min(h2, h2 - int(h2 * glitch_height_frac)))
        if y0 < h2:
            num_rows = h2 - y0
            seed = (int(abs(float(scanline_phase_px)) * 0.05) + (w2 << 10) + (h2 << 1)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            rows_idx = np.arange(num_rows, dtype=np.float32)
            amp_rows = np.asarray(float(glitch_amp_px) * np.exp(-3.0 * (rows_idx / max(1.0, float(num_rows)))), dtype=np.float32)
            base = rng.normal(loc=0.0, scale=0.5, size=num_rows).astype(np.float32)
            base = np.clip(base, -1.0, 1.0)
            jump_mask = rng.random(num_rows).astype(np.float32) < 0.03
            jump_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=num_rows)
            base = base + jump_mask * jump_sign
            offs_row = np.clip(base * amp_rows, -amp_rows, amp_rows)
            bottom = img[y0:, :, :]
            x = np.arange(w2, dtype=np.int32)[None, :]
            xi = (x + np.rint(offs_row)[:, None].astype(np.int32)) % w2
            idx = np.broadcast_to(xi[:, :, None], bottom.shape)
            bottom = np.take_along_axis(bottom, idx, axis=1)
            img[y0:, :, :] = bottom
    if state_prev is not None and persistence > 0.0:
        if state_prev.shape != img.shape:
            prev_im = Image.fromarray(np.clip(state_prev * 255.0, 0, 255).astype(np.uint8))
            prev_im = prev_im.resize((w, h), Image.BILINEAR)
            prev_arr = np.asarray(prev_im).astype(np.float32) / 255.0
        else:
            prev_arr = state_prev
        img = np.clip(persistence * prev_arr + (1.0 - persistence) * img, 0.0, 1.0)
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return out, img


def apply_static_effects(
    frame: np.ndarray,
    scanline_strength: float,
    triad_mask: Optional[np.ndarray],
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    noise_strength: float,
    vignette_mask: Optional[np.ndarray],
    scanline_period_px: float,
    scanline_phase_px: float,
    fast_bloom: bool,
    pixel_size: int,
    glitch_amp_px: int,
    glitch_height_frac: float,
) -> np.ndarray:
    h, w = frame.shape[0], frame.shape[1]
    img = frame.astype(np.float32) / 255.0
    if aberration_px != 0:
        r = shift_channel(img[:, :, 0], aberration_px, 0)
        g = img[:, :, 1]
        b = shift_channel(img[:, :, 2], -aberration_px, 0)
        img = np.stack([r, g, b], axis=2)
    if pixel_size > 1:
        im_px = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        sw = max(1, w // int(pixel_size))
        sh = max(1, h // int(pixel_size))
        small = im_px.resize((sw, sh), Image.NEAREST)
        big = small.resize((w, h), Image.NEAREST)
        img = np.asarray(big).astype(np.float32) / 255.0
    if bloom_strength > 0.0 and (bloom_sigma > 0.0 or fast_bloom):
        if fast_bloom:
            im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            ds = im.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
            us = ds.resize((w, h), Image.BILINEAR)
            blurf = np.asarray(us).astype(np.float32) / 255.0
        else:
            im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            blur = im.filter(ImageFilter.GaussianBlur(radius=bloom_sigma))
            blurf = np.asarray(blur).astype(np.float32) / 255.0
        img = np.clip(img + bloom_strength * blurf, 0.0, 1.0)
    if triad_mask is not None:
        img = np.clip(img * triad_mask, 0.0, 1.0)
    if scanline_strength > 0.0:
        sl = make_scanline_mask_dynamic(h, scanline_strength, scanline_period_px, scanline_phase_px)
        img = np.clip(img * sl[:, None], 0.0, 1.0)
    if vignette_mask is not None:
        img = np.clip(img * vignette_mask[:, :, None], 0.0, 1.0)
    if noise_strength > 0.0:
        noise = np.random.standard_normal(size=img.shape[:2]).astype(np.float32)
        noise = noise * noise_strength / 255.0
        img = np.clip(img + noise[:, :, None], 0.0, 1.0)
    if glitch_amp_px > 0 and glitch_height_frac > 0.0:
        h2, w2 = img.shape[0], img.shape[1]
        y0 = max(0, min(h2, h2 - int(h2 * glitch_height_frac)))
        if y0 < h2:
            num_rows = h2 - y0
            seed = (int(abs(float(scanline_phase_px)) * 2.0) + (w2 << 10) + (h2 << 1)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            seg_len = max(8, min(32, w2 // 120 if w2 >= 120 else 8))
            num_segs = (w2 + seg_len - 1) // seg_len
            rows_idx = np.arange(num_rows, dtype=np.float32)
            amp_rows = float(glitch_amp_px) * (1.0 - (rows_idx / max(1.0, float(num_rows))))
            seg_offsets = rng.standard_normal((num_rows, num_segs)).astype(np.float32) * (amp_rows[:, None] * 0.7)
            base_rw = rng.standard_normal(num_rows).astype(np.float32)
            base = np.cumsum(base_rw) * 0.1
            base = np.clip(base, -amp_rows * 0.4, amp_rows * 0.4)
            seg_index = (np.arange(w2, dtype=np.int32) // int(seg_len)).astype(np.int32)
            bottom = img[y0:, :, :]
            offs_pp = base[:, None] + seg_offsets[np.arange(num_rows)[:, None], seg_index[None, :]]
            x = np.arange(w2, dtype=np.int32)[None, :]
            xi = (x + np.rint(offs_pp).astype(np.int32)) % w2
            idx = np.broadcast_to(xi[:, :, None], bottom.shape)
            bottom = np.take_along_axis(bottom, idx, axis=1)
            img[y0:, :, :] = bottom
    return img


def process_video(
    input_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scanline_strength: float,
    triad_strength: float,
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    noise_strength: float,
    vignette_strength: float,
    persistence: float,
    fps: Optional[int],
    crf: int,
    scanline_speed_px_s: float,
    scanline_period_px: float,
    fast_bloom: bool,
    pixel_size: int,
    gpu: bool,
    nvenc_preset: str,
    glitch_amp_px: int = 0,
    glitch_height_frac: float = 0.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> bool:
    clip = VideoFileClip(str(input_path))
    fps_out = int(fps) if fps and fps > 0 else int(clip.fps or 24)
    if width and height:
        out_w, out_h = int(width), int(height)
    else:
        out_w, out_h = clip.size
    triad_mask = make_triad_mask(out_h, out_w, triad_strength) if triad_strength > 0.0 else None
    vignette_mask = make_vignette(out_h, out_w, vignette_strength) if vignette_strength > 0.0 else None
    try:
        import math, tempfile
        total_frames = max(1, int(math.ceil((clip.duration or 0) * fps_out)))
        audio_path: Optional[str] = None
        if clip.audio is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".aac")
            tmp_path = tmp.name
            tmp.close()
            try:
                clip.audio.write_audiofile(tmp_path, fps=44100, nbytes=2, codec="aac", verbose=False, logger=None)
                audio_path = tmp_path
            except Exception:
                audio_path = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "h264_nvenc" if gpu else "libx264"
        used_gpu = codec == "h264_nvenc"
        ffparams = (["-cq", str(crf), "-preset", str(nvenc_preset), "-pix_fmt", "yuv420p"] if used_gpu else ["-crf", str(crf), "-pix_fmt", "yuv420p"])
        writer_kwargs = dict(
            filename=str(output_path),
            size=(out_w, out_h),
            fps=fps_out,
            codec=codec,
            audiofile=audio_path,
            threads=os.cpu_count() or 4,
            ffmpeg_params=ffparams,
        )
        if not used_gpu:
            writer_kwargs["preset"] = "medium"
        writer = FFMPEG_VideoWriter(**writer_kwargs)
        max_workers = max(os.cpu_count() - 2, 1)
        queue_cap = max_workers * 4
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        next_write = 0
        prev_state = None
        i = 0
        for frame in clip.iter_frames(fps=fps_out, dtype="uint8"):
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                im = Image.fromarray(frame)
                frame = np.asarray(im.resize((out_w, out_h), Image.BILINEAR))
            phase = (i / float(fps_out)) * scanline_speed_px_s
            def submit_job(idx: int, f: np.ndarray, ph: float):
                return executor.submit(
                    apply_static_effects,
                    f,
                    scanline_strength,
                    triad_mask,
                    aberration_px,
                    bloom_sigma,
                    bloom_strength,
                    noise_strength,
                    vignette_mask,
                    scanline_period_px,
                    ph,
                    fast_bloom,
                    pixel_size,
                    int(glitch_amp_px),
                    float(glitch_height_frac),
                )
            futures[i] = submit_job(i, frame, phase)
            i += 1
            while len(futures) >= queue_cap or next_write in futures:
                if next_write in futures:
                    static_img = futures.pop(next_write).result()
                    if prev_state is not None and persistence > 0.0:
                        if prev_state.shape != static_img.shape:
                            prev_im = Image.fromarray(np.clip(prev_state * 255.0, 0, 255).astype(np.uint8))
                            prev_im = prev_im.resize((out_w, out_h), Image.BILINEAR)
                            prev_state = np.asarray(prev_im).astype(np.float32) / 255.0
                        blended = np.clip(persistence * prev_state + (1.0 - persistence) * static_img, 0.0, 1.0)
                    else:
                        blended = static_img
                    prev_state = blended
                    out_frame = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
                    writer.write_frame(out_frame)
                    next_write += 1
                    if progress_cb is not None:
                        progress_cb(min(1.0, next_write / float(total_frames)))
                else:
                    break
        while next_write in futures:
            static_img = futures.pop(next_write).result()
            if prev_state is not None and persistence > 0.0:
                if prev_state.shape != static_img.shape:
                    prev_im = Image.fromarray(np.clip(prev_state * 255.0, 0, 255).astype(np.uint8))
                    prev_im = prev_im.resize((out_w, out_h), Image.BILINEAR)
                    prev_state = np.asarray(prev_im).astype(np.float32) / 255.0
                blended = np.clip(persistence * prev_state + (1.0 - persistence) * static_img, 0.0, 1.0)
            else:
                blended = static_img
            prev_state = blended
            out_frame = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
            writer.write_frame(out_frame)
            next_write += 1
            if progress_cb is not None:
                progress_cb(min(1.0, next_write / float(total_frames)))
        writer.close()
        executor.shutdown(wait=True)
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass
        if progress_cb is not None:
            progress_cb(1.0)
        return used_gpu
    finally:
        clip.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="")
    p.add_argument("--output", type=str)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--scanline-strength", type=float, default=0.6)
    p.add_argument("--triad-strength", type=float, default=0.35)
    p.add_argument("--aberration-px", type=int, default=1)
    p.add_argument("--bloom-sigma", type=float, default=1.2)
    p.add_argument("--bloom-strength", type=float, default=0.25)
    p.add_argument("--noise-strength", type=float, default=1.5)
    p.add_argument("--vignette-strength", type=float, default=0.25)
    p.add_argument("--persistence", type=float, default=0.2)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--scanline-speed", type=float, default=30.0)
    p.add_argument("--scanline-period", type=float, default=2.0)
    p.add_argument("--fast-bloom", action="store_true")
    p.add_argument("--no-fast-bloom", dest="fast_bloom", action="store_false")
    p.set_defaults(fast_bloom=True)
    p.add_argument("--pixel-size", type=int, default=2)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--nvenc-preset", type=str, default="p4")
    p.add_argument("--glitch-amp", type=int, default=0)
    p.add_argument("--glitch-height", type=float, default=0.0)
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.gui or not a.input:
        launch_gui()
        return
    inp = Path(a.input)
    if not inp.exists():
        raise SystemExit("input not found")
    out = Path(a.output) if a.output else inp.with_name(inp.stem + "_crt.mp4")
    used_gpu = process_video(
        input_path=inp,
        output_path=out,
        width=a.width if a.width > 0 else None,
        height=a.height if a.height > 0 else None,
        scanline_strength=float(max(0.0, min(1.0, a.scanline_strength))),
        triad_strength=float(max(0.0, min(1.0, a.triad_strength))),
        aberration_px=int(max(-8, min(8, a.aberration_px))),
        bloom_sigma=max(0.0, a.bloom_sigma),
        bloom_strength=max(0.0, a.bloom_strength),
        noise_strength=max(0.0, a.noise_strength),
        vignette_strength=float(max(0.0, min(1.0, a.vignette_strength))),
        persistence=float(max(0.0, min(0.95, a.persistence))),
        fps=a.fps if a.fps > 0 else None,
        crf=int(max(12, min(28, a.crf))),
        scanline_speed_px_s=float(a.scanline_speed),
        scanline_period_px=max(1.0, float(a.scanline_period)),
        fast_bloom=bool(a.fast_bloom),
        pixel_size=max(1, int(a.pixel_size)),
        gpu=bool(a.gpu),
        nvenc_preset=str(a.nvenc_preset),
        glitch_amp_px=max(0, int(a.glitch_amp)),
        glitch_height_frac=float(max(0.0, min(1.0, a.glitch_height))),
    )
    print("GPU NVENC used" if used_gpu else "CPU x264 used")


def launch_gui() -> None:
    from PySide6 import QtCore, QtGui, QtWidgets

    class ExportDialog(QtWidgets.QDialog):
        def __init__(self, parent: QtWidgets.QWidget, src: Path) -> None:
            super().__init__(parent)
            self.setWindowTitle("Export")
            self.setModal(True)
            self.setFixedWidth(420)
            self.src = src
            self.out_edit = QtWidgets.QLineEdit(str(src.with_name(src.stem + "_crt.mp4")))
            browse = QtWidgets.QPushButton("Browse")
            browse.clicked.connect(self.browse)
            path_row = QtWidgets.QHBoxLayout()
            path_row.addWidget(self.out_edit, 1)
            path_row.addWidget(browse)
            self.width = QtWidgets.QSpinBox()
            self.width.setRange(0, 8192)
            self.width.setValue(0)
            self.height = QtWidgets.QSpinBox()
            self.height.setRange(0, 8192)
            self.height.setValue(0)
            self.fps = QtWidgets.QSpinBox()
            self.fps.setRange(0, 240)
            self.fps.setValue(0)
            self.gpu = QtWidgets.QCheckBox("Use GPU (NVENC)")
            self.gpu.setChecked(parent.gpu_cb.isChecked())
            form = QtWidgets.QFormLayout()
            form.addRow("output path", path_row)
            form.addRow("width (0 keep)", self.width)
            form.addRow("height (0 keep)", self.height)
            form.addRow("fps (0 keep)", self.fps)
            form.addRow("gpu", self.gpu)
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(form)
            layout.addWidget(btns)

        def browse(self) -> None:
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Output", self.out_edit.text(), "MP4 (*.mp4)")
            if out_path:
                self.out_edit.setText(out_path)

        def get_output_path(self) -> str:
            return self.out_edit.text()

        def get_options(self) -> dict:
            w = int(self.width.value()) or None
            h = int(self.height.value()) or None
            f = int(self.fps.value()) or None
            return {"width": w, "height": h, "fps": f, "gpu": bool(self.gpu.isChecked())}

    class CRTWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("CRT Filter")
            self.resize(1160, 760)
            self.video_label = QtWidgets.QLabel()
            self.video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.preview_frame = QtWidgets.QFrame()
            self.preview_frame.setObjectName("PreviewFrame")
            pf_layout = QtWidgets.QVBoxLayout(self.preview_frame)
            pf_layout.setContentsMargins(12, 12, 12, 12)
            pf_layout.addWidget(self.video_label, 1)
            shadow = QtWidgets.QGraphicsDropShadowEffect(self.preview_frame)
            shadow.setBlurRadius(24)
            shadow.setColor(QtGui.QColor(0, 0, 0, 200))
            shadow.setOffset(0, 8)
            self.preview_frame.setGraphicsEffect(shadow)
            self.gpu_cb = QtWidgets.QCheckBox("GPU (NVENC)")
            self.fast_bloom_cb = QtWidgets.QCheckBox("Fast Bloom")
            self.fast_bloom_cb.setChecked(True)
            self.scanline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.scanline_slider.setMinimum(0)
            self.scanline_slider.setMaximum(100)
            self.scanline_slider.setValue(60)
            self.scanline_val = QtWidgets.QDoubleSpinBox()
            self.scanline_val.setRange(0.0, 1.0)
            self.scanline_val.setSingleStep(0.01)
            self.scanline_val.setValue(0.6)
            self.triad_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.triad_slider.setMinimum(0)
            self.triad_slider.setMaximum(100)
            self.triad_slider.setValue(35)
            self.triad_val = QtWidgets.QDoubleSpinBox()
            self.triad_val.setRange(0.0, 1.0)
            self.triad_val.setSingleStep(0.01)
            self.triad_val.setValue(0.35)
            self.pixel_size = QtWidgets.QSpinBox()
            self.pixel_size.setRange(1, 16)
            self.pixel_size.setValue(2)
            self.aberration = QtWidgets.QSpinBox()
            self.aberration.setRange(-8, 8)
            self.aberration.setValue(1)
            self.noise_val = QtWidgets.QDoubleSpinBox()
            self.noise_val.setRange(0.0, 16.0)
            self.noise_val.setSingleStep(0.1)
            self.noise_val.setValue(1.5)
            self.bloom_sigma = QtWidgets.QDoubleSpinBox()
            self.bloom_sigma.setRange(0.0, 10.0)
            self.bloom_sigma.setSingleStep(0.1)
            self.bloom_sigma.setValue(1.2)
            self.bloom_strength = QtWidgets.QDoubleSpinBox()
            self.bloom_strength.setRange(0.0, 2.0)
            self.bloom_strength.setSingleStep(0.05)
            self.bloom_strength.setValue(0.25)
            self.vignette_val = QtWidgets.QDoubleSpinBox()
            self.vignette_val.setRange(0.0, 1.0)
            self.vignette_val.setSingleStep(0.01)
            self.vignette_val.setValue(0.25)
            self.persistence_val = QtWidgets.QDoubleSpinBox()
            self.persistence_val.setRange(0.0, 0.95)
            self.persistence_val.setSingleStep(0.01)
            self.persistence_val.setValue(0.2)
            self.scanline_speed = QtWidgets.QDoubleSpinBox()
            self.scanline_speed.setRange(0.0, 200.0)
            self.scanline_speed.setSingleStep(1.0)
            self.scanline_speed.setValue(60.0)
            self.scanline_period = QtWidgets.QDoubleSpinBox()
            self.scanline_period.setRange(1.0, 16.0)
            self.scanline_period.setSingleStep(0.1)
            self.scanline_period.setValue(2.0)
            self.glitch_amp = QtWidgets.QSpinBox()
            self.glitch_amp.setRange(0, 128)
            self.glitch_amp.setValue(0)
            self.glitch_height = QtWidgets.QDoubleSpinBox()
            self.glitch_height.setRange(0.0, 1.0)
            self.glitch_height.setSingleStep(0.01)
            self.glitch_height.setValue(0.0)
            self.crf_val = QtWidgets.QSpinBox()
            self.crf_val.setRange(12, 28)
            self.crf_val.setValue(18)
            self.nvenc_preset = QtWidgets.QLineEdit("p4")
            effects_form = QtWidgets.QFormLayout()
            effects_form.setLabelAlignment(QtCore.Qt.AlignRight)
            effects_form.addRow("scanline", self.scanline_slider)
            effects_form.addRow("scanline value", self.scanline_val)
            effects_form.addRow("triad", self.triad_slider)
            effects_form.addRow("triad value", self.triad_val)
            effects_form.addRow("pixel size", self.pixel_size)
            effects_form.addRow("aberration px", self.aberration)
            effects_form.addRow("noise", self.noise_val)
            effects_form.addRow("bloom sigma", self.bloom_sigma)
            effects_form.addRow("bloom strength", self.bloom_strength)
            effects_form.addRow("vignette", self.vignette_val)
            effects_form.addRow("persistence", self.persistence_val)
            motion_form = QtWidgets.QFormLayout()
            motion_form.setLabelAlignment(QtCore.Qt.AlignRight)
            motion_form.addRow("scanline speed", self.scanline_speed)
            motion_form.addRow("scanline period", self.scanline_period)
            motion_form.addRow("glitch amp (px)", self.glitch_amp)
            motion_form.addRow("glitch height (0-1)", self.glitch_height)
            output_form = QtWidgets.QFormLayout()
            output_form.setLabelAlignment(QtCore.Qt.AlignRight)
            output_form.addRow("gpu (nvenc)", self.gpu_cb)
            output_form.addRow("fast bloom", self.fast_bloom_cb)
            output_form.addRow("crf/cq", self.crf_val)
            output_form.addRow("nvenc preset", self.nvenc_preset)
            self.render_btn = QtWidgets.QPushButton("Render")
            self.reset_btn = QtWidgets.QPushButton("Reset")
            out_buttons = QtWidgets.QHBoxLayout()
            out_buttons.addStretch(1)
            out_buttons.addWidget(self.reset_btn)
            out_buttons.addWidget(self.render_btn)
            output_col = QtWidgets.QVBoxLayout()
            output_col.addLayout(output_form)
            output_col.addStretch(1)
            output_col.addLayout(out_buttons)
            effects_tab = QtWidgets.QWidget()
            effects_tab.setLayout(effects_form)
            motion_tab = QtWidgets.QWidget()
            motion_tab.setLayout(motion_form)
            output_tab = QtWidgets.QWidget()
            output_tab.setLayout(output_col)
            tabs = QtWidgets.QTabWidget()
            tabs.addTab(effects_tab, "Effects")
            tabs.addTab(motion_tab, "Motion")
            tabs.addTab(output_tab, "Output")
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self.preview_frame)
            splitter.addWidget(tabs)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            central = QtWidgets.QWidget()
            central_layout = QtWidgets.QHBoxLayout(central)
            central_layout.setContentsMargins(12, 12, 12, 12)
            central_layout.addWidget(splitter)
            self.setCentralWidget(central)
            bar = QtWidgets.QToolBar()
            bar.setMovable(False)
            bar.setIconSize(QtCore.QSize(20, 20))
            self.addToolBar(QtCore.Qt.TopToolBarArea, bar)
            self.actOpen = QtGui.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), "Open", self)
            self.actPlay = QtGui.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay), "Play", self)
            self.actRender = QtGui.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Render", self)
            self.actGPU = QtGui.QAction("GPU", self)
            self.actGPU.setCheckable(True)
            self.actFast = QtGui.QAction("Fast Bloom", self)
            self.actFast.setCheckable(True)
            self.actFast.setChecked(True)
            bar.addAction(self.actOpen)
            bar.addAction(self.actPlay)
            bar.addAction(self.actRender)
            bar.addSeparator()
            bar.addAction(self.actGPU)
            bar.addAction(self.actFast)
            self.status = self.statusBar()
            self.time_label = QtWidgets.QLabel("00:00 / 00:00")
            self.status.addPermanentWidget(self.time_label)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setFixedWidth(300)
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            self.progress.setVisible(False)
            self.status.addPermanentWidget(self.progress)
            self.actOpen.triggered.connect(self.on_open)
            self.actPlay.triggered.connect(self.on_play_pause)
            self.actRender.triggered.connect(self.on_render)
            self.actGPU.toggled.connect(self.gpu_cb.setChecked)
            self.gpu_cb.toggled.connect(self.actGPU.setChecked)
            self.actFast.toggled.connect(self.fast_bloom_cb.setChecked)
            self.fast_bloom_cb.toggled.connect(self.actFast.setChecked)
            self.scanline_slider.valueChanged.connect(self.on_scanline_slider)
            self.triad_slider.valueChanged.connect(self.on_triad_slider)
            self.scanline_val.valueChanged.connect(self.on_scanline_val)
            self.triad_val.valueChanged.connect(self.on_triad_val)
            self.reset_btn.clicked.connect(self.on_reset)
            self.clip = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.playing = False
            self.t = 0.0
            self.prev_img = None
            self.preview_max_w = 960
            self.preview_max_h = 540

        def on_scanline_slider(self, v: int) -> None:
            self.scanline_val.setValue(float(v) / 100.0)

        def on_triad_slider(self, v: int) -> None:
            self.triad_val.setValue(float(v) / 100.0)

        def on_scanline_val(self, v: float) -> None:
            self.scanline_slider.setValue(int(v * 100.0))

        def on_triad_val(self, v: float) -> None:
            self.triad_slider.setValue(int(v * 100.0))

        def on_open(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", str(Path.cwd()), "Videos (*.mp4 *.mov *.mkv *.avi)")
            if not path:
                return
            self.load_clip(Path(path))

        def load_clip(self, p: Path) -> None:
            self.prev_img = None
            self.clip = VideoFileClip(str(p))
            fps = max(1, int(self.clip.fps or 24))
            self.timer.setInterval(int(1000 / fps))
            self.t = 0.0

        def on_play_pause(self) -> None:
            if self.clip is None:
                return
            self.playing = not self.playing
            self.actPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause) if self.playing else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.actPlay.setText("Pause" if self.playing else "Play")
            if self.playing:
                if not self.timer.isActive():
                    self.timer.start()
            else:
                self.timer.stop()

        def on_tick(self) -> None:
            if self.clip is None:
                return
            frame = self.clip.get_frame(self.t)
            w, h = frame.shape[1], frame.shape[0]
            scale = min(self.preview_max_w / max(1, w), self.preview_max_h / max(1, h), 1.0)
            if scale < 1.0:
                im = Image.fromarray(frame)
                frame = np.asarray(im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR))
            out, self.prev_img = apply_crt_effect(
                frame=frame,
                scanline_strength=float(self.scanline_val.value()),
                triad_mask=make_triad_mask(frame.shape[0], frame.shape[1], float(self.triad_val.value())) if self.triad_val.value() > 0.0 else None,
                aberration_px=int(self.aberration.value()),
                bloom_sigma=float(self.bloom_sigma.value()),
                bloom_strength=float(self.bloom_strength.value()),
                noise_strength=float(self.noise_val.value()),
                vignette_mask=make_vignette(frame.shape[0], frame.shape[1], float(self.vignette_val.value())) if self.vignette_val.value() > 0.0 else None,
                persistence=float(self.persistence_val.value()),
                state_prev=self.prev_img,
                scanline_period_px=float(self.scanline_period.value()),
                scanline_phase_px=float(self.scanline_speed.value()) * self.t,
                fast_bloom=bool(self.fast_bloom_cb.isChecked()),
                pixel_size=int(self.pixel_size.value()),
                glitch_amp_px=int(self.glitch_amp.value()),
                glitch_height_frac=float(self.glitch_height.value()),
            )
            qimg = QtGui.QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix)
            self.t += 1.0 / max(1.0, self.clip.fps or 24.0)
            if self.t >= float(self.clip.duration or 1.0):
                self.t = 0.0
            tl = int(self.t)
            dur = int(self.clip.duration or 0)
            self.time_label.setText(f"{tl//60:02d}:{tl%60:02d} / {dur//60:02d}:{dur%60:02d}")

        def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
            super().resizeEvent(e)
            self.preview_max_w = max(1, self.preview_frame.width() - 24)
            self.preview_max_h = max(1, self.preview_frame.height() - 24)

        def on_render(self) -> None:
            if self.clip is None:
                return
            src_path = Path(self.clip.filename)
            dlg = ExportDialog(self, src_path)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            out_path = dlg.get_output_path()
            opts = dlg.get_options()
            self.setEnabled(False)
            def run_render() -> None:
                try:
                    QtCore.QMetaObject.invokeMethod(self.progress, "setVisible", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
                    QtCore.QMetaObject.invokeMethod(self.progress, "setRange", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 0), QtCore.Q_ARG(int, 100))
                    used_gpu = process_video(
                        input_path=src_path,
                        output_path=Path(out_path),
                        width=opts["width"],
                        height=opts["height"],
                        scanline_strength=float(self.scanline_val.value()),
                        triad_strength=float(self.triad_val.value()),
                        aberration_px=int(self.aberration.value()),
                        bloom_sigma=float(self.bloom_sigma.value()),
                        bloom_strength=float(self.bloom_strength.value()),
                        noise_strength=float(self.noise_val.value()),
                        vignette_strength=float(self.vignette_val.value()),
                        persistence=float(self.persistence_val.value()),
                        fps=opts["fps"],
                        crf=int(self.crf_val.value()),
                        scanline_speed_px_s=float(self.scanline_speed.value()),
                        scanline_period_px=float(self.scanline_period.value()),
                        fast_bloom=bool(self.fast_bloom_cb.isChecked()),
                        pixel_size=int(self.pixel_size.value()),
                        gpu=bool(self.gpu_cb.isChecked()) if opts["gpu"] is None else bool(opts["gpu"]),
                        nvenc_preset=str(self.nvenc_preset.text()),
                         glitch_amp_px=int(self.glitch_amp.value()),
                         glitch_height_frac=float(self.glitch_height.value()),
                        progress_cb=lambda f: QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, int(f * 100))),
                    )
                    QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 100))
                    QtCore.QMetaObject.invokeMethod(self.progress, "setVisible", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, False))
                    QtCore.QMetaObject.invokeMethod(self, "_show_done", QtCore.Qt.QueuedConnection)
                    QtCore.QMetaObject.invokeMethod(self.status, "showMessage", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "GPU NVENC used" if used_gpu else "CPU x264 used"))
                finally:
                    QtCore.QMetaObject.invokeMethod(self, "setEnabled", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
            th = threading.Thread(target=run_render, daemon=True)
            th.start()

        def _show_done(self) -> None:
            QtWidgets.QMessageBox.information(self, "Done", "Render complete")

        def on_reset(self) -> None:
            self.scanline_slider.setValue(60)
            self.scanline_val.setValue(0.6)
            self.triad_slider.setValue(35)
            self.triad_val.setValue(0.35)
            self.pixel_size.setValue(2)
            self.aberration.setValue(1)
            self.noise_val.setValue(1.5)
            self.bloom_sigma.setValue(1.2)
            self.bloom_strength.setValue(0.25)
            self.vignette_val.setValue(0.25)
            self.persistence_val.setValue(0.2)
            self.scanline_speed.setValue(60.0)
            self.scanline_period.setValue(2.0)
            self.crf_val.setValue(18)
            self.nvenc_preset.setText("p4")
            self.fast_bloom_cb.setChecked(True)
            self.gpu_cb.setChecked(False)
            self.glitch_amp.setValue(0)
            self.glitch_height.setValue(0.0)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(22, 24, 28))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(28, 30, 34))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(34, 36, 40))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(34, 36, 40))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 170, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(palette)
    qss = """
    QMainWindow { background: #16181c; }
    QToolBar { background: #1e2126; border: none; padding: 6px; spacing: 8px; }
    QToolBar QToolButton { color: #e6e6e6; background: #2a2e35; border: 1px solid #3a4048; border-radius: 6px; padding: 6px 10px; }
    QToolBar QToolButton:checked { background: #0d6efd; border-color: #0b62d8; }
    QToolBar QToolButton:hover { background: #343943; }
    QLabel { color: #e6e6e6; }
    QTabBar::tab { background: #1e2126; color: #cfd2d6; padding: 8px 12px; border: 1px solid #2b2f36; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }
    QTabBar::tab:selected { background: #252a31; color: #e6e6e6; }
    QTabWidget::pane { border: 1px solid #2b2f36; top: -1px; }
    QFrame#PreviewFrame { background: #0f1115; border: 1px solid #2b2f36; border-radius: 12px; }
    QPushButton { color: #e6e6e6; background: #2a2e35; border: 1px solid #3a4048; border-radius: 8px; padding: 8px 14px; }
    QPushButton:hover { background: #343943; }
    QPushButton:pressed { background: #1f2329; }
    QSlider::groove:horizontal { height: 6px; background: #2a2e35; border-radius: 3px; }
    QSlider::handle:horizontal { background: #0d6efd; width: 16px; margin: -6px 0; border-radius: 8px; }
    QSpinBox, QDoubleSpinBox, QLineEdit { background: #1e2126; color: #e6e6e6; border: 1px solid #3a4048; border-radius: 6px; padding: 4px 6px; }
    QCheckBox { color: #cfd2d6; }
    QStatusBar { background: #1a1d22; color: #cfd2d6; }
    """
    app.setStyleSheet(qss)
    w = CRTWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()


