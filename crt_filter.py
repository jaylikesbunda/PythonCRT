from __future__ import annotations
import argparse
import os
import sys
import subprocess
import importlib
from pathlib import Path
import json
from typing import Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import threading
import time
from collections import defaultdict


def ensure_deps() -> None:
    try:
        import importlib.util as _iu
        need = ["numpy", "PIL", "moviepy", "PySide6", "cv2"]
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
                "opencv-python-headless>=4.8.0",
            ]
        subprocess.run(cmd, check=True)
        importlib.invalidate_caches()


ensure_deps()

from PIL import Image, ImageFilter
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import imageio_ffmpeg as iio_ffmpeg
from PIL import ImageDraw, ImageFont, ImageFont
import subprocess as _sub


_perf_lock = threading.Lock()
_perf_totals = defaultdict(float)
_perf_counts = defaultdict(int)


def perf_add(name: str, dt: float) -> None:
    with _perf_lock:
        _perf_totals[name] += float(dt)
        _perf_counts[name] = _perf_counts.get(name, 0) + 1


def perf_report(total_frames: int, total_seconds: float) -> None:
    items = sorted(_perf_totals.items(), key=lambda kv: kv[1], reverse=True)
    print(f"perf total {total_seconds:.3f}s")
    print(f"perf frames {total_frames}")
    for k, v in items:
        c = _perf_counts.get(k, 0)
        avg = (v / c * 1000.0) if c else 0.0
        print(f"{k} total={v:.3f}s count={c} avg_ms={avg:.2f}")


def perf_timed_iter(iterable, name: str):
    it = iter(iterable)
    while True:
        t0 = time.perf_counter()
        try:
            v = next(it)
        except StopIteration:
            return
        perf_add(name, time.perf_counter() - t0)
        yield v


def perf_report_auto() -> None:
    total_frames = _perf_counts.get("crt.total", 0) + _perf_counts.get("fx.total", 0)
    total_seconds = _perf_totals.get("crt.total", 0.0) + _perf_totals.get("fx.total", 0.0)
    perf_report(total_frames=total_frames, total_seconds=total_seconds)


def perf_reset() -> None:
    with _perf_lock:
        _perf_totals.clear()
        _perf_counts.clear()


def normalize_nvenc_preset(preset: str) -> str:
    """Map NVENC preset aliases to values supported by older ffmpeg builds.

    Accepts p1..p7 (newer ffmpeg) and maps to older tokens; if unsupported, fallback to 'medium'.
    """
    if not preset:
        return "medium"
    p = str(preset).strip().lower()
    # Allowed legacy presets
    legacy_allowed = {
        "default",
        "slow",
        "medium",
        "fast",
        "hp",
        "hq",
        "bd",
        "ll",
        "llhq",
        "llhp",
        "lossless",
        "losslesshp",
    }
    if p in legacy_allowed:
        return p
    # Map p1..p7 to reasonable legacy equivalents
    p_map = {
        "p1": "hp",        # fastest
        "p2": "fast",
        "p3": "medium",
        "p4": "default",
        "p5": "hq",
        "p6": "bd",
        "p7": "slow",      # highest quality
    }
    return p_map.get(p, "medium")


def can_use_nvenc() -> bool:
    """Return True if ffmpeg can actually encode with h264_nvenc at runtime.

    Some builds list the encoder but fail at runtime (e.g., missing nvcuda.dll).
    We probe by attempting a tiny encode to the null muxer.
    """
    try:
        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = None
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        return False
    try:
        # Tiny 16x16, 0.05s color source, encode with h264_nvenc, discard output
        # Cross-platform null sink: use '-' with null muxer
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.05",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False


def can_use_amf() -> bool:
    """Return True if ffmpeg can encode with h264_amf (AMD) at runtime."""
    try:
        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = None
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        return False
    try:
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.05",
            "-c:v",
            "h264_amf",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False


def shift_channel(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return arr
    return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)


def make_scanline_mask_dynamic(h: int, strength: float, period_px: float, phase_px: float) -> np.ndarray:
    y = np.arange(h, dtype=np.float32)
    s = 0.5 * (1.0 + np.sin((2.0 * np.pi / max(1e-6, period_px)) * (y + phase_px)))
    line = 1.0 - strength * s
    return line


def make_triad_mask(h: int, w: int, strength: float, softness_px: float = 0.0) -> np.ndarray:
    x = np.arange(w)[None, :]
    m0 = (x % 3 == 0).astype(np.float32)
    m1 = (x % 3 == 1).astype(np.float32)
    m2 = (x % 3 == 2).astype(np.float32)
    base = 1.0 - float(strength)
    r = base + float(strength) * m0
    g = base + float(strength) * m1
    b = base + float(strength) * m2
    mask = np.stack([r, g, b], axis=2).astype(np.float32)
    mask = np.repeat(mask, h, axis=0)
    s = float(max(0.0, softness_px))
    if s > 0.0:
        k = max(3, int(round(s * 3)) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, 1), sigmaX=s, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    return mask.astype(np.float32)


def _apply_triad_mask(img: np.ndarray, mask: np.ndarray, gamma: float = 2.2, preserve_luma: bool = True) -> np.ndarray:
    g = float(gamma)
    if (not preserve_luma) and abs(g - 1.0) < 1e-3:
        out = img * mask
        return np.clip(out, 0.0, 1.0)
    if g <= 0.0:
        out = img * mask
        return np.clip(out, 0.0, 1.0)
    lut_size = 1024
    scale = float(lut_size)
    lut_x = np.linspace(0.0, 1.0, lut_size + 1, dtype=np.float32)
    lut_g = np.power(lut_x, g, dtype=np.float32)
    idx = np.clip((np.clip(img, 0.0, 1.0) * scale).astype(np.int32), 0, lut_size)
    lin = lut_g[idx]
    out_lin = lin * mask
    if preserve_luma:
        w_r, w_g, w_b = 0.2126, 0.7152, 0.0722
        y_before = w_r * lin[:, :, 0] + w_g * lin[:, :, 1] + w_b * lin[:, :, 2]
        y_after = w_r * out_lin[:, :, 0] + w_g * out_lin[:, :, 1] + w_b * out_lin[:, :, 2]
        ratio = y_before / np.maximum(y_after, 1e-6)
        ratio = np.clip(ratio, 0.5, 2.0)
        out_lin = out_lin * ratio[:, :, None]
    lut_inv = np.power(lut_x, 1.0 / g, dtype=np.float32)
    idx2 = np.clip((np.clip(out_lin, 0.0, 1.0) * scale).astype(np.int32), 0, lut_size)
    out = lut_inv[idx2]
    return np.clip(out, 0.0, 1.0)


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


def apply_color_adjustments(
    img: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
    saturation: float,
    temperature: float,
) -> np.ndarray:
    # Saturation
    if saturation != 1.0:
        luma = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        img = np.clip(luma[:, :, None] + (img - luma[:, :, None]) * float(saturation), 0.0, 1.0)
    # Temperature: warm/cool via R/B gains
    if temperature != 0.0:
        t = float(temperature)
        r_gain = float(np.clip(1.0 + 0.5 * t, 0.5, 1.5))
        b_gain = float(np.clip(1.0 - 0.5 * t, 0.5, 1.5))
        img[:, :, 0] = np.clip(img[:, :, 0] * r_gain, 0.0, 1.0)
        img[:, :, 2] = np.clip(img[:, :, 2] * b_gain, 0.0, 1.0)
    # Brightness/contrast
    if brightness != 0.0 or contrast != 1.0:
        img = np.clip((img - 0.5) * float(contrast) + 0.5 + float(brightness), 0.0, 1.0)
    # Gamma
    if gamma != 1.0 and gamma > 0.0:
        inv_g = 1.0 / float(gamma)
        img = np.clip(np.power(img, inv_g, dtype=np.float32), 0.0, 1.0)
    return img


def make_scanline_mask_2d(
    h: int,
    w: int,
    strength: float,
    period_px: float,
    phase_px: float,
    angle_deg: float,
    thickness: float,
) -> np.ndarray:
    if strength <= 0.0:
        return np.ones((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    theta = np.deg2rad(float(angle_deg))
    slanted = yy + np.tan(theta) * xx
    omega = 2.0 * np.pi / max(1e-6, float(period_px))
    s = 0.5 * (1.0 + np.sin(omega * (slanted + float(phase_px))))
    # thickness maps to sharpness: >1 widens bright bands, <1 narrows
    sharp = np.clip(float(thickness), 0.1, 4.0)
    s_shaped = np.power(s, 1.0 / sharp)
    mask = 1.0 - float(strength) * s_shaped
    return mask.astype(np.float32)


def apply_barrel_warp(img: np.ndarray, strength: float) -> np.ndarray:
    s = float(strength)
    if s == 0.0:
        return img
    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    x = (np.arange(w, dtype=np.float32) - cx) / max(1.0, cx)
    y = (np.arange(h, dtype=np.float32) - cy) / max(1.0, cy)
    xv, yv = np.meshgrid(x, y)
    r2 = xv * xv + yv * yv
    k = s * 0.5  # scale factor
    # radial barrel distortion: r' = r * (1 + k*r^2)
    factor = 1.0 + k * r2
    map_x = (xv * factor * cx + cx).astype(np.float32)
    map_y = (yv * factor * cy + cy).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def _parse_hex_color(s: str) -> Tuple[int, int, int]:
    try:
        st = s.strip()
        if st.startswith("#"):
            st = st[1:]
        if len(st) == 6:
            r = int(st[0:2], 16)
            g = int(st[2:4], 16)
            b = int(st[4:6], 16)
            return r, g, b
    except Exception:
        pass
    return 255, 255, 255


def _make_text_overlay_rgba(w: int, h: int, text: str, font_family: str, size: int, color_hex: str, pos: Tuple[int, int]) -> np.ndarray:
    if not text:
        return np.zeros((h, w, 4), dtype=np.uint8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = None
    # First try: if a file path was provided
    if font_family and (os.path.isfile(font_family)):
        try:
            font = ImageFont.truetype(font_family, size)
        except Exception:
            font = None
    # Second: try common Windows fonts mapped from family
    if font is None:
        candidates = []
        fam = (font_family or "").lower()
        win_dir = os.environ.get("WINDIR", "C:\\Windows")
        fonts_dir = os.path.join(win_dir, "Fonts")
        # Map some known families to TTF
        mapping = {
            "arial": "arial.ttf",
            "segoe ui": "segoeui.ttf",
            "consolas": "consola.ttf",
            "tahoma": "tahoma.ttf",
            "times new roman": "times.ttf",
            "courier new": "cour.ttf",
        }
        if fam in mapping:
            candidates.append(os.path.join(fonts_dir, mapping[fam]))
        # Also try raw family + .ttf
        if fam:
            candidates.append(os.path.join(fonts_dir, f"{fam}.ttf"))
        for path in candidates:
            try:
                if os.path.isfile(path):
                    font = ImageFont.truetype(path, size)
                    break
            except Exception:
                font = None
    # Fallbacks
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    r, g, b = _parse_hex_color(color_hex)
    x, y = int(pos[0]), int(pos[1])
    draw.text((x, y), text, font=font, fill=(r, g, b, 255))
    return np.asarray(img, dtype=np.uint8)


def _make_text_overlay_rgba_qt(w: int, h: int, text: str, font_family: str, size_px: int, color_hex: str, pos: Tuple[int, int]) -> np.ndarray:
    try:
        from PySide6 import QtGui, QtCore
    except Exception:
        # Fallback to PIL path if Qt not available
        return _make_text_overlay_rgba(w, h, text, font_family, size_px, color_hex, pos)
    img = QtGui.QImage(w, h, QtGui.QImage.Format_RGBA8888)
    img.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(img)
    try:
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing | QtGui.QPainter.SmoothPixmapTransform, True)
        # Resolve font: accept direct file path or family name
        resolved_family = None
        if font_family and os.path.isfile(font_family):
            try:
                from PySide6 import QtGui as _QtGui
                fid = _QtGui.QFontDatabase.addApplicationFont(font_family)
                fams = _QtGui.QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                if fams:
                    resolved_family = fams[0]
            except Exception:
                resolved_family = None
        if not resolved_family and font_family:
            resolved_family = font_family
        font = QtGui.QFont(resolved_family) if resolved_family else QtGui.QFont()
        # Use pixel size for consistency with preview/export coords
        font.setPixelSize(int(max(1, size_px)))
        painter.setFont(font)
        r, g, b = _parse_hex_color(color_hex)
        painter.setPen(QtGui.QColor(int(r), int(g), int(b), 255))
        x, y = int(pos[0]), int(pos[1])
        painter.drawText(x, y + int(font.pixelSize() or size_px), text)
    finally:
        painter.end()
    # Extract pixel data; QImage rows may be padded, so respect bytesPerLine
    bpl = int(img.bytesPerLine())
    mv = img.bits()
    try:
        buf = mv.tobytes()
    except AttributeError:
        # Fallback for older bindings
        buf = bytes(mv)
    arr = np.frombuffer(buf, dtype=np.uint8)
    # Guard against incomplete buffers
    expected = bpl * h
    if arr.size < expected:
        arr = np.pad(arr, (0, max(0, expected - arr.size)))
    arr = arr.reshape((h, bpl // 4, 4))
    arr = arr[:, :w, :]
    return arr.copy()


class FFmpegRawReader:
    def __init__(self, src_path: str, out_w: int, out_h: int, fps: int, hwaccel: Optional[str] = None) -> None:
        self.src_path = src_path
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.fps = int(max(1, fps))
        self.hwaccel = hwaccel
        self.proc = None
        self._start()

    def _start(self) -> None:
        ffmpeg = iio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        if self.hwaccel and self.hwaccel != "auto":
            cmd += ["-hwaccel", self.hwaccel]
        cmd += [
            "-i", self.src_path,
            "-vf", f"scale={self.out_w}:{self.out_h}",
            "-r", str(self.fps),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        self.proc = _sub.Popen(cmd, stdout=_sub.PIPE, stderr=_sub.PIPE)

    def iter_frames(self):
        assert self.proc is not None and self.proc.stdout is not None
        frame_size = self.out_w * self.out_h * 3
        while True:
            buf = self.proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            yield arr.reshape((self.out_h, self.out_w, 3))

    def close(self) -> None:
        if self.proc is not None:
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


def _map_decoder_to_hwaccel(pref: str) -> Optional[str]:
    p = (pref or "auto").strip().lower()
    if p in ("", "auto"):
        return None
    if p == "nvidia":
        return "cuda"
    if p == "amd":
        return "dxva2"  # or d3d11va on newer builds
    if p == "intel":
        return "d3d11va"
    if p == "cpu":
        return None
    return None

def apply_crt_effect(
    frame: np.ndarray,
    scanline_strength: float,
    triad_mask: Optional[np.ndarray],
    triad_gamma: float,
    triad_preserve_luma: bool,
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    bloom_threshold: float,
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
    time_sec: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    flicker_strength: float = 0.0,
    flicker_hz: float = 0.0,
    grain_size: int = 1,
    scanline_angle: float = 0.0,
    scanline_thickness: float = 1.0,
    warp_strength: float = 0.0,
    # EFX new
    beam_spread_strength: float = 0.0,
    hbleed_sigma: float = 0.0,
    hbleed_strength: float = 0.0,
    vbleed_sigma: float = 0.0,
    vbleed_strength: float = 0.0,
    jitter_amp_px: int = 0,
    jitter_speed_hz: float = 0.0,
    screen_jitter_amp_px: int = 0,
    screen_jitter_speed_hz: float = 0.0,
    rfi_strength: float = 0.0,
    rfi_freq: float = 0.0,
    rfi_speed_hz: float = 0.0,
    rfi_angle_deg: float = 45.0,
    waves_amp_px: int = 0,
    waves_freq: float = 0.0,
    waves_speed_hz: float = 0.0,
    strobe_hz: float = 0.0,
    strobe_duty: float = 0.5,
    anaglyph_offset_px: int = 0,
    anaglyph_mode: str = "red_cyan",
    text_overlay_rgba: Optional[np.ndarray] = None,
    text_overlay_after: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    t_all = time.perf_counter()
    h, w = frame.shape[0], frame.shape[1]
    t0 = time.perf_counter()
    img = frame.astype(np.float32) / 255.0
    perf_add("crt.astype", time.perf_counter() - t0)
    if aberration_px != 0:
        t0 = time.perf_counter()
        r = shift_channel(img[:, :, 0], aberration_px, 0)
        g = img[:, :, 1]
        b = shift_channel(img[:, :, 2], -aberration_px, 0)
        img = np.stack([r, g, b], axis=2)
        perf_add("crt.aberration", time.perf_counter() - t0)
    if pixel_size > 1:
        t0 = time.perf_counter()
        sw = max(1, w // int(pixel_size))
        sh = max(1, h // int(pixel_size))
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        perf_add("crt.pixelate", time.perf_counter() - t0)
    t0 = time.perf_counter()
    img = apply_color_adjustments(img, brightness, contrast, gamma, saturation, temperature)
    perf_add("crt.color", time.perf_counter() - t0)
    if text_overlay_rgba is not None and not text_overlay_after:
        t0 = time.perf_counter()
        ov = text_overlay_rgba
        if ov.dtype != np.uint8:
            ov = np.clip(ov, 0, 255).astype(np.uint8)
        if ov.shape[0] != h or ov.shape[1] != w:
            ov = np.asarray(Image.fromarray(ov, mode="RGBA").resize((w, h), Image.BILINEAR))
        alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = ov[:, :, :3].astype(np.float32) / 255.0
        img = np.clip(img * (1.0 - alpha) + rgb * alpha, 0.0, 1.0)
        perf_add("crt.text_before", time.perf_counter() - t0)
    if bloom_strength > 0.0 and (bloom_sigma > 0.0 or fast_bloom):
        t0 = time.perf_counter()
        src = img
        if bloom_threshold > 0.0:
            thr = float(min(0.99, max(0.0, bloom_threshold)))
            src = np.clip((img - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
        if fast_bloom:
            ds = cv2.resize(src, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
            blurf = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            k = max(1, int(round(bloom_sigma * 3)) * 2 + 1)
            blurf = cv2.GaussianBlur(src, (k, k), sigmaX=bloom_sigma, sigmaY=bloom_sigma, borderType=cv2.BORDER_REPLICATE)
        img = np.clip(img + bloom_strength * blurf, 0.0, 1.0)
        perf_add("crt.bloom", time.perf_counter() - t0)
    if triad_mask is not None:
        t0 = time.perf_counter()
        img = _apply_triad_mask(img, triad_mask, triad_gamma, triad_preserve_luma)
        perf_add("crt.triad", time.perf_counter() - t0)
    # Directional bleed (before scanlines for softer look)
    if hbleed_strength > 0.0 and hbleed_sigma > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(hbleed_sigma) * 3)) * 2 + 1)
        hb = cv2.GaussianBlur(img, (k, 1), sigmaX=float(hbleed_sigma), sigmaY=0, borderType=cv2.BORDER_REPLICATE)
        img = np.clip((1.0 - float(hbleed_strength)) * img + float(hbleed_strength) * hb, 0.0, 1.0)
        perf_add("crt.hbleed", time.perf_counter() - t0)
    if vbleed_strength > 0.0 and vbleed_sigma > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(vbleed_sigma) * 3)) * 2 + 1)
        vb = cv2.GaussianBlur(img, (1, k), sigmaX=0, sigmaY=float(vbleed_sigma), borderType=cv2.BORDER_REPLICATE)
        img = np.clip((1.0 - float(vbleed_strength)) * img + float(vbleed_strength) * vb, 0.0, 1.0)
        perf_add("crt.vbleed", time.perf_counter() - t0)
    if scanline_strength > 0.0:
        t0 = time.perf_counter()
        if not hasattr(apply_crt_effect, "_scan_cache"):
            apply_crt_effect._scan_cache = {}
        # Quantize phase to reduce cache churn
        phase_q = float(round(scanline_phase_px * 4.0) / 4.0)
        key = (h, w, float(scanline_strength), float(scanline_period_px), float(scanline_angle), float(scanline_thickness), phase_q)
        cache = apply_crt_effect._scan_cache
        mask3 = cache.get(key)
        if mask3 is None:
            if scanline_angle == 0.0 and scanline_thickness == 1.0:
                sl = make_scanline_mask_dynamic(h, scanline_strength, scanline_period_px, phase_q)
                mask3 = sl[:, None, None].astype(np.float32)
            else:
                sl2d = make_scanline_mask_2d(h, w, scanline_strength, scanline_period_px, phase_q, scanline_angle, scanline_thickness)
                mask3 = sl2d[:, :, None].astype(np.float32)
            if len(cache) > 256:
                cache.clear()
            cache[key] = mask3
        img = np.clip(img * mask3, 0.0, 1.0)
        perf_add("crt.scanlines", time.perf_counter() - t0)
    # Beam width modulation (bright areas spread vertically)
    if beam_spread_strength > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(beam_spread_strength) * 6)) * 2 + 1)
        vb = cv2.GaussianBlur(img, (1, k), sigmaX=0, sigmaY=max(0.5, float(beam_spread_strength) * 2.0), borderType=cv2.BORDER_REPLICATE)
        img = np.clip(np.maximum(img, vb), 0.0, 1.0)
        perf_add("crt.beam_spread", time.perf_counter() - t0)
    if vignette_mask is not None:
        t0 = time.perf_counter()
        img = np.clip(img * vignette_mask[:, :, None], 0.0, 1.0)
        perf_add("crt.vignette", time.perf_counter() - t0)
    if flicker_strength > 0.0 and flicker_hz > 0.0:
        t0 = time.perf_counter()
        factor = 1.0 + 0.25 * float(flicker_strength) * np.sin(2.0 * np.pi * float(flicker_hz) * float(time_sec))
        img = np.clip(img * factor, 0.0, 1.0)
        perf_add("crt.flicker", time.perf_counter() - t0)
    if noise_strength > 0.0:
        t0 = time.perf_counter()
        if grain_size and grain_size > 1:
            gh = max(1, h // int(grain_size))
            gw = max(1, w // int(grain_size))
            small_noise = np.empty((gh, gw), dtype=np.float32)
            cv2.randn(small_noise, 0.0, 1.0)
            noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            noise = np.empty((h, w), dtype=np.float32)
            cv2.randn(noise, 0.0, 1.0)
        noise = noise * (noise_strength / 255.0)
        img = np.clip(img + noise[:, :, None], 0.0, 1.0)
        perf_add("crt.noise", time.perf_counter() - t0)
    if warp_strength != 0.0:
        t0 = time.perf_counter()
        img = apply_barrel_warp(img, warp_strength)
        perf_add("crt.warp", time.perf_counter() - t0)
    # Screen (global) vertical jitter
    if screen_jitter_amp_px and abs(int(screen_jitter_amp_px)) > 0:
        t0 = time.perf_counter()
        amp = int(abs(int(screen_jitter_amp_px)))
        jumps_per_sec = float(max(0.1, screen_jitter_speed_hz))
        seg = int(np.floor(float(time_sec) * jumps_per_sec))
        seed = (seg * 1664525 + (w << 4) + (h << 1) + 1013904223) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        dy = int(rng.integers(-amp, amp + 1))
        img = np.roll(img, dy, axis=0)
        perf_add("crt.screen_jitter", time.perf_counter() - t0)
    # Line jitter (small per-row offsets animated over time)
    if jitter_amp_px and abs(int(jitter_amp_px)) > 0:
        t0 = time.perf_counter()
        amp = float(jitter_amp_px)
        speed = float(max(0.0, jitter_speed_hz))
        y = np.arange(h, dtype=np.float32)
        offs_row = amp * np.sin(2.0 * np.pi * (y / max(1.0, float(h)) + speed * float(time_sec)))
        x = np.arange(w, dtype=np.int32)[None, :]
        xi = (x + np.rint(offs_row)[:, None].astype(np.int32)) % w
        idx = np.broadcast_to(xi[:, :, None], img.shape)
        img = np.take_along_axis(img, idx, axis=1)
        perf_add("crt.jitter", time.perf_counter() - t0)
    # RF interference (diagonal sinusoidal modulation)
    if rfi_strength > 0.0 and rfi_freq > 0.0:
        t0 = time.perf_counter()
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.deg2rad(float(rfi_angle_deg))
        axis = (xx * np.cos(theta) + yy * np.sin(theta)) / max(1.0, float(w))
        phase = 2.0 * np.pi * (float(rfi_freq) * axis + float(rfi_speed_hz) * float(time_sec))
        mod = 1.0 + float(rfi_strength) * 0.08 * np.sin(phase)
        img = np.clip(img * mod[:, :, None], 0.0, 1.0)
        perf_add("crt.rf", time.perf_counter() - t0)
    # Waves
    if waves_amp_px and abs(float(waves_amp_px)) > 0.0 and waves_freq > 0.0:
        t0 = time.perf_counter()
        A = float(waves_amp_px)
        f = float(waves_freq)
        w2 = img.shape[1]
        y = np.arange(h, dtype=np.float32)
        phase_t = 2.0 * np.pi * float(max(0.0, waves_speed_hz)) * float(time_sec)
        offs_row = A * np.sin(2.0 * np.pi * f * (y / max(1.0, float(h))) + phase_t)
        x = np.arange(w2, dtype=np.int32)[None, :]
        xi = (x + np.rint(offs_row)[:, None].astype(np.int32)) % w2
        idx = np.broadcast_to(xi[:, :, None], img.shape)
        img = np.take_along_axis(img, idx, axis=1)
        perf_add("crt.waves", time.perf_counter() - t0)
    # Strobe
    if strobe_hz and float(strobe_hz) > 0.0:
        t0 = time.perf_counter()
        duty = float(np.clip(strobe_duty, 0.0, 1.0))
        phase = (float(time_sec) * float(strobe_hz)) % 1.0
        on = 1.0 if phase < duty else 0.0
        img = np.clip(img * on, 0.0, 1.0)
        perf_add("crt.strobe", time.perf_counter() - t0)
    # Anaglyph 3D
    if anaglyph_offset_px and abs(int(anaglyph_offset_px)) > 0:
        t0 = time.perf_counter()
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
        left = shift_channel(gray, int(anaglyph_offset_px), 0)
        right = shift_channel(gray, -int(anaglyph_offset_px), 0)
        mode = (anaglyph_mode or "red_cyan").lower()
        if mode == "red_cyan":
            r = left
            g = right
            b = right
        else:
            r = left
            g = right
            b = right
        img = np.stack([r, g, b], axis=2)
        perf_add("crt.anaglyph", time.perf_counter() - t0)
    if text_overlay_rgba is not None and text_overlay_after:
        t0 = time.perf_counter()
        ov = text_overlay_rgba
        if ov.dtype != np.uint8:
            ov = np.clip(ov, 0, 255).astype(np.uint8)
        if ov.shape[0] != h or ov.shape[1] != w:
            ov = np.asarray(Image.fromarray(ov, mode="RGBA").resize((w, h), Image.BILINEAR))
        alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = ov[:, :, :3].astype(np.float32) / 255.0
        img = np.clip(img * (1.0 - alpha) + rgb * alpha, 0.0, 1.0)
        perf_add("crt.text_after", time.perf_counter() - t0)
    if glitch_amp_px > 0 and glitch_height_frac > 0.0:
        t0 = time.perf_counter()
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
        perf_add("crt.glitch", time.perf_counter() - t0)
    if state_prev is not None and persistence > 0.0:
        t0 = time.perf_counter()
        if state_prev.shape != img.shape:
            prev_arr = cv2.resize(state_prev, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            prev_arr = state_prev
        # Ensure matching types for addWeighted
        if prev_arr.dtype != np.float32:
            prev_arr = prev_arr.astype(np.float32, copy=False)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        img = cv2.addWeighted(prev_arr, float(persistence), img, float(1.0 - persistence), 0.0)
        perf_add("crt.persistence", time.perf_counter() - t0)
    t0 = time.perf_counter()
    out = cv2.convertScaleAbs(img, alpha=255.0, beta=0)
    perf_add("crt.to_uint8", time.perf_counter() - t0)
    perf_add("crt.total", time.perf_counter() - t_all)
    return out, img


def apply_static_effects(
    frame: np.ndarray,
    scanline_strength: float,
    triad_mask: Optional[np.ndarray],
    triad_gamma: float,
    triad_preserve_luma: bool,
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    bloom_threshold: float,
    noise_strength: float,
    vignette_mask: Optional[np.ndarray],
    scanline_period_px: float,
    scanline_phase_px: float,
    fast_bloom: bool,
    pixel_size: int,
    glitch_amp_px: int,
    glitch_height_frac: float,
    time_sec: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    flicker_strength: float = 0.0,
    flicker_hz: float = 0.0,
    grain_size: int = 1,
    scanline_angle: float = 0.0,
    scanline_thickness: float = 1.0,
    warp_strength: float = 0.0,
    beam_spread_strength: float = 0.0,
    hbleed_sigma: float = 0.0,
    hbleed_strength: float = 0.0,
    vbleed_sigma: float = 0.0,
    vbleed_strength: float = 0.0,
    jitter_amp_px: int = 0,
    jitter_speed_hz: float = 0.0,
    screen_jitter_amp_px: int = 0,
    screen_jitter_speed_hz: float = 0.0,
    rfi_strength: float = 0.0,
    rfi_freq: float = 0.0,
    rfi_speed_hz: float = 0.0,
    rfi_angle_deg: float = 45.0,
    waves_amp_px: int = 0,
    waves_freq: float = 0.0,
    waves_speed_hz: float = 0.0,
    strobe_hz: float = 0.0,
    strobe_duty: float = 0.5,
    anaglyph_offset_px: int = 0,
    anaglyph_mode: str = "red_cyan",
    text_overlay_rgba: Optional[np.ndarray] = None,
    text_overlay_after: bool = True,
) -> np.ndarray:
    t_all = time.perf_counter()
    # Skip-unchanged caches (module-level or attached via function attributes)
    if not hasattr(apply_static_effects, "_waves_cache"):
        apply_static_effects._waves_cache = {}
    h, w = frame.shape[0], frame.shape[1]
    t0 = time.perf_counter()
    img = frame.astype(np.float32) / 255.0
    perf_add("fx.astype", time.perf_counter() - t0)
    if aberration_px != 0:
        t0 = time.perf_counter()
        r = shift_channel(img[:, :, 0], aberration_px, 0)
        g = img[:, :, 1]
        b = shift_channel(img[:, :, 2], -aberration_px, 0)
        img = np.stack([r, g, b], axis=2)
        perf_add("fx.aberration", time.perf_counter() - t0)
    if pixel_size > 1:
        t0 = time.perf_counter()
        sw = max(1, w // int(pixel_size))
        sh = max(1, h // int(pixel_size))
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        perf_add("fx.pixelate", time.perf_counter() - t0)
    t0 = time.perf_counter()
    img = apply_color_adjustments(img, brightness, contrast, gamma, saturation, temperature)
    perf_add("fx.color", time.perf_counter() - t0)
    # Text overlay before effects
    if text_overlay_rgba is not None and not text_overlay_after:
        t0 = time.perf_counter()
        ov = text_overlay_rgba
        if ov.dtype != np.uint8:
            ov = np.clip(ov, 0, 255).astype(np.uint8)
        if ov.shape[0] != h or ov.shape[1] != w:
            ov = np.asarray(Image.fromarray(ov, mode="RGBA").resize((w, h), Image.BILINEAR))
        alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = ov[:, :, :3].astype(np.float32) / 255.0
        img = np.clip(img * (1.0 - alpha) + rgb * alpha, 0.0, 1.0)
        perf_add("fx.text_before", time.perf_counter() - t0)
    if bloom_strength > 0.0 and (bloom_sigma > 0.0 or fast_bloom):
        t0 = time.perf_counter()
        src = img
        if bloom_threshold > 0.0:
            thr = float(min(0.99, max(0.0, bloom_threshold)))
            src = np.clip((img - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
        if fast_bloom:
            ds = cv2.resize(src, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
            blurf = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            k = max(1, int(round(bloom_sigma * 3)) * 2 + 1)
            blurf = cv2.GaussianBlur(src, (k, k), sigmaX=bloom_sigma, sigmaY=bloom_sigma, borderType=cv2.BORDER_REPLICATE)
        img = np.clip(img + bloom_strength * blurf, 0.0, 1.0)
        perf_add("fx.bloom", time.perf_counter() - t0)
    if triad_mask is not None:
        t0 = time.perf_counter()
        img = _apply_triad_mask(img, triad_mask, triad_gamma, triad_preserve_luma)
        perf_add("fx.triad", time.perf_counter() - t0)
    if hbleed_strength > 0.0 and hbleed_sigma > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(hbleed_sigma) * 3)) * 2 + 1)
        hb = cv2.GaussianBlur(img, (k, 1), sigmaX=float(hbleed_sigma), sigmaY=0, borderType=cv2.BORDER_REPLICATE)
        img = np.clip((1.0 - float(hbleed_strength)) * img + float(hbleed_strength) * hb, 0.0, 1.0)
        perf_add("fx.hbleed", time.perf_counter() - t0)
    if vbleed_strength > 0.0 and vbleed_sigma > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(vbleed_sigma) * 3)) * 2 + 1)
        vb = cv2.GaussianBlur(img, (1, k), sigmaX=0, sigmaY=float(vbleed_sigma), borderType=cv2.BORDER_REPLICATE)
        img = np.clip((1.0 - float(vbleed_strength)) * img + float(vbleed_strength) * vb, 0.0, 1.0)
        perf_add("fx.vbleed", time.perf_counter() - t0)
    if scanline_strength > 0.0:
        t0 = time.perf_counter()
        if scanline_angle == 0.0 and scanline_thickness == 1.0:
            sl = make_scanline_mask_dynamic(h, scanline_strength, scanline_period_px, scanline_phase_px)
            img = np.clip(img * sl[:, None, None], 0.0, 1.0)
        else:
            sl2d = make_scanline_mask_2d(h, w, scanline_strength, scanline_period_px, scanline_phase_px, scanline_angle, scanline_thickness)
            img = np.clip(img * sl2d[:, :, None], 0.0, 1.0)
        perf_add("fx.scanlines", time.perf_counter() - t0)
    if beam_spread_strength > 0.0:
        t0 = time.perf_counter()
        k = max(1, int(round(float(beam_spread_strength) * 6)) * 2 + 1)
        vb = cv2.GaussianBlur(img, (1, k), sigmaX=0, sigmaY=max(0.5, float(beam_spread_strength) * 2.0), borderType=cv2.BORDER_REPLICATE)
        img = np.clip(np.maximum(img, vb), 0.0, 1.0)
        perf_add("fx.beam_spread", time.perf_counter() - t0)
    if vignette_mask is not None:
        t0 = time.perf_counter()
        img = np.clip(img * vignette_mask[:, :, None], 0.0, 1.0)
        perf_add("fx.vignette", time.perf_counter() - t0)
    if flicker_strength > 0.0 and flicker_hz > 0.0:
        t0 = time.perf_counter()
        factor = 1.0 + 0.25 * float(flicker_strength) * np.sin(2.0 * np.pi * float(flicker_hz) * float(time_sec))
        img = np.clip(img * factor, 0.0, 1.0)
        perf_add("fx.flicker", time.perf_counter() - t0)
    if noise_strength > 0.0:
        t0 = time.perf_counter()
        if grain_size and grain_size > 1:
            gh = max(1, h // int(grain_size))
            gw = max(1, w // int(grain_size))
            small_noise = np.empty((gh, gw), dtype=np.float32)
            cv2.randn(small_noise, 0.0, 1.0)
            noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            noise = np.empty((h, w), dtype=np.float32)
            cv2.randn(noise, 0.0, 1.0)
        noise = noise * (noise_strength / 255.0)
        img = np.clip(img + noise[:, :, None], 0.0, 1.0)
        perf_add("fx.noise", time.perf_counter() - t0)
    if warp_strength != 0.0:
        t0 = time.perf_counter()
        img = apply_barrel_warp(img, warp_strength)
        perf_add("fx.warp", time.perf_counter() - t0)
    if screen_jitter_amp_px and abs(int(screen_jitter_amp_px)) > 0:
        t0 = time.perf_counter()
        amp = int(abs(int(screen_jitter_amp_px)))
        jumps_per_sec = float(max(0.1, screen_jitter_speed_hz))
        seg = int(np.floor(float(time_sec) * jumps_per_sec))
        seed = (seg * 1664525 + (w << 4) + (h << 1) + 1013904223) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        dy = int(rng.integers(-amp, amp + 1))
        img = np.roll(img, dy, axis=0)
        perf_add("fx.screen_jitter", time.perf_counter() - t0)
    if jitter_amp_px and abs(int(jitter_amp_px)) > 0:
        t0 = time.perf_counter()
        amp = float(jitter_amp_px)
        speed = float(max(0.0, jitter_speed_hz))
        y = np.arange(h, dtype=np.float32)
        offs_row = amp * np.sin(2.0 * np.pi * (y / max(1.0, float(h)) + speed * float(time_sec)))
        x = np.arange(w, dtype=np.int32)[None, :]
        xi = (x + np.rint(offs_row)[:, None].astype(np.int32)) % w
        idx = np.broadcast_to(xi[:, :, None], img.shape)
        img = np.take_along_axis(img, idx, axis=1)
        perf_add("fx.jitter", time.perf_counter() - t0)
    if rfi_strength > 0.0 and rfi_freq > 0.0:
        t0 = time.perf_counter()
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.deg2rad(float(rfi_angle_deg))
        axis = (xx * np.cos(theta) + yy * np.sin(theta)) / max(1.0, float(w))
        phase = 2.0 * np.pi * (float(rfi_freq) * axis + float(rfi_speed_hz) * float(time_sec))
        mod = 1.0 + float(rfi_strength) * 0.08 * np.sin(phase)
        img = np.clip(img * mod[:, :, None], 0.0, 1.0)
        perf_add("fx.rf", time.perf_counter() - t0)
    # Waves
    if waves_amp_px and abs(float(waves_amp_px)) > 0.0 and waves_freq > 0.0:
        t0 = time.perf_counter()
        A = float(waves_amp_px)
        f = float(waves_freq)
        w2 = img.shape[1]
        y = np.arange(h, dtype=np.float32)
        phase_t = 2.0 * np.pi * float(max(0.0, waves_speed_hz)) * float(time_sec)
        key = (h, w2, round(A, 3), round(f, 4), int(np.rint(phase_t)))
        cache = apply_static_effects._waves_cache
        xi = None
        if key in cache:
            xi = cache[key]
        else:
            offs_row = A * np.sin(2.0 * np.pi * f * (y / max(1.0, float(h))) + phase_t)
            x = np.arange(w2, dtype=np.int32)[None, :]
            xi = (x + np.rint(offs_row)[:, None].astype(np.int32)) % w2
            # cap cache size
            if len(cache) > 64:
                cache.clear()
            cache[key] = xi
        idx = np.broadcast_to(xi[:, :, None], img.shape)
        img = np.take_along_axis(img, idx, axis=1)
        perf_add("fx.waves", time.perf_counter() - t0)
    # Strobe
    if strobe_hz and float(strobe_hz) > 0.0:
        t0 = time.perf_counter()
        duty = float(np.clip(strobe_duty, 0.0, 1.0))
        phase = (float(time_sec) * float(strobe_hz)) % 1.0
        on = 1.0 if phase < duty else 0.0
        img = np.clip(img * on, 0.0, 1.0)
        perf_add("fx.strobe", time.perf_counter() - t0)
    # Anaglyph 3D
    if anaglyph_offset_px and abs(int(anaglyph_offset_px)) > 0:
        t0 = time.perf_counter()
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
        left = shift_channel(gray, int(anaglyph_offset_px), 0)
        right = shift_channel(gray, -int(anaglyph_offset_px), 0)
        mode = (anaglyph_mode or "red_cyan").lower()
        if mode == "red_cyan":
            r = left
            g = right
            b = right
        else:
            r = left
            g = right
            b = right
        img = np.stack([r, g, b], axis=2)
        perf_add("fx.anaglyph", time.perf_counter() - t0)
    # Text overlay after effects
    if text_overlay_rgba is not None and text_overlay_after:
        t0 = time.perf_counter()
        ov = text_overlay_rgba
        if ov.dtype != np.uint8:
            ov = np.clip(ov, 0, 255).astype(np.uint8)
        if ov.shape[0] != h or ov.shape[1] != w:
            ov = np.asarray(Image.fromarray(ov, mode="RGBA").resize((w, h), Image.BILINEAR))
        alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = ov[:, :, :3].astype(np.float32) / 255.0
        img = np.clip(img * (1.0 - alpha) + rgb * alpha, 0.0, 1.0)
        perf_add("fx.text_after", time.perf_counter() - t0)
    if glitch_amp_px > 0 and glitch_height_frac > 0.0:
        t0 = time.perf_counter()
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
        perf_add("fx.glitch", time.perf_counter() - t0)
    perf_add("fx.total", time.perf_counter() - t_all)
    return img


def process_video(
    input_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scanline_strength: float,
    triad_strength: float,
    triad_gamma: float,
    triad_preserve_luma: bool,
    triad_softness: float,
    aberration_px: int,
    bloom_sigma: float,
    bloom_strength: float,
    noise_strength: float,
    vignette_strength: float,
    persistence: float,
    fps: Optional[int],
    crf: int,
    target_bitrate_kbps: int,
    scanline_speed_px_s: float,
    scanline_period_px: float,
    fast_bloom: bool,
    pixel_size: int,
    gpu: bool,
    nvenc_preset: str,
    glitch_amp_px: int = 0,
    glitch_height_frac: float = 0.0,
    encoder_preference: str = "auto",
    decoder_preference: str = "auto",
    bloom_threshold: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    flicker_strength: float = 0.0,
    flicker_hz: float = 0.0,
    grain_size: int = 1,
    scanline_angle: float = 0.0,
    scanline_thickness: float = 1.0,
    warp_strength: float = 0.0,
    beam_spread_strength: float = 0.0,
    hbleed_sigma: float = 0.0,
    hbleed_strength: float = 0.0,
    vbleed_sigma: float = 0.0,
    vbleed_strength: float = 0.0,
    text: str = "",
    text_font: str = "",
    text_size: int = 36,
    text_color: str = "#FFFFFF",
    text_pos: Tuple[int, int] = (32, 32),
    text_after: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
    offload_filters: bool = False,
) -> bool:
    clip = VideoFileClip(str(input_path))
    fps_out = int(fps) if fps and fps > 0 else int(clip.fps or 24)
    if width and height:
        out_w, out_h = int(width), int(height)
    else:
        out_w, out_h = clip.size
    triad_mask = make_triad_mask(out_h, out_w, triad_strength, triad_softness) if triad_strength > 0.0 else None
    vignette_mask = make_vignette(out_h, out_w, vignette_strength) if vignette_strength > 0.0 else None
    try:
        import math, tempfile
        total_frames = max(1, int(math.ceil((clip.duration or 0) * fps_out)))
        t_pipeline_start = time.perf_counter()
        audio_path: Optional[str] = None
        if clip.audio is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".aac")
            tmp_path = tmp.name
            tmp.close()
            try:
                # quicker audio write
                clip.audio.write_audiofile(tmp_path, fps=44100, nbytes=2, codec="aac", bitrate="128k", verbose=False, logger=None)
                audio_path = tmp_path
            except Exception:
                audio_path = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Choose codec based on preference and availability
        pref = (encoder_preference or "auto").strip().lower()
        requested_gpu = bool(gpu)
        codec = "libx264"
        if pref == "nvidia":
            codec = "h264_nvenc" if can_use_nvenc() else "libx264"
        elif pref == "amd":
            codec = "h264_amf" if can_use_amf() else "libx264"
        elif pref == "cpu":
            codec = "libx264"
        else:  # auto
            if requested_gpu and can_use_nvenc():
                codec = "h264_nvenc"
            elif requested_gpu and can_use_amf():
                codec = "h264_amf"
            else:
                codec = "libx264"
        used_gpu = codec in ("h264_nvenc", "h264_amf")
        kbps = int(max(0, target_bitrate_kbps or 0))
        if codec == "h264_nvenc":
            nv_preset = normalize_nvenc_preset(nvenc_preset)
            if kbps > 0:
                ffparams = [
                    "-b:v",
                    f"{kbps}k",
                    "-maxrate",
                    f"{kbps}k",
                    "-bufsize",
                    f"{kbps * 2}k",
                    "-rc",
                    "vbr",
                    "-preset",
                    nv_preset,
                    "-pix_fmt",
                    "yuv420p",
                ]
            else:
                ffparams = ["-cq", str(crf), "-preset", nv_preset, "-pix_fmt", "yuv420p"]
        elif codec == "h264_amf":
            if kbps > 0:
                ffparams = [
                    "-b:v",
                    f"{kbps}k",
                    "-maxrate",
                    f"{kbps}k",
                    "-bufsize",
                    f"{kbps * 2}k",
                    "-pix_fmt",
                    "yuv420p",
                ]
            else:
                ffparams = ["-pix_fmt", "yuv420p"]
        else:
            if kbps > 0:
                ffparams = [
                    "-b:v",
                    f"{kbps}k",
                    "-maxrate",
                    f"{kbps}k",
                    "-bufsize",
                    f"{kbps * 2}k",
                    "-pix_fmt",
                    "yuv420p",
                ]
            else:
                ffparams = ["-crf", str(crf), "-pix_fmt", "yuv420p"]
        # Optionally offload H/V bleed to FFmpeg filtergraph
        vf_filters: Optional[str] = None
        if offload_filters and ((hbleed_strength > 0.0 and hbleed_sigma > 0.0) or (vbleed_strength > 0.0 and vbleed_sigma > 0.0)):
            parts = ["split[orig][tmp]"]
            last = "orig"
            if hbleed_strength > 0.0 and hbleed_sigma > 0.0:
                parts.append(f"[{last}]gblur=sigmaX={max(0.1, hbleed_sigma)}:sigmaY=0[hb]")
                parts.append(f"[tmp][hb]blend=all_mode=normal:all_opacity={min(1.0, max(0.0, hbleed_strength))}[mixh]")
                last = "mixh"
            if vbleed_strength > 0.0 and vbleed_sigma > 0.0:
                parts.append(f"[{last}]gblur=sigmaX=0:sigmaY={max(0.1, vbleed_sigma)}[vb]")
                parts.append(f"[tmp][vb]blend=all_mode=normal:all_opacity={min(1.0, max(0.0, vbleed_strength))}[mixv]")
                last = "mixv"
            parts.append(f"[{last}]copy[out]")
            vf_filters = ",".join(parts)
        writer_kwargs = dict(
            filename=str(output_path),
            size=(out_w, out_h),
            fps=fps_out,
            codec=codec,
            audiofile=audio_path,
            threads=os.cpu_count() or 4,
            ffmpeg_params=ffparams,
        )
        if vf_filters is not None:
            writer_kwargs["ffmpeg_params"] = ffparams + ["-filter_complex", vf_filters, "-map", "[out]"]
        if not used_gpu:
            writer_kwargs["preset"] = "medium"
        writer = FFMPEG_VideoWriter(**writer_kwargs)
        # Use CPU-1 worker threads; numpy/OpenCV release the GIL during heavy ops
        max_workers = max(1, (os.cpu_count() or 4) - 1)
        queue_cap = max_workers * 6
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        next_write = 0
        prev_state = None
        i = 0
        # Optional hardware-accelerated decode via ffmpeg raw reader
        hw_reader = None
        def _open_hw_reader() -> Optional[object]:
            accel = _map_decoder_to_hwaccel(decoder_preference)
            if accel is None:
                return None
            try:
                return FFmpegRawReader(str(input_path), out_w, out_h, fps_out, accel)
            except Exception:
                return None
        hw_reader = _open_hw_reader()
        if hw_reader is not None:
            frame_iter = perf_timed_iter(hw_reader.iter_frames(), "io.hw_decode")
        else:
            frame_iter = perf_timed_iter(clip.iter_frames(fps=fps_out, dtype="uint8"), "io.decode")
        # Precompute static text overlay once for the whole render
        overlay_rgba = _make_text_overlay_rgba_qt(out_w, out_h, text, text_font, text_size, text_color, text_pos) if text else None
        for frame in frame_iter:
            t_submit = time.perf_counter()
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                im = Image.fromarray(frame)
                frame = np.asarray(im.resize((out_w, out_h), Image.BILINEAR))
            perf_add("io.resize_in", time.perf_counter() - t_submit)
            phase = (i / float(fps_out)) * scanline_speed_px_s
            def submit_job(idx: int, f: np.ndarray, ph: float):
                return executor.submit(
                    apply_static_effects,
                    f,
                    scanline_strength,
                    triad_mask,
                    float(triad_gamma),
                    bool(triad_preserve_luma),
                    aberration_px,
                    bloom_sigma,
                    bloom_strength,
                    float(bloom_threshold),
                    noise_strength,
                    vignette_mask,
                    scanline_period_px,
                    ph,
                    fast_bloom,
                    pixel_size,
                    int(glitch_amp_px),
                    float(glitch_height_frac),
                    time_sec=(idx / float(fps_out)),
                    brightness=float(brightness),
                    contrast=float(contrast),
                    gamma=float(gamma),
                    saturation=float(saturation),
                    temperature=float(temperature),
                    flicker_strength=float(flicker_strength),
                    flicker_hz=float(flicker_hz),
                    grain_size=int(grain_size),
                    scanline_angle=float(scanline_angle),
                    scanline_thickness=float(scanline_thickness),
                    warp_strength=float(warp_strength),
                    beam_spread_strength=float(beam_spread_strength),
                    hbleed_sigma=0.0 if vf_filters is not None else float(hbleed_sigma),
                    hbleed_strength=0.0 if vf_filters is not None else float(hbleed_strength),
                    vbleed_sigma=0.0 if vf_filters is not None else float(vbleed_sigma),
                    vbleed_strength=0.0 if vf_filters is not None else float(vbleed_strength),
                    text_overlay_rgba=overlay_rgba,
                    text_overlay_after=bool(text_after),
                )
            futures[i] = submit_job(i, frame, phase)
            i += 1
            while len(futures) >= queue_cap or next_write in futures:
                if next_write in futures:
                    t0 = time.perf_counter()
                    static_img = futures.pop(next_write).result()
                    perf_add("fx.future_wait", time.perf_counter() - t0)
                    if prev_state is not None and persistence > 0.0:
                        t0 = time.perf_counter()
                        if prev_state.shape != static_img.shape:
                            prev_im = Image.fromarray(np.clip(prev_state * 255.0, 0, 255).astype(np.uint8))
                            prev_im = prev_im.resize((out_w, out_h), Image.BILINEAR)
                            prev_state = np.asarray(prev_im).astype(np.float32) / 255.0
                        blended = np.clip(persistence * prev_state + (1.0 - persistence) * static_img, 0.0, 1.0)
                        perf_add("fx.persistence_blend", time.perf_counter() - t0)
                    else:
                        blended = static_img
                    prev_state = blended
                    t0 = time.perf_counter()
                    out_frame = cv2.convertScaleAbs(blended, alpha=255.0, beta=0)
                    perf_add("io.to_uint8_out", time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    writer.write_frame(out_frame)
                    perf_add("io.encode", time.perf_counter() - t0)
                    next_write += 1
                    if progress_cb is not None:
                        progress_cb(min(1.0, next_write / float(total_frames)))
                else:
                    break
        while next_write in futures:
            t0 = time.perf_counter()
            static_img = futures.pop(next_write).result()
            perf_add("fx.future_wait", time.perf_counter() - t0)
            if prev_state is not None and persistence > 0.0:
                t0 = time.perf_counter()
                if prev_state.shape != static_img.shape:
                    prev_im = Image.fromarray(np.clip(prev_state * 255.0, 0, 255).astype(np.uint8))
                    prev_im = prev_im.resize((out_w, out_h), Image.BILINEAR)
                    prev_state = np.asarray(prev_im).astype(np.float32) / 255.0
                blended = np.clip(persistence * prev_state + (1.0 - persistence) * static_img, 0.0, 1.0)
                perf_add("fx.persistence_blend", time.perf_counter() - t0)
            else:
                blended = static_img
            prev_state = blended
            t0 = time.perf_counter()
            out_frame = cv2.convertScaleAbs(blended, alpha=255.0, beta=0)
            perf_add("io.to_uint8_out", time.perf_counter() - t0)
            t0 = time.perf_counter()
            writer.write_frame(out_frame)
            perf_add("io.encode", time.perf_counter() - t0)
            next_write += 1
            if progress_cb is not None:
                progress_cb(min(1.0, next_write / float(total_frames)))
        writer.close()
        total_seconds = time.perf_counter() - t_pipeline_start
        perf_report(total_frames=total_frames, total_seconds=total_seconds)
        if hw_reader is not None:
            try:
                hw_reader.close()
            except Exception:
                pass
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
    p.add_argument("--triad-gamma", type=float, default=2.2)
    p.add_argument("--triad-preserve-luma", action="store_true")
    p.add_argument("--triad-softness", type=float, default=0.5)
    p.add_argument("--aberration-px", type=int, default=1)
    p.add_argument("--bloom-sigma", type=float, default=1.2)
    p.add_argument("--bloom-strength", type=float, default=0.25)
    p.add_argument("--bloom-threshold", type=float, default=0.0)
    p.add_argument("--noise-strength", type=float, default=1.5)
    p.add_argument("--vignette-strength", type=float, default=0.25)
    p.add_argument("--persistence", type=float, default=0.2)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--bitrate", type=int, default=0)
    p.add_argument("--scanline-speed", type=float, default=30.0)
    p.add_argument("--scanline-period", type=float, default=2.0)
    p.add_argument("--fast-bloom", action="store_true")
    p.add_argument("--no-fast-bloom", dest="fast_bloom", action="store_false")
    p.set_defaults(fast_bloom=True)
    p.add_argument("--pixel-size", type=int, default=2)
    # Advanced
    p.add_argument("--brightness", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--saturation", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--flicker-strength", type=float, default=0.0)
    p.add_argument("--flicker-hz", type=float, default=0.0)
    p.add_argument("--grain-size", type=int, default=1)
    p.add_argument("--scanline-angle", type=float, default=0.0)
    p.add_argument("--scanline-thickness", type=float, default=1.0)
    p.add_argument("--warp-strength", type=float, default=0.0)
    p.add_argument("--beam-spread", type=float, default=0.0)
    p.add_argument("--hbleed-sigma", type=float, default=0.0)
    p.add_argument("--hbleed-strength", type=float, default=0.0)
    p.add_argument("--vbleed-sigma", type=float, default=0.0)
    p.add_argument("--vbleed-strength", type=float, default=0.0)
    # Text overlay
    p.add_argument("--text", type=str, default="")
    p.add_argument("--text-font", type=str, default="")
    p.add_argument("--text-size", type=int, default=36)
    p.add_argument("--text-color", type=str, default="#FFFFFF")
    p.add_argument("--text-x", type=int, default=32)
    p.add_argument("--text-y", type=int, default=32)
    p.add_argument("--text-after", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--nvenc-preset", type=str, default="p4")
    p.add_argument("--encoder", type=str, default="auto", choices=["auto", "nvidia", "amd", "cpu"])
    p.add_argument("--decoder", type=str, default="auto", choices=["auto", "nvidia", "amd", "intel", "cpu"])
    p.add_argument("--glitch-amp", type=int, default=0)
    p.add_argument("--glitch-height", type=float, default=0.0)
    p.add_argument("--offload-filters", action="store_true")
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.gui or not a.input:
        launch_gui()
        return
    t_main = time.perf_counter()
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
        triad_gamma=float(max(0.1, a.triad_gamma)),
        triad_preserve_luma=bool(a.triad_preserve_luma),
        triad_softness=float(max(0.0, a.triad_softness)),
        aberration_px=int(max(-8, min(8, a.aberration_px))),
        bloom_sigma=max(0.0, a.bloom_sigma),
        bloom_strength=max(0.0, a.bloom_strength),
        noise_strength=max(0.0, a.noise_strength),
        vignette_strength=float(max(0.0, min(1.0, a.vignette_strength))),
        persistence=float(max(0.0, min(0.95, a.persistence))),
        fps=a.fps if a.fps > 0 else None,
        crf=int(max(12, min(28, a.crf))),
        target_bitrate_kbps=int(max(0, getattr(a, "bitrate", 0))),
        scanline_speed_px_s=float(a.scanline_speed),
        scanline_period_px=max(1.0, float(a.scanline_period)),
        fast_bloom=bool(a.fast_bloom),
        pixel_size=max(1, int(a.pixel_size)),
        gpu=bool(a.gpu),
        nvenc_preset=str(a.nvenc_preset),
        glitch_amp_px=max(0, int(a.glitch_amp)),
        glitch_height_frac=float(max(0.0, min(1.0, a.glitch_height))),
        encoder_preference=str(a.encoder),
        decoder_preference=str(a.decoder),
        bloom_threshold=float(max(0.0, min(1.0, a.bloom_threshold))),
        brightness=float(a.brightness),
        contrast=float(a.contrast),
        gamma=float(max(1e-3, a.gamma)),
        saturation=float(max(0.0, a.saturation)),
        temperature=float(max(-1.0, min(1.0, a.temperature))),
        flicker_strength=float(max(0.0, min(1.0, a.flicker_strength))),
        flicker_hz=float(max(0.0, a.flicker_hz)),
        grain_size=max(1, int(a.grain_size)),
        scanline_angle=float(a.scanline_angle),
        scanline_thickness=float(max(0.1, a.scanline_thickness)),
        warp_strength=float(max(-1.0, min(1.0, a.warp_strength))),
        beam_spread_strength=float(max(0.0, a.beam_spread)),
        hbleed_sigma=float(max(0.0, a.hbleed_sigma)),
        hbleed_strength=float(max(0.0, min(1.0, a.hbleed_strength))),
        vbleed_sigma=float(max(0.0, a.vbleed_sigma)),
        vbleed_strength=float(max(0.0, min(1.0, a.vbleed_strength))),
        text=str(a.text),
        text_font=str(a.text_font),
        text_size=int(a.text_size),
        text_color=str(a.text_color),
        text_pos=(int(a.text_x), int(a.text_y)),
        text_after=bool(a.text_after),
        offload_filters=bool(a.offload_filters),
    )
    print("Hardware encoder used" if used_gpu else "CPU x264 used")
    print(f"elapsed {time.perf_counter() - t_main:.3f}s")


def launch_gui() -> None:
    from PySide6 import QtCore, QtGui, QtWidgets

    class HWPreviewReader:
        def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
            self.path = str(path)
            self.width = int(width)
            self.height = int(height)
            self.fps = int(max(1, fps))
            self.cap = None
            self.running = False

        def start(self) -> None:
            self.stop()
            self.running = True
            # Try preferred hardware backends
            cap = None
            try:
                cap = cv2.cudacodec.createVideoReader(self.path)  # type: ignore[attr-defined]
            except Exception:
                cap = None
            if cap is not None:
                self.cap = ("cuda", cap)
                return
            # Try D3D11/DXVA2 via FFmpeg filters using VideoCapture fallback
            cap2 = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
            if cap2 is not None and cap2.isOpened():
                # Set desired fps/size as hints; actual decode will be original, we will resize
                self.cap = ("ffmpeg", cap2)
                return
            # Fallback to default VideoCapture
            cap3 = cv2.VideoCapture(self.path)
            if cap3 is not None and cap3.isOpened():
                self.cap = ("default", cap3)
            else:
                self.cap = None

        def stop(self) -> None:
            if self.cap is not None:
                kind, handle = self.cap
                try:
                    handle.release()
                except Exception:
                    pass
            self.cap = None
            self.running = False

        def read_next(self) -> Optional[np.ndarray]:
            if self.cap is None:
                return None
            kind, handle = self.cap
            try:
                if kind == "cuda":
                    ok, gpu_mat = handle.nextFrame()  # returns GpuMat
                    if not ok:
                        return None
                    frame = gpu_mat.download()
                else:
                    ok, frame = handle.read()
                    if not ok:
                        return None
                if frame is None:
                    return None
                # Convert BGR->RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Do not scale here; preview scales dynamically per current widget size
                return frame.astype(np.uint8)
            except Exception:
                return None

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
            self.gpu = QtWidgets.QCheckBox("Use hardware encoder")
            self.gpu.setChecked(parent.gpu_cb.isChecked())
            form = QtWidgets.QFormLayout()
            form.addRow("output path", path_row)
            form.addRow("width (0 keep)", self.width)
            form.addRow("height (0 keep)", self.height)
            form.addRow("fps (0 keep)", self.fps)
            form.addRow("hardware encode", self.gpu)
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
            self.video_label.setScaledContents(False)
            self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            self.video_label.setMinimumSize(1, 1)
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
            self.gpu_cb = QtWidgets.QCheckBox("Enable hardware encode")
            self.encoder_choice = QtWidgets.QComboBox()
            self.encoder_choice.addItems(["Auto", "NVIDIA", "AMD", "CPU"]) 
            self.encoder_choice.setCurrentIndex(0)
            self.fast_bloom_cb = QtWidgets.QCheckBox("Fast Bloom")
            self.fast_bloom_cb.setChecked(True)
            # Advanced FX controls
            self.brightness = QtWidgets.QDoubleSpinBox(); self.brightness.setRange(-1.0, 1.0); self.brightness.setSingleStep(0.01); self.brightness.setValue(0.0)
            self.contrast = QtWidgets.QDoubleSpinBox(); self.contrast.setRange(0.1, 3.0); self.contrast.setSingleStep(0.05); self.contrast.setValue(1.0)
            self.gamma = QtWidgets.QDoubleSpinBox(); self.gamma.setRange(0.1, 5.0); self.gamma.setSingleStep(0.05); self.gamma.setValue(1.0)
            self.saturation = QtWidgets.QDoubleSpinBox(); self.saturation.setRange(0.0, 3.0); self.saturation.setSingleStep(0.05); self.saturation.setValue(1.0)
            self.temperature = QtWidgets.QDoubleSpinBox(); self.temperature.setRange(-1.0, 1.0); self.temperature.setSingleStep(0.05); self.temperature.setValue(0.0)
            self.bloom_threshold = QtWidgets.QDoubleSpinBox(); self.bloom_threshold.setRange(0.0, 1.0); self.bloom_threshold.setSingleStep(0.01); self.bloom_threshold.setValue(0.0)
            self.flicker_strength = QtWidgets.QDoubleSpinBox(); self.flicker_strength.setRange(0.0, 1.0); self.flicker_strength.setSingleStep(0.01); self.flicker_strength.setValue(0.0)
            self.flicker_hz = QtWidgets.QDoubleSpinBox(); self.flicker_hz.setRange(0.0, 120.0); self.flicker_hz.setSingleStep(0.5); self.flicker_hz.setValue(0.0)
            self.grain_size = QtWidgets.QSpinBox(); self.grain_size.setRange(1, 64); self.grain_size.setValue(1)
            self.scanline_angle = QtWidgets.QDoubleSpinBox(); self.scanline_angle.setRange(-45.0, 45.0); self.scanline_angle.setSingleStep(0.5); self.scanline_angle.setValue(0.0)
            self.scanline_thickness = QtWidgets.QDoubleSpinBox(); self.scanline_thickness.setRange(0.1, 4.0); self.scanline_thickness.setSingleStep(0.1); self.scanline_thickness.setValue(1.0)
            self.warp_strength = QtWidgets.QDoubleSpinBox(); self.warp_strength.setRange(-1.0, 1.0); self.warp_strength.setSingleStep(0.05); self.warp_strength.setValue(0.0)
            # EFX controls
            self.waves_amp = QtWidgets.QSpinBox(); self.waves_amp.setRange(-256, 256); self.waves_amp.setValue(0)
            self.waves_freq = QtWidgets.QDoubleSpinBox(); self.waves_freq.setRange(0.0, 10.0); self.waves_freq.setSingleStep(0.05); self.waves_freq.setValue(0.0)
            self.waves_speed = QtWidgets.QDoubleSpinBox(); self.waves_speed.setRange(0.0, 240.0); self.waves_speed.setSingleStep(0.5); self.waves_speed.setValue(0.0)
            self.strobe_hz = QtWidgets.QDoubleSpinBox(); self.strobe_hz.setRange(0.0, 240.0); self.strobe_hz.setSingleStep(0.5); self.strobe_hz.setValue(0.0)
            self.strobe_duty = QtWidgets.QDoubleSpinBox(); self.strobe_duty.setRange(0.0, 1.0); self.strobe_duty.setSingleStep(0.05); self.strobe_duty.setValue(0.5)
            self.anaglyph_offset = QtWidgets.QSpinBox(); self.anaglyph_offset.setRange(-64, 64); self.anaglyph_offset.setValue(0)
            self.anaglyph_mode = QtWidgets.QComboBox(); self.anaglyph_mode.addItems(["red_cyan"]) ; self.anaglyph_mode.setCurrentIndex(0)
            self.beam_spread = QtWidgets.QDoubleSpinBox(); self.beam_spread.setRange(0.0, 8.0); self.beam_spread.setSingleStep(0.1); self.beam_spread.setValue(0.0)
            self.hbleed_sigma = QtWidgets.QDoubleSpinBox(); self.hbleed_sigma.setRange(0.0, 8.0); self.hbleed_sigma.setSingleStep(0.1); self.hbleed_sigma.setValue(0.0)
            self.hbleed_strength = QtWidgets.QDoubleSpinBox(); self.hbleed_strength.setRange(0.0, 1.0); self.hbleed_strength.setSingleStep(0.05); self.hbleed_strength.setValue(0.0)
            self.vbleed_sigma = QtWidgets.QDoubleSpinBox(); self.vbleed_sigma.setRange(0.0, 8.0); self.vbleed_sigma.setSingleStep(0.1); self.vbleed_sigma.setValue(0.0)
            self.vbleed_strength = QtWidgets.QDoubleSpinBox(); self.vbleed_strength.setRange(0.0, 1.0); self.vbleed_strength.setSingleStep(0.05); self.vbleed_strength.setValue(0.0)
            # removed line jitter controls
            self.rfi_strength = QtWidgets.QDoubleSpinBox(); self.rfi_strength.setRange(0.0, 1.0); self.rfi_strength.setSingleStep(0.05); self.rfi_strength.setValue(0.0)
            self.rfi_freq = QtWidgets.QDoubleSpinBox(); self.rfi_freq.setRange(0.0, 50.0); self.rfi_freq.setSingleStep(0.5); self.rfi_freq.setValue(0.0)
            self.rfi_speed = QtWidgets.QDoubleSpinBox(); self.rfi_speed.setRange(0.0, 240.0); self.rfi_speed.setSingleStep(0.5); self.rfi_speed.setValue(0.0)
            self.rfi_angle = QtWidgets.QDoubleSpinBox(); self.rfi_angle.setRange(-90.0, 90.0); self.rfi_angle.setSingleStep(1.0); self.rfi_angle.setValue(45.0)
            self.screen_jitter_amp = QtWidgets.QSpinBox(); self.screen_jitter_amp.setRange(-64, 64); self.screen_jitter_amp.setValue(0)
            self.screen_jitter_speed = QtWidgets.QDoubleSpinBox(); self.screen_jitter_speed.setRange(0.0, 240.0); self.screen_jitter_speed.setSingleStep(0.5); self.screen_jitter_speed.setValue(0.0)
            # Text overlay controls
            self.text_input = QtWidgets.QLineEdit()
            self.text_font_combo = QtWidgets.QFontComboBox()
            # Show only scalable (TrueType/OpenType) fonts to avoid DirectWrite issues
            self.text_font_combo.setFontFilters(QtWidgets.QFontComboBox.ScalableFonts)
            self.text_size = QtWidgets.QSpinBox(); self.text_size.setRange(6, 256); self.text_size.setValue(36)
            self.text_color = QtWidgets.QLineEdit("#FFFFFF")
            self.text_x = QtWidgets.QSpinBox(); self.text_x.setRange(0, 10000); self.text_x.setValue(32)
            self.text_y = QtWidgets.QSpinBox(); self.text_y.setRange(0, 10000); self.text_y.setValue(32)
            self.text_after = QtWidgets.QCheckBox("Draw text after effects")
            self.text_after.setChecked(True)
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
            self.triad_gamma = QtWidgets.QDoubleSpinBox(); self.triad_gamma.setRange(0.1, 5.0); self.triad_gamma.setSingleStep(0.05); self.triad_gamma.setValue(2.2)
            self.triad_softness = QtWidgets.QDoubleSpinBox(); self.triad_softness.setRange(0.0, 4.0); self.triad_softness.setSingleStep(0.05); self.triad_softness.setValue(0.5)
            self.triad_preserve_luma = QtWidgets.QCheckBox("Triad preserve luma")
            self.triad_preserve_luma.setChecked(True)
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
            effects_form.addRow("triad gamma", self.triad_gamma)
            effects_form.addRow("triad softness px", self.triad_softness)
            effects_form.addRow(self.triad_preserve_luma)
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
            output_form.addRow("hardware encode", self.gpu_cb)
            output_form.addRow("encoder", self.encoder_choice)
            self.decoder_choice = QtWidgets.QComboBox()
            self.decoder_choice.addItems(["Auto", "NVIDIA", "AMD", "Intel", "CPU"]) 
            self.decoder_choice.setCurrentIndex(0)
            output_form.addRow("decoder", self.decoder_choice)
            output_form.addRow("fast bloom", self.fast_bloom_cb)
            output_form.addRow("crf/cq", self.crf_val)
            self.bitrate_kbps = QtWidgets.QSpinBox(); self.bitrate_kbps.setRange(0, 200000); self.bitrate_kbps.setSingleStep(100); self.bitrate_kbps.setValue(0)
            output_form.addRow("bitrate kbps (0 auto)", self.bitrate_kbps)
            output_form.addRow("nvenc preset", self.nvenc_preset)
            self.render_btn = QtWidgets.QPushButton("Render")
            self.reset_btn = QtWidgets.QPushButton("Reset")
            self.save_btn = QtWidgets.QPushButton("Save Preset")
            self.load_btn = QtWidgets.QPushButton("Load Preset")
            out_buttons = QtWidgets.QHBoxLayout()
            out_buttons.addStretch(1)
            out_buttons.addWidget(self.load_btn)
            out_buttons.addWidget(self.save_btn)
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
            adv_form = QtWidgets.QFormLayout()
            adv_form.setLabelAlignment(QtCore.Qt.AlignRight)
            adv_form.addRow("brightness", self.brightness)
            adv_form.addRow("contrast", self.contrast)
            adv_form.addRow("gamma", self.gamma)
            adv_form.addRow("saturation", self.saturation)
            adv_form.addRow("temperature", self.temperature)
            adv_form.addRow("bloom threshold", self.bloom_threshold)
            adv_form.addRow("flicker strength", self.flicker_strength)
            adv_form.addRow("flicker hz", self.flicker_hz)
            adv_form.addRow("grain size", self.grain_size)
            adv_form.addRow("scanline angle", self.scanline_angle)
            adv_form.addRow("scanline thickness", self.scanline_thickness)
            adv_form.addRow("warp strength", self.warp_strength)
            # EFX tab
            efx_form = QtWidgets.QFormLayout()
            efx_form.setLabelAlignment(QtCore.Qt.AlignRight)
            efx_form.addRow("waves amp (px)", self.waves_amp)
            efx_form.addRow("waves freq", self.waves_freq)
            efx_form.addRow("waves speed (hz)", self.waves_speed)
            efx_form.addRow("strobe hz", self.strobe_hz)
            efx_form.addRow("strobe duty (0-1)", self.strobe_duty)
            efx_form.addRow("3D offset (px)", self.anaglyph_offset)
            efx_form.addRow("3D mode", self.anaglyph_mode)
            efx_form.addRow("beam spread", self.beam_spread)
            efx_form.addRow("h bleed sigma", self.hbleed_sigma)
            efx_form.addRow("h bleed strength", self.hbleed_strength)
            efx_form.addRow("v bleed sigma", self.vbleed_sigma)
            efx_form.addRow("v bleed strength", self.vbleed_strength)
            # removed line jitter rows
            efx_form.addRow("RF strength", self.rfi_strength)
            efx_form.addRow("RF freq", self.rfi_freq)
            efx_form.addRow("RF speed (hz)", self.rfi_speed)
            efx_form.addRow("RF angle (deg)", self.rfi_angle)
            efx_form.addRow("screen jitter amp (px)", self.screen_jitter_amp)
            efx_form.addRow("screen jitter speed (hz)", self.screen_jitter_speed)
            efx_tab = QtWidgets.QWidget(); efx_tab.setLayout(efx_form)
            # Text: move to its own tab
            text_form = QtWidgets.QFormLayout()
            text_form.setLabelAlignment(QtCore.Qt.AlignRight)
            text_form.addRow("text", self.text_input)
            # Custom font file picker
            self.text_font_path = QtWidgets.QLineEdit("")
            self.text_font_path.setPlaceholderText("Optional: .ttf/.otf file path")
            self.text_font_browse = QtWidgets.QPushButton("Browse Font…")
            font_row = QtWidgets.QHBoxLayout()
            font_row.addWidget(self.text_font_combo, 1)
            font_row.addWidget(self.text_font_browse)
            text_form.addRow("font", font_row)
            text_form.addRow("font file", self.text_font_path)
            text_form.addRow("text size", self.text_size)
            text_form.addRow("text color (#RRGGBB)", self.text_color)
            text_form.addRow("text x", self.text_x)
            text_form.addRow("text y", self.text_y)
            text_form.addRow("text after effects", self.text_after)
            # Text preset save/load
            self.text_save_btn = QtWidgets.QPushButton("Save Text Preset")
            self.text_load_btn = QtWidgets.QPushButton("Load Text Preset")
            text_btns = QtWidgets.QHBoxLayout()
            text_btns.addStretch(1)
            text_btns.addWidget(self.text_load_btn)
            text_btns.addWidget(self.text_save_btn)
            text_form.addRow(text_btns)
            text_tab = QtWidgets.QWidget(); text_tab.setLayout(text_form)
            adv_tab = QtWidgets.QWidget(); adv_tab.setLayout(adv_form)
            output_tab = QtWidgets.QWidget()
            output_tab.setLayout(output_col)
            tabs = QtWidgets.QTabWidget()
            tabs.addTab(effects_tab, "Effects")
            tabs.addTab(motion_tab, "Motion")
            tabs.addTab(adv_tab, "Advanced")
            tabs.addTab(efx_tab, "EFX")
            tabs.addTab(text_tab, "Text")
            tabs.addTab(output_tab, "Output")
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(self.preview_frame)
            splitter.addWidget(tabs)
            # Lock the right-side tabs width so only the preview grows/shrinks
            self.sidebar_width = 420
            tabs.setFixedWidth(self.sidebar_width)
            tabs.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)
            # Initial sizes: left takes remaining space, right is fixed
            splitter.setSizes([max(1, self.width() - self.sidebar_width - 48), self.sidebar_width])
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
            self.actGPU = QtGui.QAction("HW Encode", self)
            self.actGPU.setCheckable(True)
            self.actHWDec = QtGui.QAction("HW Decode", self)
            self.actHWDec.setCheckable(True)
            self.actFast = QtGui.QAction("Fast Bloom", self)
            self.actFast.setCheckable(True)
            self.actFast.setChecked(True)
            bar.addAction(self.actOpen)
            bar.addAction(self.actRender)
            bar.addSeparator()
            bar.addAction(self.actGPU)
            bar.addAction(self.actHWDec)
            bar.addAction(self.actFast)
            # Transport controls
            self.actPrevFrame = QtGui.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward), "Prev", self)
            self.actNextFrame = QtGui.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSkipForward), "Next", self)
            self.actBack5 = QtGui.QAction("-5s", self)
            self.actFwd5 = QtGui.QAction("+5s", self)
            bar.addSeparator()
            bar.addAction(self.actBack5)
            bar.addAction(self.actPrevFrame)
            bar.addAction(self.actPlay)
            bar.addAction(self.actNextFrame)
            bar.addAction(self.actFwd5)
            self.loop_cb = QtWidgets.QCheckBox("Loop")
            self.loop_cb.setChecked(True)
            self.speed_combo = QtWidgets.QComboBox(); self.speed_combo.addItems(["0.25x","0.5x","1x","1.5x","2x","4x"]); self.speed_combo.setCurrentText("1x")
            bar.addWidget(QtWidgets.QLabel("Speed:"))
            bar.addWidget(self.speed_combo)
            bar.addWidget(self.loop_cb)
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
            self.actHWDec.toggled.connect(self.on_hwdec_toggle)
            self.actFast.toggled.connect(self.fast_bloom_cb.setChecked)
            self.fast_bloom_cb.toggled.connect(self.actFast.setChecked)
            self.actPrevFrame.triggered.connect(self.on_prev_frame)
            self.actNextFrame.triggered.connect(self.on_next_frame)
            self.actBack5.triggered.connect(lambda: self.nudge_time(-5.0))
            self.actFwd5.triggered.connect(lambda: self.nudge_time(5.0))
            self.scanline_slider.valueChanged.connect(self.on_scanline_slider)
            self.triad_slider.valueChanged.connect(self.on_triad_slider)
            self.scanline_val.valueChanged.connect(self.on_scanline_val)
            self.triad_val.valueChanged.connect(self.on_triad_val)
            self.reset_btn.clicked.connect(self.on_reset)
            self.save_btn.clicked.connect(self.on_save_preset)
            self.load_btn.clicked.connect(self.on_load_preset)
            self.clip = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.playing = False
            self.t = 0.0
            self.prev_img = None
            self.preview_max_w = 960
            self.preview_max_h = 540
            self.hw_reader = None

            # Capture defaults for Reset
            self._defaults = self._collect_settings()
            # Seek slider
            self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.seek_slider.setRange(0, 1000)
            self.seek_slider.setTracking(False)
            self.status.addPermanentWidget(self.seek_slider)
            self._seeking = False
            self.seek_slider.sliderPressed.connect(self.on_seek_press)
            self.seek_slider.sliderReleased.connect(self.on_seek_release)
            self.seek_slider.valueChanged.connect(self.on_seek_change)

            # Live refresh when text controls change
            self.text_input.textChanged.connect(lambda _=None: self._render_current_frame())
            self.text_font_combo.currentFontChanged.connect(lambda _=None: self._render_current_frame())
            self.text_font_browse.clicked.connect(self.on_browse_font)
            self.text_font_path.textChanged.connect(lambda _=None: self._render_current_frame())
            self.text_size.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.text_color.textChanged.connect(lambda _=None: self._render_current_frame())
            self.text_x.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.text_y.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.text_after.toggled.connect(lambda _=None: self._render_current_frame())
            self.text_save_btn.clicked.connect(self.on_save_text_preset)
            self.text_load_btn.clicked.connect(self.on_load_text_preset)
            # Live refresh for effect controls when paused
            self.scanline_val.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.triad_val.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.triad_gamma.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.triad_softness.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.triad_preserve_luma.toggled.connect(lambda _=None: self._render_current_frame())
            self.pixel_size.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.aberration.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.noise_val.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.bloom_sigma.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.bloom_strength.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.vignette_val.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.persistence_val.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.scanline_speed.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.scanline_period.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.fast_bloom_cb.toggled.connect(lambda _=None: self._render_current_frame())
            self.brightness.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.contrast.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.gamma.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.saturation.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.temperature.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.flicker_strength.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.flicker_hz.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.grain_size.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.scanline_angle.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.scanline_thickness.valueChanged.connect(lambda _=None: self._render_current_frame())
            self.warp_strength.valueChanged.connect(lambda _=None: self._render_current_frame())

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
            # Clean up any existing clip/hw reader
            try:
                self._stop_hw_reader()
            except Exception:
                pass
            if self.clip is not None:
                try:
                    self.clip.close()
                except Exception:
                    try:
                        if hasattr(self.clip, "reader") and self.clip.reader:
                            self.clip.reader.close()
                    except Exception:
                        pass
                    try:
                        if getattr(self.clip, "audio", None) is not None and hasattr(self.clip.audio, "reader"):
                            self.clip.audio.reader.close_proc()
                    except Exception:
                        pass
            self.prev_img = None
            self.clip = VideoFileClip(str(p))
            fps = max(1, int(self.clip.fps or 24))
            self.timer.setInterval(int(1000 / fps))
            self.t = 0.0
            self._restart_hw_reader()
            # Reset seek slider
            try:
                dur = float(self.clip.duration or 0.0)
                self.seek_slider.setEnabled(dur > 0.0)
                self.seek_slider.setValue(0)
            except Exception:
                pass
            self._render_current_frame()

        def on_play_pause(self) -> None:
            if self.clip is None:
                return
            self.playing = not self.playing
            self.actPlay.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause) if self.playing else self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.actPlay.setText("Pause" if self.playing else "Play")
            if self.playing:
                if not self.timer.isActive():
                    self.timer.start()
                if self.actHWDec.isChecked() and self.hw_reader is None:
                    self._restart_hw_reader()
            else:
                self.timer.stop()

        def on_tick(self) -> None:
            if self.clip is None:
                return
            frame = None
            if self.actHWDec.isChecked() and self.hw_reader is not None:
                frame = self.hw_reader.read_next()
                if frame is None:
                    # try restart (loop)
                    self._restart_hw_reader()
                    frame = self.hw_reader.read_next() if self.hw_reader else None
            if frame is None:
                frame = self.clip.get_frame(self.t)
            w, h = frame.shape[1], frame.shape[0]
            avail_w = max(1, self.video_label.width())
            avail_h = max(1, self.video_label.height())
            scale = min(avail_w / max(1, w), avail_h / max(1, h))
            if scale != 1.0:
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
            # Compute scaled text overlay size/position so UI values represent output pixels
            txt_scale = float(scale)
            txt_size_px = max(1, int(self.text_size.value() * txt_scale))
            txt_pos = (int(self.text_x.value() * txt_scale), int(self.text_y.value() * txt_scale))
            out, self.prev_img = apply_crt_effect(
                frame=frame,
                scanline_strength=float(self.scanline_val.value()),
                triad_mask=make_triad_mask(frame.shape[0], frame.shape[1], float(self.triad_val.value()), float(self.triad_softness.value())) if self.triad_val.value() > 0.0 else None,
                triad_gamma=float(self.triad_gamma.value()),
                triad_preserve_luma=bool(self.triad_preserve_luma.isChecked()),
                aberration_px=int(self.aberration.value()),
                bloom_sigma=float(self.bloom_sigma.value()),
                bloom_strength=float(self.bloom_strength.value()),
                bloom_threshold=float(self.bloom_threshold.value()),
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
                time_sec=self.t,
                brightness=float(self.brightness.value()),
                contrast=float(self.contrast.value()),
                gamma=float(self.gamma.value()),
                saturation=float(self.saturation.value()),
                temperature=float(self.temperature.value()),
                flicker_strength=float(self.flicker_strength.value()),
                flicker_hz=float(self.flicker_hz.value()),
                grain_size=int(self.grain_size.value()),
                scanline_angle=float(self.scanline_angle.value()),
                scanline_thickness=float(self.scanline_thickness.value()),
                warp_strength=float(self.warp_strength.value()),
                beam_spread_strength=float(self.beam_spread.value()),
                hbleed_sigma=float(self.hbleed_sigma.value()),
                hbleed_strength=float(self.hbleed_strength.value()),
                vbleed_sigma=float(self.vbleed_sigma.value()),
                vbleed_strength=float(self.vbleed_strength.value()),
                jitter_amp_px=0,
                jitter_speed_hz=0.0,
                screen_jitter_amp_px=int(self.screen_jitter_amp.value()),
                screen_jitter_speed_hz=float(self.screen_jitter_speed.value()),
                rfi_strength=float(self.rfi_strength.value()),
                rfi_freq=float(self.rfi_freq.value()),
                rfi_speed_hz=float(self.rfi_speed.value()),
                rfi_angle_deg=float(self.rfi_angle.value()),
                waves_amp_px=int(self.waves_amp.value()),
                waves_freq=float(self.waves_freq.value()),
                waves_speed_hz=float(self.waves_speed.value()),
                strobe_hz=float(self.strobe_hz.value()),
                strobe_duty=float(self.strobe_duty.value()),
                anaglyph_offset_px=int(self.anaglyph_offset.value()),
                anaglyph_mode=str(self.anaglyph_mode.currentText()),
                text_overlay_rgba=_make_text_overlay_rgba_qt(
                    frame.shape[1],
                    frame.shape[0],
                    self.text_input.text(),
                    self._effective_font_spec(),
                    int(txt_size_px),
                    self.text_color.text(),
                    txt_pos,
                ) if self.text_input.text() else None,
                text_overlay_after=bool(self.text_after.isChecked()),
            )
            qimg = QtGui.QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix)
            self.t += self._speed_factor() * (1.0 / max(1.0, self.clip.fps or 24.0))
            if self.t >= float(self.clip.duration or 1.0):
                if self.loop_cb.isChecked():
                    self.t = 0.0
                else:
                    self.t = float(self.clip.duration or 0.0)
                    self.playing = False
                    self.timer.stop()
            tl = int(self.t)
            dur = int(self.clip.duration or 0)
            self.time_label.setText(f"{tl//60:02d}:{tl%60:02d} / {dur//60:02d}:{dur%60:02d}")
            # update seek
            try:
                durf = float(self.clip.duration or 0.0)
                if durf > 0 and not self._seeking:
                    self.seek_slider.blockSignals(True)
                    self.seek_slider.setValue(int((self.t / durf) * 1000))
                    self.seek_slider.blockSignals(False)
            except Exception:
                pass

        def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
            super().resizeEvent(e)
            self.preview_max_w = max(1, self.preview_frame.width() - 24)
            self.preview_max_h = max(1, self.preview_frame.height() - 24)
            # Force relayout of label pixmap (guard against null pixmap)
            pm = self.video_label.pixmap()
            if pm is not None and not pm.isNull():
                self.video_label.setPixmap(
                    pm.scaled(
                        self.preview_max_w,
                        self.preview_max_h,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
            # No restart on resize; scaling is done dynamically in on_tick

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
                        triad_gamma=float(self.triad_gamma.value()),
                        triad_preserve_luma=bool(self.triad_preserve_luma.isChecked()),
                        triad_softness=float(self.triad_softness.value()),
                        aberration_px=int(self.aberration.value()),
                        bloom_sigma=float(self.bloom_sigma.value()),
                        bloom_strength=float(self.bloom_strength.value()),
                        noise_strength=float(self.noise_val.value()),
                        vignette_strength=float(self.vignette_val.value()),
                        persistence=float(self.persistence_val.value()),
                        fps=opts["fps"],
                        crf=int(self.crf_val.value()),
                        target_bitrate_kbps=int(self.bitrate_kbps.value()),
                        scanline_speed_px_s=float(self.scanline_speed.value()),
                        scanline_period_px=float(self.scanline_period.value()),
                        fast_bloom=bool(self.fast_bloom_cb.isChecked()),
                        pixel_size=int(self.pixel_size.value()),
                        gpu=bool(self.gpu_cb.isChecked()) if opts["gpu"] is None else bool(opts["gpu"]),
                        nvenc_preset=str(self.nvenc_preset.text()),
                        glitch_amp_px=int(self.glitch_amp.value()),
                        glitch_height_frac=float(self.glitch_height.value()),
                        encoder_preference=self._encoder_pref_gui(),
                        decoder_preference=self._decoder_pref_gui(),
                        bloom_threshold=float(self.bloom_threshold.value()),
                        brightness=float(self.brightness.value()),
                        contrast=float(self.contrast.value()),
                        gamma=float(self.gamma.value()),
                        saturation=float(self.saturation.value()),
                        temperature=float(self.temperature.value()),
                        flicker_strength=float(self.flicker_strength.value()),
                        flicker_hz=float(self.flicker_hz.value()),
                        grain_size=int(self.grain_size.value()),
                        scanline_angle=float(self.scanline_angle.value()),
                        scanline_thickness=float(self.scanline_thickness.value()),
                        warp_strength=float(self.warp_strength.value()),
                        beam_spread_strength=float(self.beam_spread.value()),
                        hbleed_sigma=float(self.hbleed_sigma.value()),
                        hbleed_strength=float(self.hbleed_strength.value()),
                        vbleed_sigma=float(self.vbleed_sigma.value()),
                        vbleed_strength=float(self.vbleed_strength.value()),
                        offload_filters=bool(self.fast_bloom_cb.isChecked() and False),
                        text=str(self.text_input.text()),
                        text_font=str(self._effective_font_spec()),
                        text_size=int(self.text_size.value()),
                        text_color=str(self.text_color.text()),
                        text_pos=(int(self.text_x.value()), int(self.text_y.value())),
                        text_after=bool(self.text_after.isChecked()),
                        progress_cb=lambda f: QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, int(f * 100))),
                    )
                    QtCore.QMetaObject.invokeMethod(self.progress, "setValue", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, 100))
                    QtCore.QMetaObject.invokeMethod(self.progress, "setVisible", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, False))
                    QtCore.QMetaObject.invokeMethod(self, "_show_done", QtCore.Qt.QueuedConnection)
                    QtCore.QMetaObject.invokeMethod(self.status, "showMessage", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "Hardware encoder used" if used_gpu else "CPU x264 used"))
                finally:
                    QtCore.QMetaObject.invokeMethod(self, "setEnabled", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, True))
            th = threading.Thread(target=run_render, daemon=True)
            th.start()

        @QtCore.Slot()
        def _show_done(self) -> None:
            QtWidgets.QMessageBox.information(self, "Done", "Render complete")

        def _render_current_frame(self) -> None:
            if self.clip is None:
                return
            try:
                frame = self.clip.get_frame(self.t)
            except Exception:
                return
            w, h = frame.shape[1], frame.shape[0]
            avail_w = max(1, self.video_label.width())
            avail_h = max(1, self.video_label.height())
            scale = min(avail_w / max(1, w), avail_h / max(1, h))
            if scale != 1.0:
                im = Image.fromarray(frame)
                frame = np.asarray(im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR))
            out, _ = apply_crt_effect(
                frame=frame,
                scanline_strength=float(self.scanline_val.value()),
                triad_mask=make_triad_mask(frame.shape[0], frame.shape[1], float(self.triad_val.value()), float(self.triad_softness.value())) if self.triad_val.value() > 0.0 else None,
                triad_gamma=float(self.triad_gamma.value()),
                triad_preserve_luma=bool(self.triad_preserve_luma.isChecked()),
                aberration_px=int(self.aberration.value()),
                bloom_sigma=float(self.bloom_sigma.value()),
                bloom_strength=float(self.bloom_strength.value()),
                bloom_threshold=float(self.bloom_threshold.value()),
                noise_strength=float(self.noise_val.value()),
                vignette_mask=make_vignette(frame.shape[0], frame.shape[1], float(self.vignette_val.value())) if self.vignette_val.value() > 0.0 else None,
                persistence=0.0,
                state_prev=None,
                scanline_period_px=float(self.scanline_period.value()),
                scanline_phase_px=float(self.scanline_speed.value()) * self.t,
                fast_bloom=bool(self.fast_bloom_cb.isChecked()),
                pixel_size=int(self.pixel_size.value()),
                glitch_amp_px=int(self.glitch_amp.value()),
                glitch_height_frac=float(self.glitch_height.value()),
                time_sec=self.t,
                brightness=float(self.brightness.value()),
                contrast=float(self.contrast.value()),
                gamma=float(self.gamma.value()),
                saturation=float(self.saturation.value()),
                temperature=float(self.temperature.value()),
                flicker_strength=float(self.flicker_strength.value()),
                flicker_hz=float(self.flicker_hz.value()),
                grain_size=int(self.grain_size.value()),
                scanline_angle=float(self.scanline_angle.value()),
                scanline_thickness=float(self.scanline_thickness.value()),
                warp_strength=float(self.warp_strength.value()),
                text_overlay_rgba=_make_text_overlay_rgba(
                    frame.shape[1],
                    frame.shape[0],
                    self.text_input.text(),
                    self.text_font_combo.currentFont().family(),
                    int(self.text_size.value()),
                    self.text_color.text(),
                    (int(self.text_x.value()), int(self.text_y.value())),
                ) if self.text_input.text() else None,
                text_overlay_after=bool(self.text_after.isChecked()),
            )
            qimg = QtGui.QImage(out.data, out.shape[1], out.shape[0], out.strides[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pix)

        def _speed_factor(self) -> float:
            s = self.speed_combo.currentText().lower().replace("x", "").strip()
            try:
                return float(s) if s else 1.0
            except Exception:
                return 1.0

        def on_prev_frame(self) -> None:
            if self.clip is None:
                return
            fps = max(1.0, float(self.clip.fps or 24.0))
            self.t = max(0.0, self.t - 1.0 / fps)
            self._render_current_frame()

        def on_next_frame(self) -> None:
            if self.clip is None:
                return
            fps = max(1.0, float(self.clip.fps or 24.0))
            self.t = min(float(self.clip.duration or 0.0), self.t + 1.0 / fps)
            self._render_current_frame()

        def nudge_time(self, dt: float) -> None:
            if self.clip is None:
                return
            dur = float(self.clip.duration or 0.0)
            self.t = float(np.clip(self.t + float(dt), 0.0, dur))
            self._render_current_frame()

        def on_seek_press(self) -> None:
            self._seeking = True

        def on_seek_release(self) -> None:
            self._seeking = False
            self.on_seek_change(self.seek_slider.value())

        def on_seek_change(self, v: int) -> None:
            if self.clip is None:
                return
            dur = float(self.clip.duration or 0.0)
            if dur <= 0:
                return
            self.t = (float(v) / 1000.0) * dur
            self._render_current_frame()

        def on_reset(self) -> None:
            if hasattr(self, "_defaults") and isinstance(self._defaults, dict):
                self._apply_settings(self._defaults)
            else:
                # fallback minimal reset
                self.scanline_slider.setValue(60)
                self.triad_slider.setValue(35)
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
                self.nvenc_preset.setText(self.nvenc_preset.text())
                self.fast_bloom_cb.setChecked(True)
                self.gpu_cb.setChecked(False)
                self.bitrate_kbps.setValue(0)
                self.glitch_amp.setValue(0)
                self.glitch_height.setValue(0.0)

        def _collect_settings(self) -> dict:
            return {
                "scanline": float(self.scanline_val.value()),
                "triad": float(self.triad_val.value()),
                "triad_gamma": float(self.triad_gamma.value()),
                "triad_softness": float(self.triad_softness.value()),
                "triad_preserve_luma": bool(self.triad_preserve_luma.isChecked()),
                "pixel_size": int(self.pixel_size.value()),
                "aberration_px": int(self.aberration.value()),
                "noise": float(self.noise_val.value()),
                "bloom_sigma": float(self.bloom_sigma.value()),
                "bloom_strength": float(self.bloom_strength.value()),
                "bloom_threshold": float(self.bloom_threshold.value()),
                "vignette": float(self.vignette_val.value()),
                "persistence": float(self.persistence_val.value()),
                "scanline_speed": float(self.scanline_speed.value()),
                "scanline_period": float(self.scanline_period.value()),
                "glitch_amp": int(self.glitch_amp.value()),
                "glitch_height": float(self.glitch_height.value()),
                "crf": int(self.crf_val.value()),
                "bitrate_kbps": int(self.bitrate_kbps.value()),
                "nvenc_preset": str(self.nvenc_preset.text()),
                "fast_bloom": bool(self.fast_bloom_cb.isChecked()),
                "gpu": bool(self.gpu_cb.isChecked()),
                "encoder": self._encoder_pref_gui(),
                # Advanced
                "brightness": float(self.brightness.value()),
                "contrast": float(self.contrast.value()),
                "gamma": float(self.gamma.value()),
                "saturation": float(self.saturation.value()),
                "temperature": float(self.temperature.value()),
                "flicker_strength": float(self.flicker_strength.value()),
                "flicker_hz": float(self.flicker_hz.value()),
                "grain_size": int(self.grain_size.value()),
                "scanline_angle": float(self.scanline_angle.value()),
                "scanline_thickness": float(self.scanline_thickness.value()),
                "warp_strength": float(self.warp_strength.value()),
            }

        def _set_encoder_pref_gui(self, pref: str) -> None:
            mapping = ["auto", "nvidia", "amd", "cpu"]
            try:
                idx = mapping.index(str(pref).strip().lower())
            except ValueError:
                idx = 0
            self.encoder_choice.setCurrentIndex(idx)

        def _apply_settings(self, s: dict) -> None:
            if not isinstance(s, dict):
                return
            if "scanline" in s:
                self.scanline_val.setValue(float(s["scanline"]))
            if "triad" in s:
                self.triad_val.setValue(float(s["triad"]))
            if "triad_gamma" in s:
                self.triad_gamma.setValue(float(s["triad_gamma"]))
            if "triad_softness" in s:
                self.triad_softness.setValue(float(s["triad_softness"]))
            if "triad_preserve_luma" in s:
                self.triad_preserve_luma.setChecked(bool(s["triad_preserve_luma"]))
            if "pixel_size" in s:
                self.pixel_size.setValue(int(s["pixel_size"]))
            if "aberration_px" in s:
                self.aberration.setValue(int(s["aberration_px"]))
            if "noise" in s:
                self.noise_val.setValue(float(s["noise"]))
            if "bloom_sigma" in s:
                self.bloom_sigma.setValue(float(s["bloom_sigma"]))
            if "bloom_strength" in s:
                self.bloom_strength.setValue(float(s["bloom_strength"]))
            if "bloom_threshold" in s:
                self.bloom_threshold.setValue(float(s["bloom_threshold"]))
            if "vignette" in s:
                self.vignette_val.setValue(float(s["vignette"]))
            if "persistence" in s:
                self.persistence_val.setValue(float(s["persistence"]))
            if "scanline_speed" in s:
                self.scanline_speed.setValue(float(s["scanline_speed"]))
            if "scanline_period" in s:
                self.scanline_period.setValue(float(s["scanline_period"]))
            if "glitch_amp" in s:
                self.glitch_amp.setValue(int(s["glitch_amp"]))
            if "glitch_height" in s:
                self.glitch_height.setValue(float(s["glitch_height"]))
            if "crf" in s:
                self.crf_val.setValue(int(s["crf"]))
            if "bitrate_kbps" in s:
                self.bitrate_kbps.setValue(int(s["bitrate_kbps"]))
            if "nvenc_preset" in s:
                self.nvenc_preset.setText(str(s["nvenc_preset"]))
            if "fast_bloom" in s:
                self.fast_bloom_cb.setChecked(bool(s["fast_bloom"]))
            if "gpu" in s:
                self.gpu_cb.setChecked(bool(s["gpu"]))
            if "encoder" in s:
                self._set_encoder_pref_gui(str(s["encoder"]))
            # Advanced
            if "brightness" in s:
                self.brightness.setValue(float(s["brightness"]))
            if "contrast" in s:
                self.contrast.setValue(float(s["contrast"]))
            if "gamma" in s:
                self.gamma.setValue(float(s["gamma"]))
            if "saturation" in s:
                self.saturation.setValue(float(s["saturation"]))
            if "temperature" in s:
                self.temperature.setValue(float(s["temperature"]))
            if "flicker_strength" in s:
                self.flicker_strength.setValue(float(s["flicker_strength"]))
            if "flicker_hz" in s:
                self.flicker_hz.setValue(float(s["flicker_hz"]))
            if "grain_size" in s:
                self.grain_size.setValue(int(s["grain_size"]))
            if "scanline_angle" in s:
                self.scanline_angle.setValue(float(s["scanline_angle"]))
            if "scanline_thickness" in s:
                self.scanline_thickness.setValue(float(s["scanline_thickness"]))
            if "warp_strength" in s:
                self.warp_strength.setValue(float(s["warp_strength"]))

        def on_save_preset(self) -> None:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Preset", str(Path.cwd() / "preset.json"), "JSON (*.json)")
            if not path:
                return
            data = self._collect_settings()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self.status.showMessage("Preset saved")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save preset:\n{e}")

        def on_load_preset(self) -> None:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Preset", str(Path.cwd()), "JSON (*.json)")
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_settings(data)
                self.status.showMessage("Preset loaded")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load preset:\n{e}")

        def _encoder_pref_gui(self) -> str:
            idx = int(self.encoder_choice.currentIndex()) if hasattr(self, "encoder_choice") else 0
            return ["auto", "nvidia", "amd", "cpu"][idx]

        def _decoder_pref_gui(self) -> str:
            idx = int(self.decoder_choice.currentIndex()) if hasattr(self, "decoder_choice") else 0
            return ["auto", "nvidia", "amd", "intel", "cpu"][idx]

        def _effective_font_spec(self) -> str:
            path = self.text_font_path.text().strip()
            if path:
                return path
            return self.text_font_combo.currentFont().family()

        def on_browse_font(self) -> None:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Font", str(Path.cwd()), "Fonts (*.ttf *.otf)")
            if path:
                self.text_font_path.setText(path)

        def on_save_text_preset(self) -> None:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Text Preset", str(Path.cwd() / "text_preset.json"), "JSON (*.json)")
            if not path:
                return
            data = {
                "text": self.text_input.text(),
                "font": self._effective_font_spec(),
                "size": int(self.text_size.value()),
                "color": self.text_color.text(),
                "x": int(self.text_x.value()),
                "y": int(self.text_y.value()),
                "after": bool(self.text_after.isChecked()),
            }
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self.status.showMessage("Text preset saved")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save text preset:\n{e}")

        def on_load_text_preset(self) -> None:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Text Preset", str(Path.cwd()), "JSON (*.json)")
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.text_input.setText(str(data.get("text", "")))
                self.text_font_path.setText(str(data.get("font", "")))
                self.text_size.setValue(int(data.get("size", 36)))
                self.text_color.setText(str(data.get("color", "#FFFFFF")))
                self.text_x.setValue(int(data.get("x", 32)))
                self.text_y.setValue(int(data.get("y", 32)))
                self.text_after.setChecked(bool(data.get("after", True)))
                self.status.showMessage("Text preset loaded")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load text preset:\n{e}")

        def on_hwdec_toggle(self, enabled: bool) -> None:
            if enabled:
                self._restart_hw_reader()
            else:
                self._stop_hw_reader()

        def closeEvent(self, e: QtGui.QCloseEvent) -> None:
            try:
                self.timer.stop()
            except Exception:
                pass
            try:
                self._stop_hw_reader()
            except Exception:
                pass
            if self.clip is not None:
                try:
                    self.clip.close()
                except Exception:
                    try:
                        if hasattr(self.clip, "reader") and self.clip.reader:
                            self.clip.reader.close()
                    except Exception:
                        pass
                    try:
                        if getattr(self.clip, "audio", None) is not None and hasattr(self.clip.audio, "reader"):
                            self.clip.audio.reader.close_proc()
                    except Exception:
                        pass
            super().closeEvent(e)

        def _stop_hw_reader(self) -> None:
            r = getattr(self, "hw_reader", None)
            if r is not None:
                try:
                    r.stop()
                except Exception:
                    pass
            self.hw_reader = None

        def _restart_hw_reader(self) -> None:
            self._stop_hw_reader()
            try:
                if not self.clip or not self.actHWDec.isChecked():
                    return
                src_path = Path(self.clip.filename)
                if not src_path.exists():
                    return
                w, h = self.clip.size
                # scale to preview to reduce bandwidth
                scale = min(self.preview_max_w / max(1, w), self.preview_max_h / max(1, h), 1.0)
                sw = max(1, int(w * scale))
                sh = max(1, int(h * scale))
                fps = max(1, int(self.clip.fps or 24))
                self.hw_reader = HWPreviewReader(src_path, sw, sh, fps)
                self.hw_reader.start()
            except Exception:
                self.hw_reader = None


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


