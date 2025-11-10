# Copyright 2025 Noam Rotstein
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys, cv2, numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import shutil, subprocess


from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QPainterPath, QPen, QColor, QPainter, QPolygonF, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsLineItem,
    QToolBar, QLabel, QSpinBox, QWidget, QMessageBox,
    QComboBox, QPushButton, QGraphicsEllipseItem, QFrame, QVBoxLayout, QSlider, QHBoxLayout,
    QPlainTextEdit
)
import imageio

# ------------------------------
# Utility: numpy <-> QPixmap
# ------------------------------

def np_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def np_rgba_to_qpixmap(img_rgba: np.ndarray) -> QPixmap:
    h, w = img_rgba.shape[:2]
    qimg = QImage(img_rgba.data, w, h, img_rgba.strides[0], QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())

# ------------------------------
# Image I/O + fit helpers
# ------------------------------

def load_first_frame(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    low = path.lower()
    if low.endswith((".mp4", ".mov", ".avi", ".mkv")):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame from video")
        return frame
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to read image")
    return img

def resize_then_center_crop(img: np.ndarray, target_h: int, target_w: int, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(target_w / float(w), target_h / float(h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    y0 = (new_h - target_h) // 2
    x0 = (new_w - target_w) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]

def fit_center_pad(img: np.ndarray, target_h: int, target_w: int, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    h, w = img.shape[:2]
    scale_h = target_h / float(h)
    new_w_hfirst = int(round(w * scale_h))
    new_h_hfirst = target_h
    if new_w_hfirst <= target_w:
        resized = cv2.resize(img, (new_w_hfirst, new_h_hfirst), interpolation=interpolation)
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - new_w_hfirst) // 2
        result[:, x0:x0 + new_w_hfirst] = resized
        return result
    scale_w = target_w / float(w)
    new_w_wfirst = target_w
    new_h_wfirst = int(round(h * scale_w))
    resized = cv2.resize(img, (new_w_wfirst, new_h_wfirst), interpolation=interpolation)
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h_wfirst) // 2
    result[y0:y0 + new_h_wfirst, :] = resized
    return result

# ------------------------------
# Hue utilities
# ------------------------------

def apply_hue_shift_bgr(img_bgr: np.ndarray, hue_deg: float) -> np.ndarray:
    """Rotate hue by hue_deg (degrees) in HSV space. S and V unchanged."""
    if abs(hue_deg) < 1e-6:
        return img_bgr.copy()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    offset = int(round((hue_deg / 360.0) * 179.0))
    h = (h + offset) % 180
    hsv[:, :, 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ------------------------------
# Compositing / warping (final)
# ------------------------------

def alpha_over(bg_bgr: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    a = (fg_rgba[:, :, 3:4].astype(np.float32) / 255.0)
    if a.max() == 0:
        return bg_bgr.copy()
    fg = fg_rgba[:, :, :3].astype(np.float32)
    bg = bg_bgr.astype(np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

def inpaint_background(image_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    mask = (mask_bool.astype(np.uint8) * 255)
    return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)

def animate_polygon(image_bgr, polygon_xy, path_xy, scales, rotations_deg, interp=cv2.INTER_LINEAR, origin_xy=None):
    """
    Returns list of RGBA frames and list of transformed polygons per frame.
    Uses BORDER_REPLICATE so off-canvas doesn't appear black.
    """
    h, w = image_bgr.shape[:2]
    frames_rgba = []
    polys_per_frame = []

    if origin_xy is None:
        if len(path_xy) == 0:
            raise ValueError("animate_polygon: path_xy is empty and origin_xy not provided.")
        origin = np.asarray(path_xy[0], dtype=np.float32)
    else:
        origin = np.asarray(origin_xy, dtype=np.float32)

    for i in range(len(path_xy)):
        theta = np.deg2rad(rotations_deg[i]).astype(np.float32)
        s = float(scales[i])
        a11 = s * np.cos(theta); a12 = -s * np.sin(theta)
        a21 = s * np.sin(theta); a22 =  s * np.cos(theta)
        tx = path_xy[i, 0] - (a11 * origin[0] + a12 * origin[1])
        ty = path_xy[i, 1] - (a21 * origin[0] + a22 * origin[1])
        M = np.array([[a11, a12, tx], [a21, a22, ty]], dtype=np.float32)

        warped = cv2.warpAffine(image_bgr, M, (w, h), flags=interp,
                                borderMode=cv2.BORDER_REPLICATE)

        poly = np.asarray(polygon_xy, dtype=np.float32)
        pts1 = np.hstack([poly, np.ones((len(poly), 1), dtype=np.float32)])
        poly_t = (M @ pts1.T).T
        polys_per_frame.append(poly_t.astype(np.float32))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_t.astype(np.int32)], 255)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = warped
        rgba[:, :, 3] = mask
        frames_rgba.append(rgba)

    return frames_rgba, polys_per_frame

def composite_frames(background_bgr, list_of_layer_frame_lists):
    frames = []
    T = len(list_of_layer_frame_lists[0]) if list_of_layer_frame_lists else 0
    for t in range(T):
        frame = background_bgr.copy()
        for layer in list_of_layer_frame_lists:
            frame = alpha_over(frame, layer[t])
        frames.append(frame)
    return frames

def save_video_mp4(frames_bgr, path, fps=24):
    """
    Write MP4 using imageio (FFmpeg backend) with H.264 + yuv420p so it works on macOS/QuickTime.
    - Converts BGR->RGB (imageio expects RGB)
    - Enforces even width/height (needed for yuv420p)
    - Tags BT.709 and faststart for smooth playback
    """
    if not frames_bgr:
        raise ValueError("No frames to save")

    # Validate and normalize frames (to RGB uint8 and consistent size)
    h, w = frames_bgr[0].shape[:2]
    out_frames = []
    for f in frames_bgr:
        if f is None:
            raise RuntimeError("Encountered None frame")
        # Accept gray/BGR/BGRA; convert to BGR then to RGB
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        elif f.shape[2] != 3:
            raise RuntimeError("Frames must be gray, BGR, or BGRA")
        if f.shape[:2] != (h, w):
            raise RuntimeError("Frame size mismatch during save.")
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        # BGR -> RGB for imageio/ffmpeg
        out_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    # Enforce even dims for yuv420p
    hh = h - (h % 2)
    ww = w - (w % 2)
    if (hh != h) or (ww != w):
        out_frames = [frm[:hh, :ww] for frm in out_frames]
        h, w = hh, ww

    # Try libx264 first; fall back to MPEG-4 Part 2 if libx264 missing
    ffmpeg_common = ['-movflags', '+faststart',
                     '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                     '-tag:v', 'avc1']  # helps QuickTime recognize H.264 properly
    try:
        writer = imageio.get_writer(
            path, format='ffmpeg', fps=float(fps),
            codec='libx264', pixelformat='yuv420p',
            ffmpeg_params=ffmpeg_common
        )
    except Exception:
        # Fallback: MPEG-4 (still Mac-friendly, a bit larger/softer)
        writer = imageio.get_writer(
            path, format='ffmpeg', fps=float(fps),
            codec='mpeg4', pixelformat='yuv420p',
            ffmpeg_params=['-movflags', '+faststart']
        )

    try:
        for frm in out_frames:
            writer.append_data(frm)
    finally:
        writer.close()

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("imageio/ffmpeg produced an empty file. Check that FFmpeg is available.")
    return path



# ------------------------------
# Data structures
# ------------------------------

PALETTE = [
    QColor(255, 99, 99),   # red
    QColor(99, 155, 255),  # blue
    QColor(120, 220, 120), # green
    QColor(255, 200, 80),  # orange
    QColor(200, 120, 255), # purple
    QColor(120, 255, 255)  # cyan
]

@dataclass
class Keyframe:
    pos: np.ndarray          # (2,)
    rot_deg: float
    scale: float
    hue_deg: float = 0.0

@dataclass
class Layer:
    name: str
    source_bgr: np.ndarray
    polygon_xy: Optional[np.ndarray] = None
    origin_local_xy: Optional[np.ndarray] = None  # bbox center in item coords
    is_external: bool = False
    pixmap_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
    outline_item: Optional[QGraphicsPathItem] = None
    handle_items: List[QtWidgets.QGraphicsItem] = field(default_factory=list)
    keyframes: List[Keyframe] = field(default_factory=list)
    path_lines: List[QGraphicsLineItem] = field(default_factory=list)
    preview_line: Optional[QGraphicsLineItem] = None
    color: QColor = field(default_factory=lambda: QColor(255, 99, 99))

    def has_polygon(self) -> bool:
        return self.polygon_xy is not None and len(self.polygon_xy) >= 3

# ------------------------------
# Handles (scale corners + rotate dot)
# ------------------------------

class HandleBase(QGraphicsEllipseItem):
    def __init__(self, r: float, color: QColor, parent=None):
        super().__init__(-r, -r, 2*r, 2*r, parent)
        self.setBrush(color)
        pen = QPen(QColor(0, 0, 0), 1)
        pen.setCosmetic(True)  # (optional) keep 1px outline at any zoom
        self.setPen(pen)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setZValue(2000)
        self._item: Optional[QGraphicsPixmapItem] = None

        # 👇 The key line: draw in device coords (no scaling with the polygon)
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)

    def set_item(self, item: QGraphicsPixmapItem):
        self._item = item

    def origin_scene(self) -> QPointF:
        return self._item.mapToScene(self._item.transformOriginPoint())

class ScaleHandle(HandleBase):
    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if not self._item: return super().mousePressEvent(event)
        self._start_scale = self._item.scale() if self._item.scale() != 0 else 1.0
        self._origin_scene = self.origin_scene()
        v0 = event.scenePos() - self._origin_scene
        self._d0 = max(1e-6, (v0.x()*v0.x() + v0.y()*v0.y())**0.5)
        event.accept()
    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if not self._item: return super().mouseMoveEvent(event)
        v = event.scenePos() - self._origin_scene
        d = max(1e-6, (v.x()*v.x() + v.y()*v.y())**0.5)
        s = float(self._start_scale * (d / self._d0))
        s = float(np.clip(s, 0.05, 10.0))
        self._item.setScale(s)
        event.accept()

class RotateHandle(HandleBase):
    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if not self._item: return super().mousePressEvent(event)
        self._start_rot = self._item.rotation()
        self._origin_scene = self.origin_scene()
        v0 = event.scenePos() - self._origin_scene
        self._a0 = np.degrees(np.arctan2(v0.y(), v0.x()))
        event.accept()
    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if not self._item: return super().mouseMoveEvent(event)
        v = event.scenePos() - self._origin_scene
        a = np.degrees(np.arctan2(v.y(), v.x()))
        delta = a - self._a0
        r = self._start_rot + delta
        if r > 180: r -= 360
        if r < -180: r += 360
        self._item.setRotation(r)
        event.accept()

# ------------------------------
# Pixmap item that notifies on transform changes
# ------------------------------

class NotifyingPixmapItem(QGraphicsPixmapItem):
    def __init__(self, pm: QPixmap, on_change_cb=None):
        super().__init__(pm)
        self._on_change_cb = on_change_cb
        self.setFlag(QGraphicsPixmapItem.ItemSendsGeometryChanges, True)
    def itemChange(self, change, value):
        ret = super().itemChange(change, value)
        if change in (QGraphicsPixmapItem.ItemPositionHasChanged,
                      QGraphicsPixmapItem.ItemRotationHasChanged,
                      QGraphicsPixmapItem.ItemScaleHasChanged):
            if callable(self._on_change_cb):
                self._on_change_cb()
        return ret

# ------------------------------
# Canvas
# ------------------------------

class Canvas(QGraphicsView):
    MODE_IDLE = 0
    MODE_DRAW_POLY = 1

    polygon_finished = Signal(bool)
    end_segment_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)  # NN preview
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.base_bgr = None
        self.base_preview_bgr = None
        self.base_item = None
        self.layers: List[Layer] = []
        self.current_layer: Optional[Layer] = None
        self.layer_index = 0

        self.mode = Canvas.MODE_IDLE
        self.temp_points: List[QPointF] = []
        self.temp_path_item: Optional[QGraphicsPathItem] = None
        self.first_click_marker: Optional[QGraphicsEllipseItem] = None

        self.fit_mode_combo = None
        self.target_w = 720
        self.target_h = 480

        # hue preview for current segment (degrees)
        self.current_segment_hue_deg: float = 0.0

        # Demo playback
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self.play_frames: List[np.ndarray] = []
        self.play_index = 0
        self.player_item: Optional[QGraphicsPixmapItem] = None

        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


    # ------------ small helpers ------------
    def _remove_if_in_scene(self, item):
        if item is None:
            return
        try:
            sc = item.scene()
            if sc is None:
                return  # already detached or removed
            if sc is self.scene:
                self.scene.removeItem(item)
            else:
                # If the item belongs to a different scene, remove it from THAT scene.
                sc.removeItem(item)
        except Exception:
            pass


    def _apply_pose_from_origin_scene(self, item, origin_scene_qp: QPointF, rot: float, scale: float):
        item.setRotation(float(rot))
        item.setScale(float(scale) if scale != 0 else 1.0)
        new_origin = item.mapToScene(item.transformOriginPoint())
        d = origin_scene_qp - new_origin
        item.setPos(item.pos() + d)

    # ------------ Icons ------------
    def make_pentagon_icon(self) -> QIcon:
        pm = QPixmap(22, 22)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(40, 40, 40)); pen.setWidth(2)
        p.setPen(pen)
        r = 8; cx, cy = 11, 11
        pts = []
        for i in range(5):
            ang = -90 + i * 72
            rad = np.radians(ang)
            pts.append(QPointF(cx + r * np.cos(rad), cy + r * np.sin(rad)))
        p.drawPolygon(QPolygonF(pts))
        p.end()
        return QIcon(pm)

    # -------- Fit helpers --------
    def _apply_fit(self, img: np.ndarray) -> np.ndarray:
        mode = 'Center Crop'
        if self.fit_mode_combo is not None:
            mode = self.fit_mode_combo.currentText()
        if mode == 'Center Pad':
            return fit_center_pad(img, self.target_h, self.target_w, interpolation=cv2.INTER_NEAREST)
        else:
            return resize_then_center_crop(img, self.target_h, self.target_w, interpolation=cv2.INTER_NEAREST)

    def _refresh_inpaint_preview(self):
        if self.base_bgr is None:
            return
        H, W = self.base_bgr.shape[:2]
        total_mask = np.zeros((H, W), dtype=bool)
        for L in self.layers:
            if not L.has_polygon() or L.is_external:
                continue
            poly0 = L.polygon_xy.astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly0], 255)
            total_mask |= (mask > 0)
        inpainted = inpaint_background(self.base_bgr, total_mask)
        self.base_preview_bgr = inpainted.copy()
        if self.base_item is not None:
            self.base_item.setPixmap(np_bgr_to_qpixmap(self.base_preview_bgr))

    # -------- Scene expansion helpers --------
    def _expand_scene_to_item(self, item: QtWidgets.QGraphicsItem, margin: int = 120, center: bool = True):
        if item is None:
            return
        try:
            local_rect = item.boundingRect()
            poly = item.mapToScene(local_rect)
            r = poly.boundingRect()
        except Exception:
            r = self.scene.sceneRect()
        sr = self.scene.sceneRect().united(r.adjusted(-margin, -margin, margin, margin))
        self.scene.setSceneRect(sr)
        self.ensureVisible(r.adjusted(-20, -20, 20, 20))
        if center:
            try:
                self.centerOn(item)
            except Exception:
                pass

    # -------- Base image --------
    def set_base_image(self, bgr_original: np.ndarray):
        self.scene.clear()
        for L in self.layers:
            L.handle_items.clear(); L.path_lines.clear(); L.preview_line = None
        self.layers.clear(); self.current_layer = None
        base_for_save = resize_then_center_crop(bgr_original, self.target_h, self.target_w, interpolation=cv2.INTER_AREA)
        self.base_bgr = base_for_save.copy()
        self.base_preview_bgr = self._apply_fit(bgr_original)
        pm = np_bgr_to_qpixmap(self.base_preview_bgr)
        self.base_item = self.scene.addPixmap(pm)
        self.base_item.setZValue(0)
        self.base_item.setTransformationMode(Qt.FastTransformation)  # NN
        self.setSceneRect(0, 0, pm.width(), pm.height())

    # -------- External sprite layer (no keyframe yet) --------
    def add_external_sprite_layer(self, raw_bgr: np.ndarray) -> 'Layer':
        if self.base_bgr is None:
            return None
        H, W = self.base_bgr.shape[:2]
        h0, w0 = raw_bgr.shape[:2]
        target_h = int(0.6 * H)
        scale = target_h / float(h0)
        ew = int(round(w0 * scale))
        eh = int(round(h0 * scale))
        ext_small = cv2.resize(raw_bgr, (ew, eh), interpolation=cv2.INTER_AREA)

        # pack onto same canvas size as base
        px = (W - ew) // 2
        py = (H - eh) // 2
        source_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = max(px, 0); y0 = max(py, 0)
        x1 = min(px + ew, W); y1 = min(py + eh, H)
        if x0 < x1 and y0 < y1:
            sx0 = x0 - px; sy0 = y0 - py
            sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
            source_bgr[y0:y1, x0:x1] = ext_small[sy0:sy1, sx0:sx1]

        rect_poly = np.array([[px, py], [px+ew, py], [px+ew, py+eh], [px, py+eh]], dtype=np.float32)
        cx, cy = px + ew/2.0, py + eh/2.0

        mask_rect = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_rect, [rect_poly.astype(np.int32)], 255)
        rgba = np.dstack([cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB), mask_rect])
        pm = np_rgba_to_qpixmap(rgba)

        color = PALETTE[self.layer_index % len(PALETTE)]; self.layer_index += 1
        L = Layer(name=f"Layer {len(self.layers)+1} (ext)", source_bgr=source_bgr, is_external=True,
                  polygon_xy=rect_poly.copy(), origin_local_xy=np.array([cx, cy], dtype=np.float32), color=color)
        self.layers.append(L); self.current_layer = L

        def on_change():
            if L.keyframes:
                self._ensure_preview_line(L)
            self._relayout_handles(L)

        item = NotifyingPixmapItem(pm, on_change_cb=on_change)
        item.setZValue(10 + len(self.layers))
        item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)   # place first
        item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
        item.setTransformationMode(Qt.FastTransformation)
        item.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
        item.setTransformOriginPoint(QPointF(cx, cy))
        self.scene.addItem(item); L.pixmap_item = item

        # start slightly to the left, still visible
        min_vis = min(max(40, ew // 5), W // 2)
        outside_x = (min_vis - (px + ew))
        item.setPos(outside_x, 0)

        # rect outline + handles (for initial placement)
        qpath = QPainterPath(QPointF(rect_poly[0,0], rect_poly[0,1]))
        for i in range(1, len(rect_poly)): qpath.lineTo(QPointF(rect_poly[i,0], rect_poly[i,1]))
        qpath.closeSubpath()
        outline = QGraphicsPathItem(qpath, parent=item)
        outline.setPen(QPen(L.color, 2, Qt.DashLine))
        outline.setZValue(item.zValue() + 1)
        L.outline_item = outline
        self._create_handles_for_layer(L)

        self._expand_scene_to_item(item, center=True)
        return L

    def place_external_initial_keyframe(self, L: 'Layer'):
        if not (L and L.pixmap_item): return
        origin_scene = L.pixmap_item.mapToScene(L.pixmap_item.transformOriginPoint())
        L.keyframes.append(Keyframe(
            pos=np.array([origin_scene.x(), origin_scene.y()], dtype=np.float32),
            rot_deg=float(L.pixmap_item.rotation()),
            scale=float(L.pixmap_item.scale()) if L.pixmap_item.scale()!=0 else 1.0,
            hue_deg=0.0
        ))
        self._ensure_preview_line(L)

    # -------- Polygon authoring --------
    def new_layer_from_source(self, name: str, source_bgr: np.ndarray, is_external: bool):
        color = PALETTE[self.layer_index % len(PALETTE)]; self.layer_index += 1
        layer = Layer(name=name, source_bgr=source_bgr.copy(), is_external=is_external, color=color)
        self.layers.append(layer); self.current_layer = layer
        self.start_draw_polygon(preserve_motion=False)

    def start_draw_polygon(self, preserve_motion: bool):
        L = self.current_layer
        if L is None: return
        if preserve_motion:
            for it in L.handle_items:
                it.setVisible(False)
        else:
            # We are re-drawing on the current (base) layer: clear visuals first.
            # REMOVE CHILDREN FIRST (outline/handles/preview), THEN parent pixmap.
            if L.outline_item is not None:
                self._remove_if_in_scene(L.outline_item)
                L.outline_item = None
            for it in L.handle_items:
                self._remove_if_in_scene(it)
            L.handle_items = []
            if L.preview_line is not None:
                self._remove_if_in_scene(L.preview_line)
                L.preview_line = None
            if L.pixmap_item is not None:
                self._remove_if_in_scene(L.pixmap_item)
                L.pixmap_item = None

            L.path_lines = []
            L.keyframes.clear()
        L.polygon_xy = None
        self.mode = Canvas.MODE_DRAW_POLY
        self.temp_points = []
        if self.temp_path_item is not None:
            self.scene.removeItem(self.temp_path_item); self.temp_path_item = None
        if self.first_click_marker is not None:
            self.scene.removeItem(self.first_click_marker); self.first_click_marker = None
        # reset hue preview for new segment
        self.current_segment_hue_deg = 0.0

    def _compute_ext_rect_from_source(self, src_bgr: np.ndarray) -> np.ndarray:
        ys, xs = np.where(np.any(src_bgr != 0, axis=2))
        if len(xs) == 0 or len(ys) == 0:
            return np.array([[0,0],[0,0],[0,0],[0,0]], dtype=np.float32)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)

    def _make_rgba_from_bgr_and_maskpoly(self, bgr: np.ndarray, poly: np.ndarray) -> np.ndarray:
        H, W = bgr.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        if poly is not None and poly.size:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask])
        return rgba

    def _add_static_external_item(self, bgr_inpainted: np.ndarray, rect_poly: np.ndarray,
                                  kf0: 'Keyframe', z_under: float, color: QColor) -> QGraphicsPixmapItem:
        rgba = self._make_rgba_from_bgr_and_maskpoly(bgr_inpainted, rect_poly)
        pm = np_rgba_to_qpixmap(rgba)
        item = QGraphicsPixmapItem(pm)
        item.setZValue(max(1.0, z_under - 0.2))
        item.setTransformationMode(Qt.FastTransformation)
        item.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
        cx = (rect_poly[:,0].min() + rect_poly[:,0].max())/2.0
        cy = (rect_poly[:,1].min() + rect_poly[:,1].max())/2.0
        item.setTransformOriginPoint(QPointF(cx, cy))
        self.scene.addItem(item)
        self._apply_pose_from_origin_scene(item, QPointF(kf0.pos[0], kf0.pos[1]), kf0.rot_deg, kf0.scale)
        path = QPainterPath(QPointF(rect_poly[0,0], rect_poly[0,1]))
        for i in range(1, len(rect_poly)): path.lineTo(QPointF(rect_poly[i,0], rect_poly[i,1]))
        path.closeSubpath()
        outline = QGraphicsPathItem(path, parent=item)
        outline.setPen(QPen(color, 1, Qt.DashLine))
        outline.setZValue(item.zValue() + 0.01)
        self._expand_scene_to_item(item, center=False)
        return item

    def _update_current_item_hue_preview(self):
        """Live hue preview for current moving polygon sprite."""
        L = self.current_layer
        if not (L and L.pixmap_item and L.polygon_xy is not None):
            return
        # Rebuild pixmap of moving item with current hue (full strength preview)
        bgr = L.source_bgr
        if abs(self.current_segment_hue_deg) > 1e-6:
            bgr = apply_hue_shift_bgr(bgr, self.current_segment_hue_deg)
        rgba = self._make_rgba_from_bgr_and_maskpoly(bgr, L.polygon_xy)
        L.pixmap_item.setPixmap(np_rgba_to_qpixmap(rgba))

    def finish_polygon(self, preserve_motion: bool) -> bool:
        L = self.current_layer
        if L is None or self.mode != Canvas.MODE_DRAW_POLY: return False
        if len(self.temp_points) < 3: return False

        pts_scene = [QtCore.QPointF(p) for p in self.temp_points]

        if preserve_motion and L.pixmap_item is not None and L.is_external:
            # ===== EXTERNAL split: remove old rect item, add static rect-with-hole and new moving polygon =====
            old_item = L.pixmap_item

            # polygon in the old item's LOCAL coords (source_bgr space)
            pts_local_qt = [old_item.mapFromScene(p) for p in pts_scene]
            pts_local = np.array([[p.x(), p.y()] for p in pts_local_qt], dtype=np.float32)

            # origin for moving poly = polygon bbox center
            x0, y0 = pts_local.min(axis=0)
            x1, y1 = pts_local.max(axis=0)
            cx_local, cy_local = (x0 + x1) / 2.0, (y0 + y1) / 2.0

            rect_poly_prev = (L.polygon_xy.copy()
                              if (L.polygon_xy is not None and len(L.polygon_xy) >= 3)
                              else self._compute_ext_rect_from_source(L.source_bgr))

            # cache pose / z
            old_origin_scene = old_item.mapToScene(old_item.transformOriginPoint())
            old_rot = old_item.rotation()
            old_scale = old_item.scale() if old_item.scale() != 0 else 1.0
            old_z = old_item.zValue()

            # build RGBA for moving polygon & static rect-with-hole
            H, W = L.source_bgr.shape[:2]
            rgb_full = cv2.cvtColor(L.source_bgr, cv2.COLOR_BGR2RGB)

            mov_mask = np.zeros((H, W), dtype=np.uint8); cv2.fillPoly(mov_mask, [pts_local.astype(np.int32)], 255)
            mov_rgba = np.dstack([rgb_full, mov_mask])

            hole_mask = np.zeros((H, W), dtype=np.uint8); cv2.fillPoly(hole_mask, [pts_local.astype(np.int32)], 255)
            inpainted_ext = inpaint_background(L.source_bgr, hole_mask > 0)
            rect_mask = np.zeros((H, W), dtype=np.uint8); cv2.fillPoly(rect_mask, [rect_poly_prev.astype(np.int32)], 255)
            static_rgba = np.dstack([cv2.cvtColor(inpainted_ext, cv2.COLOR_BGR2RGB), rect_mask])

            # remove old outline/handles and the old item itself
            if L.outline_item is not None:
                self._remove_if_in_scene(L.outline_item); L.outline_item = None
            for it in L.handle_items:
                self._remove_if_in_scene(it)
            L.handle_items = []
            self._remove_if_in_scene(old_item)

            # STATIC rect (non-movable, below)
            kf0 = L.keyframes[0] if L.keyframes else Keyframe(
                pos=np.array([old_origin_scene.x(), old_origin_scene.y()], dtype=np.float32),
                rot_deg=old_rot, scale=old_scale, hue_deg=0.0
            )
            static_item = QGraphicsPixmapItem(np_rgba_to_qpixmap(static_rgba))
            static_item.setZValue(max(1.0, old_z - 0.2))
            static_item.setTransformationMode(Qt.FastTransformation)
            static_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
            rcx = (rect_poly_prev[:,0].min() + rect_poly_prev[:,0].max())/2.0
            rcy = (rect_poly_prev[:,1].min() + rect_poly_prev[:,1].max())/2.0
            static_item.setTransformOriginPoint(QPointF(rcx, rcy))
            self.scene.addItem(static_item)
            self._apply_pose_from_origin_scene(static_item, QPointF(kf0.pos[0], kf0.pos[1]), kf0.rot_deg, kf0.scale)
            # dashed outline for static
            qpath_rect = QPainterPath(QPointF(rect_poly_prev[0,0], rect_poly_prev[0,1]))
            for i in range(1, len(rect_poly_prev)): qpath_rect.lineTo(QPointF(rect_poly_prev[i,0], rect_poly_prev[i,1]))
            qpath_rect.closeSubpath()
            outline_static = QGraphicsPathItem(qpath_rect, parent=static_item)
            outline_static.setPen(QPen(L.color, 1, Qt.DashLine))
            outline_static.setZValue(static_item.zValue() + 0.01)

            # NEW MOVING polygon (on top)
            def on_change():
                if L.keyframes:
                    self._ensure_preview_line(L)
                self._relayout_handles(L)

            poly_item = NotifyingPixmapItem(np_rgba_to_qpixmap(mov_rgba), on_change_cb=on_change)
            poly_item.setZValue(old_z + 0.2)
            poly_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
            poly_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
            poly_item.setTransformationMode(Qt.FastTransformation)
            poly_item.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
            poly_item.setTransformOriginPoint(QPointF(cx_local, cy_local))
            self.scene.addItem(poly_item)
            self._apply_pose_from_origin_scene(poly_item, old_origin_scene, old_rot, old_scale)

            # outline/handles on moving polygon
            qpath = QPainterPath(QPointF(pts_local[0,0], pts_local[0,1]))
            for i in range(1, len(pts_local)): qpath.lineTo(QPointF(pts_local[i,0], pts_local[i,1]))
            qpath.closeSubpath()
            outline_move = QGraphicsPathItem(qpath, parent=poly_item)
            outline_move.setPen(QPen(L.color, 2))
            outline_move.setZValue(poly_item.zValue() + 1)

            L.polygon_xy = pts_local
            L.origin_local_xy = np.array([cx_local, cy_local], dtype=np.float32)
            L.pixmap_item = poly_item
            L.outline_item = outline_move
            self._create_handles_for_layer(L)
            self._ensure_preview_line(L)

            # live hue preview starts neutral for new poly
            self.current_segment_hue_deg = 0.0
            self._expand_scene_to_item(poly_item, center=True)

        else:
            # ===== BASE image polygon path =====
            pts = np.array([[p.x(), p.y()] for p in pts_scene], dtype=np.float32)
            x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
            cx, cy = (x0+x1)/2.0, (y0+y1)/2.0

            L.polygon_xy = pts
            L.origin_local_xy = np.array([cx, cy], dtype=np.float32)

            rgb = cv2.cvtColor(L.source_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            rgba = np.dstack([rgb, mask])
            pm = np_rgba_to_qpixmap(rgba)

            def on_change():
                if L.keyframes:
                    self._ensure_preview_line(L)
                self._relayout_handles(L)
            item = NotifyingPixmapItem(pm, on_change_cb=on_change)
            item.setZValue(10 + len(self.layers))
            item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
            item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
            item.setTransformationMode(Qt.FastTransformation)
            item.setShapeMode(QGraphicsPixmapItem.ShapeMode.MaskShape)
            item.setTransformOriginPoint(QPointF(cx, cy))
            self.scene.addItem(item); L.pixmap_item = item

            qpath = QPainterPath(QPointF(pts[0,0], pts[0,1]))
            for i in range(1, len(pts)): qpath.lineTo(QPointF(pts[i,0], pts[i,1]))
            qpath.closeSubpath()
            outline = QGraphicsPathItem(qpath, parent=item)
            outline.setPen(QPen(L.color, 2))
            outline.setZValue(item.zValue() + 1)
            L.outline_item = outline
            self._create_handles_for_layer(L)

            origin_scene = item.mapToScene(item.transformOriginPoint())
            L.keyframes.append(Keyframe(pos=np.array([origin_scene.x(), origin_scene.y()], dtype=np.float32),
                                        rot_deg=float(item.rotation()),
                                        scale=float(item.scale()) if item.scale()!=0 else 1.0,
                                        hue_deg=0.0))

        if not (L.is_external):
            self._refresh_inpaint_preview()

        if self.temp_path_item is not None: self._remove_if_in_scene(self.temp_path_item); self.temp_path_item = None
        if self.first_click_marker is not None: self._remove_if_in_scene(self.first_click_marker); self.first_click_marker = None
        self.temp_points = []
        self.mode = Canvas.MODE_IDLE
        # ensure the current hue preview is applied (neutral at first)
        self._update_current_item_hue_preview()
        return True

    # -------- UI helpers --------
    def _create_handles_for_layer(self, L: Layer):
        if L.polygon_xy is None or L.pixmap_item is None:
            return
        x0, y0 = L.polygon_xy.min(axis=0)
        x1, y1 = L.polygon_xy.max(axis=0)
        corners = [QPointF(x0,y0), QPointF(x1,y0), QPointF(x1,y1), QPointF(x0,y1)]
        top_center = QPointF((x0+x1)/2.0, y0)
        rot_pos = QPointF(top_center.x(), top_center.y() - 24)

        box_path = QPainterPath(corners[0])
        for p in corners[1:]:
            box_path.lineTo(p)
        box_path.closeSubpath()
        # bbox (dashed) around the polygon
        bbox_item = QGraphicsPathItem(box_path, parent=L.pixmap_item)
        pen = QPen(L.color, 1, Qt.DashLine)
        pen.setCosmetic(True)
        bbox_item.setPen(pen)
        bbox_item.setZValue(L.pixmap_item.zValue() + 0.5)
        L.handle_items.append(bbox_item)

        for c in corners:
            h = ScaleHandle(6, L.color, parent=L.pixmap_item)
            h.setPos(c); h.set_item(L.pixmap_item)
            L.handle_items.append(h)
        rot_dot = RotateHandle(6, L.color, parent=L.pixmap_item)
        rot_dot.setPos(rot_pos); rot_dot.set_item(L.pixmap_item)
        L.handle_items.append(rot_dot)
        tether = QGraphicsLineItem(QtCore.QLineF(top_center, rot_pos), L.pixmap_item)
        pen_tether = QPen(L.color, 1)
        pen_tether.setCosmetic(True)
        tether.setPen(pen_tether)
        tether.setZValue(L.pixmap_item.zValue() + 0.4)
        L.handle_items.append(tether)

    def _relayout_handles(self, L: Layer):
        if L.polygon_xy is None or L.pixmap_item is None or not L.handle_items:
            return
        x0, y0 = L.polygon_xy.min(axis=0); x1, y1 = L.polygon_xy.max(axis=0)
        corners = [QPointF(x0,y0), QPointF(x1,y0), QPointF(x1,y1), QPointF(x0,y1)]
        top_center = QPointF((x0+x1)/2.0, y0)
        rot_pos = QPointF(top_center.x(), top_center.y() - 24)
        bbox_item = L.handle_items[0]
        if isinstance(bbox_item, QGraphicsPathItem):
            box_path = QPainterPath(corners[0])
            for p in corners[1:]: box_path.lineTo(p)
            box_path.closeSubpath(); bbox_item.setPath(box_path)
        for i in range(4):
            h = L.handle_items[1+i]
            if isinstance(h, QGraphicsEllipseItem):
                h.setPos(corners[i])
        rot_dot = L.handle_items[5]
        if isinstance(rot_dot, QGraphicsEllipseItem):
            rot_dot.setPos(rot_pos)
        tether = L.handle_items[6]
        if isinstance(tether, QGraphicsLineItem):
            tether.setLine(QtCore.QLineF(top_center, rot_pos))

    def _ensure_preview_line(self, L: Layer):
        if L.pixmap_item is None or not L.keyframes:
            return
        origin_scene = L.pixmap_item.mapToScene(L.pixmap_item.transformOriginPoint())
        p0 = L.keyframes[-1].pos
        p1 = np.array([origin_scene.x(), origin_scene.y()], dtype=np.float32)
        if L.preview_line is None:
            line = QGraphicsLineItem(p0[0], p0[1], p1[0], p1[1])
            line.setPen(QPen(L.color, 1, Qt.DashLine))
            line.setZValue(950)
            self.scene.addItem(line)
            L.preview_line = line
        else:
            L.preview_line.setLine(p0[0], p0[1], p1[0], p1[1])

    def _update_temp_path_item(self, color: QColor):
        if self.temp_path_item is None:
            self.temp_path_item = QGraphicsPathItem()
            pen = QPen(color, 2)
            self.temp_path_item.setPen(pen)
            self.temp_path_item.setZValue(1000)
            self.scene.addItem(self.temp_path_item)
        if not self.temp_points:
            self.temp_path_item.setPath(QPainterPath())
            return
        path = QPainterPath(self.temp_points[0])
        for p in self.temp_points[1:]:
            path.lineTo(p)
        path.lineTo(self.temp_points[0])
        self.temp_path_item.setPath(path)

    # -------------- Mouse / Keys for polygon drawing --------------
    def mousePressEvent(self, event):
        # Right-click = End Segment (ONLY when not drawing a polygon)
        if self.mode != Canvas.MODE_DRAW_POLY and event.button() == Qt.RightButton:
            self.end_segment_requested.emit()
            event.accept()
            return

        if self.mode == Canvas.MODE_DRAW_POLY:
            try:
                p = event.position()
                scene_pos = self.mapToScene(int(p.x()), int(p.y()))
            except AttributeError:
                scene_pos = self.mapToScene(event.pos())

            if event.button() == Qt.LeftButton:
                self.temp_points.append(scene_pos)
                if len(self.temp_points) == 1:
                    if self.first_click_marker is not None:
                        self.scene.removeItem(self.first_click_marker)
                    self.first_click_marker = QGraphicsEllipseItem(-3, -3, 6, 6)
                    self.first_click_marker.setBrush(QColor(0, 220, 0))
                    self.first_click_marker.setPen(QPen(QColor(0, 0, 0), 1))
                    self.first_click_marker.setZValue(1200)
                    self.scene.addItem(self.first_click_marker)
                    self.first_click_marker.setPos(scene_pos)
                color = self.current_layer.color if self.current_layer else QColor(255, 0, 0)
                self._update_temp_path_item(color)
            elif event.button() == Qt.RightButton:
                # Finish polygon with right-click
                preserve = (
                    self.current_layer is not None
                    and self.current_layer.pixmap_item is not None
                    and self.current_layer.is_external
                )
                ok = self.finish_polygon(preserve_motion=preserve)
                self.polygon_finished.emit(ok)
                if not ok:
                    QMessageBox.information(self, "Polygon", "Need at least 3 points.")
                event.accept()
                return
                
            return

        super().mousePressEvent(event)


    def mouseDoubleClickEvent(self, event):
        if self.mode == Canvas.MODE_DRAW_POLY:
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.mode == Canvas.MODE_DRAW_POLY:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                # Polygons are finished with right-click now; ignore Enter.
                return
            elif event.key() == Qt.Key_Backspace:
                if self.temp_points:
                    self.temp_points.pop()
                    color = self.current_layer.color if self.current_layer else QColor(255,0,0)
                    self._update_temp_path_item(color)
                return
            elif event.key() == Qt.Key_Escape:
                if self.temp_path_item is not None: self.scene.removeItem(self.temp_path_item); self.temp_path_item = None
                if self.first_click_marker is not None: self.scene.removeItem(self.first_click_marker); self.first_click_marker = None
                self.temp_points = []
                self.mode = Canvas.MODE_IDLE
                return
        super().keyPressEvent(event)

    # keyframes
    def end_segment_add_keyframe(self):
        if not (self.current_layer and self.current_layer.pixmap_item and (self.current_layer.polygon_xy is not None) and self.current_layer.keyframes):
            return False
        item = self.current_layer.pixmap_item
        origin_scene = item.mapToScene(item.transformOriginPoint())
        kf = Keyframe(
            pos=np.array([origin_scene.x(), origin_scene.y()], dtype=np.float32),
            rot_deg=float(item.rotation()),
            scale=float(item.scale()) if item.scale()!=0 else 1.0,
            hue_deg=float(self.current_segment_hue_deg)
        )
        L = self.current_layer
        if len(L.keyframes) >= 1:
            p0 = L.keyframes[-1].pos; p1 = kf.pos
            if L.preview_line is not None:
                self.scene.removeItem(L.preview_line); L.preview_line = None
            line = QGraphicsLineItem(p0[0], p0[1], p1[0], p1[1])
            line.setPen(QPen(L.color, 2)); line.setZValue(900); self.scene.addItem(line)
            L.path_lines.append(line)
        L.keyframes.append(kf)
        self._ensure_preview_line(L)
        # reset hue for next leg
        self.current_segment_hue_deg = 0.0
        # refresh preview back to neutral
        self._update_current_item_hue_preview()
        return True

    def has_pending_transform(self) -> bool:
        L = self.current_layer
        if not (L and L.pixmap_item and L.keyframes): return False
        last = L.keyframes[-1]
        item = L.pixmap_item
        origin_scene = item.mapToScene(item.transformOriginPoint())
        pos = np.array([origin_scene.x(), origin_scene.y()], dtype=np.float32)
        dpos = np.linalg.norm(pos - last.pos)
        drot = abs(float(item.rotation()) - last.rot_deg)
        dscale = abs((float(item.scale()) if item.scale()!=0 else 1.0) - last.scale)
        # hue preview doesn’t count as a “transform” until you end the segment
        return (dpos > 0.5) or (drot > 0.1) or (dscale > 1e-3)

    def revert_to_last_keyframe(self, L: Optional[Layer] = None):
        if L is None: L = self.current_layer
        if not (L and L.pixmap_item and L.keyframes): return
        last = L.keyframes[-1]
        item = L.pixmap_item
        item.setRotation(last.rot_deg); item.setScale(last.scale)
        origin_scene = item.mapToScene(item.transformOriginPoint())
        d = QPointF(last.pos[0]-origin_scene.x(), last.pos[1]-origin_scene.y())
        item.setPos(item.pos() + d)
        self._ensure_preview_line(L)
        # restore hue preview to last keyframe hue
        self.current_segment_hue_deg = last.hue_deg
        self._update_current_item_hue_preview()

    def _sample_keyframes_constant_speed_with_seg(self, keyframes: List[Keyframe], T: int):
        """
        Allocate frames to segments proportional to their Euclidean length so that
        translation happens at constant speed across the whole path.
        Returns (pos[T,2], scl[T], rot[T], seg_idx[T], t[T]).
        """
        K = len(keyframes)
        assert K >= 1
        import math
        if T <= 0:
            # degenerate: return just the first pose
            p0 = keyframes[0].pos.astype(np.float32)
            return (np.repeat(p0[None, :], 0, axis=0),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32),
                    np.zeros((0,), np.float32))

        if K == 1:
            p0 = keyframes[0].pos.astype(np.float32)
            pos = np.repeat(p0[None, :], T, axis=0)
            scl = np.full((T,), float(keyframes[0].scale), dtype=np.float32)
            rot = np.full((T,), float(keyframes[0].rot_deg), dtype=np.float32)
            seg_idx = np.zeros((T,), dtype=np.int32)
            t = np.zeros((T,), dtype=np.float32)
            return pos, scl, rot, seg_idx, t

        # Segment lengths (translation only)
        P = np.array([kf.pos for kf in keyframes], dtype=np.float32)  # [K,2]
        seg_vec = P[1:] - P[:-1]                                      # [K-1,2]
        lengths = np.linalg.norm(seg_vec, axis=1)                     # [K-1]
        total_len = float(lengths.sum())

        def _per_seg_counts_uniform():
            # fallback: equal frames per segment
            base = np.zeros((K-1,), dtype=np.int32)
            if T > 0:
                # spread as evenly as possible
                q, r = divmod(T, K-1)
                base[:] = q
                base[:r] += 1
            return base

        if total_len <= 1e-6:
            counts = _per_seg_counts_uniform()
        else:
            # Proportional allocation by length, rounded with largest-remainder
            raw = (lengths / total_len) * T
            base = np.floor(raw).astype(np.int32)
            remainder = T - int(base.sum())
            if remainder > 0:
                order = np.argsort(-(raw - base))  # largest fractional parts first
                base[order[:remainder]] += 1
            counts = base  # may contain zeros for ~zero-length segments

        # Build arrays
        pos_list, scl_list, rot_list, seg_idx_list, t_list = [], [], [], [], []

        for s in range(K - 1):
            n = int(counts[s])
            if n <= 0:
                continue
            # Local times in [0,1) to avoid s+1 overflow in hue blending
            ts = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

            p0, p1 = P[s], P[s + 1]
            s0, s1 = max(1e-6, float(keyframes[s].scale)), max(1e-6, float(keyframes[s + 1].scale))
            r0, r1 = float(keyframes[s].rot_deg), float(keyframes[s + 1].rot_deg)

            # Interpolate
            pos_seg = (1 - ts)[:, None] * p0[None, :] + ts[:, None] * p1[None, :]
            scl_seg = np.exp((1 - ts) * math.log(s0) + ts * math.log(s1))
            rot_seg = (1 - ts) * r0 + ts * r1

            pos_list.append(pos_seg.astype(np.float32))
            scl_list.append(scl_seg.astype(np.float32))
            rot_list.append(rot_seg.astype(np.float32))
            seg_idx_list.append(np.full((n,), s, dtype=np.int32))
            t_list.append(ts.astype(np.float32))

        # If counts summed to < T (can happen if T < #segments), pad with final pose of last used seg
        N = sum(int(c) for c in counts)
        if N < T:
            # use final keyframe as hold
            p_end = P[-1].astype(np.float32)
            extra = T - N
            pos_list.append(np.repeat(p_end[None, :], extra, axis=0))
            scl_list.append(np.full((extra,), float(keyframes[-1].scale), dtype=np.float32))
            rot_list.append(np.full((extra,), float(keyframes[-1].rot_deg), dtype=np.float32))
            # Use the last real segment index (K-2), with t=0 (blend start of final seg)
            seg_idx_list.append(np.full((extra,), max(0, K - 2), dtype=np.int32))
            t_list.append(np.zeros((extra,), dtype=np.float32))

        pos = np.vstack(pos_list) if pos_list else np.zeros((T, 2), dtype=np.float32)
        scl = np.concatenate(scl_list) if scl_list else np.zeros((T,), dtype=np.float32)
        rot = np.concatenate(rot_list) if rot_list else np.zeros((T,), dtype=np.float32)
        seg_idx = np.concatenate(seg_idx_list) if seg_idx_list else np.zeros((T,), dtype=np.int32)
        t = np.concatenate(t_list) if t_list else np.zeros((T,), dtype=np.float32)

        # Truncate in case of rounding over-alloc (very rare), or pad if still short
        if len(pos) > T:
            pos, scl, rot, seg_idx, t = pos[:T], scl[:T], rot[:T], seg_idx[:T], t[:T]
        elif len(pos) < T:
            pad = T - len(pos)
            pos = np.vstack([pos, np.repeat(pos[-1:,:], pad, axis=0)])
            scl = np.concatenate([scl, np.repeat(scl[-1:], pad)])
            rot = np.concatenate([rot, np.repeat(rot[-1:], pad)])
            seg_idx = np.concatenate([seg_idx, np.repeat(seg_idx[-1:], pad)])
            t = np.concatenate([t, np.repeat(t[-1:], pad)])

        return pos.astype(np.float32), scl.astype(np.float32), rot.astype(np.float32), seg_idx.astype(np.int32), t.astype(np.float32)


    def undo(self) -> bool:
        if self.mode == Canvas.MODE_DRAW_POLY and self.temp_points:
            self.temp_points.pop()
            color = self.current_layer.color if self.current_layer else QColor(255,0,0)
            self._update_temp_path_item(color)
            return True
        if self.has_pending_transform():
            self.revert_to_last_keyframe()
            return True
        if self.current_layer and len(self.current_layer.keyframes) > 1:
            L = self.current_layer
            if L.path_lines:
                line = L.path_lines.pop(); self.scene.removeItem(line)
            L.keyframes.pop()
            self.revert_to_last_keyframe(L)
            return True
        if self.current_layer:
            L = self.current_layer
            if (L.pixmap_item is not None) and (len(L.keyframes) <= 1) and (len(L.path_lines) == 0):
                if L.preview_line is not None:
                    self.scene.removeItem(L.preview_line); L.preview_line = None
                if L.outline_item is not None:
                    self.scene.removeItem(L.outline_item); L.outline_item = None
                for it in L.handle_items:
                    self.scene.removeItem(it)
                L.handle_items.clear()
                self.scene.removeItem(L.pixmap_item); L.pixmap_item = None
                try:
                    idx = self.layers.index(L)
                    self.layers.pop(idx)
                except ValueError:
                    pass
                self.current_layer = self.layers[-1] if self.layers else None
                if (L.is_external is False):
                    self._refresh_inpaint_preview()
                return True
        return False

    # -------- Demo playback (with hue crossfade) --------
    def build_preview_frames(self, T_total: int) -> Optional[List[np.ndarray]]:
        if self.base_bgr is None:
            return None
        H, W = self.base_bgr.shape[:2]
        total_mask = np.zeros((H, W), dtype=bool)
        for L in self.layers:
            if not L.has_polygon() or L.is_external:
                continue
            poly0 = L.polygon_xy.astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8); cv2.fillPoly(mask, [poly0], 255)
            total_mask |= (mask > 0)
        background = inpaint_background(self.base_bgr, total_mask)

        all_layer_frames = []
        has_any = False

        # def sample_keyframes_uniform_with_seg(keyframes: List[Keyframe], T: int):
        #     K = len(keyframes); assert K >= 1
        #     if K == 1:
        #         pos = np.repeat(keyframes[0].pos[None, :], T, axis=0).astype(np.float32)
        #         scl = np.full((T,), keyframes[0].scale, dtype=np.float32)
        #         rot = np.full((T,), keyframes[0].rot_deg, dtype=np.float32)
        #         seg_idx = np.zeros((T,), dtype=np.int32)
        #         t = np.zeros((T,), dtype=np.float32)
        #         return pos, scl, rot, seg_idx, t
        #     segs = K - 1
        #     u = np.linspace(0.0, float(segs), T, dtype=np.float32)
        #     seg_idx = np.minimum(np.floor(u).astype(int), segs - 1)
        #     t = u - seg_idx
        #     k0 = np.array([[keyframes[i].pos[0], keyframes[i].pos[1], keyframes[i].scale, keyframes[i].rot_deg] for i in seg_idx], dtype=np.float32)
        #     k1 = np.array([[keyframes[i+1].pos[0], keyframes[i+1].pos[1], keyframes[i+1].scale, keyframes[i+1].rot_deg] for i in seg_idx], dtype=np.float32)
        #     pos0 = k0[:, :2]; pos1 = k1[:, :2]
        #     s0 = np.maximum(1e-6, k0[:, 2]); s1 = np.maximum(1e-6, k1[:, 2])
        #     r0 = k0[:, 3]; r1 = k1[:, 3]
        #     pos = (1 - t)[:, None] * pos0 + t[:, None] * pos1
        #     scl = np.exp((1 - t) * np.log(s0) + t * np.log(s1))
        #     rot = (1 - t) * r0 + t * r1
        #     return pos.astype(np.float32), scl.astype(np.float32), rot.astype(np.float32), seg_idx, t

        for L in self.layers:
            if not L.has_polygon() or len(L.keyframes) < 2:
                continue
            has_any = True

            # path_xy, scales, rots, seg_idx, t = sample_keyframes_uniform_with_seg(L.keyframes, T_total)
            path_xy, scales, rots, seg_idx, t = self._sample_keyframes_constant_speed_with_seg(L.keyframes, T_total)

            origin_xy = L.origin_local_xy if L.origin_local_xy is not None else L.polygon_xy.mean(axis=0)

            # Precompute animations for each keyframe’s hue (crossfade per segment)
            K = len(L.keyframes)
            hue_values = [L.keyframes[k].hue_deg for k in range(K)]
            hue_to_frames: Dict[int, List[np.ndarray]] = {}
            for k in range(K):
                bgr_h = apply_hue_shift_bgr(L.source_bgr, hue_values[k])
                frames_h, _ = animate_polygon(bgr_h, L.polygon_xy, path_xy, scales, rots,
                                              interp=cv2.INTER_LINEAR, origin_xy=origin_xy)
                hue_to_frames[k] = frames_h

            # Mix per frame using seg_idx / t
            frames_rgba = []
            for i in range(T_total):
                s = int(seg_idx[i])
                w = float(t[i])
                A = hue_to_frames[s][i].astype(np.float32)
                B = hue_to_frames[s+1][i].astype(np.float32)
                mix = (1.0 - w) * A + w * B
                frames_rgba.append(np.clip(mix, 0, 255).astype(np.uint8))
            all_layer_frames.append(frames_rgba)

        if not has_any:
            return None
        frames_out = composite_frames(background, all_layer_frames)
        return frames_out

    def play_demo(self, fps: int, T_total: int):
        frames = self.build_preview_frames(T_total)
        if not frames:
            QMessageBox.information(self, "Play Demo", "Nothing to play yet. Add a polygon and keyframes.")
            return
        self.play_frames = frames
        self.play_index = 0
        if self.player_item is None:
            self.player_item = QGraphicsPixmapItem()
            self.player_item.setZValue(5000)
            self.scene.addItem(self.player_item)
        self.player_item.setVisible(True)
        self._on_play_tick()
        interval_ms = max(1, int(1000 / max(1, fps)))
        self.play_timer.start(interval_ms)

    def _on_play_tick(self):
        if not self.play_frames or self.play_index >= len(self.play_frames):
            self.play_timer.stop()
            if self.player_item is not None:
                self.player_item.setVisible(False)
            return
        frame = self.play_frames[self.play_index]
        self.play_index += 1
        self.player_item.setPixmap(np_bgr_to_qpixmap(frame))

# ------------------------------
# Main window / controls
# ------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time-to-Move: Cut & Drag")
        self.resize(1180, 840)

        self.canvas = Canvas(self)
        self.canvas.polygon_finished.connect(self._on_canvas_polygon_finished)
        self.canvas.end_segment_requested.connect(self.on_end_segment)

        # -------- Instruction banner above canvas (CENTERED) --------
        self.instruction_label = QLabel()
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                background: #f7f7fa;
                border-bottom: 1px solid #ddd;
                padding: 10px 12px;
                font-size: 15px;
                color: #222;
            }
        """)
        self._set_instruction("Welcome! • Select Image to begin.")

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0,0,0,0); v.setSpacing(0)
        v.addWidget(self.instruction_label)
        v.addWidget(self.canvas)
        self.setCentralWidget(central)

        # state: external placing mode?
        self.placing_external: bool = False
        self.placing_layer: Optional[Layer] = None

        # -------- Vertical toolbar on the LEFT --------
        tb = QToolBar("Tools")
        self.addToolBar(Qt.LeftToolBarArea, tb)
        tb.setOrientation(Qt.Vertical)

        def add_btn(text: str, slot, icon: Optional[QIcon] = None):
            btn = QPushButton(text)
            if icon: btn.setIcon(icon)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumWidth(180)
            btn.clicked.connect(slot)
            tb.addWidget(btn); return btn

        # Fit dropdown (default: Center Crop)
        self.cmb_fit = QComboBox(); self.cmb_fit.addItems(["Center Crop", "Center Pad"])
        tb.addWidget(self.cmb_fit)
        self.canvas.fit_mode_combo = self.cmb_fit

        # Dotted separator
        line_sep = QFrame(); line_sep.setFrameShape(QFrame.HLine); line_sep.setFrameShadow(QFrame.Plain)
        line_sep.setStyleSheet("color: #888; border-top: 1px dotted #888; margin: 8px 0;")
        tb.addWidget(line_sep)

        # Select Image
        self.btn_select = add_btn("🖼️ Select Image", self.on_select_base)

        # Add Polygon (toggles to Finish)
        self.pent_icon = self.canvas.make_pentagon_icon()
        self.btn_add_poly = add_btn("Add Polygon", self.on_add_polygon_toggled, icon=self.pent_icon)
        self.add_poly_active = False

        # Add External (two-step: file → Place)
        self.btn_add_external = add_btn("🖼️➕ Add External Image", self.on_add_or_place_external)

        # HUE TRANSFORM (slider + Default) ABOVE End Segment
        tb.addSeparator()
        tb.addWidget(QLabel("Hue Transform"))
        hue_row = QWidget(); row = QHBoxLayout(hue_row); row.setContentsMargins(0,0,0,0)
        self.sld_hue = QSlider(Qt.Horizontal); self.sld_hue.setRange(-180, 180); self.sld_hue.setValue(0)
        btn_default = QPushButton("Default"); btn_default.setCursor(Qt.PointingHandCursor); btn_default.setFixedWidth(70)
        row.addWidget(self.sld_hue, 1); row.addWidget(btn_default, 0)
        tb.addWidget(hue_row)
        self.sld_hue.valueChanged.connect(self.on_hue_changed)
        btn_default.clicked.connect(lambda: self.sld_hue.setValue(0))

        # End Segment and Undo
        self.btn_end_seg = add_btn("🎯 End Segment", self.on_end_segment)
        self.btn_undo   = add_btn("↩️ Undo", self.on_undo)

        tb.addSeparator()
        tb.addWidget(QLabel("Total Frames:"))
        self.spn_total_frames = QSpinBox(); self.spn_total_frames.setRange(1, 2000); self.spn_total_frames.setValue(81)
        tb.addWidget(self.spn_total_frames)
        tb.addWidget(QLabel("FPS:"))
        self.spn_fps = QSpinBox(); self.spn_fps.setRange(1, 120); self.spn_fps.setValue(16)
        tb.addWidget(self.spn_fps)

        tb.addSeparator()
        self.btn_play = add_btn("▶️ Play Demo", self.on_play_demo)
        tb.addWidget(QLabel("Prompt"))
        self.txt_prompt = QPlainTextEdit()
        self.txt_prompt.setFixedHeight(80)      # ~3–5 lines tall; tweak if you like
        self.txt_prompt.setMinimumWidth(180)    # matches your button width
        # (Optional) If your PySide6 supports it, you can uncomment the next line:
        # self.txt_prompt.setPlaceholderText("Write your prompt here…")
        tb.addWidget(self.txt_prompt)
        self.btn_save = add_btn("💾 Save", self.on_save)
        self.btn_new  = add_btn("🆕 New", self.on_new)
        self.btn_exit = add_btn("⏹️ Exit", self.close)

        # Status strip at bottom
        status = QToolBar("Status")
        self.addToolBar(Qt.BottomToolBarArea, status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label)

    # ---------- Instruction helper ----------
    def _set_instruction(self, text: str):
        self.instruction_label.setText(text)

    # ---------- Pending-segment guards ----------
    def _block_if_pending_segment(self, action_label: str) -> bool:
        if self.canvas.current_layer and self.canvas.has_pending_transform():
            QMessageBox.information(
                self, "Finish Segment",
                f"Please end the current segment (click '🎯 End Segment') before {action_label}."
            )
            self._set_instruction("Finish current segment: drag/scale/rotate as needed, adjust Hue, then click 🎯 End Segment.")
            return True
        return False

    # ------------- Actions -------------
    def on_select_base(self):
        if self._block_if_pending_segment("changing the base image"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.mov *.avi *.mkv)")
        if not path:
            self._set_instruction("No image selected. Click ‘Select Image’ to begin.")
            return
        try:
            raw = load_first_frame(path)
        except Exception as e:
            QMessageBox.critical(self, "Load", f"Failed to load: {e}")
            return
        self.canvas.set_base_image(raw)
        self.add_poly_active = False
        self.btn_add_poly.setText("Add Polygon")
        self.placing_external = False; self.placing_layer = None
        self.btn_add_external.setText("🖼️➕ Add External Image")
        self.status_label.setText("Base loaded.")
        self._set_instruction("Step 1: Add a polygon (Add Polygon), or add an external sprite (Add External Image).")

    def on_add_polygon_toggled(self):
        if self.placing_external:
            QMessageBox.information(self, "Place External First",
                                    "Please place the external image first (click ‘✅ Place External Image’).")
            self._set_instruction("Place External: drag/scale/rotate to choose starting pose, then click ‘✅ Place External Image’.")
            return

        if (not self.add_poly_active) and self._block_if_pending_segment("adding a polygon"):
            return

        if not self.add_poly_active:
            if self.canvas.base_bgr is None:
                QMessageBox.information(self, "Add Polygon", "Please select an image first.")
                self._set_instruction("Click ‘Select Image’ to begin.")
                return

            # --- KEY CHANGE ---
            # If there's no current layer OR the current layer is BASE -> make a NEW BASE layer (new color).
            # If the current layer is EXTERNAL -> split/cut that external (preserve motion).
            if (self.canvas.current_layer is None) or (not self.canvas.current_layer.is_external):
                # New polygon on the base image => new layer with a fresh color
                self.canvas.new_layer_from_source(
                    name=f"Layer {len(self.canvas.layers)+1}",
                    source_bgr=self.canvas.base_bgr,
                    is_external=False
                )
            else:
                # Current layer is external: go into "draw polygon to cut external" mode
                self.canvas.start_draw_polygon(preserve_motion=True)
            # --- END KEY CHANGE ---

            self.add_poly_active = True
            self.btn_add_poly.setText("✅ Finish Polygon Selection")
            self.status_label.setText("Drawing polygon…")
            self._set_instruction("Polygon mode: Left-click to add points. Backspace = undo point. Right-click = finish. Esc = cancel.")
        else:
            # Finish current polygon selection
            preserve = (self.canvas.current_layer is not None and
                        self.canvas.current_layer.pixmap_item is not None and
                        self.canvas.current_layer.is_external)
            ok = self.canvas.finish_polygon(preserve_motion=preserve)
            if not ok:
                QMessageBox.information(self, "Polygon", "Need at least 3 points (keep adding).")
                self._set_instruction("Keep adding polygon points (≥3). Right-click to finish.")
                return
            self.add_poly_active = False
            self.btn_add_poly.setText("Add Polygon")
            self.status_label.setText("Polygon ready.")
            self._set_instruction(
                "Drag to move, use corner circles to scale, top dot to rotate. "
                "Adjust Hue if you like, then click ‘🎯 End Segment’ or Right Click to record a move."
            )
    def on_add_or_place_external(self):
        # If we're already in "placing" mode, finalize the initial keyframe.
        if self.placing_external and self.placing_layer is not None:
            try:
                # Lock the initial pose as keyframe #1
                self.canvas.place_external_initial_keyframe(self.placing_layer)
                # Make sure this layer stays selected
                self.canvas.current_layer = self.placing_layer
                # Draw the dashed preview line if relevant
                self.canvas._ensure_preview_line(self.placing_layer)
            finally:
                self.placing_external = False
                self.placing_layer = None
                self.btn_add_external.setText("🖼️➕ Add External Image")
                self.status_label.setText("External starting pose locked.")
                self._set_instruction("Now drag/scale/rotate and click ‘🎯 End Segment’ to record movement.")
            return

        # Otherwise, begin adding a new external image.
        if self._block_if_pending_segment("adding an external image"):
            return
        if self.canvas.base_bgr is None:
            QMessageBox.information(self, "External", "Select a base image first.")
            self._set_instruction("Click ‘Select Image’ to begin.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select external image", "",
            "Images/Videos (*.png *.jpg *.jpeg *.bmp *.mp4 *.mov *.avi *.mkv)"
        )
        if not path:
            self._set_instruction("External not chosen. You can Add External Image later.")
            return

        try:
            raw = load_first_frame(path)
        except Exception as e:
            QMessageBox.critical(self, "Load", f"Failed to load external: {e}")
            return

        L = self.canvas.add_external_sprite_layer(raw)  # no keyframe yet
        if L is None:
            QMessageBox.critical(self, "External", "Failed to create external layer.")
            return

        self.placing_external = True
        self.placing_layer = L
        self.canvas.current_layer = L  # keep selection consistent
        self.btn_add_external.setText("✅ Place External Image")
        self.status_label.setText("Place external image.")
        self._set_instruction("Place External: drag into view, scale with corner circles, rotate with top dot. Then click ‘✅ Place External Image’.")


    def _on_canvas_polygon_finished(self, ok: bool):
        if ok:
            self.add_poly_active = False
            self.btn_add_poly.setText("Add Polygon")
            self.status_label.setText("Polygon ready.")
            self._set_instruction(
                "Drag to move, use corner circles to scale, top dot to rotate. "
                "Adjust Hue if you like, then click ‘🎯 End Segment’ or Right Click to record a move."
            )
        else:
            # keep your existing “need ≥3 points” behavior; nothing else to do here
            pass

    def on_hue_changed(self, val: int):
        self.canvas.current_segment_hue_deg = float(val)
        self.canvas._update_current_item_hue_preview()

    def on_end_segment(self):
        if self.placing_external:
            QMessageBox.information(self, "Place External First",
                                    "Please place the external image first (click ‘✅ Place External Image’).")
            self._set_instruction("Place External: drag/scale/rotate to choose starting pose, then click ‘✅ Place External Image’.")
            return
        ok = self.canvas.end_segment_add_keyframe()
        if ok:
            n = len(self.canvas.current_layer.keyframes) if self.canvas.current_layer else 0
            self.status_label.setText(f"Keyframe #{n} added.")
            self._set_instruction(
                "Segment added! Move again for the next leg, adjust Hue if you like, "
                "then click ‘🎯 End Segment’ or Right Click to record a move."
            )
        else:
            QMessageBox.information(self, "End Segment", "Nothing to record yet. Add/finish a polygon or add/place an external sprite first.")
            self._set_instruction("Add a polygon (base/external) or place an external image, then drag and click ‘🎯 End Segment’.")

    def on_undo(self):
        if self.placing_external and self.placing_layer is not None:
            L = self.placing_layer
            if L.pixmap_item is not None: self.canvas.scene.removeItem(L.pixmap_item)
            if L.outline_item is not None: self.canvas.scene.removeItem(L.outline_item)
            for it in L.handle_items: self.canvas.scene.removeItem(it)
            try:
                idx = self.canvas.layers.index(L)
                self.canvas.layers.pop(idx)
            except ValueError:
                pass
            self.canvas.current_layer = self.canvas.layers[-1] if self.canvas.layers else None
            self.placing_layer = None
            self.placing_external = False
            self.btn_add_external.setText("🖼️➕ Add External Image")
            self.status_label.setText("External placement canceled.")
            self._set_instruction("External placement canceled. Add External Image again or continue editing.")
            return

        if self.canvas.undo():
            self.status_label.setText("Undo applied.")
            self._set_instruction("Undone. Continue editing, or click ‘🎯 End Segment’ to record movement.")
        else:
            self.status_label.setText("Nothing to undo.")
            self._set_instruction("Nothing to undo. Drag/scale/rotate and click ‘🎯 End Segment’, or add new shapes.")

    def _sample_keyframes_uniform(self, keyframes: List[Keyframe], T: int):
        K = len(keyframes); assert K >= 2
        segs = K - 1
        u = np.linspace(0.0, float(segs), T, dtype=np.float32)
        seg_idx = np.minimum(np.floor(u).astype(int), segs - 1)
        t = u - seg_idx
        k0 = np.array([[keyframes[i].pos[0], keyframes[i].pos[1], keyframes[i].scale, keyframes[i].rot_deg] for i in seg_idx], dtype=np.float32)
        k1 = np.array([[keyframes[i+1].pos[0], keyframes[i+1].pos[1], keyframes[i+1].scale, keyframes[i+1].rot_deg] for i in seg_idx], dtype=np.float32)
        pos0 = k0[:, :2]; pos1 = k1[:, :2]
        s0 = np.maximum(1e-6, k0[:, 2]); s1 = np.maximum(1e-6, k1[:, 2])
        r0 = k0[:, 3]; r1 = k1[:, 3]
        pos = (1 - t)[:, None] * pos0 + t[:, None] * pos1
        scl = np.exp((1 - t) * np.log(s0) + t * np.log(s1))
        rot = (1 - t) * r0 + t * r1
        return pos.astype(np.float32), scl.astype(np.float32), rot.astype(np.float32)

    def on_play_demo(self):
        if self.canvas.base_bgr is None:
            QMessageBox.information(self, "Play Demo", "Select an image first.")
            self._set_instruction("Click ‘Select Image’ to begin.")
            return
        has_segments = any((L.polygon_xy is not None and len(L.keyframes) >= 2) for L in self.canvas.layers)
        if not has_segments:
            QMessageBox.information(self, "Play Demo", "No motion segments yet. Drag something and click ‘🎯 End Segment’ at least once.")
            self._set_instruction("Create at least one movement: drag/scale/rotate then click ‘🎯 End Segment’.")
            return
        fps = int(self.spn_fps.value())
        T_total = int(self.spn_total_frames.value())
        self.canvas.play_demo(fps=fps, T_total=T_total)
        self._set_instruction("Playing demo… When it ends, you’ll return to the editor. Tweak and play again, or 💾 Save.")

    def on_new(self):
        if self._block_if_pending_segment("starting a new project"):
            return
        self.canvas.scene.clear()
        self.canvas.layers.clear()
        self.canvas.current_layer = None
        self.canvas.base_bgr = None
        self.canvas.base_preview_bgr = None
        self.canvas.base_item = None
        self.add_poly_active = False
        self.btn_add_poly.setText("Add Polygon")
        self.placing_external = False; self.placing_layer = None
        self.btn_add_external.setText("🖼️➕ Add External Image")
        if hasattr(self, "txt_prompt"):
            self.txt_prompt.clear()
        self.status_label.setText("Ready")
        self._set_instruction("New project. Click ‘Select Image’ to begin.")
        self.on_select_base()

    def on_save(self):
        if self._block_if_pending_segment("saving"):
            return
        if self.canvas.base_bgr is None or not self.canvas.layers:
            QMessageBox.information(self, "Save", "Load an image and add at least one polygon/sprite first.")
            self._set_instruction("Add a polygon (base/external), record segments (🎯 End Segment), then Save.")
            return

        # If any layer has exactly one keyframe, auto-add the current pose as a second keyframe
        for L in self.canvas.layers:
            if L.pixmap_item and L.polygon_xy is not None and len(L.keyframes) == 1:
                self.canvas.current_layer = L
                self.canvas.end_segment_add_keyframe()

        # 1) Pick a parent directory
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", ""
        )
        if not base_dir:
            self._set_instruction("Save canceled. You can keep editing or try ▶️ Play Demo.")
            return

        # 2) Ask for a subdirectory name
        subdir_name, ok = QtWidgets.QInputDialog.getText(
            self, "Subfolder Name", "Create a new subfolder in the selected directory:"
        )
        if not ok or not subdir_name.strip():
            self._set_instruction("Save canceled (no subfolder name).")
            return
        subdir_name = subdir_name.strip()

        final_dir = os.path.join(base_dir, subdir_name)
        if os.path.exists(final_dir):
            resp = QMessageBox.question(
                self, "Folder exists",
                f"'{subdir_name}' already exists in the selected directory.\n"
                f"Use it and overwrite files?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if resp != QMessageBox.Yes:
                self._set_instruction("Save canceled. Choose another name or directory next time.")
                return
        else:
            try:
                os.makedirs(final_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save", f"Failed to create folder:\n{e}")
                return

        try:
            prompt_text = self.txt_prompt.toPlainText()
        except Exception:
            prompt_text = ""
        try:
            with open(os.path.join(final_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt_text)
        except Exception as e:
            # Non-fatal: continue saving the rest if prompt write fails
            print(f"[warn] Failed to write prompt.txt: {e}")

        # Output paths
        first_frame_path = os.path.join(final_dir, "first_frame.png")
        motion_path      = os.path.join(final_dir, "motion_signal.mp4")
        mask_path        = os.path.join(final_dir, "mask.mp4")
        base_title       = subdir_name  # for optional numpy save below
        npy_path         = os.path.join(final_dir, f"{base_title}_polygons.npy")

        fps = int(self.spn_fps.value())
        T_total = int(self.spn_total_frames.value())

        # Build background (inpaint base regions belonging to non-external layers)
        H, W = self.canvas.base_bgr.shape[:2]
        total_mask = np.zeros((H, W), dtype=bool)
        for L in self.canvas.layers:
            if L.polygon_xy is None: 
                continue
            if L.is_external: 
                continue
            poly0 = L.polygon_xy.astype(np.int32)
            m = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(m, [poly0], 255)
            total_mask |= (m > 0)
        background = inpaint_background(self.canvas.base_bgr, total_mask)

        # Collect animated frames for each layer (with hue crossfade as before)
        all_layer_frames = []
        layer_polys = []  # kept for the optional numpy block below
        for L in self.canvas.layers:
            if L.polygon_xy is None or len(L.keyframes) < 2:
                continue

            def sample_keyframes_uniform_with_seg(keyframes: List[Keyframe], T: int):
                K = len(keyframes); assert K >= 1
                if K == 1:
                    pos = np.repeat(keyframes[0].pos[None, :], T, axis=0).astype(np.float32)
                    scl = np.full((T,), keyframes[0].scale, dtype=np.float32)
                    rot = np.full((T,), keyframes[0].rot_deg, dtype=np.float32)
                    seg_idx = np.zeros((T,), dtype=np.int32)
                    t = np.zeros((T,), dtype=np.float32)
                    return pos, scl, rot, seg_idx, t
                segs = K - 1
                u = np.linspace(0.0, float(segs), T, dtype=np.float32)
                seg_idx = np.minimum(np.floor(u).astype(int), segs - 1)
                t = u - seg_idx
                k0 = np.array([[keyframes[i].pos[0], keyframes[i].pos[1], keyframes[i].scale, keyframes[i].rot_deg] for i in seg_idx], dtype=np.float32)
                k1 = np.array([[keyframes[i+1].pos[0], keyframes[i+1].pos[1], keyframes[i+1].scale, keyframes[i+1].rot_deg] for i in seg_idx], dtype=np.float32)
                pos0 = k0[:, :2]; pos1 = k1[:, :2]
                s0 = np.maximum(1e-6, k0[:, 2]); s1 = np.maximum(1e-6, k1[:, 2])
                r0 = k0[:, 3]; r1 = k1[:, 3]
                pos = (1 - t)[:, None] * pos0 + t[:, None] * pos1
                scl = np.exp((1 - t) * np.log(s0) + t * np.log(s1))
                rot = (1 - t) * r0 + t * r1
                return pos.astype(np.float32), scl.astype(np.float32), rot.astype(np.float32), seg_idx, t

            path_xy, scales, rots, seg_idx, t = sample_keyframes_uniform_with_seg(L.keyframes, T_total)
            origin_xy = L.origin_local_xy if L.origin_local_xy is not None else L.polygon_xy.mean(axis=0)

            # Precompute one animation per keyframe hue
            K = len(L.keyframes)
            hue_values = [L.keyframes[k].hue_deg for k in range(K)]
            hue_to_frames: Dict[int, List[np.ndarray]] = {}
            polys_for_layer = None
            for k in range(K):
                bgr_h = apply_hue_shift_bgr(L.source_bgr, hue_values[k])
                frames_h, polys = animate_polygon(
                    bgr_h, L.polygon_xy, path_xy, scales, rots,
                    interp=cv2.INTER_LINEAR, origin_xy=origin_xy
                )
                hue_to_frames[k] = frames_h
                if polys_for_layer is None:  # same polys for all hues
                    polys_for_layer = np.array(polys, dtype=np.float32)
            if polys_for_layer is not None:
                layer_polys.append(polys_for_layer)

            # Mix per frame using seg_idx / t
            frames_rgba = []
            for i in range(T_total):
                s = int(seg_idx[i])
                w = float(t[i])
                A = hue_to_frames[s][i].astype(np.float32)
                B = hue_to_frames[s+1][i].astype(np.float32)
                mix = (1.0 - w) * A + w * B
                frames_rgba.append(np.clip(mix, 0, 255).astype(np.uint8))
            all_layer_frames.append(frames_rgba)

        if not all_layer_frames:
            QMessageBox.information(self, "Save", "No motion segments found. Add keyframes with ‘🎯 End Segment’.")
            self._set_instruction("Record at least one segment on a layer, then Save.")
            return

        frames_out = composite_frames(background, all_layer_frames)

        # Build mask frames (union of alpha across layers per frame)
        mask_frames = []
        for t in range(T_total):
            m = np.zeros((H, W), dtype=np.uint16)
            for Lframes in all_layer_frames:
                m += Lframes[t][:, :, 3].astype(np.uint16)
            m = np.clip(m, 0, 255).astype(np.uint8)
            mask_frames.append(m)

        # --- Actual saving ---
        try:
            # first_frame.png (copy of the base image used for saving)
            cv2.imwrite(first_frame_path, self.canvas.base_bgr)

            # motion_signal.mp4 = composited warped video
            save_video_mp4(frames_out, motion_path, fps=fps)

            # mask.mp4 = grayscale mask video
            save_video_mp4([cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in mask_frames], mask_path, fps=fps)

            # Optional: polygons.npy — disabled by default
            if False:
                # Pad and save polygons
                Vmax = 0
                for P in layer_polys:
                    if P.size:
                        Vmax = max(Vmax, P.shape[1])

                def pad_poly(P: np.ndarray, Vmax_: int) -> np.ndarray:
                    if P.size == 0:
                        return np.zeros((T_total, Vmax_, 2), dtype=np.float32)
                    T_, V, _ = P.shape
                    out = np.zeros((T_, Vmax_, 2), dtype=np.float32)
                    out[:, :V, :] = P
                    if V > 0:
                        out[:, V:, :] = P[:, V-1:V, :]
                    return out

                polys_uniform = np.stack([pad_poly(P, Vmax) for P in layer_polys], axis=0)
                np.save(npy_path, polys_uniform)

        except Exception as e:
            QMessageBox.critical(self, "Save", f"Failed to save:\n{e}")
            return

        QMessageBox.information(self, "Saved", f"Saved to:\n{final_dir}")
        self._set_instruction("Saved! You can keep editing, play demo again, or start a New project.")


# ------------------------------
# Entry
# ------------------------------

def main():
    if sys.version_info < (3, 8):
        print("[Warning] PySide6 officially supports Python 3.8+. You're on %d.%d." % (sys.version_info.major, sys.version_info.minor))
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
