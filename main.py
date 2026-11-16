"""
Real-time vehicle speed detection at a campus crosswalk.

YOLOv8n (cars) -> ByteTrack -> perspective transform to real metres -> speed,
with license-plate OCR (iApp API) running on a background thread.

Usage:
	python main.py --video videos/your_clip.mp4
"""
# ── Standard library ──────────────────────────────────────────────────────────
import os
import argparse
import json
import time
import threading
import queue
from collections import defaultdict, deque

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import requests
import supervision as sv
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PATHS  (auto-derived from script location)
# ══════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# Secrets (IAPP_API_KEY) come from a local .env file that is never committed
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

_parser = argparse.ArgumentParser(description="Vehicle speed detection from a fixed CCTV camera")
_parser.add_argument("--video", default="videos/ControlDataset.mp4",
					 help="input video, relative to this script or absolute")
_parser.add_argument("--no-display", action="store_true",
					 help="do not open the preview window (headless run)")
_ARGS = _parser.parse_args()

# Outputs go to output/<video name>/ (git-ignored: they contain real plates)
_SCRIPT_NAME = os.path.splitext(os.path.basename(_ARGS.video))[0]
_OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "output", _SCRIPT_NAME)
_CARS_DIR    = os.path.join(_OUTPUT_DIR, "detected_cars")
os.makedirs(_CARS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
	# ── Input paths ───────────────────────────────────────────────────────────
	"video_path":               _ARGS.video,
	"font_path":                "fonts/THSarabun-Bold.ttf",
	"yolo_model_path":          "models/yolov8n.pt",
	"license_plate_model_path": "models/license_plate_detector.pt",

	# ── Output paths (auto-set from script name) ──────────────────────────────
	"target_video_path":   os.path.join(_OUTPUT_DIR, f"{_SCRIPT_NAME}_video.mp4"),
	"records_output_path": os.path.join(_OUTPUT_DIR, f"{_SCRIPT_NAME}_records.json"),
	"car_image_save_dir":  _CARS_DIR,

	# ── iApp OCR API ──────────────────────────────────────────────────────────
	"iapp_api_key": os.getenv("IAPP_API_KEY", ""),   # set in .env (see .env.example)
	"iapp_ocr_url": "https://api.iapp.co.th/v3/store/smart-city/license-plate-ocr",
	"api_timeout":  10,

	# ── Detection & tracking ──────────────────────────────────────────────────
	"frame_skip":             5,      # process every Nth frame
	"car_class_id":           2,      # COCO class for 'car'
	"yolo_imgsz":             1280,   # PERF: inference at 1280 instead of native 2560
	"byte_track_threshold":   0.25,
	"byte_track_lost_buffer": 30,     

	# ── Speed thresholds (km/h) ───────────────────────────────────────────────
	"speed_thresholds": {"safe": 40, "caution": 60},

	# ── Perspective transform ─────────────────────────────────────────────────
	"source_polygon":  np.array([[2000, 144], [2720, 210], [1724, 1316], [-32, 1013]]),
	"target_width_m":  5.50,
	"target_height_m": 24.30,

	# ── UI layout (all in one place) ──────────────────────────────────────────
	"sidebar_width":          490,
	"debug_bar_height":       100,

	# Font sizes
	"font_size_legend":       50,
	"font_size_debug":        50,
	"font_size_label":        50,
	"font_size_sidebar_info": 35,

	# Legend geometry
	"legend_start_x_box":    20,
	"legend_start_x_text":   100,
	"legend_start_y":        20,
	"legend_box_w":          50,
	"legend_box_h":          50,
	"legend_spacing":        25,
	"legend_y_text_offset":  -5,

	# Debug bar — x position of the right-side stats text
	"debug_stats_start_x":   1950,
}

FONT_PATH = os.path.join(_SCRIPT_DIR, CONFIG["font_path"])
if not os.path.exists(FONT_PATH):
	raise FileNotFoundError(f"Font not found: {FONT_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# Font cache: load each size from disk only once
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
	if size not in _font_cache:
		_font_cache[size] = ImageFont.truetype(FONT_PATH, size)
	return _font_cache[size]


def draw_thai_text(image: np.ndarray, text: str, position: tuple,
				   font_size: int, color_bgr: tuple) -> np.ndarray:
	"""
	Render Thai / English text onto a BGR OpenCV image using Pillow.
	PERF: converts only a small ROI around the text instead of the full frame.
	"""
	font      = _get_font(font_size)
	color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

	# Estimate text bounding box to define the ROI
	# bbox = (x_offset, y_offset, x_end, y_end) — y_offset may be >0 (top bearing)
	# Use font_size as height budget so descenders/ascenders are never clipped
	bbox   = font.getbbox(text)
	tw     = bbox[2] - bbox[0]
	th     = font_size
	pad    = 4
	x0     = max(0, position[0] - pad)
	y0     = max(0, position[1] - pad)
	x1     = min(image.shape[1], position[0] + tw + pad * 2)
	y1     = min(image.shape[0], position[1] + th + pad * 2)

	if x1 <= x0 or y1 <= y0:
		return image  # nothing to draw

	roi       = image[y0:y1, x0:x1]
	pil_roi   = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
	ImageDraw.Draw(pil_roi).text(
		(position[0] - x0, position[1] - y0), text, font=font, fill=color_rgb
	)
	image[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
	return image


class NumpyEncoder(json.JSONEncoder):
	"""JSON encoder that handles NumPy scalar / array types."""
	def default(self, obj):
		if isinstance(obj, np.floating): return float(obj)
		if isinstance(obj, np.integer):  return int(obj)
		if isinstance(obj, np.ndarray):  return obj.tolist()
		return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class ViewTransformer:
	"""Maps pixel coordinates to real-world metres via perspective transform."""

	def __init__(self, source: np.ndarray, target: np.ndarray):
		self.m = cv2.getPerspectiveTransform(
			source.astype(np.float32), target.astype(np.float32)
		)

	def transform_points(self, points: np.ndarray) -> np.ndarray:
		pts = points.reshape(-1, 1, 2).astype(np.float32)
		return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


class SpeedEstimator:
	_MAX_POSITIONS = 150  # 5 s × 30 fps = 150 max positions per vehicle

	def __init__(self, calculate_interval: float = 0.5, smoothing_window: int = 3):
		self.tracker_positions = defaultdict(lambda: deque(maxlen=self._MAX_POSITIONS))
		self.tracker_speeds    = {}
		self.speed_history     = defaultdict(lambda: deque(maxlen=smoothing_window))
		self.average_speeds    = {}
		self.calculate_interval = calculate_interval

	def update_position(self, tracker_id: int, position: np.ndarray, timestamp: float):
		self.tracker_positions[tracker_id].append((timestamp, position[0], position[1]))
		# PERF: deque(maxlen) auto-evicts old entries — no list rebuild needed
		self._calculate_speed(tracker_id, timestamp)

	def _calculate_speed(self, tracker_id: int, current_time: float):
		positions = self.tracker_positions[tracker_id]
		if len(positions) < 2:
			self.tracker_speeds[tracker_id] = 0.0
			return

		recent_position   = positions[-1]
		previous_position = None
		for pos in reversed(list(positions)[:-1]):
			if recent_position[0] - pos[0] >= self.calculate_interval:
				previous_position = pos
				break

		if previous_position is None:
			self.tracker_speeds[tracker_id] = 0.0
			return

		dx        = recent_position[1] - previous_position[1]
		dy        = recent_position[2] - previous_position[2]
		distance  = np.sqrt(dx * dx + dy * dy)
		time_diff = recent_position[0] - previous_position[0]
		speed_kmh = (distance / time_diff) * 3.6

		self.speed_history[tracker_id].append(speed_kmh)
		self.tracker_speeds[tracker_id] = (
			sum(self.speed_history[tracker_id]) / len(self.speed_history[tracker_id])
		)

		# v2.4 running average
		if tracker_id not in self.average_speeds:
			self.average_speeds[tracker_id] = speed_kmh
		else:
			self.average_speeds[tracker_id] = (
				self.average_speeds[tracker_id] + speed_kmh
			) / 2

	def get_speed(self, tracker_id: int)         -> float: return self.tracker_speeds.get(tracker_id, 0.0)
	def get_average_speed(self, tracker_id: int) -> float: return self.average_speeds.get(tracker_id, 0.0)

	def get_speed_color(self, speed: float) -> tuple:
		t = CONFIG["speed_thresholds"]
		if speed < t["safe"]:    return (0, 255, 0)
		if speed < t["caution"]: return (0, 255, 255)
		return (0, 0, 255)

	def get_speed_label(self, speed: float) -> str:
		t = CONFIG["speed_thresholds"]
		if speed < t["safe"]:    return "safe"
		if speed < t["caution"]: return "caution"
		return "speeding"


class BackgroundWorker:
	"""
	Saves car images and calls the OCR API in a background thread.
	Main processing loop is never stalled by I/O or network latency.
	"""

	def __init__(self, records: dict, records_path: str):
		self._records      = records
		self._records_path = records_path
		self._queue        = queue.Queue()
		self._thread       = threading.Thread(target=self._run, daemon=True)

	def start(self): self._thread.start()
	def stop(self):  self._queue.put(None); self._thread.join()

	def submit(self, tracker_id: int, display_id,
			   avg_speed: float, image: np.ndarray, speed_label: str):
		self._queue.put((tracker_id, display_id, avg_speed, image.copy(), speed_label))

	def _run(self):
		while True:
			item = self._queue.get()
			if item is None:
				break
			self._process(*item)

	def _process(self, tracker_id, display_id, avg_speed, image, speed_label):
		record = {
			"display_id":    display_id,
			"avg_speed_kmh": round(avg_speed, 1),
			"speed_label":   speed_label,
			"image_path":    None,
			"lp_number":     None,
			"province":      None,
			"vehicle_brand": None,
			"vehicle_model": None,
			"vehicle_year":  None,
			"vehicle_color": None,
			"ocr_confidence":None,
			"ocr_raw":       None,
		}

		# ── Save image ────────────────────────────────────────────────────────
		save_path = os.path.join(
			CONFIG["car_image_save_dir"],
			f"ID_{display_id}_AVG_{avg_speed:.1f}.jpg"
		)
		cv2.imwrite(save_path, image)
		record["image_path"] = save_path
		print(f"[SAVE] ID {display_id} → {os.path.basename(save_path)}")

		# ── OCR API call (skipped when no API key is configured) ──────────────
		if not CONFIG["iapp_api_key"]:
			self._save(display_id, record)
			return
		try:
			with open(save_path, "rb") as f:
				resp = requests.post(
					CONFIG["iapp_ocr_url"],
					headers={"apikey": CONFIG["iapp_api_key"]},
					files={"file": f},
					timeout=CONFIG["api_timeout"],
				)
			if resp.status_code == 200:
				ocr = resp.json()
				record.update({
					"lp_number":      ocr.get("lp_number"),
					"province":       ocr.get("province"),
					"vehicle_brand":  ocr.get("vehicle_brand"),
					"vehicle_model":  ocr.get("vehicle_model"),
					"vehicle_year":   ocr.get("vehicle_year"),
					"vehicle_color":  ocr.get("vehicle_color"),
					"ocr_confidence": ocr.get("conf"),
					"ocr_raw":        ocr,
				})
				print(f"[OCR]  ID {display_id} → {record['lp_number']} ({record['province']})")
			else:
				print(f"[OCR]  ID {display_id} → API error {resp.status_code}")
		except Exception as exc:
			print(f"[OCR]  ID {display_id} → Request failed: {exc}")

		self._save(display_id, record)

	def _save(self, display_id, record):
		# ── Persist to JSON ───────────────────────────────────────────────────
		self._records[display_id] = record
		with open(self._records_path, "w", encoding="utf-8") as jf:
			json.dump(list(self._records.values()), jf,
					  ensure_ascii=False, indent=2, cls=NumpyEncoder)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STATE
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
	def __init__(self):
		self.car_images:          dict = {}
		self.car_records:         dict = {}
		self.license_plate_cache: dict = {}
		self.display_id_map:      dict = {}
		self.cars_in_zone:        set  = set()
		self.next_display_id:     int  = 1
		self.safe_count           = 0
		self.caution_count        = 0
		self.speeding_count       = 0
		self.highest_speed        = 0.0
		self.highest_speed_id     = None
		# frame_count increments EVERY frame (keeps frame_time accurate)
		self.frame_count          = 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — UI DRAWING  (v3.5 style)
# ══════════════════════════════════════════════════════════════════════════════

def draw_speed_legend(frame: np.ndarray) -> np.ndarray:
	"""Speed legend — large colored boxes + Thai-capable font"""
	cfg  = CONFIG
	bx   = cfg["legend_start_x_box"]
	tx   = cfg["legend_start_x_text"]
	y    = cfg["legend_start_y"]
	bw   = cfg["legend_box_w"]
	bh   = cfg["legend_box_h"]
	sp   = cfg["legend_spacing"]
	yo   = cfg["legend_y_text_offset"]
	fs   = cfg["font_size_legend"]
	t    = cfg["speed_thresholds"]

	# Black background behind legend
	cv2.rectangle(frame, (0, 0), (cfg["sidebar_width"] - 10, 250), (0, 0, 0), -1)

	for color, text in [
		((0, 255, 0),   f"< {t['safe']} km/h  (Safe)"),
		((0, 255, 255), f"{t['safe']}–{t['caution']} km/h  (Caution)"),
		((0, 0, 255),   f"> {t['caution']} km/h  (Speeding)"),
	]:
		cv2.rectangle(frame, (bx, y), (bx + bw, y + bh), color, -1)
		frame = draw_thai_text(frame, text, (tx, y + yo), fs, color)
		y += bh + sp

	return frame


def draw_debug_bar(frame: np.ndarray, state: AppState, current_fps: float) -> np.ndarray:
	"""Bottom debug bar — frame count, FPS, max speed left; category counts right"""
	cfg = CONFIG
	bar = np.zeros((cfg["debug_bar_height"], frame.shape[1], 3), dtype=np.uint8)

	display_highest_id = state.display_id_map.get(state.highest_speed_id, "N/A")
	left_text  = (f"Frame: {state.frame_count} | "
				  f"FPS: {current_fps:.1f} | "
				  f"Max: {state.highest_speed:.1f} km/h (ID: {display_highest_id})")

	bar = draw_thai_text(bar, left_text,  (20, 15),                         cfg["font_size_debug"], (255, 255, 255))

	# Draw each speed-category count in its own color
	font       = _get_font(cfg["font_size_debug"])
	rx         = cfg["debug_stats_start_x"]
	safe_txt   = f"Safe: {state.safe_count}"
	caut_txt   = f"Caution: {state.caution_count}"
	spd_txt    = f"Speeding: {state.speeding_count}"
	sep_txt    = " | "
	bar = draw_thai_text(bar, safe_txt,  (rx, 15), cfg["font_size_debug"], (0, 255, 0))
	rx += font.getbbox(safe_txt)[2] - font.getbbox(safe_txt)[0]
	bar = draw_thai_text(bar, sep_txt,   (rx, 15), cfg["font_size_debug"], (255, 255, 255))
	rx += font.getbbox(sep_txt)[2]  - font.getbbox(sep_txt)[0]
	bar = draw_thai_text(bar, caut_txt,  (rx, 15), cfg["font_size_debug"], (0, 255, 255))
	rx += font.getbbox(caut_txt)[2] - font.getbbox(caut_txt)[0]
	bar = draw_thai_text(bar, sep_txt,   (rx, 15), cfg["font_size_debug"], (255, 255, 255))
	rx += font.getbbox(sep_txt)[2]  - font.getbbox(sep_txt)[0]
	bar = draw_thai_text(bar, spd_txt,   (rx, 15), cfg["font_size_debug"], (0, 0, 255))
	return np.vstack((frame, bar))


def draw_sidebar(frame: np.ndarray, state: AppState,
				 speed_est: SpeedEstimator) -> np.ndarray:
	"""Right sidebar — 4 car thumbnails with merged info line below each."""
	cfg     = CONFIG
	panel   = np.zeros((frame.shape[0], cfg["sidebar_width"], 3), dtype=np.uint8)
	y       = 5
	img_w   = 450
	img_h   = 220
	border  = 4
	spacing = 8
	fs      = cfg["font_size_sidebar_info"]

	# PERF: only take last 4 items — no point resizing images that won't be shown
	visible_items = list(state.car_images.items())[-4:]

	for tid, img in reversed(visible_items):
		thumb = cv2.resize(img, (img_w, img_h))

		# Overlay license plate crop in top-right corner of thumbnail
		if tid in state.license_plate_cache:
			lp = state.license_plate_cache[tid]
			h, w = lp.shape[:2]
			thumb[10:10 + h, img_w - w - 10:img_w - 10] = lp

		avg_speed    = speed_est.get_average_speed(tid)
		border_color = speed_est.get_speed_color(avg_speed)
		bordered     = cv2.copyMakeBorder(thumb, border, border, border, border,
										  cv2.BORDER_CONSTANT, value=border_color)

		if y + bordered.shape[0] + fs + spacing > frame.shape[0]:
			break

		panel[y:y + bordered.shape[0], 20:20 + bordered.shape[1]] = bordered

		display_id = state.display_id_map.get(tid, "N/A")
		record     = state.car_records.get(display_id, {})
		lp_number  = record.get("lp_number") or "N/A"

		# Single merged line: ID | Avg | Plate — colored to match border/speed
		info   = f"ID: {display_id} | Avg: {avg_speed:.1f} km/h | Plate: {lp_number}"
		y_base = y + bordered.shape[0]
		panel  = draw_thai_text(panel, info, (20, y_base + 4), fs, border_color)

		y += bordered.shape[0] + fs + spacing + 10

	return np.hstack((frame, panel))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PER-FRAME PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def draw_annotations(source_frame: np.ndarray,
					 target_frame: np.ndarray,
					 tracked_detections: sv.Detections,
					 frame_time: float,
					 state: AppState,
					 speed_est: SpeedEstimator,
					 view_tf: ViewTransformer,
					 lp_model: YOLO,
					 worker: BackgroundWorker) -> np.ndarray:
	"""Annotate frame, update speed estimates, handle exits."""

	if tracked_detections.tracker_id is None:
		return target_frame

	# Load font once per frame call (served from cache)
	label_font   = _get_font(CONFIG["font_size_label"])
	car_class_id = CONFIG["car_class_id"]
	current_cars: set = set()

	for box, tracker_id, class_id in zip(
		tracked_detections.xyxy,
		tracked_detections.tracker_id,
		tracked_detections.class_id,
	):
		x1, y1, x2, y2 = map(int, box)

		# ── Assign sequential display ID ──────────────────────────────────────
		if tracker_id not in state.display_id_map:
			state.display_id_map[tracker_id] = state.next_display_id
			state.next_display_id += 1
		display_id = state.display_id_map[tracker_id]

		# ── Perspective transform ─────────────────────────────────────────────
		bottom_cx = int((x1 + x2) / 2)
		bottom_cy = int(y2)
		real_pt   = view_tf.transform_points(np.array([[bottom_cx, bottom_cy]]))[0]

		if int(class_id) == car_class_id:
			# ── Crop & store car image ────────────────────────────────────────
			crop = source_frame[y1:y2, x1:x2]
			if crop.size > 0:
				state.car_images[tracker_id] = crop

				# LP detector: run on first appearance and every 5*frame_skip frames
				if (tracker_id not in state.license_plate_cache or
						state.frame_count % (5 * CONFIG["frame_skip"]) == 0):
					try:
						lp_res   = lp_model(crop, verbose=False)[0]
						lp_boxes = lp_res.boxes.data.cpu().numpy()
						if len(lp_boxes) > 0:
							lx1, ly1, lx2, ly2 = map(int, lp_boxes[0][:4])
							plate_crop = crop[ly1:ly2, lx1:lx2]
							if plate_crop.size > 0:
								state.license_plate_cache[tracker_id] = cv2.resize(plate_crop, (200, 100))
					except Exception:
						pass

			# ── Speed update ──────────────────────────────────────────────────
			speed_est.update_position(tracker_id, real_pt, frame_time)
			speed       = speed_est.get_speed(tracker_id)
			speed_color = speed_est.get_speed_color(speed)

			if speed > state.highest_speed:
				state.highest_speed    = speed
				state.highest_speed_id = tracker_id

			# ── Draw bounding box ─────────────────────────────────────────────
			cv2.rectangle(target_frame, (x1, y1), (x2, y2), speed_color, 4)

			# ── Draw label — background rect sits just inside the top of the box ─
			label = f"#{display_id} {speed:.1f} km/h"
			bbox  = label_font.getbbox(label)
			tw    = bbox[2] - bbox[0]
			th    = bbox[3] - bbox[1]
			pad   = 4
			# rect starts AT y1 (top edge of bounding box), text is inside it
			cv2.rectangle(target_frame,
						  (x1, y1), (x1 + tw + pad * 2, y1 + th + pad * 2),
						  speed_color, -1)
			target_frame = draw_thai_text(
				target_frame, label, (x1 + pad, y1 + pad - 20),
				CONFIG["font_size_label"], (0, 0, 0)
			)

			current_cars.add(tracker_id)

	# ── Handle vehicles that left the zone ───────────────────────────────────
	for tid in state.cars_in_zone - current_cars:
		_handle_exit(tid, state, speed_est, worker)

	state.cars_in_zone = current_cars
	return target_frame


def _handle_exit(tracker_id: int, state: AppState,
				 speed_est: SpeedEstimator, worker: BackgroundWorker):
	"""Called when a tracked vehicle leaves the detection zone."""
	avg_speed  = speed_est.get_average_speed(tracker_id)
	display_id = state.display_id_map.get(tracker_id, "Unknown")
	lbl        = speed_est.get_speed_label(avg_speed)

	if lbl == "safe":      state.safe_count     += 1
	elif lbl == "caution": state.caution_count  += 1
	else:                  state.speeding_count += 1

	if avg_speed > 0 and tracker_id in state.car_images:
		worker.submit(tracker_id, display_id, avg_speed, state.car_images[tracker_id], lbl)
	else:
		state.car_records[display_id] = {
			"display_id":    display_id,
			"avg_speed_kmh": 0.0,
			"speed_label":   lbl,
		}
		print(f"[SKIP] ID {display_id} → avg speed 0, not saved.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
	# ── Load models ───────────────────────────────────────────────────────────
	yolo_model = YOLO(os.path.join(_SCRIPT_DIR, CONFIG["yolo_model_path"]))
	lp_model   = YOLO(os.path.join(_SCRIPT_DIR, CONFIG["license_plate_model_path"]))

	# ── Open video ────────────────────────────────────────────────────────────
	video_path = os.path.join(_SCRIPT_DIR, CONFIG["video_path"])
	cap    = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise FileNotFoundError(f"Cannot open video: {video_path}")
	if not CONFIG["iapp_api_key"]:
		print("[OCR]  IAPP_API_KEY not set → plate OCR disabled (speeds are still measured).")
	fps    = int(cap.get(cv2.CAP_PROP_FPS))
	width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	final_w = width  + CONFIG["sidebar_width"]
	final_h = height + CONFIG["debug_bar_height"]
	if final_w % 2: final_w += 1
	if final_h % 2: final_h += 1

	out = cv2.VideoWriter(
		CONFIG["target_video_path"],
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps / CONFIG["frame_skip"],
		(final_w, final_h),
	)

	# ── Build helpers ─────────────────────────────────────────────────────────
	tracker = sv.ByteTrack(
		track_activation_threshold=CONFIG["byte_track_threshold"],
		lost_track_buffer=CONFIG["byte_track_lost_buffer"],
		frame_rate=fps / CONFIG["frame_skip"],
	)
	speed_est   = SpeedEstimator()
	target_poly = np.array([
		[0, 0],
		[CONFIG["target_width_m"] - 1, 0],
		[CONFIG["target_width_m"] - 1, CONFIG["target_height_m"] - 1],
		[0, CONFIG["target_height_m"] - 1],
	])
	view_tf  = ViewTransformer(source=CONFIG["source_polygon"], target=target_poly)
	src_poly = CONFIG["source_polygon"]
	src_poly_i32 = src_poly.astype(np.int32)   # PERF: pre-cast once for cv2.polylines
	state    = AppState()
	worker   = BackgroundWorker(state.car_records, CONFIG["records_output_path"])
	worker.start()

	start_time = time.time()
	print(f"Processing video → {_OUTPUT_DIR}")
	print(f"Algorithm v2.4 | frame_skip={CONFIG['frame_skip']} | "
		  f"lost_track_buffer={CONFIG['byte_track_lost_buffer']}")
	print("Press 'q' to quit.")

	while cap.isOpened():
		# KEY: frame_time from video position (immune to CPU load)
		frame_time  = start_time + (state.frame_count / (fps if fps > 0 else 30))
		current_fps = (state.frame_count / (time.time() - start_time)) if (time.time() - start_time) > 0 else 0.0

		# KEY: increment BEFORE the skip check so frame_time stays accurate
		state.frame_count += 1

		if state.frame_count % CONFIG["frame_skip"] != 0:
			# PERF: grab() advances the file pointer without decoding the frame (~10× faster)
			cap.grab()
			continue

		ret, frame = cap.read()
		if not ret:
			break

		display_frame = frame.copy()

		# ── Detect → filter to zone → track ───────────────────────────────────
		# PERF: imgsz=1280 halves inference pixels vs native 2560, ~2× faster
		results    = yolo_model(frame, classes=[CONFIG["car_class_id"], 5, 7],
								imgsz=CONFIG["yolo_imgsz"], verbose=False)[0]
		detections = sv.Detections.from_ultralytics(results)
		detections = detections[detections.class_id == CONFIG["car_class_id"]]

		if len(detections) > 0:
			# PERF: vectorized polygon test — build centers array once, test in batch
			centers = np.column_stack([
				(detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,
				detections.xyxy[:, 3]
			])
			mask = np.array([
				cv2.pointPolygonTest(src_poly, (float(c[0]), float(c[1])), False) >= 0
				for c in centers
			])
			detections = detections[mask]

		tracked = tracker.update_with_detections(detections)

		# ── Annotate ──────────────────────────────────────────────────────────
		display_frame = draw_annotations(
			frame, display_frame, tracked, frame_time,
			state, speed_est, view_tf, lp_model, worker
		)
		# PERF: use pre-cast int32 polygon — no per-frame alloc
		cv2.polylines(display_frame, [src_poly_i32], True, (255, 255, 0), 2)
		display_frame = draw_speed_legend(display_frame)
		display_frame = draw_debug_bar(display_frame, state, current_fps)
		display_frame = draw_sidebar(display_frame, state, speed_est)

		# ── Write & display ───────────────────────────────────────────────────
		canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
		canvas[:display_frame.shape[0], :display_frame.shape[1]] = display_frame
		out.write(canvas)
		if not _ARGS.no_display:
			cv2.imshow("Traffic Monitor", canvas)
			if cv2.waitKey(1) == ord("q"):
				break

	# ── Cleanup ───────────────────────────────────────────────────────────────
	worker.stop()
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	print(f"Done. Files saved to: {_OUTPUT_DIR}")


if __name__ == "__main__":
	main()