"""
人脸识别与追踪模块
基于 InsightFace + buffalo_s，实现摄像头中心人脸实时监测
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger


@dataclass
class FaceTrack:
	"""单目标人脸追踪状态"""

	bbox: np.ndarray
	label: str = "识别中"
	is_identity_locked: bool = False
	best_quality: float = 0.0
	missed_frames: int = 0
	similarity: float = 0.0


class FaceRecognitionModule:
	"""人脸监测模块（中心人脸 + 追踪 + 首帧高质量识别）"""

	IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

	def __init__(
		self,
		known_faces_dir: str,
		model_name: str = "buffalo_s",
		det_size: Tuple[int, int] = (640, 640),
		recognition_threshold: float = 0.38,
		quality_threshold: float = 0.46,
		iou_threshold: float = 0.25,
		max_missed_frames: int = 15,
		use_gpu: bool = True,
	):
		"""
		初始化人脸识别模块

		Args:
			known_faces_dir: 已知人脸库目录（按图片文件名作为身份名）
			model_name: InsightFace 模型名称，默认 buffalo_s
			det_size: 检测输入尺寸
			recognition_threshold: 相似度阈值（低于则判定为陌生人）
			quality_threshold: 触发首次识别的图像质量阈值
			iou_threshold: 追踪时判定同一目标的 IoU 阈值
			max_missed_frames: 最多允许丢帧数量
			use_gpu: 是否优先使用 GPU
		"""
		self.known_faces_dir = Path(known_faces_dir)
		self.recognition_threshold = recognition_threshold
		self.quality_threshold = quality_threshold
		self.iou_threshold = iou_threshold
		self.max_missed_frames = max_missed_frames

		self.current_track: Optional[FaceTrack] = None
		self.known_embeddings: Dict[str, List[np.ndarray]] = {}

		ctx_id = 0 if use_gpu else -1
		self.app = FaceAnalysis(name=model_name)
		try:
			self.app.prepare(ctx_id=ctx_id, det_size=det_size)
		except Exception as e:
			if use_gpu:
				logger.warning(f"GPU 初始化失败，切换 CPU: {e}")
				self.app.prepare(ctx_id=-1, det_size=det_size)
			else:
				raise

		self._build_known_faces_index()

	def _build_known_faces_index(self) -> None:
		"""从 known_faces 目录构建已知人脸 embedding 索引"""
		self.known_embeddings.clear()

		if not self.known_faces_dir.exists():
			logger.warning(f"known_faces 目录不存在: {self.known_faces_dir}")
			return

		image_paths = [
			p for p in self.known_faces_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS
		]

		if not image_paths:
			logger.warning(f"known_faces 中未找到图片文件: {self.known_faces_dir}")
			return

		loaded_count = 0
		for image_path in image_paths:
			image = cv2.imread(str(image_path))
			if image is None:
				logger.warning(f"图片读取失败，跳过: {image_path}")
				continue

			faces = self.app.get(image)
			if not faces:
				logger.warning(f"未检测到人脸，跳过: {image_path}")
				continue

			best_face = max(faces, key=lambda face: self._face_quality_score(image, face))
			embedding = self._normalize_embedding(best_face.embedding)
			label = image_path.stem

			self.known_embeddings.setdefault(label, []).append(embedding)
			loaded_count += 1

		logger.info(
			f"已加载 known_faces 人脸库：{len(self.known_embeddings)} 个身份，{loaded_count} 张有效图片"
		)

	@staticmethod
	def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
		norm = np.linalg.norm(embedding)
		if norm <= 1e-8:
			return embedding
		return embedding / norm

	@staticmethod
	def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
		x1 = max(float(box1[0]), float(box2[0]))
		y1 = max(float(box1[1]), float(box2[1]))
		x2 = min(float(box1[2]), float(box2[2]))
		y2 = min(float(box1[3]), float(box2[3]))

		inter_w = max(0.0, x2 - x1)
		inter_h = max(0.0, y2 - y1)
		inter_area = inter_w * inter_h

		area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
		area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
		union = area1 + area2 - inter_area

		if union <= 1e-8:
			return 0.0
		return inter_area / union

	@staticmethod
	def _distance_to_frame_center(frame_shape: Tuple[int, int, int], bbox: np.ndarray) -> float:
		h, w = frame_shape[:2]
		cx = (float(bbox[0]) + float(bbox[2])) / 2.0
		cy = (float(bbox[1]) + float(bbox[3])) / 2.0
		return float(np.hypot(cx - w / 2.0, cy - h / 2.0))

	def _face_quality_score(self, frame: np.ndarray, face) -> float:
		"""
		计算识别图像质量分数
		组合项：检测置信度、清晰度、脸框面积占比
		"""
		h, w = frame.shape[:2]
		x1, y1, x2, y2 = face.bbox.astype(int)
		x1 = max(0, min(x1, w - 1))
		y1 = max(0, min(y1, h - 1))
		x2 = max(0, min(x2, w - 1))
		y2 = max(0, min(y2, h - 1))

		if x2 <= x1 or y2 <= y1:
			return 0.0

		crop = frame[y1:y2, x1:x2]
		if crop.size == 0:
			return 0.0

		blur_score = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
		blur_score = float(np.clip(blur_score / 250.0, 0.0, 1.0))

		area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(1, h * w))
		area_score = float(np.clip(area_ratio * 8.0, 0.0, 1.0))

		det_score = float(getattr(face, "det_score", 0.0))
		det_score = float(np.clip(det_score, 0.0, 1.0))

		quality = 0.45 * det_score + 0.35 * blur_score + 0.20 * area_score
		return float(np.clip(quality, 0.0, 1.0))

	def _match_known_identity(self, embedding: np.ndarray) -> Tuple[str, float]:
		"""将 embedding 与已知人脸库匹配，返回 (标签, 相似度)"""
		if not self.known_embeddings:
			return "陌生人", 0.0

		emb = self._normalize_embedding(embedding)
		best_label = "陌生人"
		best_similarity = -1.0

		for label, vectors in self.known_embeddings.items():
			sims = [float(np.dot(emb, vec)) for vec in vectors]
			sim = max(sims) if sims else -1.0
			if sim > best_similarity:
				best_similarity = sim
				best_label = label

		if best_similarity >= self.recognition_threshold:
			return best_label, best_similarity
		return "陌生人", best_similarity

	def _pick_center_face(self, frame: np.ndarray, faces: list) -> Optional[object]:
		if not faces:
			return None
		return min(
			faces,
			key=lambda face: self._distance_to_frame_center(frame.shape, face.bbox),
		)

	def _update_track(self, frame: np.ndarray, center_face) -> None:
		"""追踪更新：仅跟踪“当前最靠近画面中心”的人脸"""
		bbox = center_face.bbox.astype(float)

		if self.current_track is None:
			self.current_track = FaceTrack(bbox=bbox)
			return

		iou = self._iou(self.current_track.bbox, bbox)
		if iou < self.iou_threshold:
			self.current_track = FaceTrack(bbox=bbox)
			return

		self.current_track.bbox = bbox
		self.current_track.missed_frames = 0

	def _maybe_lock_identity(self, frame: np.ndarray, face) -> None:
		"""当追踪目标尚未锁定身份时，等待高质量首图触发识别"""
		if self.current_track is None or self.current_track.is_identity_locked:
			return

		quality = self._face_quality_score(frame, face)
		if quality > self.current_track.best_quality:
			self.current_track.best_quality = quality

		if quality < self.quality_threshold:
			return

		label, similarity = self._match_known_identity(face.embedding)
		self.current_track.label = label
		self.current_track.similarity = similarity
		self.current_track.is_identity_locked = True

	def process_frame(self, frame: np.ndarray) -> Dict[str, object]:
		"""
		处理一帧图像

		Returns:
			dict: {
				"label": str,
				"bbox": Optional[np.ndarray],
				"quality": float,
				"similarity": float,
				"tracked": bool,
				"frame": np.ndarray,
			}
		"""
		faces = self.app.get(frame)

		if not faces:
			if self.current_track is not None:
				self.current_track.missed_frames += 1
				if self.current_track.missed_frames > self.max_missed_frames:
					self.current_track = None
			return {
				"label": "无人脸",
				"bbox": None,
				"quality": 0.0,
				"similarity": 0.0,
				"tracked": False,
				"frame": frame,
			}

		center_face = self._pick_center_face(frame, faces)
		if center_face is None:
			return {
				"label": "无人脸",
				"bbox": None,
				"quality": 0.0,
				"similarity": 0.0,
				"tracked": False,
				"frame": frame,
			}

		self._update_track(frame, center_face)
		self._maybe_lock_identity(frame, center_face)

		if self.current_track is None:
			return {
				"label": "无人脸",
				"bbox": None,
				"quality": 0.0,
				"similarity": 0.0,
				"tracked": False,
				"frame": frame,
			}

		quality = self._face_quality_score(frame, center_face)
		label = self.current_track.label if self.current_track.is_identity_locked else "识别中"

		result = {
			"label": label,
			"bbox": self.current_track.bbox.copy(),
			"quality": quality,
			"similarity": self.current_track.similarity,
			"tracked": True,
			"frame": frame,
		}
		return result

	def annotate_frame(self, frame: np.ndarray, result: Dict[str, object]) -> np.ndarray:
		"""在图像上绘制识别结果"""
		output = frame.copy()
		bbox = result.get("bbox")
		if bbox is None:
			return output

		x1, y1, x2, y2 = [int(v) for v in bbox]
		label = str(result.get("label", "未知"))
		quality = float(result.get("quality", 0.0))
		sim = float(result.get("similarity", 0.0))

		color = (0, 200, 0) if label not in ("陌生人", "识别中") else (0, 165, 255)
		cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
		text = f"{label} | q={quality:.2f} | sim={sim:.2f}"
		cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
		return output

	def monitor_camera(self, camera_index: int = 0, window_name: str = "Face Monitor") -> None:
		"""启动实时监测（按 q 退出）"""
		cap = cv2.VideoCapture(camera_index)
		if not cap.isOpened():
			raise RuntimeError(f"无法打开摄像头: index={camera_index}")

		logger.info("人脸监测启动，按 q 退出")
		try:
			while True:
				ok, frame = cap.read()
				if not ok:
					logger.warning("摄像头读取失败")
					break

				result = self.process_frame(frame)
				show = self.annotate_frame(frame, result)
				cv2.imshow(window_name, show)

				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		finally:
			cap.release()
			cv2.destroyAllWindows()
			logger.info("人脸监测已停止")


if __name__ == "__main__":
	current_dir = Path(__file__).resolve().parent
	known_dir = current_dir / "known_faces"

	module = FaceRecognitionModule(
		known_faces_dir=str(known_dir),
		model_name="buffalo_s",
		use_gpu=True,
	)
	module.monitor_camera(camera_index=0)
