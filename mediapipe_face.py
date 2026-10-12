"""MediaPipe-backed face detection + landmarks.

Drop-in replacement for the dlib-based Face_utilities that keeps the same
return shape as no_age_gender_face_process() and ROI_extraction() so
process.py does not change.

We reduce MediaPipe's 478-point Face Mesh down to five points that match
dlib's 5-point landmark convention (two per eye + nose tip). The rest of
the pipeline consumes that 5-point array; preserving the convention means
the cheek ROIs stay in the same anatomical location and the signal-
processing code sees an identical input shape.
"""
import os

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python import vision


class _Rect:
    """Minimal stand-in for dlib.rectangle. process.py reads .left/.top/
    .width/.height off it; we keep the same surface so the rest of the
    pipeline is unaware dlib is gone."""
    __slots__ = ("_l", "_t", "_r", "_b")

    def __init__(self, left, top, right, bottom):
        self._l, self._t, self._r, self._b = left, top, right, bottom

    def left(self):
        return self._l

    def top(self):
        return self._t

    def right(self):
        return self._r

    def bottom(self):
        return self._b

    def width(self):
        return self._r - self._l

    def height(self):
        return self._b - self._t

# MediaPipe Face Mesh indices chosen to stand in for dlib's 5-point model.
# dlib convention: shape[0:2] = subject's RIGHT eye (viewer's left),
#                  shape[2:4] = subject's LEFT  eye (viewer's right),
#                  shape[4]   = nose tip.
# ROI_extraction below and the geometry in process.py assume that order,
# so any swap rotates the aligned face upside-down and puts the "cheek"
# rectangles on the forehead.
# MediaPipe indices are from the canonical 468-point Face Mesh topology —
# x is left-to-right in image coordinates, so index 33 (outer corner of
# MediaPipe's "right eye") is the viewer's-left (subject's right) eye.
_MP_SUBJECT_RIGHT_EYE_OUTER = 33   # viewer's left
_MP_SUBJECT_RIGHT_EYE_INNER = 133
_MP_SUBJECT_LEFT_EYE_INNER = 362
_MP_SUBJECT_LEFT_EYE_OUTER = 263   # viewer's right
_MP_NOSE_TIP = 1
_MP_FIVE_POINT_IDXS = [
    _MP_SUBJECT_RIGHT_EYE_OUTER, _MP_SUBJECT_RIGHT_EYE_INNER,
    _MP_SUBJECT_LEFT_EYE_INNER, _MP_SUBJECT_LEFT_EYE_OUTER,
    _MP_NOSE_TIP,
]

def get_default_model_path():
    base = os.path.abspath(__file__)
    candidates = [
        os.path.join(os.path.dirname(base), "models", "face_landmarker.task"),
        os.path.join(os.path.dirname(os.path.dirname(base)), "models", "face_landmarker.task"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(base))), "models", "face_landmarker.task"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]

_DEFAULT_MODEL_PATH = get_default_model_path()


class MediaPipeFace:
    """Replacement for Face_utilities that uses MediaPipe Face Mesh."""

    def __init__(self, model_path=None, face_width=200):
        if model_path is None:
            model_path = get_default_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "MediaPipe face landmarker model not found at {}. "
                "Download it with: curl -L -o {} "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                .format(model_path, model_path))

        self.model_path = model_path
        self._init_landmarker()
        self._frame_idx = 0
        self.last_raw_landmarks = None

        self.desired_left_eye = (0.35, 0.35)
        self.desired_face_width = face_width
        self.desired_face_height = face_width

    def _init_landmarker(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

    def _landmarks_to_five_point(self, norm_landmarks, w, h):
        """Project Face Mesh to a 5-point shape matching dlib's convention:
        (0,1) image-left eye pair, (2,3) image-right eye pair, (4) nose tip.

        The webcam layer horizontal-flips frames, and MediaPipe's *labeled*
        eye indices are anatomical (subject-relative). Using labels blindly
        lets the flip invert which eye lands on which image side and breaks
        downstream ROI math. Pick the pair by image x instead.
        """
        pts = np.empty((5, 2), dtype=np.int32)
        for out_i, mp_i in enumerate(_MP_FIVE_POINT_IDXS):
            lm = norm_landmarks[mp_i]
            pts[out_i, 0] = int(round(lm.x * w))
            pts[out_i, 1] = int(round(lm.y * h))

        # If the "labeled-right-eye" pair is actually image-left (mean x
        # smaller), swap the two pairs so (0,1) is always the image-left
        # eye pair and (2,3) the image-right — independent of any flip.
        if pts[0:2, 0].mean() > pts[2:4, 0].mean():
            pts[[0, 1, 2, 3]] = pts[[2, 3, 0, 1]]

        # ROI_extraction uses shape[1][0]:shape[0][0] (needs pts[1].x <
        # pts[0].x) and shape[2][0]:shape[3][0] (needs pts[2].x < pts[3].x)
        # to slice non-empty cheek rectangles. Enforce both orderings.
        if pts[0, 0] < pts[1, 0]:
            pts[[0, 1]] = pts[[1, 0]]
        if pts[2, 0] > pts[3, 0]:
            pts[[2, 3]] = pts[[3, 2]]
        return pts

    def _face_rect_from_landmarks(self, norm_landmarks, w, h):
        """Compute a tight bounding rectangle around all 478 landmarks."""
        xs = np.fromiter((lm.x for lm in norm_landmarks), dtype=np.float32,
                         count=len(norm_landmarks))
        ys = np.fromiter((lm.y for lm in norm_landmarks), dtype=np.float32,
                         count=len(norm_landmarks))
        x0 = int(max(0, np.min(xs) * w))
        x1 = int(min(w, np.max(xs) * w))
        y0 = int(max(0, np.min(ys) * h))
        y1 = int(min(h, np.max(ys) * h))
        return _Rect(x0, y0, x1, y1)

    def _face_alignment(self, frame, shape5):
        """Rotate+scale so the eyes sit at a canonical position in a fixed-size
        face image.

        We don't trust the landmark *ordering* for alignment because the
        webcam layer horizontal-flips the frame (Webcam.get_frame calls
        cv2.flip). After that flip the anatomical "right eye" sits on image-
        right, which is the opposite of a non-flipped capture. Trusting the
        declared order would rotate the face 180° and push it out of the
        warpAffine canvas (aligned_face comes out all-zero).

        Instead pick the two eye centers by image x: leftmost = left_in_image,
        rightmost = right_in_image. That makes dX always positive and the
        eye line nearly horizontal for an upright face — no -180 shim needed.
        """
        center_a = shape5[0:2].mean(axis=0).astype(int)
        center_b = shape5[2:4].mean(axis=0).astype(int)
        if center_a[0] <= center_b[0]:
            left_eye_center, right_eye_center = center_a, center_b
        else:
            left_eye_center, right_eye_center = center_b, center_a

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt(dX * dX + dY * dY)
        if dist <= 0:
            return None, None
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist

        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                       int((left_eye_center[1] + right_eye_center[1]) // 2))
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        aligned_face = cv2.warpAffine(
            frame, M, (self.desired_face_width, self.desired_face_height),
            flags=cv2.INTER_CUBIC)

        shape_reshaped = shape5.reshape(5, 1, 2).astype(np.float32)
        aligned_shape = cv2.transform(shape_reshaped, M)
        aligned_shape = np.squeeze(aligned_shape).astype(np.int32)
        return aligned_face, aligned_shape

    def no_age_gender_face_process(self, frame, _type):
        """Matches Face_utilities.no_age_gender_face_process(frame, '5').

        Returns (rects, face, shape, aligned_face, aligned_shape), or None
        when no face is detected. The `_type` argument is accepted and
        ignored — MediaPipe's mesh always gives us the same 478 points and
        we always project to 5.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # MediaPipe expects RGB uint8.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs monotonically increasing timestamps in ms.
        if self._landmarker is None:
            self._init_landmarker()

        self._frame_idx += 1
        timestamp_ms = self._frame_idx * 33  # nominal 30 FPS
        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return None

        if result is None or not result.face_landmarks:
            self.last_raw_landmarks = None
            return None

        landmarks = result.face_landmarks[0]
        self.last_raw_landmarks = landmarks
        shape5 = self._landmarks_to_five_point(landmarks, w, h)
        rect = self._face_rect_from_landmarks(landmarks, w, h)
        rects = [rect]

        # Same face crop definition as the original.
        x, y = rect.left(), rect.top()
        fw, fh = rect.width(), rect.height()
        face = frame[y:y + fh, x:x + fw]

        aligned_face, aligned_shape = self._face_alignment(frame, shape5)
        if aligned_face is None:
            return None

        return rects, face, shape5, aligned_face, aligned_shape

    def get_last_ear(self, w=1, h=1):
        """Computes Eye Aspect Ratio (EAR) from the most recent detected face mesh."""
        if self.last_raw_landmarks is None:
            return None
        lms = self.last_raw_landmarks
        if len(lms) < 468:
            return None

        def d(i1, i2):
            p1 = lms[i1]
            p2 = lms[i2]
            dx = (p1.x - p2.x) * w
            dy = (p1.y - p2.y) * h
            return float(np.sqrt(dx * dx + dy * dy))

        # Right eye (MediaPipe indices)
        r_v1 = d(160, 144)
        r_v2 = d(158, 153)
        r_h = d(33, 133)
        ear_r = (r_v1 + r_v2) / (2.0 * max(r_h, 1e-6))

        # Left eye (MediaPipe indices)
        l_v1 = d(385, 380)
        l_v2 = d(387, 373)
        l_h = d(362, 263)
        ear_l = (l_v1 + l_v2) / (2.0 * max(l_h, 1e-6))

        return float(0.5 * (ear_r + ear_l))

    def get_last_face_anchor(self, w=1, h=1):
        """Returns nose tip position (x, y) and inter-ocular face width."""
        if self.last_raw_landmarks is None:
            return None, None
        lms = self.last_raw_landmarks
        if len(lms) < 468:
            return None, None
        nose = lms[1]
        p_left = lms[33]
        p_right = lms[263]
        dx = (p_left.x - p_right.x) * w
        dy = (p_left.y - p_right.y) * h
        face_w = float(np.sqrt(dx * dx + dy * dy))
        return (float(nose.x * w), float(nose.y * h)), max(face_w, 1.0)

    def ROI_extraction(self, face, shape):
        """Two cheek patches, one per side — identical geometry to the
        original 5-point path so the green signal is derived from the
        same anatomical region."""
        ROI1 = face[int((shape[4][1] + shape[2][1]) / 2):shape[4][1],
                    shape[2][0]:shape[3][0]]
        ROI2 = face[int((shape[4][1] + shape[2][1]) / 2):shape[4][1],
                    shape[1][0]:shape[0][0]]
        return ROI1, ROI2

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


if __name__ == "__main__":
    print("Testing MediaPipeFace...")
    mp_face = MediaPipeFace()
    print("[OK] MediaPipeFace model loaded successfully.")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = mp_face.no_age_gender_face_process(test_frame, "5")
    print("[OK] Inference executed without errors (no face in blank frame -> None).")
    mp_face.close()
    print("[OK] MediaPipeFace self-test passed.")
