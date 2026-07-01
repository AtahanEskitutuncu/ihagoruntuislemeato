import threading
import time
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------
# Yardımcı veri yapısı
# ------------------------------------------------------------------

class _SquareResult(NamedTuple):
    """
    _is_square_contour() başarılı olduğunda döndürülen sonuç.
    boundingRect'in iki kez hesaplanmasını önler.
    """
    approx: np.ndarray               # Onaylanan 4 köşeli kontur noktaları
    bbox:   Tuple[int, int, int, int]  # (x, y, w, h) — ROI koordinatı


# ------------------------------------------------------------------
# Parlaklık profilleri
# ------------------------------------------------------------------

_COLOR_PROFILES = {
    "low": {
        "red": [
            (np.array([0,   70,  40]), np.array([8,  255, 255])),
            (np.array([155, 65,  40]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([95,  60, 40]), np.array([135, 255, 255])),
        ],
        "orange": [
            (np.array([8, 70, 40]), np.array([30, 255, 255])),
        ],
    },
    "normal": {
        "red": [
            (np.array([0,   100, 60]), np.array([8,  255, 255])),
            (np.array([160, 95,  60]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([95,  80, 60]), np.array([130, 255, 255])),
        ],
        "orange": [
            (np.array([8, 100, 60]), np.array([28, 255, 255])),
        ],
    },
    "high": {
        "red": [
            (np.array([0,   120, 80]), np.array([8,  255, 255])),
            (np.array([162, 115, 80]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 100, 80]), np.array([130, 255, 255])),
        ],
        "orange": [
            (np.array([8, 120, 80]), np.array([30, 255, 255])),
        ],
    },
}

# Parlaklık eşikleri (frame ortalama V kanalı)
_BRIGHTNESS_LOW_MAX  = 80    # V < 80  → low
_BRIGHTNESS_HIGH_MIN = 170   # V > 170 → high
                              # arası   → normal


# ------------------------------------------------------------------
# Modül seviyesi yardımcı fonksiyon
# ------------------------------------------------------------------

def _angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """p2 köşesindeki açıyı hesaplar (derece cinsinden)."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _select_profile(hsv_roi: np.ndarray) -> dict:
    """ROI'nin ortalama parlaklığına (V kanalı) göre renk profilini seçer."""
    mean_v = cv2.mean(hsv_roi[:, :, 2])[0]

    if mean_v < _BRIGHTNESS_LOW_MAX:
        return _COLOR_PROFILES["low"]
    elif mean_v > _BRIGHTNESS_HIGH_MIN:
        return _COLOR_PROFILES["high"]
    else:
        return _COLOR_PROFILES["normal"]


def _build_mask(
    hsv: np.ndarray,
    ranges: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Verilen HSV aralık listesinden birleşik bir maske üretir."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for alt, ust in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, alt, ust))
    return mask


# ------------------------------------------------------------------
# Ana sınıf
# ------------------------------------------------------------------

class ColorDetector:
    """
    Kamera karesinde kırmızı ve mavi renkteki kareleri tespit eder.

    Ortam parlaklığına göre otomatik olarak üç HSV profili (low / normal / high)
    arasında geçiş yapar. Turuncu bölgeler kırmızı maskesinden çıkarılır.
    Eğimli/döndürülmüş kareleri (≈20° eğime kadar) tolere edecek şekilde
    açı, kenar oranı ve doluluk eşikleri gevşetilmiştir.

    Attributes:
        min_area:            Konturu değerlendirmeye almak için gereken minimum piksel alanı.
        side_ratio_max:      Kare kabulü için izin verilen maksimum kenar oranı (uzun/kısa).
                              Eğim arttıkça perspektifte kenarlar eşitsizleşir,
                              bu yüzden bu değer yüksek tutulur.
        bbox_fill_ratio_min: Kontur alanı / minAreaRect alanı eşiği.
                              minAreaRect kullanıldığı için kare hangi açıda
                              dönmüş olursa olsun oran ~1.0'a yakın kalır;
                              gürültü/pürüz payı için eşik biraz gevşetilmiştir.
        min_saturation:      HSV doygunluk eşiği.
        angle_tolerance:     90° etrafındaki kabul edilebilir sapma (derece).
                              Eğimli karelerde perspektif nedeniyle köşe açıları
                              90°'den sapar; bu yüzden tolerans yüksek tutulur.
        kernel_size:         Morfoloji işlemi için çekirdek boyutu.
        log_cooldown_sec:    Aynı renk için log mesajları arasındaki minimum süre.
    """

    _BGR_RED  = (0, 0, 255)
    _BGR_BLUE = (255, 0, 0)

    def __init__(
        self,
        logger=None,
        min_area: int = 400,
        side_ratio_max: float = 1.55,      # 20° eğim için yükseltildi (1.35 → 1.55)
        bbox_fill_ratio_min: float = 0.60, # minAreaRect ile uyumlu, biraz gevşetildi (0.65 → 0.60)
        min_saturation: int = 120,
        angle_tolerance: float = 28.0,     # 20° eğim için yükseltildi (20.0 → 28.0)
        kernel_size: int = 3,
        log_cooldown_sec: float = 3.0,
    ) -> None:
        self.logger = logger
        self.min_area = min_area
        self.side_ratio_max = side_ratio_max
        self.bbox_fill_ratio_min = bbox_fill_ratio_min
        self.min_saturation = min_saturation
        self.angle_tolerance = angle_tolerance
        self.log_cooldown_sec = log_cooldown_sec

        self._last_log_time: dict[str, float] = {}
        self._log_lock = threading.Lock()
        self._kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Açma → Kapama morfolojik işlemlerini uygular."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)
        return mask

    def _throttled_log(self, key: str, message: str) -> None:
        """Aynı anahtar için belirli aralıktan daha sık log yazmaz. Thread-safe."""
        if not self.logger:
            return
        now = time.time()
        with self._log_lock:
            if now - self._last_log_time.get(key, 0.0) > self.log_cooldown_sec:
                self.logger.info(message)
                self._last_log_time[key] = now

    def _is_square_contour(self, cnt: np.ndarray) -> Optional[_SquareResult]:
        """
        Konturun kare olup olmadığını doğrular.

        Kontrol adımları:
          1. Alan eşiği
          2. 4 köşeli poligon yaklaşımı
          3. Konvekslik
          4. Açı kontrolü (90° ± angle_tolerance)
          5. Kenar oranı kontrolü (side_ratio_max)
          6. minAreaRect doluluk oranı (bbox_fill_ratio_min)
             — boundingRect DEĞİL: kare döndürüldüğünde axis-aligned
             bounding box çok büyür ve oran düşer (yanlışlıkla elenir).
             minAreaRect, nesneyi açısına göre sıkıca sardığı için
             dönüklükten bağımsız doğru bir doluluk oranı verir.

        Returns:
            _SquareResult (approx, bbox) ya da None.
        """
        area = cv2.contourArea(cnt)
        if area < self.min_area:
            return None

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return None

        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        if len(approx) != 4:
            return None

        if not cv2.isContourConvex(approx):
            return None

        pts = approx.reshape(4, 2)
        angles = [
            _angle_between(pts[i - 1], pts[i], pts[(i + 1) % 4])
            for i in range(4)
        ]
        lo = 90.0 - self.angle_tolerance
        hi = 90.0 + self.angle_tolerance
        if any(a < lo or a > hi for a in angles):
            return None

        side_lengths = [
            np.linalg.norm(pts[i] - pts[(i + 1) % 4])
            for i in range(4)
        ]
        min_side = min(side_lengths)
        if min_side == 0:
            return None
        if max(side_lengths) / min_side > self.side_ratio_max:
            return None

        # minAreaRect: kare hangi açıda dönmüş olursa olsun doğru alanı verir.
        _, (rw, rh), _ = cv2.minAreaRect(approx)
        rot_rect_area = rw * rh
        if rot_rect_area == 0:
            return None
        if area / rot_rect_area < self.bbox_fill_ratio_min:
            return None

        bbox = cv2.boundingRect(approx)  # bbox sadece çizim/log için kullanılır

        return _SquareResult(approx=approx, bbox=bbox)

    # ------------------------------------------------------------------
    # Ana metodlar
    # ------------------------------------------------------------------

    def _detect_and_draw(
        self,
        frame: np.ndarray,
        roi: np.ndarray,
        hsv_roi: np.ndarray,
        mask: np.ndarray,
        color_name: str,
        roi_x_start: int,
        roi_y_start: int,
    ) -> None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bgr_color = self._BGR_RED if color_name == "Red" else self._BGR_BLUE

        for cnt in contours:
            result = self._is_square_contour(cnt)
            if result is None:
                continue

            approx = result.approx
            x, y, w, h = result.bbox

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [approx], -1, 255, -1)

            mean_s = cv2.mean(hsv_roi[:, :, 1], mask=contour_mask)[0]
            if mean_s < self.min_saturation:
                continue

            if color_name == "Red":
                mean_b = cv2.mean(roi[:, :, 0], mask=contour_mask)[0]
                mean_g = cv2.mean(roi[:, :, 1], mask=contour_mask)[0]
                mean_r = cv2.mean(roi[:, :, 2], mask=contour_mask)[0]
                if mean_g > mean_r * 0.55:
                    continue
                if mean_b > mean_r * 0.75:
                    continue

            approx_shifted = approx.copy()
            approx_shifted[:, :, 0] += roi_x_start
            approx_shifted[:, :, 1] += roi_y_start
            cv2.drawContours(frame, [approx_shifted], -1, bgr_color, 3)

            cx = x + w // 2 + roi_x_start
            cy = y + h // 2 + roi_y_start
            cv2.circle(frame, (cx, cy), 4, bgr_color, -1)
            cv2.putText(
                frame,
                f"{color_name} Square",
                (x + roi_x_start, y + roi_y_start - 7),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                bgr_color,
                2,
            )

            self._throttled_log(
                color_name,
                f"{color_name} Square detected at ({x + roi_x_start}, {y + roi_y_start})",
            )

    def process(
        self,
        frame: np.ndarray,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Karedeki kırmızı ve mavi kareleri tespit eder.

        Args:
            frame:    İşlenecek BGR kamera karesi.
            roi_rect: İsteğe bağlı (x_start, y_start, x_end, y_end) ROI dikdörtgeni.
                      Verilmezse tüm kare kullanılır.

        Returns:
            frame:         Üzerine çizim yapılmış BGR kare.
            combined_mask: Kırmızı | Mavi ikili maske.
            kirmizi_maske: Yalnızca kırmızı ikili maske.
            mavi_maske:    Yalnızca mavi ikili maske.
        """
        h, w = frame.shape[:2]

        if roi_rect is not None:
            rx0, ry0, rx1, ry1 = roi_rect
        else:
            rx0, ry0, rx1, ry1 = 0, 0, w, h

        rx0 = max(0, min(rx0, w))
        rx1 = max(0, min(rx1, w))
        ry0 = max(0, min(ry0, h))
        ry1 = max(0, min(ry1, h))

        if rx1 <= rx0 or ry1 <= ry0:
            return (
                frame,
                np.zeros((h, w), dtype=np.uint8),
                np.zeros((h, w), dtype=np.uint8),
                np.zeros((h, w), dtype=np.uint8),
            )

        roi = frame[ry0:ry1, rx0:rx1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- DEBUG BAŞLANGIC (gerekirse yorumdan çıkar) ---
        # def _mouse_hsv(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         h_val, s_val, v_val = param[y, x]
        #         mean_v = cv2.mean(param[:, :, 2])[0]
        #         print(f"Tıklanan HSV → H:{h_val}  S:{s_val}  V:{v_val} | Frame ort. V:{mean_v:.1f}")
        #
        # cv2.imshow("HSV_debug", hsv)
        # cv2.setMouseCallback("HSV_debug", _mouse_hsv, hsv)
        # cv2.waitKey(1)
        # --- DEBUG SONU ---

        # Parlaklığa göre profil seç
        profile = _select_profile(hsv)

        # Profildeki aralıklardan maskeler üret
        kirmizi_roi_maske = _build_mask(hsv, profile["red"])
        mavi_roi_maske    = _build_mask(hsv, profile["blue"])
        turuncu_roi_maske = _build_mask(hsv, profile["orange"])

        # Morfoloji
        kirmizi_roi_maske = self._apply_morphology(kirmizi_roi_maske)
        mavi_roi_maske    = self._apply_morphology(mavi_roi_maske)
        turuncu_roi_maske = self._apply_morphology(turuncu_roi_maske)

        # Turuncu sınır piksellerini genişlet, kırmızıdan çıkar
        turuncu_dilated = cv2.dilate(turuncu_roi_maske, self._kernel, iterations=1)
        kirmizi_roi_maske = cv2.bitwise_and(
            kirmizi_roi_maske,
            cv2.bitwise_not(turuncu_dilated)
        )

        self._detect_and_draw(frame, roi, hsv, kirmizi_roi_maske, "Red",  rx0, ry0)
        self._detect_and_draw(frame, roi, hsv, mavi_roi_maske,    "Blue", rx0, ry0)

        combined_roi_mask = cv2.bitwise_or(kirmizi_roi_maske, mavi_roi_maske)

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        kirmizi_maske = np.zeros((h, w), dtype=np.uint8)
        mavi_maske    = np.zeros((h, w), dtype=np.uint8)

        combined_mask[ry0:ry1, rx0:rx1] = combined_roi_mask
        kirmizi_maske[ry0:ry1, rx0:rx1] = kirmizi_roi_maske
        mavi_maske[ry0:ry1,    rx0:rx1] = mavi_roi_maske

        return frame, combined_mask, kirmizi_maske, mavi_maske
