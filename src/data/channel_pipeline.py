import cv2
import numpy as np
from PIL import Image


def _illumination_correction(channel: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(channel, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = cv2.subtract(channel, blurred)
    corrected = cv2.add(corrected, np.full_like(corrected, 128))
    return corrected


def _clahe(channel: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return clahe.apply(channel)


def _white_tophat(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
    return cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)


class ChannelDecomposition:
    """Decompoe a imagem RGB em tres canais engenheirados focados em exsudatos.

    - Canal 0 (slot R): correcao de iluminacao no R (subtracao de fundo gaussiano).
    - Canal 1 (slot G): CLAHE no G (realce de contraste local).
    - Canal 2 (slot B): top-hat morfologico no G (realce de pequenas estruturas brilhantes).
    """

    def __init__(
        self,
        illumination_sigma: float = 30.0,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid: int = 8,
        tophat_kernel_size: int = 15,
    ):
        self.illumination_sigma = illumination_sigma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.tophat_kernel_size = tophat_kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.uint8)
        r, g, _ = cv2.split(arr)

        ch0 = _illumination_correction(r, self.illumination_sigma)
        ch1 = _clahe(g, self.clahe_clip_limit, self.clahe_tile_grid)
        ch2 = _white_tophat(g, self.tophat_kernel_size)

        out = cv2.merge([ch0, ch1, ch2])
        return Image.fromarray(out)
