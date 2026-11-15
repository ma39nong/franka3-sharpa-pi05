import math
import numpy as np

def valid_box(box, width, height):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected finite pixel xyxy coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box must be ordered and inside the native image")
    return values


def square_box(box, width, height, scale=1.5):
    x1, y1, x2, y2 = valid_box(box, width, height)
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("Context scale must be finite and at least one")
    required = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1), 2)
    side = max(required, math.ceil(max(x2 - x1, y2 - y1) * scale))
    side = min(side, min(width, height) if required <= min(width, height) else max(width, height))

    def fit(center, low, high, length):
        start = math.floor(center - side / 2)
        lower = max(0, math.ceil(high - side)) if side <= length else length - side
        upper = min(length - side, math.floor(low)) if side <= length else 0
        if lower > upper:
            raise ValueError("Cannot contain interaction box")
        return min(max(start, lower), upper)

    left = fit((x1 + x2) / 2, x1, x2, width)
    top = fit((y1 + y2) / 2, y1, y2, height)
    return left, top, left + side, top + side


def native_crop(image, box):
    left, top, right, bottom = box
    side = right - left
    if side <= 0 or side != bottom - top:
        raise ValueError("Expected nonempty square crop")
    height, width = image.shape[:2]
    result = np.zeros((side, side, 3), dtype=np.uint8)
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    result[y1 - top : y2 - top, x1 - left : x2 - left] = image[y1:y2, x1:x2]
    return result


