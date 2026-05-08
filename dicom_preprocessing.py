import cv2
import numpy as np
import pydicom

def dicom_to_rgb(ds):
    try:
        img = ds.pixel_array.astype(float)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope = getattr(ds, 'RescaleSlope', 1)
        img = img * slope + intercept
        center = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        width = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        low, high = center - width // 2, center + width // 2
        img = np.clip(img, low, high)
        img = (img - low) / (high - low) * 255.0
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Ошибка чтения DICOM: {e}")
        return np.zeros((512, 512, 3), dtype=np.uint8)
