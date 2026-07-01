import cv2
import numpy as np


def _gray_arrays(img_pil):
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    return img, gray, gray_f, edge_mag


def extract_lightweight_features(img_pil):
    img, gray, gray_f, edge_mag = _gray_arrays(img_pil)
    h, _ = gray.shape

    cy_min, cy_max = h // 4, 3 * h // 4
    cx_min, cx_max = gray.shape[1] // 4, 3 * gray.shape[1] // 4
    center_area = gray[cy_min:cy_max, cx_min:cx_max]
    center_sum = np.sum(center_area)
    total_sum = np.sum(gray)
    surround_count = gray.size - center_area.size
    surround_mean = (total_sum - center_sum) / surround_count if surround_count else 0.0

    return [
        float(np.var(gray)),
        float(np.mean(edge_mag)),
        float(np.sum(edge_mag > 50) / gray.size),
        float((np.var(img[:, :, 0]) + np.var(img[:, :, 1]) + np.var(img[:, :, 2])) / 3.0),
        float(np.sum((gray < 15) | (gray > 240)) / gray.size),
        float(np.mean(np.abs(gray_f[:, 1:] - gray_f[:, :-1]))),
        float(abs(np.mean(center_area) - surround_mean)),
        float(np.sum(np.abs(gray_f[:, 1:] - gray_f[:, :-1]) < 3) / (gray.size - h)),
    ]


def extract_grid_features(img_pil, grid_size):
    img, gray, gray_f, edge_mag = _gray_arrays(img_pil)
    h, w = gray.shape
    features = extract_lightweight_features(img_pil)
    names = [
        "Global_Variance",
        "Global_EdgeMean",
        "Global_EdgeDensity",
        "Global_ColorVar",
        "Global_ExtremePix",
        "Global_TV",
        "Global_CenterSurround",
        "Global_Flatness",
    ]

    step_h, step_w = h // grid_size, w // grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            prefix = f"Grid_{row}_{col}"
            patch = gray[row * step_h:(row + 1) * step_h, col * step_w:(col + 1) * step_w]
            patch_edge = edge_mag[row * step_h:(row + 1) * step_h, col * step_w:(col + 1) * step_w]
            patch_f = gray_f[row * step_h:(row + 1) * step_h, col * step_w:(col + 1) * step_w]

            features.append(float(np.var(patch)))
            names.append(f"{prefix}_Var")
            features.append(float(np.mean(patch_edge)))
            names.append(f"{prefix}_Edge")
            tv = np.mean(np.abs(patch_f[:, 1:] - patch_f[:, :-1])) if patch_f.shape[1] > 1 else 0.0
            features.append(float(tv))
            names.append(f"{prefix}_TV")

            a, b = patch_f[0::2, 0::2], patch_f[0::2, 1::2]
            c, d = patch_f[1::2, 0::2], patch_f[1::2, 1::2]
            if a.shape == b.shape == c.shape == d.shape and a.size:
                high_freq = np.mean(np.abs((a + c) - (b + d)) + np.abs((a + b) - (c + d)) + np.abs((a + d) - (b + c)))
            else:
                high_freq = 0.0
            features.append(float(high_freq))
            names.append(f"{prefix}_HighFreq")

    return features, names


def extract_raw_pixel_features(img_pil, size=8):
    img = np.array(img_pil)
    img_tiny = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return [float(value) for value in img_tiny.flatten().tolist()]
