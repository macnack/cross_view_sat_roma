"""Display helpers. Nothing here touches the data that goes to the matcher."""
from __future__ import annotations

import cv2
import numpy as np

INVALID_GREY = 90


def enhance(rgb):
    """RGB -> brightened BGR for display: the recordings are dark and flat (overcast, wet)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255
    return (np.power(out, 0.65) * 255).astype(np.uint8)


def range_colours(r, span=30.0):
    return cv2.applyColorMap(np.clip(np.asarray(r) / span * 255, 0, 255).astype(np.uint8)[:, None],
                             cv2.COLORMAP_TURBO)[:, 0]


def bev_tile(img, valid, title, cell_m, z=3, edge=None, track=None, rings=(10, 20)):
    t = cv2.resize(enhance(img), None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    up = lambda m: cv2.resize(m.astype(np.uint8), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST) > 0
    t[~up(valid)] = INVALID_GREY
    if edge is not None:
        t[up(edge)] = (0, 0, 255)                 # display colour; the query itself uses near-black
    c = t.shape[0] // 2
    if track is None:
        for r in rings:
            rad = int(r / cell_m * z)
            cv2.circle(t, (c, c), rad, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(t, f"{r} m", (c + 4, c - rad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(t, (c, c + 10), (c, c - 22), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)
    else:
        for p in track:
            cv2.circle(t, (int(p[0] * z), int(p[1] * z)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(t, (0, 0), (t.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(t, f"{title}   valid {valid.mean():.0%}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return t


def erp_strip(erp, erp_valid, contact_row, width, title=""):
    e = enhance(erp)
    if erp_valid is not None:
        e[~erp_valid] = (0.4 * e[~erp_valid] + 0.6 * INVALID_GREY).astype(np.uint8)
    if contact_row is not None:
        for u, v in enumerate(contact_row):
            cv2.circle(e, (u, int(v)), 1, (0, 255, 255), -1)
    e = cv2.resize(e, (width, width // 2), interpolation=cv2.INTER_CUBIC)[int(0.18 * width):]
    cv2.putText(e, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return e


def grid_of(tiles, cols):
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(INVALID_GREY,) * 3) for r in rows]
    return np.vstack(rows)


def overlay_points(bgr, u, v, colours, radius=1, zoom=1.0, origin=(0, 0)):
    for uu, vv, cc in zip((u - origin[0]) * zoom, (v - origin[1]) * zoom, colours):
        cv2.circle(bgr, (int(uu), int(vv)), radius, tuple(int(t) for t in cc), -1)
    return bgr


# --- feature / pose panels shared by the Lift-Splat and hybrid layer sheets ---


def mass_to_bgr(valid_hw):
    u8 = (np.clip(valid_hw.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_BONE)


def pca_rgb(feat_bchw):
    """First 3 PCA components of channels → RGB uint8 (H, W, 3)."""
    f = feat_bchw[0].detach().float().cpu().numpy()  # C,H,W
    C, H, W = f.shape
    X = f.reshape(C, -1).T  # HW, C
    X = X - X.mean(0, keepdims=True)
    # thin PCA via covariance on channels
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    basis = vecs[:, -3:][:, ::-1]  # C,3
    Y = X @ basis  # HW,3
    for i in range(3):
        lo, hi = np.percentile(Y[:, i], [2, 98])
        Y[:, i] = np.clip((Y[:, i] - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = (Y.reshape(H, W, 3) * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def heat_to_bgr(heat_kk, ref_hw):
    h = heat_kk / (heat_kk.max() + 1e-8)
    u8 = (h * 255).astype(np.uint8)
    cm = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.resize(cm, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_NEAREST)


def pose_overlay(ref_rgb, H_gt, H_est, n):
    canvas = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR).copy()
    box(canvas, H_gt, n, (0, 220, 0), 2)
    if H_est is not None:
        box(canvas, H_est, n, (0, 0, 255), 2)
    return canvas


def box(img, H, n, color, thick):
    corners = np.array([[0, 0, 1], [n - 1, 0, 1], [n - 1, n - 1, 1], [0, n - 1, 1]], float)
    p = corners @ np.asarray(H, float).T
    pts = np.ascontiguousarray(np.round(p[:, :2] / p[:, 2:3]).astype(np.int32)).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)


def fit(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
