"""KITScenes Multimodal (KIT-MRT, arXiv:2606.02956) reader + ring-camera -> ERP panorama for PanoRoMa.

KITScenes has no 360 deg / fisheye camera: it has six undistorted pinhole "ring" cameras (3504 x 2272, f ~ 1843 px,
87 x 63 deg FOV) at yaws (ccw from forward) +1, +61, +121, -179, -118, -58 deg (60 deg apart, ~27 deg overlap).
`ErpStitcher` turns them into the panorama the PanoRoMa query expects (same convention as the Dur360BEV / Mapillary
ERPs: centre column = ego forward, azimuth increases to the right, top row = zenith). Coverage is only +-31 deg of
elevation; the rest is invalid (`valid` = 0, `crop_clean` cuts to the rows that are valid at every azimuth).

All file access (calibration, images, LiDAR, timestamps, poses) goes through the KIT-MRT devkit (Apache-2.0), used
unchanged from third_party/kitscenes (wrap, never edit; `KITSCENES_DEVKIT_DIR` overrides): `SensorDataLoader` for
calib.json / JPEGs / parquet / timestamps, `load_ego_poses` for poses.txt. The devkit is imported lazily, so the pure
geometry here (stitcher, crop, alignment check) works without the submodule. Scene layout: data/README.md.

Frames (VERIFIED on scene 142f1419 against calib.json, tests/test_kitscenes.py):
  * reference = ego = `lidar_top` frame (its T_to_reference is the identity): x forward, y left, z up;
  * cameras: OpenCV (x right, y down, z forward); T_to_reference maps camera -> reference;
  * the panorama centre is `pano_centre` (default: mean ring-camera position, ~(0, 0, -0.18) m in the reference
    frame), so a reference point p lands at  p - pano_centre  in the ERP frame (`bevloc.bev.spherical.lidar_to_erp`
    with t_cl = -pano_centre).
Parallax: the cameras sit up to ~0.2 m from pano_centre; seams are feathered (`blend_power`), objects closer than a
few metres ghost in the overlaps.
"""
from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from bevloc.bev.spherical import lidar_to_erp

RING_CAMERAS = ("camera_ring_front", "camera_ring_front_right", "camera_ring_rear_right",
                "camera_ring_rear", "camera_ring_rear_left", "camera_ring_front_left")   # clockwise from the front


@functools.lru_cache(maxsize=None)
def devkit():
    """The KIT-MRT `kitscenes` package, found like the other third_party code: $KITSCENES_DEVKIT_DIR, else
    third_party/kitscenes (git submodule). Importable without Lanelet2 (only its map_api needs it)."""
    env = os.environ.get("KITSCENES_DEVKIT_DIR", "")
    if env and not (Path(env) / "kitscenes/sensors.py").exists():                  # an explicit choice must not fall back
        raise ImportError(f"$KITSCENES_DEVKIT_DIR={env!r} is not a KITScenes devkit checkout (no kitscenes/sensors.py)")
    sub = Path(__file__).resolve().parents[3] / "third_party/kitscenes"
    root = env or (str(sub) if (sub / "kitscenes/sensors.py").exists() else None)
    if root is None:
        raise ImportError("KITScenes devkit not found: `git submodule update --init third_party/kitscenes` "
                          "or set KITSCENES_DEVKIT_DIR")
    sys.path.insert(0, root)            # first, so the chosen checkout wins over a pip-installed `kitscenes`
    import kitscenes
    from kitscenes import poses
    return kitscenes, poses


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray            # (3, 3)
    T_ref_cam: np.ndarray    # (4, 4) camera -> reference
    size: tuple              # (width, height) px

    @property
    def R(self):
        return self.T_ref_cam[:3, :3]

    @property
    def centre(self):
        return self.T_ref_cam[:3, 3]

    @property
    def yaw_deg(self):
        """Bearing of the optical axis, degrees counter-clockwise from reference forward (x) towards y (left)."""
        a = self.R[:, 2]
        return float(np.degrees(np.arctan2(a[1], a[0])))


def _loader(scene):
    """A devkit `SensorDataLoader` for a scene directory (or the loader itself)."""
    sdl = devkit()[0].SensorDataLoader
    return scene if isinstance(scene, sdl) else sdl(scene)


def load_calibration(scene, names=RING_CAMERAS):
    """{name: Camera} from the devkit's calibration of a scene directory / `SensorDataLoader`."""
    ld = _loader(scene)
    out = {}
    for n in names:
        c = ld.get_camera_calibration(n)
        size = c.image_size
        if size is None:                                                      # calib.json without `resolution`: read a JPEG header
            frames = ld.get_frame_indices(n)
            if not frames:
                raise FileNotFoundError(f"{n}: calib.json has no resolution and the camera directory has no images")
            size = ld.get_camera_image_size(n, frames[0])
        out[n] = Camera(n, np.asarray(c.intrinsic, np.float64), np.asarray(c.extrinsic, np.float64),
                        (int(size[0]), int(size[1])))
    return out


def lidar_extrinsic(scene, name="lidar_top"):
    """(4, 4) T_lidar_to_reference."""
    return np.asarray(_loader(scene).get_lidar_extrinsic(name), np.float64)


def erp_rays(width, height):
    """(H, W, 3) unit rays in the ERP frame (x forward, y left, z up): centre column forward, azimuth to the right."""
    az = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    el = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    az, el = np.meshgrid(az, el)
    return np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], -1)


def project(cam: Camera, pts_ref):
    """Reference-frame points (N, 3) -> pixels (N, 2) in `cam` (full resolution) and depth along its optical axis."""
    pc = (np.asarray(pts_ref, np.float64) - cam.centre) @ cam.R           # = R^T (p - t)
    z = pc[:, 2]
    uv = (pc @ cam.K.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return uv, z


class ErpStitcher:
    """Precomputed ERP <- ring-camera lookup maps (one per camera) for one calibration and ERP size.

    `scale` pre-shrinks each camera image (cv2.INTER_AREA) before the bilinear remap: the ERP at 2048 px is ~7x
    coarser than the 3504 px cameras, so sampling the full-resolution image would alias.
    """

    def __init__(self, cams: dict, size=(2048, 1024), pano_centre="mean", scale=0.25, blend_power=4.0):
        self.cams = cams
        self.size = (int(size[0]), int(size[1]))
        self.scale = float(scale)
        self.pano_centre = (np.mean([c.centre for c in cams.values()], 0) if isinstance(pano_centre, str)
                            else np.asarray(pano_centre, np.float64))
        w, h = self.size
        d = erp_rays(w, h)
        self.maps = {}
        for n, c in cams.items():
            # ERP ray -> camera pixel for a point at infinity (translation ignored: exact for far content, the
            # parallax of near content is the ~0.2 m camera offset from pano_centre, see module doc)
            pc = d @ c.R                                                  # directions in the camera frame
            z = pc[..., 2]
            uv = pc @ c.K.T
            u, v = uv[..., 0] / np.where(z > 1e-6, z, 1.0), uv[..., 1] / np.where(z > 1e-6, z, 1.0)
            W, H = c.size
            # feather: 1 at the image centre, 0 at the border (rectangular, so corners are not over-weighted)
            f = np.minimum(1 - np.abs(2 * u / W - 1), 1 - np.abs(2 * v / H - 1))
            ok = (z > 1e-6) & (f > 0)
            wgt = np.where(ok, np.clip(f, 0, 1) ** blend_power, 0.0).astype(np.float32)
            s = self.scale
            # cv2 pixel centres sit at integer coordinates; after the area shrink by s: u' = (u + 0.5) * s - 0.5
            self.maps[n] = (((u + 0.5) * s - 0.5).astype(np.float32), ((v + 0.5) * s - 0.5).astype(np.float32), wgt)

    def __call__(self, images: dict):
        """images {camera: (H, W, 3) uint8 RGB at full resolution} -> (erp uint8 (h, w, 3), valid bool (h, w))."""
        w, h = self.size
        acc = np.zeros((h, w, 3), np.float32)
        tot = np.zeros((h, w), np.float32)
        for n, (mu, mv, wgt) in self.maps.items():
            if not wgt.any():
                continue
            img = images[n]
            if self.scale != 1.0:
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            samp = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            acc += samp * wgt[..., None]
            tot += wgt
        valid = tot > 0
        erp = np.where(valid[..., None], acc / np.maximum(tot, 1e-9)[..., None], 0)
        return np.clip(erp + 0.5, 0, 255).astype(np.uint8), valid


def clean_band(valid, ego_mask_deg=None, max_elevation_deg=None):
    """Rows (r0, r1) of the largest ERP band that has image at EVERY azimuth (no black pixel), i.e. the elevation
    range all six cameras jointly cover (the coverage narrows at the seams between cameras). Optional extra cuts:
    `ego_mask_deg` drops rows more than that many degrees below the horizon (the ego car's body / roof sensors),
    `max_elevation_deg` drops rows above that elevation. Returns (r0, r1, (el_top_deg, el_bottom_deg)), r1 exclusive;
    raises if no row is fully covered."""
    h = valid.shape[0]
    el = (0.5 - (np.arange(h) + 0.5) / h) * 180.0                      # row-centre elevation, + up
    ok = valid.all(1)
    if ego_mask_deg:
        ok &= el >= -float(ego_mask_deg)
    if max_elevation_deg is not None:
        ok &= el <= float(max_elevation_deg)
    rows = np.where(ok)[0]
    if not len(rows):
        raise ValueError("no ERP row is covered at every azimuth")
    # the longest run of consecutive ok rows (coverage is one band; guard against stray rows)
    cuts = np.where(np.diff(rows) > 1)[0]
    runs = np.split(rows, cuts + 1)
    run = max(runs, key=len)
    r0, r1 = int(run[0]), int(run[-1]) + 1
    return r0, r1, (float(90.0 - r0 * 180.0 / h), float(90.0 - r1 * 180.0 / h))


def crop_clean(erp, valid, ego_mask_deg=None, max_elevation_deg=None):
    """ERP and mask cropped to `clean_band`: a panorama with no unavailable pixel. Returns (erp, valid, info) where info
    holds the rows / elevations so the crop can be mapped back to the full ERP (row = r0 + row_in_crop)."""
    r0, r1, (top, bot) = clean_band(valid, ego_mask_deg, max_elevation_deg)
    return erp[r0:r1], valid[r0:r1], {"row0": r0, "row1": r1, "elevation_top_deg": top, "elevation_bottom_deg": bot,
                                       "size_full": [int(valid.shape[1]), int(valid.shape[0])]}


def quat_to_R(q):
    """(qx, qy, qz, qw) -> 3x3 rotation matrix (the devkit's / TUM quaternion order)."""
    return Rotation.from_quat(np.asarray(q, np.float64)).as_matrix()


class KitScene:
    """One extracted scene: ring-camera images, LiDAR sweeps, ego poses, calibration, all read by the devkit."""

    def __init__(self, scene):
        self.path = Path(scene)
        self.loader = _loader(self.path)                                       # devkit SensorDataLoader
        self.cams = load_calibration(self.loader)
        # the devkit returns empty arrays / lists for missing files instead of raising: a scene that is not fully
        # extracted must fail here, not as an empty dataset or an un-deskewed sweep later
        self.timestamps_ns = self.loader.get_reference_timestamps() if (self.path / "timestamp.reference.txt").exists() else []
        ego = devkit()[1].load_ego_poses(self.path)                            # list of EgoPose (UTM-local, TUM file)
        if not len(self.timestamps_ns):
            raise FileNotFoundError(f"{self.path}: timestamp.reference.txt missing or empty")
        if not ego:
            raise FileNotFoundError(f"{self.path}: poses.txt missing or empty (the LiDAR cannot be deskewed without it)")
        self.pose_t = np.array([p.timestamp_ns for p in ego], np.int64) / 1e9
        self.pose_xyz = np.array([p.translation for p in ego], np.float64)
        self.pose_q = np.array([p.rotation for p in ego], np.float64)          # (qx, qy, qz, qw)
        # frame index = the file stem (images, sweeps); poses and timestamps are indexed by position in the lists.
        # They coincide only if the stems are 0..N-1, so keep the map and refuse a scene where the counts disagree.
        self.frame_ids = self.loader.get_frame_indices("lidar_top")
        if not (len(self.frame_ids) == len(ego) == len(self.timestamps_ns)):
            raise ValueError(f"{self.path}: {len(self.frame_ids)} lidar_top sweeps, {len(ego)} poses, "
                             f"{len(self.timestamps_ns)} reference timestamps")
        self._pos = {f: i for i, f in enumerate(self.frame_ids)}

    def __len__(self):
        return len(self.timestamps_ns)

    def image(self, cam, idx):
        """(H, W, 3) uint8 RGB."""
        return self.loader.get_camera_image(cam, idx)

    def images(self, idx, names=None):
        return {n: self.image(n, idx) for n in (names or self.cams)}

    def lidar(self, idx, name="lidar_top", with_ring=False):
        """(N, 3) float64 points in the reference frame + (N,) reflectivity (all-zero returns dropped); with
        `with_ring` also the (N,) beam index. `idx` is the file stem, as for `image`.

        The devkit's default sweep: ego-motion deskewed with each point's timestamp and the interpolated pose (the
        sweep takes 100 ms; up to ~0.6 m of shift at speed) and moved into the reference frame by the sensor's
        extrinsic. The devkit does that only for points that have a timestamp, and returns the raw sensor-frame sweep
        if it cannot deskew at all (e.g. no poses); the points it did not move are rotated into the reference frame
        here, so every point is in the reference frame whatever the LiDAR. Those fallback points are not deskewed.
        """
        sw = self.loader.get_lidar_sweep(name, idx)
        T = lidar_extrinsic(self.loader, name)
        arr, raw = sw.points, sw.raw()
        xyz = np.stack([arr[k] for k in "xyz"], 1).astype(np.float64)
        raw_xyz = np.stack([raw[k] for k in "xyz"], 1)
        ts = raw["timestamp"] if "timestamp" in raw.dtype.names else np.zeros(len(raw))
        moved = (np.asarray(ts) > 0) & (np.abs(raw_xyz).sum(1) > 0) if sw.deskewed else np.zeros(len(raw), bool)
        xyz[~moved] = xyz[~moved] @ T[:3, :3].T + T[:3, 3]                      # what the devkit left in the sensor frame
        keep = np.abs(raw_xyz).sum(1) > 0                                       # drop all-zero returns (no echo)
        out = (xyz[keep], arr["reflectivity"][keep])
        return out + (arr["ring"][keep],) if with_ring else out

    def ego_pose(self, idx):
        """(R (3, 3) ego -> map, t (3,)) of frame `idx` (file stem) in the poses.txt frame (local metres around
        maps/origin.json)."""
        i = self._pos[idx]
        return quat_to_R(self.pose_q[i]), self.pose_xyz[i]

    def heading_vs_travel(self, min_step_m=0.3):
        """Per-step (ego-forward yaw - direction of travel) in degrees, wrapped to +-180: should centre on 0 if the
        pose frame is x-forward and the quaternion is ego -> map."""
        d = np.diff(self.pose_xyz[:, :2], axis=0)
        m = np.linalg.norm(d, axis=1) > min_step_m
        fwd = np.array([quat_to_R(q)[:2, 0] for q in self.pose_q[:-1]])
        a = np.degrees(np.arctan2(fwd[:, 1], fwd[:, 0]) - np.arctan2(d[:, 1], d[:, 0]))
        return ((a[m] + 180) % 360) - 180


def depth_edge_alignment(erp, pts, ring, pano_centre, shifts_px=range(-24, 25, 4), jump_m=1.5, max_range_m=60.0):
    """Calibration check: image gradient at LiDAR depth discontinuities as the projection is shifted sideways.

    Points are ordered by azimuth within each ring; a depth edge is a pair of consecutive returns on one ring more than
    `jump_m` apart in range (both nearer than `max_range_m`). Walls, poles and cars against the background have a
    colour edge exactly there, so the mean |d(gray)/du| sampled at the projected nearer-return of each edge peaks where
    the camera <-> LiDAR calibration (and the stitch) is right. `pts` (N, 3) reference-frame points, `ring` (N,) beam
    index. Returns {"curve": {shift_px: mean gradient}, "peak_shift_px", "n_edge_points"}.
    """
    h, w = erp.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(erp, cv2.COLOR_RGB2GRAY), (0, 0), 1.5).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    rel = np.asarray(pts, np.float64) - np.asarray(pano_centre)
    r = np.linalg.norm(rel, axis=1)
    az = np.arctan2(rel[:, 1], rel[:, 0])
    edge = []
    for k in np.unique(ring):
        i = np.where(ring == k)[0]
        i = i[np.argsort(az[i])]
        ra, rb = r[i][:-1], r[i][1:]
        m = (np.abs(ra - rb) > jump_m) & (np.maximum(ra, rb) < max_range_m) & (np.minimum(ra, rb) > 2.0)
        edge.append(np.where(ra < rb, i[:-1], i[1:])[m])                        # the nearer return of each pair
    edge = np.concatenate(edge) if edge else np.zeros(0, int)
    u, v, _ = lidar_to_erp(rel[edge], w, h)
    ok = (v > 2) & (v < h - 3)
    u, v = u[ok], v[ok]
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    curve = {int(s): float(gx[vi, np.round(u + s).astype(int) % w].mean()) for s in shifts_px}
    return {"curve": curve, "peak_shift_px": max(curve, key=curve.get), "n_edge_points": int(len(u))}
