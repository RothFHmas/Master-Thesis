"""
Visual Hull – COLMAP (Shared Intrinsics) + Roboflow COCO-Export
================================================================
Keine torchhull-Abhängigkeit – läuft auf Windows ohne CUDA-Build.
Voxel-Carving + Marching Cubes.

Abhängigkeiten:
    pip install numpy scipy scikit-image trimesh pillow opencv-python pycocotools

Verwendung:
    python visual_hull_colmap.py \\
        --colmap   ./colmap_sparse \\
        --roboflow ./roboflow_export \\
        --output   mesh.obj

    python visual_hull_colmap.py \\
        --colmap   ./colmap_sparse \\
        --roboflow ./roboflow_export \\
        --grid 256 --output mesh_hq.obj --preview
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import trimesh


# ══════════════════════════════════════════════════════════════
# 1. COLMAP laden
# ══════════════════════════════════════════════════════════════

@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: list

@dataclass
class ColmapImage:
    id: int
    qw: float; qx: float; qy: float; qz: float
    tx: float; ty: float; tz: float
    camera_id: int
    name: str


def _quat_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def load_colmap_cameras(txt):
    cams = {}
    for line in Path(txt).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = line.split()
        cams[int(p[0])] = ColmapCamera(
            id=int(p[0]), model=p[1],
            width=int(p[2]), height=int(p[3]),
            params=[float(x) for x in p[4:]],
        )
    return cams


def load_colmap_images(txt):
    imgs, lines = [], []
    for l in Path(txt).read_text().splitlines():
        l = l.strip()
        if l and not l.startswith("#"): lines.append(l)
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        imgs.append(ColmapImage(
            id=int(p[0]),
            qw=float(p[1]), qx=float(p[2]), qy=float(p[3]), qz=float(p[4]),
            tx=float(p[5]), ty=float(p[6]), tz=float(p[7]),
            camera_id=int(p[8]), name=p[9],
        ))
    return imgs


def colmap_intrinsics(cam):
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    elif cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        return p[0], p[1], p[2], p[3]
    else:
        f  = p[0]
        cx = p[1] if len(p) > 1 else cam.width  / 2.0
        cy = p[2] if len(p) > 2 else cam.height / 2.0
        return f, f, cx, cy


def get_camera_params(cam, img):
    fx, fy, cx, cy = colmap_intrinsics(cam)
    K = np.array([[fx, 0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    R = _quat_to_R(img.qw, img.qx, img.qy, img.qz)
    t = np.array([img.tx, img.ty, img.tz], dtype=np.float64)
    return K, R, t


# ══════════════════════════════════════════════════════════════
# 2. Szenen-Bounds: Objekt-Mittelpunkt per Strahlenschnitt
# ══════════════════════════════════════════════════════════════

def estimate_object_center(col_images, cameras):
    """
    Schätzt den Objektmittelpunkt durch Triangulation der Kamera-Blickrichtungen.
    Jede Kamera liefert einen Strahl: Ursprung=Kamerazentrum, Richtung=optische Achse.
    Der Punkt mit minimalem Gesamtabstand zu allen Strahlen = Objektzentrum.
    """
    origins    = []
    directions = []

    for ci in col_images:
        cam = cameras[ci.camera_id]
        R   = _quat_to_R(ci.qw, ci.qx, ci.qy, ci.qz)
        t   = np.array([ci.tx, ci.ty, ci.tz])

        # Kamerazentrum in Weltkoordinaten
        C = -R.T @ t
        # Optische Achse in Weltkoordinaten: dritte Spalte von R^T (= dritte Zeile von R), negiert
        d = R.T @ np.array([0.0, 0.0, 1.0])
        d = d / np.linalg.norm(d)

        origins.append(C)
        directions.append(d)

    origins    = np.array(origins)     # (N, 3)
    directions = np.array(directions)  # (N, 3)

    # Kleinste-Quadrate Lösung: Punkt P der den Gesamtabstand zu allen Strahlen minimiert
    # Für jeden Strahl i: Abstand = ||(P - O_i) - ((P - O_i)·d_i) * d_i||
    # Normal-Gleichungssystem aufbauen
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)
    for O, d in zip(origins, directions):
        M  = I - np.outer(d, d)
        A += M
        b += M @ O

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        center = origins.mean(axis=0)

    return center


def compute_scene_bounds(col_images, cameras, margin=1.3):
    """
    Berechnet Bounding Box des Objekts.

    Strategie:
    1. Objektzentrum per Strahlenschnitt aller optischen Achsen
    2. Radius = Median-Abstand der Kameras vom Zentrum × Skalierungsfaktor
       (Objekt ist kleiner als der Kamerakreis)
    """
    center = estimate_object_center(col_images, cameras)

    # Kamerazentren
    cam_centers = []
    for ci in col_images:
        R = _quat_to_R(ci.qw, ci.qx, ci.qy, ci.qz)
        t = np.array([ci.tx, ci.ty, ci.tz])
        cam_centers.append(-R.T @ t)
    cam_centers = np.array(cam_centers)

    # Abstand der Kameras vom geschätzten Objektzentrum
    dists = np.linalg.norm(cam_centers - center, axis=1)
    cam_radius = np.median(dists)

    # Objektgröße schätzen: Öffnungswinkel × Abstand
    # Mit fx ≈ 6328 und Bildbreite 3072 → FOV ≈ 27°
    # Objekt füllt etwa 20-40% des Bildes → Radius ≈ 0.3 × Kameraabstand
    obj_radius = cam_radius * 0.35 * margin

    lo = center - obj_radius
    hi = center + obj_radius

    return lo, hi, center, cam_radius, obj_radius


# ══════════════════════════════════════════════════════════════
# 3. Roboflow COCO-Masken laden
# ══════════════════════════════════════════════════════════════

ROBOFLOW_SPLITS = ("train", "valid", "test")
ROBOFLOW_JSON   = "_annotations.coco.json"


def load_roboflow_masks(root):
    root   = Path(root)
    splits = [root / s for s in ROBOFLOW_SPLITS if (root / s / ROBOFLOW_JSON).exists()]
    if not splits:
        for name in (ROBOFLOW_JSON, "annotations.json"):
            if (root / name).exists(): splits = [root]; break
    if not splits:
        raise FileNotFoundError(f"Kein Roboflow-Export in: {root}")

    all_masks = {}
    for split_dir in splits:
        json_path = split_dir / ROBOFLOW_JSON
        if not json_path.exists():
            cands = list(split_dir.glob("*.json"))
            if not cands: continue
            json_path = cands[0]

        with open(json_path) as f:
            data = json.load(f)

        id_info = {img["id"]: {"name": Path(img["file_name"]).name,
                                "w": img["width"], "h": img["height"]}
                   for img in data["images"]}

        for ann in data["annotations"]:
            info = id_info[ann["image_id"]]
            name, W, H = info["name"], info["w"], info["h"]
            seg = ann.get("segmentation", [])

            if isinstance(seg, dict):
                try:
                    from pycocotools import mask as M
                    rle   = M.frPyObjects(seg, H, W)
                    layer = M.decode(rle).astype(np.float32)
                except ImportError:
                    print("WARNUNG: pycocotools fehlt → pip install pycocotools")
                    continue
            else:
                layer = np.zeros((H, W), dtype=np.uint8)
                for poly in seg:
                    if len(poly) < 6: continue
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(layer, [pts], 1)
                layer = layer.astype(np.float32)

            all_masks[name] = np.clip(all_masks.get(name, 0) + layer, 0, 1)

        print(f"  Split '{split_dir.name}': "
              f"{len(data['images'])} Bilder, {len(data['annotations'])} Annotationen")

    return all_masks


def match_name(colmap_name, coco_masks):
    if colmap_name in coco_masks: return colmap_name
    stem = Path(colmap_name).stem
    for k in coco_masks:
        if Path(k).name.startswith(stem): return k
    for k in coco_masks:
        if stem in k: return k
    return None


# ══════════════════════════════════════════════════════════════
# 4. Voxel-Carving
# ══════════════════════════════════════════════════════════════

def voxel_carving(silhouettes, camera_params, grid_res, lo, hi):
    """
    Voxel-Carving: Projiziert jeden Voxel in jede Silhouette.
    Voxel außerhalb der Silhouette → herausschnitzen.
    """
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    zs = np.linspace(lo[2], hi[2], grid_res)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_world = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # (N,3)

    occupied = np.ones(len(pts_world), dtype=bool)
    total = len(silhouettes)

    for i, (sil, (K, R, t)) in enumerate(zip(silhouettes, camera_params)):
        H, W = sil.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # Weltpunkte → Kamerakoordinaten
        pts_cam = (R @ pts_world.T).T + t   # (N,3)

        z = pts_cam[:, 2]
        in_front = z > 0.01

        # Projektion
        u = np.where(in_front, fx * pts_cam[:, 0] / (z + 1e-10) + cx, -1)
        v = np.where(in_front, fy * pts_cam[:, 1] / (z + 1e-10) + cy, -1)

        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)

        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & in_front

        in_sil = np.zeros(len(pts_world), dtype=bool)
        in_sil[in_bounds] = sil[vi[in_bounds], ui[in_bounds]] > 0.5

        # Voxel herausschnitzen wenn:
        # - vor der Kamera UND im Bild UND nicht in Silhouette
        occupied &= ~(in_front & in_bounds & ~in_sil)

        if (i + 1) % 10 == 0 or i == 0 or i == total - 1:
            print(f"  Kamera [{i+1:>3}/{total}]: {occupied.sum():,} Voxel verbleiben")

        if occupied.sum() == 0:
            print(f"  ⚠ Alle Voxel herausgeschnitzt bei Kamera {i+1}!")
            break

    return occupied.reshape(grid_res, grid_res, grid_res)


def voxels_to_mesh(voxel_grid, lo, hi):
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("FEHLER: scikit-image fehlt → pip install scikit-image")
        sys.exit(1)

    vol = voxel_grid.astype(np.float32)

    try:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=1.0)
    except ImportError:
        pass

    verts, faces, normals, _ = marching_cubes(vol, level=0.5)
    grid_res = voxel_grid.shape[0]
    scale    = (hi - lo) / grid_res
    verts_world = verts * scale + lo

    return trimesh.Trimesh(vertices=verts_world, faces=faces,
                           vertex_normals=normals)


# ══════════════════════════════════════════════════════════════
# 5. Hauptprogramm
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Visual Hull – COLMAP + Roboflow COCO → .obj",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--colmap",   required=True)
    ap.add_argument("--roboflow", required=True)
    ap.add_argument("--output",   default="visual_hull.obj")
    ap.add_argument("--grid",     type=int, default=128)
    ap.add_argument("--margin",   type=float, default=1.3,
                    help="Bounds-Margin – vergrößern wenn Objekt abgeschnitten (Standard: 1.3)")
    ap.add_argument("--preview",  action="store_true")
    ap.add_argument("--list",     action="store_true")
    args = ap.parse_args()

    colmap_dir   = Path(args.colmap)
    roboflow_dir = Path(args.roboflow)

    # ── COLMAP ───────────────────────────────────────────────
    print("[1/5] COLMAP einlesen …")
    for fname in ("cameras.txt", "images.txt"):
        if not (colmap_dir / fname).exists():
            print(f"FEHLER: {colmap_dir / fname} nicht gefunden.")
            sys.exit(1)

    cameras    = load_colmap_cameras(colmap_dir / "cameras.txt")
    col_images = load_colmap_images(colmap_dir  / "images.txt")
    n_cams = len(cameras)
    print(f"  {'Shared Intrinsics ✓' if n_cams == 1 else f'{n_cams} Kameras'}  |  {len(col_images)} Bilder")

    # ── Roboflow / COCO ──────────────────────────────────────
    print("[2/5] Roboflow COCO-Masken einlesen …")
    coco_masks = load_roboflow_masks(roboflow_dir)
    print(f"  {len(coco_masks)} Masken geladen")

    # ── --list ───────────────────────────────────────────────
    if args.list:
        missing = 0
        print(f"\n  {'COLMAP-Dateiname':<45}  {'Maske'}")
        print("  " + "─" * 70)
        for ci in col_images:
            m = match_name(ci.name, coco_masks)
            note = (f"✓  {m}" if m != ci.name else "✓") if m else ("✗  KEINE MASKE", missing := missing+1)[0]
            print(f"  {ci.name:<45}  {note}")
        print(f"\n  {'✓ Alle haben Masken' if not missing else f'⚠ {missing} ohne Maske'}")
        return

    # ── Silhouetten aufbauen ─────────────────────────────────
    print("[3/5] Silhouetten + Kameraposen aufbauen …")
    silhouettes, cam_params_list, skipped = [], [], []

    for i, ci in enumerate(col_images):
        matched = match_name(ci.name, coco_masks)
        if matched is None:
            skipped.append(ci.name); continue

        mask = coco_masks[matched]
        cam  = cameras[ci.camera_id]

        if mask.shape != (cam.height, cam.width):
            mask = cv2.resize(mask, (cam.width, cam.height),
                              interpolation=cv2.INTER_NEAREST)

        silhouettes.append(mask.astype(np.float32))
        cam_params_list.append(get_camera_params(cam, ci))

        idx = len(silhouettes)
        if idx == 1 or idx % 20 == 0 or i == len(col_images) - 1:
            note = f"  → '{matched}'" if matched != ci.name else ""
            print(f"  [{idx:>3}/{len(col_images)}] {ci.name}{note}")

    if skipped:
        print(f"  ⚠  {len(skipped)} Bilder ohne Maske übersprungen")
    if not silhouettes:
        print("FEHLER: Keine verwertbaren Bilder."); sys.exit(1)
    print(f"  → {len(silhouettes)} Bilder werden verarbeitet")

    # ── Szenen-Bounds ────────────────────────────────────────
    print("[4/5] Voxel-Carving …")
    lo, hi, center, cam_r, obj_r = compute_scene_bounds(
        col_images, cameras, margin=args.margin
    )
    print(f"  Geschätztes Objektzentrum : [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Kamera-Radius (median)    : {cam_r:.3f}")
    print(f"  Objekt-Radius (geschätzt) : {obj_r:.3f}")
    print(f"  Bounds lo: [{lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}]")
    print(f"  Bounds hi: [{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}]")
    print(f"  Gitter: {args.grid}³ = {args.grid**3:,} Voxel")

    voxel_grid = voxel_carving(
        silhouettes, cam_params_list,
        grid_res=args.grid, lo=lo, hi=hi,
    )

    n_filled = voxel_grid.sum()
    print(f"  Gefüllte Voxel: {n_filled:,} ({100*n_filled/args.grid**3:.1f}%)")

    if n_filled == 0:
        print("\nFEHLER: Alle Voxel herausgeschnitzt!")
        print("  Tipp: --margin 2.0 versuchen um größere Bounds zu nutzen")
        sys.exit(1)

    # ── Marching Cubes ───────────────────────────────────────
    print("[5/5] Mesh erstellen …")
    mesh = voxels_to_mesh(voxel_grid, lo, hi)

    out = Path(args.output)
    mesh.export(str(out))
    print(f"\n✓  Gespeichert: {out.resolve()}")
    print(f"   Vertices : {len(mesh.vertices):,}")
    print(f"   Faces    : {len(mesh.faces):,}")

    if args.preview:
        mesh.show()


if __name__ == "__main__":
    main()
