# Visual Hull Reconstruction Pipeline

Silhouette-based 3D reconstruction using COLMAP camera poses and Roboflow COCO segmentation masks.
No GPU or CUDA required — runs entirely on CPU.

---

## Requirements

```bash
pip install numpy scipy scikit-image trimesh pillow opencv-python pycocotools
```

---

## Folder Structure

```
project/
├── colmap_sparse/                  ← COLMAP text export (File → Export model as text)
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt                (not needed)
│
├── roboflow_export/                ← Roboflow ZIP extracted
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   ├── image001.jpg
│   │   └── ...
│   ├── valid/                      (optional)
│   └── test/                       (optional)
│
└── visual_hull_colmap.py
```

---

## COLMAP Export

After reconstruction in the COLMAP GUI:

1. **File → Export model as text**
2. Select an empty target folder (e.g. `colmap_sparse/`)
3. COLMAP creates `cameras.txt`, `images.txt`, `points3D.txt`

> **Note:** The script requires the **text format** (`.txt`), not the binary format (`.bin`).  
> To convert existing `.bin` files:
> ```bash
> colmap model_converter \
>     --input_path  ./colmap_sparse \
>     --output_path ./colmap_sparse_txt \
>     --output_type TXT
> ```

---

## Usage

### Check image–mask matching (recommended first step)
```bash
python visual_hull_colmap.py \
    --colmap   ./colmap_sparse \
    --roboflow ./roboflow_export \
    --list
```
Prints a table showing whether each COLMAP image has a matching mask (✓/✗).

### Run reconstruction
```bash
python visual_hull_colmap.py \
    --colmap   ./colmap_sparse \
    --roboflow ./roboflow_export \
    --output   mesh.obj
```

### Higher quality output
```bash
python visual_hull_colmap.py \
    --colmap   ./colmap_sparse \
    --roboflow ./roboflow_export \
    --grid     256 \
    --output   mesh_hq.obj
```

### Show 3D preview after reconstruction
```bash
python visual_hull_colmap.py \
    --colmap   ./colmap_sparse \
    --roboflow ./roboflow_export \
    --output   mesh.obj \
    --preview
```

### If voxel grid is carved empty (0 voxels remaining)
```bash
python visual_hull_colmap.py \
    --colmap   ./colmap_sparse \
    --roboflow ./roboflow_export \
    --output   mesh.obj \
    --margin   2.0
```

---

## All Parameters

| Parameter    | Default          | Description                                                  |
|--------------|------------------|--------------------------------------------------------------|
| `--colmap`   | —                | Folder containing `cameras.txt` and `images.txt`            |
| `--roboflow` | —                | Roboflow export folder (containing `train/`, `valid/`, etc.) |
| `--output`   | `visual_hull.obj`| Output mesh file (.obj)                                      |
| `--grid`     | `128`            | Voxel grid resolution (128³ or 256³). Higher = more detail, more RAM |
| `--margin`   | `1.3`            | Bounding box margin factor. Increase if object is clipped    |
| `--preview`  | off              | Show interactive 3D preview after reconstruction             |
| `--list`     | off              | Only print image–mask matching table, do not reconstruct     |

---

## Pipeline Overview

1. **COLMAP parsing** — reads camera intrinsics (`cameras.txt`) and extrinsics (`images.txt`); supports `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`
2. **Mask loading** — reads all splits (`train/`, `valid/`, `test/`) from the Roboflow COCO export; polygon annotations are rasterized to binary masks via `cv2.fillPoly`; Roboflow hash suffixes in filenames are handled automatically
3. **Object center estimation** — least-squares intersection of optical axes from all camera views
4. **Voxel carving** — each voxel in a 128³ grid is projected into every camera view; voxels outside any silhouette are removed
5. **Mesh extraction** — Gaussian smoothing (σ=1.0) + Marching Cubes (isovalue=0.5) via `scikit-image`; vertices are transformed from grid coordinates to COLMAP world coordinates
6. **Export** — Wavefront OBJ via `trimesh`

---

## Supported COLMAP Camera Models

| Model           | Parameters              |
|-----------------|-------------------------|
| SIMPLE_PINHOLE  | f, cx, cy               |
| PINHOLE         | fx, fy, cx, cy          |
| SIMPLE_RADIAL   | f, cx, cy, k1           |
| RADIAL          | f, cx, cy, k1, k2       |
| OPENCV          | fx, fy, cx, cy, k1, k2, p1, p2 |

---

## Common Issues

**All voxels carved empty (0 voxels remaining)**  
→ Try `--margin 2.0` to expand the reconstruction volume  
→ Run `--list` to verify all images have masks  
→ Check that COLMAP image filenames match Roboflow filenames  

**`cameras.txt` not found**  
→ Use *File → Export model as text* in the COLMAP GUI (not *Save Project*)

**Mask not found for an image**  
→ Roboflow sometimes appends a hash suffix (e.g. `img.jpg` → `img_jpg.rf.abc123.jpg`).  
   The script handles this automatically as long as the original filename is contained in the Roboflow name.

**Mesh looks blocky**  
→ Increase `--grid 256` for a finer voxel grid (requires more RAM)

---

## Implementation Notes

- Developed with Python 3.10 on Windows 11
- No GPU required; all computation runs on CPU
- Implemented with assistance of Claude (Anthropic, Claude Sonnet 4.6)
