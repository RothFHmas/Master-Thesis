import os
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image

# =========================================================
# AUTO THESIS GALLERY CREATOR
# Kein unnötiger Weißraum
# 16:9 Format
# =========================================================

OUTPUT_WIDTH = 7680
OUTPUT_HEIGHT = 4320

SMALL_SIZE = 320
GAP = 8

BIG_FACTOR = 4
SECOND_FACTOR = 2

BG_COLOR = (255, 255, 255)

# =========================================================

class GalleryApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Auto Thesis Gallery Creator")

        self.image_paths = []

        self.build_gui()

    # =====================================================

    def build_gui(self):

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Button(
            frame,
            text="Bilder laden",
            command=self.load_images
        ).pack(fill="x", pady=5)

        ttk.Label(
            frame,
            text="Größtes Bild"
        ).pack(anchor="w")

        self.biggest_combo = ttk.Combobox(
            frame,
            state="readonly"
        )

        self.biggest_combo.pack(fill="x", pady=5)

        ttk.Label(
            frame,
            text="Zweitgrößtes Bild"
        ).pack(anchor="w")

        self.second_combo = ttk.Combobox(
            frame,
            state="readonly"
        )

        self.second_combo.pack(fill="x", pady=5)

        ttk.Button(
            frame,
            text="Galerie erstellen",
            command=self.create_gallery
        ).pack(fill="x", pady=10)

        self.status = ttk.Label(frame, text="")
        self.status.pack(anchor="w")

    # =====================================================

    def load_images(self):

        paths = filedialog.askopenfilenames(
            title="Bilder auswählen",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp")
            ]
        )

        if not paths:
            return

        self.image_paths = list(paths)

        names = [
            os.path.basename(p)
            for p in self.image_paths
        ]

        self.biggest_combo["values"] = names
        self.second_combo["values"] = names

        if len(names) > 0:
            self.biggest_combo.current(0)

        if len(names) > 1:
            self.second_combo.current(1)

        self.status.config(
            text=f"{len(names)} Bilder geladen"
        )

    # =====================================================

    def crop_to_fill(self, img, target_size):

        target_w, target_h = target_size

        img_ratio = img.width / img.height
        target_ratio = target_w / target_h

        # Bild zu breit
        if img_ratio > target_ratio:

            new_width = int(img.height * target_ratio)

            left = (img.width - new_width) // 2

            img = img.crop((
                left,
                0,
                left + new_width,
                img.height
            ))

        # Bild zu hoch
        else:

            new_height = int(img.width / target_ratio)

            top = (img.height - new_height) // 2

            img = img.crop((
                0,
                top,
                img.width,
                top + new_height
            ))

        return img.resize(
            (target_w, target_h),
            Image.LANCZOS
        )

    # =====================================================

    def create_gallery(self):

        if len(self.image_paths) < 3:

            messagebox.showerror(
                "Fehler",
                "Mindestens 3 Bilder nötig"
            )

            return

        biggest_idx = self.biggest_combo.current()
        second_idx = self.second_combo.current()

        biggest_path = self.image_paths[biggest_idx]
        second_path = self.image_paths[second_idx]

        remaining = [

            p for p in self.image_paths

            if p not in [
                biggest_path,
                second_path
            ]
        ]

        # =================================================
        # Größen
        # =================================================

        BIG_SIZE = SMALL_SIZE * BIG_FACTOR
        SECOND_SIZE = SMALL_SIZE * SECOND_FACTOR

        # =================================================
        # Linke Spalte Höhe
        # =================================================

        left_height = (
            BIG_SIZE
            + GAP
            + SECOND_SIZE
        )

        # =================================================
        # Rechte Grid-Breite
        # =================================================

        start_x = BIG_SIZE + GAP

        available_width = (
            OUTPUT_WIDTH
            - start_x
            - GAP
        )

        columns = available_width // (
            SMALL_SIZE + GAP
        )

        if columns < 1:
            columns = 1

        # =================================================
        # Automatische Reihen
        # =================================================

        rows = math.ceil(
            len(remaining) / columns
        )

        grid_height = (
            rows * SMALL_SIZE
            + (rows - 1) * GAP
        )

        # =================================================
        # Finale Höhe dynamisch
        # =================================================

        used_height = max(
            left_height,
            grid_height
        )

        # Canvas exakt auf Inhalt zuschneiden
        canvas = Image.new(
            "RGB",
            (OUTPUT_WIDTH, used_height),
            BG_COLOR
        )

        # =================================================
        # Größtes Bild
        # =================================================

        img_big = Image.open(
            biggest_path
        ).convert("RGB")

        img_big = self.crop_to_fill(
            img_big,
            (BIG_SIZE, BIG_SIZE)
        )

        canvas.paste(
            img_big,
            (0, 0)
        )

        # =================================================
        # Zweitgrößtes Bild
        # =================================================

        img_second = Image.open(
            second_path
        ).convert("RGB")

        img_second = self.crop_to_fill(
            img_second,
            (SECOND_SIZE, SECOND_SIZE)
        )

        canvas.paste(
            img_second,
            (0, BIG_SIZE + GAP)
        )

        # =================================================
        # Kleine Bilder rechts
        # =================================================

        for idx, path in enumerate(remaining):

            row = idx // columns
            col = idx % columns

            x = start_x + col * (
                SMALL_SIZE + GAP
            )

            y = row * (
                SMALL_SIZE + GAP
            )

            try:

                img = Image.open(path).convert("RGB")

                img = self.crop_to_fill(
                    img,
                    (SMALL_SIZE, SMALL_SIZE)
                )

                canvas.paste(
                    img,
                    (x, y)
                )

            except:
                pass

        # =================================================
        # Automatisch auf genutzten Bereich zuschneiden
        # =================================================

        bbox = canvas.getbbox()

        if bbox:
            canvas = canvas.crop(bbox)

        # =================================================
        # Speichern
        # =================================================

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png")
            ]
        )

        if not save_path:
            return

        canvas.save(
            save_path,
            quality=100
        )

        messagebox.showinfo(
            "Fertig",
            "Galerie gespeichert"
        )

# =========================================================

root = tk.Tk()

app = GalleryApp(root)

root.mainloop()