#!/usr/bin/env python3
"""
batch_center_crop.py

Recursively search a directory for image files and perform a center crop
to a requested width and height.

Usage examples:
    # Save crops into a new folder (mirrors input structure)
    python batch_center_crop.py /path/to/images 640 480 --outdir /path/to/cropped

    # Overwrite originals (use with care)
    python batch_center_crop.py /path/to/images 640 480 --inplace

    # Only process certain extensions (default common formats)
    python batch_center_crop.py ./images 224 224 --ext jpg png jpeg webp

Notes:
 - If an image is smaller than the requested size in either dimension,
   the script first resizes the image (upscales) preserving aspect ratio
   so both dimensions are >= target, then center-crops. This avoids
   empty/padded regions.
 - Requires Pillow: pip install Pillow
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageOps

# common image extensions (case-insensitive)
DEFAULT_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}


def center_crop_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Center-crop an image to target_w x target_h.
    If the image is smaller than target in any dimension, first upscale
    preserving aspect ratio so both dims >= target dims, then crop.
    """
    # Ensure image is in a mode suitable for saving (keep as-is)
    orig_w, orig_h = img.size

    # If origin smaller than target, scale up so both dims >= target
    scale_w = target_w / orig_w if orig_w < target_w else 1.0
    scale_h = target_h / orig_h if orig_h < target_h else 1.0
    scale = max(scale_w, scale_h)
    if scale > 1.0:
        new_w = int(orig_w * scale + 0.5)
        new_h = int(orig_h * scale + 0.5)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Now perform center crop
    w, h = img.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img.crop((left, top, right, bottom))


def process_file(src_path: Path, dst_path: Path, target_w: int, target_h: int, inplace: bool):
    """
    Open src_path, center-crop, and save to dst_path (or overwrite src_path if inplace).
    Takes care of EXIF orientation.
    """
    try:
        with Image.open(src_path) as im:
            # Respect EXIF orientation
            im = ImageOps.exif_transpose(im)
            cropped = center_crop_image(im, target_w, target_h)

            # Create parent dir for dst if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Save preserving format and quality when reasonable
            fmt = im.format if im.format else 'PNG'
            save_kwargs = {}
            # If JPEG, keep good quality
            if fmt.upper() in ('JPG', 'JPEG'):
                save_kwargs['quality'] = 95
                save_kwargs['subsampling'] = 0  # keep best chroma
            # Preserve PNG transparency by default

            cropped.save(dst_path, format=fmt, **save_kwargs)
            return True, None
    except Exception as e:
        return False, str(e)


def find_images(root: Path, exts):
    """
    Yield Path objects for files under root matching extensions (exts set, lowercase).
    """
    for p in root.rglob('*'):
        if p.is_file():
            if p.suffix.lower() in exts:
                yield p


def main():
    p = argparse.ArgumentParser(description="Recursively center-crop images in a directory.")
    p.add_argument("indir", type=Path, help="Input directory to search recursively for images.")
    p.add_argument("width", type=int, help="Target width (pixels) for center crop.")
    p.add_argument("height", type=int, help="Target height (pixels) for center crop.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output root directory. If omitted and --inplace not set, a subfolder named "
                        "'cropped_WxH' will be created next to indir.")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite original images in-place. If set, --outdir is ignored.")
    p.add_argument("--ext", nargs='+', default=None,
                   help="List of extensions to process (e.g. jpg png). Default: common image types.")
    p.add_argument("--dry-run", action="store_true", help="List files that would be processed, do not save.")
    p.add_argument("--verbose", action="store_true", help="Print processing information.")
    args = p.parse_args()

    indir: Path = args.indir
    if not indir.exists() or not indir.is_dir():
        print(f"Input directory does not exist or is not a directory: {indir}")
        return

    w = args.width
    h = args.height

    if args.ext:
        exts = {('.' + e.lower().lstrip('.')) for e in args.ext}
    else:
        exts = DEFAULT_EXT

    if args.inplace:
        outdir = None
    else:
        if args.outdir:
            outdir = args.outdir
        else:
            # default: create sibling folder named indir_cropped_WxH
            outdir = indir.parent / f"{indir.name}_cropped_{w}x{h}"
        outdir.mkdir(parents=True, exist_ok=True)

    # Gather files
    files = list(find_images(indir, exts))
    if not files:
        print("No images found with extensions:", ','.join(sorted(exts)))
        return

    print(f"Found {len(files)} image(s). Target crop: {w}x{h}.")
    if args.dry_run:
        for f in files:
            print(f)
        return

    errors = []
    processed = 0
    for src in files:
        # Determine destination path
        if args.inplace:
            dst = src
        else:
            # mirror structure under outdir
            rel = src.relative_to(indir)
            dst = outdir / rel
            # optionally change file name to indicate crop (user didn't insist, keep same name)
            # could add suffix: dst = dst.with_name(dst.stem + f"_c{w}x{h}" + dst.suffix)
        ok, err = process_file(src, dst, w, h, inplace=args.inplace)
        if ok:
            processed += 1
            if args.verbose:
                print(f"[OK] {src} -> {dst}")
        else:
            errors.append((src, err))
            print(f"[ERR] {src}: {err}")

    print(f"Done. Processed: {processed}. Errors: {len(errors)}.")
    if errors:
        print("Errors (sample):")
        for e in errors[:10]:
            print(" ", e[0], ":", e[1])


# python3 resize.py flowers 128 128 --outdir=flowers-cropped
if __name__ == "__main__":
    main()