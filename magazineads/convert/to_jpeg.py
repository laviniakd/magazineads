"""Convert downloaded magazine archives to per-page JPEG images.

Supports two input formats:
    - **PDF**: Rendered via ``pdftoppm`` (Poppler) at 600 dpi.
    - **JP2 zip**: Extracted and converted via Pillow.

Two-page spread scans (width > height) are optionally split vertically
into separate left/right page images, with sequential page numbering.

Output naming convention: ``{issue_stem}-{page:03d}.jpg``

Usage:
    python -m magazineads.convert.to_jpeg \\
        --magazine esquire \\
        --data_path /data/magazines \\
        --split_pages

    python -m magazineads.convert.to_jpeg \\
        --magazine variety \\
        --data_path /data/magazines \\
        --jp2
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import zipfile
from functools import partial
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# ── Helpers ──────────────────────────────────────────────────────────────────

def _split_jpeg_vertically(jpeg_path: str, quality: int = 100) -> tuple[str, str]:
    """Split a landscape JPEG vertically into left and right halves."""
    with Image.open(jpeg_path) as img:
        width, height = img.size
        left = img.crop((0, 0, width // 2, height))
        right = img.crop((width // 2, 0, width, height))
        left_path = jpeg_path.replace('.jpg', '_left.jpg')
        right_path = jpeg_path.replace('.jpg', '_right.jpg')
        left.save(left_path, quality=quality, subsampling=0, optimize=False)
        right.save(right_path, quality=quality, subsampling=0, optimize=False)
    return left_path, right_path


def _renumber_with_splits(
    file_paths: list[str],
    output_folder: str,
    prefix: str,
    skip_cover_split: bool = True,
) -> int:
    """Renumber converted pages, splitting spreads into two pages.

    Returns the final page count.
    """
    page_num = 1
    for idx, file_path in enumerate(file_paths, start=1):
        with Image.open(file_path) as img:
            width, height = img.size
            is_spread = width > height

        # Cover page (idx == 1) is never split
        if (skip_cover_split and idx == 1) or not is_spread:
            new_name = f"{prefix}-{page_num:03d}.jpg"
            os.rename(file_path, os.path.join(output_folder, new_name))
            page_num += 1
        else:
            left_path, right_path = _split_jpeg_vertically(file_path)
            os.remove(file_path)
            os.rename(left_path, os.path.join(output_folder, f"{prefix}-{page_num:03d}.jpg"))
            os.rename(right_path, os.path.join(output_folder, f"{prefix}-{page_num + 1:03d}.jpg"))
            page_num += 2

    return page_num - 1


# ── PDF conversion ──────────────────────────────────────────────────────────

def process_single_pdf(
    pdf_file: str,
    input_folder: str,
    output_folder: str,
    split_pages: bool = False,
) -> str:
    """Convert one PDF to sequentially-numbered page JPEGs.

    Requires ``pdftoppm`` from Poppler to be installed.
    """
    pdf_path = os.path.join(input_folder, pdf_file)
    prefix = os.path.splitext(pdf_file)[0]
    output_path = os.path.join(output_folder, prefix)

    # Skip if already converted
    if os.path.exists(f"{output_path}-001.jpg") or os.path.exists(f"{output_path}_01.jpg"):
        return f"Skipped {pdf_file} — already converted"

    try:
        subprocess.run(
            ['pdftoppm', pdf_path, output_path, '-jpeg', '-r', '600',
             '-jpegopt', 'quality=100'],
            capture_output=True, text=True, check=True,
        )

        if split_pages:
            # Collect pdftoppm outputs and renumber with splits
            created = sorted(
                os.path.join(output_folder, f)
                for f in os.listdir(output_folder)
                if f.startswith(prefix) and f.endswith('.jpg')
            )
            _renumber_with_splits(created, output_folder, prefix)

        return f"Converted {pdf_file}"

    except subprocess.CalledProcessError as e:
        return f"Error converting {pdf_file}: {e}"
    except Exception as e:
        return f"Error processing {pdf_file}: {e}"


# ── JP2 zip conversion ──────────────────────────────────────────────────────

def process_single_zip(
    zip_file: str,
    input_folder: str,
    output_folder: str,
    split_pages: bool = False,
) -> str:
    """Convert one JP2 zip archive to sequentially-numbered page JPEGs."""
    zip_path = os.path.join(input_folder, zip_file)
    prefix = os.path.splitext(zip_file)[0]
    if prefix.endswith('_jp2'):
        prefix = prefix[:-4]

    # Skip if already converted
    if os.path.exists(os.path.join(output_folder, f"{prefix}-001.jpg")):
        return f"Skipped {zip_file} — already converted"

    temp_dir = os.path.join(output_folder, f"_temp_{prefix}")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        jp2_files = sorted(Path(temp_dir).rglob('*.jp2'))
        if not jp2_files:
            return f"No JP2 files found in {zip_file}"

        # Convert each JP2 to temporary JPEG
        temp_jpegs = []
        for idx, jp2 in enumerate(jp2_files, start=1):
            out = os.path.join(output_folder, f"{prefix}_temp_{idx:03d}.jpg")
            with Image.open(jp2) as img:
                img.save(out, 'JPEG', quality=100, subsampling=0, optimize=False)
            temp_jpegs.append(out)

        if split_pages:
            _renumber_with_splits(temp_jpegs, output_folder, prefix)
        else:
            for idx, path in enumerate(temp_jpegs, start=1):
                os.rename(path, os.path.join(output_folder, f"{prefix}-{idx:03d}.jpg"))

        shutil.rmtree(temp_dir)
        return f"Converted {zip_file} ({len(jp2_files)} pages)"

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return f"Error processing {zip_file}: {e}"


# ── Parallel driver ──────────────────────────────────────────────────────────

def convert_magazine(
    input_folder: str,
    output_folder: str,
    *,
    jp2: bool = False,
    split_pages: bool = False,
    num_processes: int | None = None,
    limit: int | None = None,
) -> None:
    """Convert all archives in *input_folder* to page JPEGs in *output_folder*.

    Args:
        input_folder: Directory containing PDFs or JP2 zips.
        output_folder: Directory for output JPEGs.
        jp2: If True, process JP2 zip files; otherwise, process PDFs.
        split_pages: If True, split two-page spreads into individual pages.
        num_processes: Worker count (default: CPU count).
        limit: Process at most this many files (for testing).
    """
    os.makedirs(output_folder, exist_ok=True)
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Search recursively — IA scraper nests files in per-item subdirectories
    suffix = '_jp2.zip' if jp2 else '.pdf'
    found = sorted(Path(input_folder).rglob(f'*{suffix}'))

    if jp2:
        # Each file carries its own parent dir as input_folder
        files = [(str(p.parent), p.name) for p in found]
        desc = "Converting JP2 zips"
    else:
        files = [(str(p.parent), p.name) for p in found]
        desc = "Converting PDFs"

    if limit:
        files = files[:limit]

    print(f"Converting {len(files)} files with {num_processes} workers")
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")

    process_fn = process_single_zip if jp2 else process_single_pdf

    # Build args tuples: (filename, parent_dir, output_folder, split_pages)
    args_list = [(name, parent, output_folder, split_pages) for parent, name in files]

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.starmap(process_fn, args_list), total=len(files), desc=desc))

    errors = [r for r in results if r.startswith("Error")]
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for e in errors[:10]:
            print(f"  {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert downloaded magazine archives to per-page JPEGs",
    )
    parser.add_argument("--magazine", required=True,
                        help="Magazine name (determines output subdirectory)")
    parser.add_argument("--data_path", required=True,
                        help="Base data directory (contains originals/ and images/)")
    parser.add_argument("--input_dir",
                        help="Explicit input directory (overrides auto-detection)")
    parser.add_argument("--jp2", action="store_true",
                        help="Process JP2 zip files instead of PDFs")
    parser.add_argument("--split_pages", action="store_true",
                        help="Split two-page spread scans into individual pages")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files (for testing)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    magazine = args.magazine

    if args.input_dir:
        input_folder = args.input_dir
    elif (data_path / "originals" / magazine).exists():
        input_folder = str(data_path / "originals" / magazine)
    elif (data_path / magazine).exists():
        input_folder = str(data_path / magazine)
    else:
        # Fall back to data_path itself (for flat layouts)
        input_folder = str(data_path)

    output_folder = str(data_path / "images" / magazine)

    convert_magazine(
        input_folder, output_folder,
        jp2=args.jp2,
        split_pages=args.split_pages,
        num_processes=args.num_processes,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
