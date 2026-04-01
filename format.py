from __future__ import annotations

from pathlib import Path
import re
import shutil
from zipfile import ZipFile


def extract_wesad_zip_files(base_dir: Path) -> None:
    """Extract every .zip file under base_dir into its parent folder."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    zip_files = sorted(base_dir.rglob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in: {base_dir}")
        return

    print(f"Found {len(zip_files)} zip file(s) in: {base_dir}")
    for zip_path in zip_files:
        extract_to = zip_path.parent

        print(f"Extracting: {zip_path}")
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)
        print(f"Done -> {extract_to}")


def move_old_extracted_folders(base_dir: Path) -> None:
    """Move files from old *_E4_Data folders into their parent subject folder."""
    old_folders = sorted(base_dir.rglob("*_E4_Data"))
    if not old_folders:
        return

    for folder in old_folders:
        if not folder.is_dir():
            continue

        destination = folder.parent
        print(f"Moving existing extracted files: {folder} -> {destination}")
        for item in folder.iterdir():
            target = destination / item.name
            if target.exists():
                print(f"Skipping existing file/folder: {target}")
                continue
            shutil.move(str(item), str(target))

        try:
            folder.rmdir()
            print(f"Removed empty folder: {folder}")
        except OSError:
            print(f"Folder not empty, kept: {folder}")


def decrement_subject_index_in_folders(base_dir: Path) -> None:
    """Rename files/folders so subject index matches the WESAD_S* folder index."""
    subject_folders = sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and re.fullmatch(r"WESAD_S\d+", path.name)]
    )

    for subject_folder in subject_folders:
        folder_index = int(subject_folder.name.split("_S")[-1])
        wrong_index = folder_index + 1
        old_prefix = f"S{wrong_index}"
        new_prefix = f"S{folder_index}"

        entries = sorted(subject_folder.rglob("*"), key=lambda path: (len(path.parts), str(path)), reverse=True)
        for entry in entries:
            if not entry.name.startswith(old_prefix):
                continue

            renamed = entry.with_name(entry.name.replace(old_prefix, new_prefix, 1))
            if renamed.exists():
                print(f"Skipping rename (already exists): {renamed}")
                continue

            print(f"Renaming: {entry.name} -> {renamed.name}")
            entry.rename(renamed)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    wesad_dir = project_root / "data" / "WESAD"
    move_old_extracted_folders(wesad_dir)
    extract_wesad_zip_files(wesad_dir)
    decrement_subject_index_in_folders(wesad_dir)
