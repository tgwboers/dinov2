# Copyright (c) Tim Boers, Inc. and affiliates.
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from zipfile import ZipFile, ZipInfo
from io import BytesIO
from mmap import ACCESS_READ, mmap
import os
from typing import Any, Callable, List, Optional, Set, Tuple
import warnings
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .extended import ExtendedVisionDataset


_Labels = int

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors

@dataclass
class _ClassEntry:
    block_offset: int
    maybe_filename: Optional[str] = None


@dataclass
class _Entry:
    class_index: int  # noqa: E701
    start_offset: int
    end_offset: int
    filename: str


class _Split(Enum):
    TRAIN = "train"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 5_000_000,
        }[self]

class GastroNet5M(ExtendedVisionDataset):
    """
    Adapts the structure of the personal dataset into the DINOv2 ImageNet21k framework.
    """

    def __init__(
        self,
        *,
        root: str,  # Directory containing ZIP files
        extra: str,  # Path for saving/loading additional metadata
        image_suffixes: List[str] = [".jpg", ".png"],
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self.image_suffixes = image_suffixes
        print(root)
        print(os.listdir(root))
        self.zip_files = list(Path(root).rglob("*.zip"))  # Find all ZIP files in root
        print(self.zip_files)

        self._entries, self._class_ids = self._load_or_generate_metadata()
        
        print('entries: ', self._entries)
        print('self._class_ids: ', self._class_ids)
        
        print('len entries: ', len(self._entries))
        print('len self._class_ids: ', len(self._class_ids))
        
        self._mmap_cache = {}  # Cache for ZIP file handles

    def _load_or_generate_metadata(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads or generates metadata for ZIP files and their contents.
        """
        entries_path = Path(self._extra_root) / "entries.npy"
        class_ids_path = Path(self._extra_root) / "class_ids.npy"

        if entries_path.exists() and class_ids_path.exists():
            # Load metadata if it already exists
            entries = np.load(entries_path, mmap_mode="r")
            class_ids = np.load(class_ids_path, mmap_mode="r")
        else:
            # Generate metadata if it doesn't exist
            entries, class_ids = self._generate_metadata()
            np.save(entries_path, entries)
            np.save(class_ids_path, class_ids)

        return entries, class_ids

    def _generate_metadata(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates metadata (entries and class IDs) for ZIP files.
        """
        class_ids = []
        class_index_map = {}  # Maps ZIP file names to class indices
        all_entries = []  # Collect entries for all ZIP files
    
        def process_zip(class_index, zip_file):
            """Processes a single ZIP file and returns its entries."""
            entries = []
            with ZipFile(zip_file) as zf:
                for member in sorted(zf.namelist()):
                    if any(member.endswith(suffix) for suffix in self.image_suffixes):
                        entries.append((class_index, zip_file, member))
            return entries
    
        zip_files_sorted = sorted(self.zip_files)
        with ThreadPoolExecutor() as executor:
            future_to_zip = {
                executor.submit(process_zip, class_index, zip_file): (class_index, zip_file)
                for class_index, zip_file in enumerate(zip_files_sorted)
            }
    
            for future in future_to_zip:
                class_index, zip_file = future_to_zip[future]
                class_id = os.path.splitext(os.path.basename(zip_file))[0]
                class_ids.append(class_id)
                class_index_map[class_id] = class_index
                all_entries.extend(future.result())
    
        dtype = np.dtype(
            [("class_index", np.uint16), ("class_id", np.uint16), ("zip_file", "U20"), ("image_path", "U20")]
        )
        entries_array = np.array(
            [(entry[0], class_ids[entry[0]], str(entry[1]), entry[2]) for entry in all_entries],
            dtype=dtype,
        )
        return entries_array, np.array(class_ids, dtype=f"U{max(len(cid) for cid in class_ids)}")

    def get_image_data(self, index: int) -> bytes:
        """
        Fetches raw image data from the ZIP file for the given index.
        """
        entry = self._entries[index]
        zip_file, image_path = entry["zip_file"], entry["image_path"]

        if zip_file not in self._mmap_cache:
            self._mmap_cache[zip_file] = ZipFile(zip_file, mode="r")

        with self._mmap_cache[zip_file].open(image_path) as f:
            image_data = f.read()

        return image_data

    def get_target(self, index: int) -> int:
        """
        Fetches the class index for the given image.
        """
        return int(self._entries[index]["class_index"])

    def get_targets(self) -> np.ndarray:
        """
        Returns all class indices.
        """
        return self._entries["class_index"]

    def get_class_id(self, index: int) -> str:
        """
        Fetches the class ID for the given image.
        """
        return str(self._entries[index]["class_id"])

    def get_class_ids(self) -> np.ndarray:
        """
        Returns all class IDs.
        """
        return self._entries["class_id"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super().__getitem__(index)

    def __len__(self) -> int:
        """
        Returns the total number of entries.
        """
        return len(self._entries)
