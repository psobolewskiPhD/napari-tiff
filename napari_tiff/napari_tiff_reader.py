"""
This modeul is a napari reader for TIFF image files.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import dask.array as da
import numpy as np
from dask.array.core import normalize_chunks
from dask.utils import parse_bytes
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tifffile import TIFF, TiffFile, TiffSequence

from napari_tiff.napari_tiff_metadata import get_metadata

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Implements napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
    path = path.lower()
    if path.endswith("zip"):
        return zip_reader
    for ext in TIFF.FILE_EXTENSIONS:
        if path.endswith(ext):
            return reader_function
    return None


def reader_function(path: PathLike) -> List[LayerData]:
    """Return a list of LayerData tuples from path or list of paths."""
    # TODO: LSM
    with TiffFile(path) as tif:
        try:
            layerdata = tifffile_reader(tif)
        except Exception as exc:
            # fallback to imagecodecs
            log_warning(f"tifffile: {exc}")
            layerdata = imagecodecs_reader(path)
    return layerdata


def zip_reader(path: PathLike) -> List[LayerData]:
    """Return napari LayerData from sequence of TIFF in ZIP file."""
    with TiffSequence(container=path) as ims:
        data = ims.asarray()
    return [(data, {}, "image")]


def tifffile_reader(tif: TiffFile) -> List[LayerData]:
    """Return napari LayerData from image series in TIFF file."""
    nlevels = len(tif.series[0].levels)
    if nlevels > 1:
        import zarr
        store = tif.aszarr(multiscales=True)
        group = zarr.open_group(store=store, mode='r')
        # using group.attrs to get multiscales is recommended by cgohlke
        # default dask chunk is 128MiB, so use 1 MiB, which is more reasonable for visualization
        data = [from_zarr_adaptive_chunks(group[path_dict['path']], target_size='1 MiB') for path_dict in group.attrs['multiscales'][0]['datasets']]
        # assert array shapes are in descending order for napari multiscale image
        shapes = [arr.shape for arr in data]
        assert shapes == list(reversed(sorted(shapes)))
    else:
        data = tif.asarray()

    metadata_kwargs = get_metadata(tif)

    return [(data, metadata_kwargs, "image")]

def from_zarr_adaptive_chunks(zarr_array, target_size='1 MiB'):
    """Load zarr array as dask array with chunks that are multiples of storage chunks.

    Compute a set of chunk sizes that are multiples of storage chunks, while
    ensuring that the chunks are reasonably isotropic."""

    storage_chunks = zarr_array.chunks
    dtype_size = zarr_array.dtype.itemsize
    target_bytes = parse_bytes(target_size)

    # If storage chunks are already large enough, use them
    storage_chunk_size = np.prod(storage_chunks) * dtype_size
    if storage_chunk_size >= target_bytes:
        return da.from_zarr(zarr_array, chunks=storage_chunks)

    # Handle RGB arrays (assume last dim == 3 is RGB channels)
    is_rgb = zarr_array.shape[-1] == 3
    spatial_chunks = storage_chunks[:-1] if is_rgb else storage_chunks
    target_elements = target_bytes // dtype_size
    if is_rgb:
        target_elements //= 3  # Account for RGB channels

    # Calculate chunk multipliers aiming for isotropic chunks
    chunk_shape = []
    remaining_elements = target_elements

    for i, storage_chunk_dim in enumerate(spatial_chunks):
        remaining_dims = len(spatial_chunks) - i

        if remaining_dims == 1:
            # Last dimension - use all remaining budget
            multiplier = max(1, remaining_elements // storage_chunk_dim)
        else:
            # Aim for isotropic chunks - each dimension gets roughly equal "chunk size"
            target_chunk_size = int(remaining_elements ** (1 / remaining_dims))
            multiplier = max(1, target_chunk_size // storage_chunk_dim)

            # Ensure we don't exceed using the remaining storage_chunk elements
            min_remaining_elements = np.prod(spatial_chunks[i+1:])
            max_elements_for_dim = remaining_elements // min_remaining_elements
            if multiplier * storage_chunk_dim > max_elements_for_dim:
                # set the multipler for this dim such that we use the remaining shape
                multiplier = max(1, max_elements_for_dim // storage_chunk_dim)

        # Don't exceed array bounds
        max_multiplier = zarr_array.shape[i] // storage_chunk_dim
        if max_multiplier > 0:
            multiplier = min(multiplier, max_multiplier)

        chunk_dim = multiplier * storage_chunk_dim
        chunk_shape.append(chunk_dim)
        remaining_elements = remaining_elements // chunk_dim

    # Add RGB channel dimension back
    if is_rgb:
        chunk_shape.append(3)

    return da.from_zarr(zarr_array, chunks=tuple(chunk_shape))


def imagecodecs_reader(path: PathLike):
    """Return napari LayerData from first page in TIFF file."""
    from imagecodecs import imread

    return [(imread(path), {}, "image")]


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger(__name__).warning(msg, *args, **kwargs)
