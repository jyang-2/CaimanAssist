""" Module with functions to help visually inspect 3D caiman motion correction results more easily.

Caiman demo code mostly, downsampling movies and viewing individual planes.

Record bad planes (planes to drop post-motion correction due to z-drift) in
{movie_folder}/caiman_mc/moco_metrics.json, and define which moco result (rigid or pw-rigid) to use.

Check the top and bottom-most planes first, to see how many planes each moco result loses. If the
movie has 16 planes, check planes 0, 1, ... until the corrected plane movie is usable; Do the
same for plane 15, 14, etc.

Example:

    {...
        "bad_planes_rig": [0, 1, 14, 15],

        "bad_planes_els": [0, 15],

        "which_movie": "m_els" (or "m_rig")
    }

To load contents of caiman_mc quickly:

>>>  m_rig, m_els, files = load_moco_results(moco_dir)

To observe moco_metrics.json:

>>> with open(files['moco_metrics'], 'r') as f:
>>>        moco_metrics = json.load(f)

"""
import time
from pathlib import Path
import caiman as cm
from caiman.motion_correction import MotionCorrect
import pydantic
from pydantic import BaseModel
import numpy as np
import dask.array as darr

from src import convert

moco_fname_templates = {
    "mc_rig": "mc_rig.npy",
    "mc_els": "mc_els.npy",
    "m_rig": "*_rig_*.mmap",
    "m_els": "*_els_*.mmap",
    "moco_metrics": "moco_metrics.json"
}


def parse_folder_contents(folder, fname_templates):
    """

    Args:
        folder (Path): folder to look for files
        fname_templates (dict): dict w/ (key, value) = (variable name, filename template string)

    Returns:
        contents (dict): dict w/ (key, value) = (variable name, filepath)
    """
    contents = {}
    for k, v in fname_templates.items():
        contents[k] = get_single_file_from_folder(folder, v)
    return contents


def get_single_file_from_folder(folder, name_template):
    fname = list(sorted(folder.glob(name_template)))

    if len(fname) == 0:
        raise ValueError(f"File ({name_template}) not found.")
    elif len(fname) > 1:
        raise ValueError(f"Multiple files ({name_template}) detected.")
    else:
        return fname[0]


def resize_movie_in_time(movie, downsample_ratio):
    if downsample_ratio == 1:
        resized_movie = movie
    elif movie.ndim == 4:
        T, d1, d2, d3 = movie.shape

        ds_movies = []
        for z in tqdm(range(d3)):
            ds_movies.append(movie[..., z].resize(1, 1, downsample_ratio))
        resized_movie = cm.concatenate(ds_movies, axis=-1)
    elif movie.ndim == 3:
        T, d1, d2 = movie.shape
        resized_movie = movie.resize(1, 1, downsample_ratio)
    return resized_movie


def load_moco_results(folder, downsample_ratio=0.05):
    files = parse_folder_contents(folder, moco_fname_templates)
    movie_rig = resize_movie_in_time(cm.load(files['m_rig']), downsample_ratio=downsample_ratio)
    movie_els = resize_movie_in_time(cm.load(files['m_els']), downsample_ratio=downsample_ratio)

    return cm.movie(movie_rig), cm.movie(movie_els), files


def darr_downsample_movie(Y, chunk_size=500):
    tic = time.perf_counter()
    dims = Y.shape

    darr_mov = darr.from_array(np.array(Y), chunks=(5, 64, 64, 16))
    darr_mov_ds = darr_mov.map_blocks(lambda x: darr.mean(x, axis=0),
                                      chunks=(chunk_size, *dims[1:]),
                                      ).compute()
    toc = time.perf_counter()
    print(f"Computed grouped mean (T={chunk_size} in {toc - tic} seconds")
    return darr_mov_ds


# Yr = np.memmap(filename, mode=mode, shape=prepare_shape((d1 * d2 * d3, T)), dtype=np.float32,
#                order=order)

# %% load motion corrected movies
# moco_dir = Path("/local/matrix/Remy-Data/projects/odor_unpredictability/processed_data/2022-09-27"
#                 "/1/unpredictable0/caiman_mc")

# ask user for directory path
moco_dir = Path(input("Enter path to caiman_mc folder: "))

files = parse_folder_contents(moco_dir, moco_fname_templates)

# load movies
m_rig = cm.load(files['m_rig'])
m_els = cm.load(files['m_els'])

# load moco objects
mc_rig = np.load(files['mc_rig'], allow_pickle=True).item()
mc_els = np.load(files['mc_rig'], allow_pickle=True).item()

# %% Inspect rigid ("m_rig") moco results
downsample_ratio = 0.1
chunk_size = round(1 / downsample_ratio)
print(f"chunk_size = {chunk_size}")
# %%
m_rig_ds = cm.movie(darr_downsample_movie(m_rig, chunk_size))
m_els_ds = cm.movie(darr_downsample_movie(m_els, chunk_size))

# %% M_RIG PLANE

plane = 0
print(f'M_RIG: downsampled plane {plane}.')
plane_ds = m_rig_ds[..., plane]

plane_ds.play(q_max=99.5, fr=30, magnification=2)

# %%
multiplanes = [0, 1, 2, 3]
multiplanes = cm.concatenate([m_rig_ds[..., z] for z in multiplanes], axis=1)

multiplanes.play(q_max=99.5, fr=30, magnification=1)

# %% Inspect pw_rigid (non-rigid, i.e. elastic/"m_els") results

plane = 15
downsample_ratio = 0.1

# downsample plane
plane_ds = m_els[:, :, :, plane].resize(1, 1, downsample_ratio)
print(f'M_ELS: downsampled plane {plane}.')

# play downsampled plane
plane_ds.play(q_max=99.5, fr=60, magnification=2)
# %%

m_els.resize()
# %% [markdown]
"""
Once `bad_planes_rig(els)` and `which_movie` are set in motion_metrics.json, save the chosen 
corrected movie as a suite2p-ordered .h5 file in folder `source_extraction_s2p`, in the same 
parent directory as motion correction folder **caiman_mc** (`moco_dir`).

```{.python}
from src import convert
convert.copy_to_suite2p_from_moco_dir(moco_dir)
```
>>> from src import convert, cnmf_3d_pipeline
>>> convert.copy_to_suite2p_from_moco_dir(moco_dir) 
>>> convert.copy_to_suite2p_from_moco_dir(moco_dir, copy_as_type='.tif')
>>> for mmap_file in moco_dir.glob("*.mmap"): convert.split_mmap_to_tiffs(mmap_file, batch_size=500)
"""
