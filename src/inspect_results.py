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

from pathlib import Path
import caiman as cm
from caiman.motion_correction import MotionCorrect
import pydantic
from pydantic import BaseModel
import numpy as np

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
        m_planes = [m[..., 0].resize(1, 1, downsample_ratio) for m in np.split(movie, d3, axis=-1)]
        resized_movie = np.stack(m_planes, axis=-1)
    elif movie.ndim == 3:
        T, d1, d2 = movie.shape
        resized_movie = movie.resize(1, 1, downsample_ratio)
    return resized_movie


def load_moco_results(folder, downsample_ratio=0.05):
    files = parse_folder_contents(folder, moco_fname_templates)
    movie_rig = resize_movie_in_time(cm.load(files['m_rig']), downsample_ratio=downsample_ratio)
    movie_els = resize_movie_in_time(cm.load(files['m_els']), downsample_ratio=downsample_ratio)

    return cm.movie(movie_rig), cm.movie(movie_els), files


# %% load motion corrected movies
# moco_dir = Path("/local/matrix/Remy-Data/projects/odor_unpredictability/processed_data/2022-09-27"
#                 "/1/unpredictable0/caiman_mc")

# ask user for directory path
moco_dir = Path(input("Enter path to caiman_mc folder: "))

files = parse_folder_contents(moco_dir, moco_fname_templates)
m_rig = cm.load(files['m_rig'])
m_els = cm.load(files['m_els'])

# %% Inspect rigid ("m_rig") moco results

plane = 1
downsample_ratio = 0.05

plane_ds = m_rig[:, :, :, plane].resize(1, 1, downsample_ratio)
print(f'M_RIG: downsampled plane {plane}.')
plane_ds.play(q_max=99.5, fr=60, magnification=2)

# %% Inspect pw_rigid (non-rigid, i.e. elastic/"m_els") results

plane = 0
downsample_ratio = 0.1

# downsample plane
plane_ds = m_els[:, :, :, plane].resize(1, 1, downsample_ratio)
print(f'M_ELS: downsampled plane {plane}.')

# play downsampled plane
plane_ds.play(q_max=99.5, fr=60, magnification=2)

# %% [markdown]
"""
Once `bad_planes_rig(els)` and `which_movie` are set in motion_metrics.json, save the chosen 
corrected movie as a suite2p-ordered .h5 file in folder `source_extraction_s2p`, in the same 
parent directory as motion correction folder **caiman_mc** (`moco_dir`).

>>> from src import convert
>>> convert.copy_to_suite2p_from_moco_dir(moco_dir) 
"""


