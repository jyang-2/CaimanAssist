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
    m_rig = resize_movie_in_time(cm.load(files['m_rig']), downsample_ratio=downsample_ratio)
    m_els = resize_movie_in_time(cm.load(files['m_els']), downsample_ratio=downsample_ratio)

    return cm.movie(m_rig), cm.movie(m_els), files


# %%

NAS_PROC_DIR = Path("/local/storage/Remy/narrow_odors/processed_data")
folder_list = sorted(list(NAS_PROC_DIR.rglob("*tsub*/*/caiman_mc")))
for i, item in enumerate(folder_list):
    print(f"{i:02d}.\t{item}")

# %%
idx = 2
folder = folder_list[idx]
print(folder.parts[-5:])
files = parse_folder_contents(folder_list[idx], moco_fname_templates)
m_rig = cm.load(files['m_rig'])
m_els = cm.load(files['m_els'])
