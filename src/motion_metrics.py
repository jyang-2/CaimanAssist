import sys

sys.path.append("/home/remy/PycharmProjects/CaimanAssist/src")
import numpy as np
from pathlib import Path
from typing import Tuple, List, Literal, Union, Optional
import pydantic
from itertools import product
import json
import pandas as pd
import argparse


# import typing
# import pprint as pp
# import utils2p


class PatchShifts3dRigid(pydantic.BaseModel):
    dims: List[int]
    overlaps: List[int]
    strides: List[int]
    shifts_xmin: float
    shifts_xmax: float
    shifts_ymin: float
    shifts_ymax: float
    shifts_zmin: float
    shifts_zmax: float

    def save_json(self, filepath):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open('w') as f:
            json.dump(self.dict(), f, indent=4)
        return filepath


class PatchShifts3dElastic(pydantic.BaseModel):
    dims: List[int]
    overlaps: List[int]
    strides: List[int]
    patch_x: List[int]
    patch_y: List[int]
    patch_z: List[int]
    shifts_xmin: List[float]
    shifts_xmax: List[float]
    shifts_ymin: List[float]
    shifts_ymax: List[float]
    shifts_zmin: List[float]
    shifts_zmax: List[float]

    def to_dataframe(self):
        attr_vars = ['dims', 'overlaps', 'strides', 'mc_els_filepath']
        df_patch_shifts = pd.DataFrame(self.dict(exclude=set(attr_vars)))
        return df_patch_shifts

    def save_json(self, filepath):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open('w') as f:
            json.dump(self.dict(), f, indent=4)
        return filepath


class ShiftLimits3d(pydantic.BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]


# %%

class MocoMetrics(pydantic.BaseModel):
    rig_shift_lims: ShiftLimits3d = pydantic.Field(...)
    els_shift_lims: ShiftLimits3d = pydantic.Field(...)
    rig_smoothness: Optional[List]
    els_smoothness: Optional[List]
    bad_planes_rig: Optional[List] = []
    bad_planes_els: Optional[List] = []
    which_movie: Optional[str]

    def save_json(self, filepath):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open('w') as f:
            json.dump(self.dict(), f, indent=4)
        return filepath


# %%
def get_1d_patch_indices(dim_size, overlap, stride):
    """
    Gives starting indices for motion correction patch grid, for given dimension parameters

    Args:
        dim_size (int): Size of numpy array along desired dimension (ex: Ym_els.shape[0])
        overlap (int): Overlap length (ex: mc_els['overlap'][0])
        stride (int): Stride length (ex: mc_els['strides'][0])

    Returns:
        patch_idx_1d (List[int]): starting indices of caiman motion correction patches
    """
    win_length = overlap + stride
    return list(range(0, dim_size - win_length, stride)) + [dim_size - win_length]


def patch_indices_from_dims(dims, overlaps, strides):
    """Gives starting indices for patches used in piecewise-rigid motion correction.
        (patch size = overlaps + strides)

    Args:
        dims (Union[List[int], np.ndarray]): Size of numpy array along desired dimension (ex: Ym_els.shape[0])
        overlaps (Union[List[int], np.ndarray]): Overlap length (ex: mc_els['overlap'][0])
        strides (Union[List[int], np.ndarray]): Stride length (ex: mc_els['strides'][0])

    Returns:
        patch_x, patch_y
        patch_idx_1d (List[int]): starting indices of caiman motion correction patches
    """
    win_size = np.add(overlaps, strides)
    patch_idx_lists = [get_1d_patch_indices(*x) for x in zip(dims, overlaps, strides)]
    return tuple(patch_idx_lists)


def get_patch_params(mc_either):
    dims = mc_either['total_template_rig'].shape
    overlaps = mc_either['overlaps']
    strides = mc_either['strides']
    return dims, overlaps, strides


def get_3d_rig_shifts(mc_rig):
    dims, overlaps, strides = get_patch_params(mc_rig)
    shifts = np.array(mc_rig['shifts_rig'])

    shifts_xmax, shifts_ymax, shifts_zmax = shifts.min(axis=0).tolist()
    shifts_xmin, shifts_ymin, shifts_zmin = shifts.max(axis=0).tolist()

    return PatchShifts3dRigid(
            dims=dims,
            overlaps=overlaps,
            strides=strides,
            shifts_xmin=shifts_xmin,
            shifts_xmax=shifts_xmax,
            shifts_ymin=shifts_ymin,
            shifts_ymax=shifts_ymax,
            shifts_zmin=shifts_zmin,
            shifts_zmax=shifts_zmax, )


def get_3d_els_patch_shifts(mc_els):
    """ From mc_els.npy, load information about caiman's motion correction results.

    Args:
        mc_els (dict): caiman motion correction object loaded from 'mc_els.npy' from .../<movie dir>/caiman_mc

    Returns:
        patch_shifts_data (dict): dict w/ keys
                                    - dims: [x, y, z] dimensions
                                    - overlaps: [x, y, z] overlaps
                                    - strides: [x, y, z] strides
                                    - patch_x, patch_y, patch_z: starting indices of grid patches
                                    - shifts_xmin, shifts_ymin, shifts_zmin: np.ndarray (# of patches)
                                    - shifts_xmax, shifts_ymax, shifts_zmax: np.ndarray (# of patches)
    """
    dims, overlaps, strides = get_patch_params(mc_els)

    # patch_idx_arr: (# patches) x (# dims)
    patch_ranges = patch_indices_from_dims(dims, overlaps, strides)
    patch_idx_arr = np.array(list(product(*patch_ranges)))

    # ..._shifts_els : (# timepoints) x (# patches)
    x_shifts_els = np.array(mc_els['x_shifts_els'])
    y_shifts_els = np.array(mc_els['y_shifts_els'])
    z_shifts_els = np.array(mc_els['z_shifts_els'])

    shifts_xmin = x_shifts_els.min(axis=0)
    shifts_xmax = x_shifts_els.max(axis=0)
    shifts_ymin = y_shifts_els.min(axis=0)
    shifts_ymax = y_shifts_els.max(axis=0)
    shifts_zmin = z_shifts_els.min(axis=0)
    shifts_zmax = z_shifts_els.max(axis=0)

    patch_x, patch_y, patch_z = zip(*patch_idx_arr)

    return PatchShifts3dElastic(
            dims=dims,
            overlaps=overlaps,
            strides=strides,
            patch_x=patch_x,
            patch_y=patch_y,
            patch_z=patch_z,
            shifts_xmin=shifts_xmin.tolist(),
            shifts_xmax=shifts_xmax.tolist(),
            shifts_ymin=shifts_ymin.tolist(),
            shifts_ymax=shifts_ymax.tolist(),
            shifts_zmin=shifts_zmin.tolist(),
            shifts_zmax=shifts_zmax.tolist(), )


def get_rig_shift_lims(patch_shifts):
    return ShiftLimits3d(
            x=[patch_shifts.shifts_xmin, patch_shifts.shifts_xmax],
            y=[patch_shifts.shifts_ymin, patch_shifts.shifts_ymax],
            z=[patch_shifts.shifts_zmin, patch_shifts.shifts_zmax],
    )


# %%
def compute_smoothness_2d(m):
    smoothness = np.sum(np.sum(np.array(np.gradient(m)) ** 2, 0))
    # smoothness = np.sqrt(
    #     np.sum(np.sum(np.array(np.gradient(np.mean(m, 0))) ** 2, 0)))
    return smoothness


def get_els_shift_lims(patch_shifts):
    """
        Args:
            patch_shifts (PatchShifts3dElastic): generated from "mc_els.npy" file in .../<movie dir>/caiman_mc
        Returns:
            shift_limits (ShiftLimits3d): largest shift at the movie limits along different axes
                                            i.e. along z axis, the largest shift at (top, bottom) patches

        Example:
            >>> mc_els = np.load(moco_dir.joinpath('mc_els.npy'), allow_pickle=True).item()
            >>> els_patch_shifts = get_3d_els_patch_shifts(mc_els)
            >>> els_shift_limits = get_els_shift_lims(els_patch_shifts)
    """
    df = patch_shifts.to_dataframe()
    return ShiftLimits3d(
            x=[df.loc[df['patch_x'] == df['patch_x'].min(), 'shifts_xmax'].max(),
               df.loc[df['patch_x'] == df['patch_x'].max(), 'shifts_xmin'].min()],

            y=[df.loc[df['patch_y'] == df['patch_y'].min(), 'shifts_ymax'].max(),
               df.loc[df['patch_y'] == df['patch_y'].max(), 'shifts_ymin'].min()],

            z=[df.loc[df['patch_z'] == df['patch_z'].min(), 'shifts_zmax'].max(),
               df.loc[df['patch_z'] == df['patch_z'].max(), 'shifts_zmin'].min()]
    )


# %%
def main(moco_dir):
    print(f"\n---")
    print(moco_dir)

    # load rigid motion correction results, and write shift info to json
    mc_rig = np.load(moco_dir.joinpath('mc_rig.npy'), allow_pickle=True).item()
    rig_patch_shifts = get_3d_rig_shifts(mc_rig)
    rig_patch_shifts.save_json(moco_dir.joinpath('rig_patch_shifts.json'))
    print('rig_patch_shifts.json saved.')

    # load elastic motion correction results, and write shift info to json
    mc_els = np.load(moco_dir.joinpath('mc_els.npy'), allow_pickle=True).item()
    els_patch_shifts = get_3d_els_patch_shifts(mc_els)
    els_patch_shifts.save_json(moco_dir.joinpath('els_patch_shifts.json'))
    print('els_patch_shifts.json saved.')

    # compute moco_metrics, and write shift info to json
    rig_shift_limits = get_rig_shift_lims(rig_patch_shifts)
    els_shift_limits = get_els_shift_lims(els_patch_shifts)
    moco_metrics = MocoMetrics(rig_shift_lims=rig_shift_limits,
                               els_shift_lims=els_shift_limits)
    moco_metrics.save_json(moco_dir.joinpath('moco_metrics.json'))
    print('moco_metrics.json saved.')
    return moco_metrics


# %%
if __name__ == '__main__':
    # NAS_PROC_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures/processed_data")
    #
    # folder_list = sorted(list(NAS_PROC_DIR.rglob('kiwi_components_again_with_partial_and_probes/caiman_mc')))
    # for folder in folder_list:
    #     print(folder)
    #     try:
    #         main(folder)
    #     except Exception as e:
    #         print("motion correction metrics did not run sucessfully.")
    #         print(e)

    USAGE = f"writes shifts from caiman's 3d motion correction to .../<moco_dir>/<els, " \
            f"rig>_patch_shifts.json,\n" \
            "and writes motion metrics to .../<moco_dir>/moco_metrics.json"
    parser = argparse.ArgumentParser(description=USAGE)
    parser.add_argument('moco_dir', type=str,
                        help='path to caiman motion correction results folder, probably w/ name '
                             'caiman_mc')
    args = parser.parse_args()

    folder = Path(args.moco_dir)

    try:
        main(folder)
    except Exception as e:
        print("motion_metrics.py did not run sucessfully.")
        print(e)
