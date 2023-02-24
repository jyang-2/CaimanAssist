"""
Run motion correction
"""
from pathlib import Path
from src import cnmf_3d_pipeline, motion_metrics


def main(folder, mc_dict=None):
    print(folder)
    try:
        moco_dir = cnmf_3d_pipeline.main(folder, mc_dict=mc_dict)
        moco_metrics = motion_metrics.main(moco_dir)
    except Exception as e:
        print("motion correction metrics did not run sucessfully.")
        print(e)


if __name__ == '__main__':
    NAS_PROC_DIR = Path('/local/matrix/Remy-Data/projects/odor_space_collab/processed_data')

    mc_opts_dict = {
        'strides': (96, 96, 4),  # start a new patch for pw-rigid motion correction every x pixels
        'overlaps': (32, 32, 2),  # overlap between pathes (size of patch strides+overlaps)
        'max_shifts': (32, 32, 5),  # maximum allowed rigid shifts (in pixels)
        'max_deviation_rigid': 5,  # maximum shifts deviation allowed for patch with respect to
        'pw_rigid': False,  # flag for performing non-rigid motion correction
        'is3D': True,
        'niter_rig': 2,
        'splits_rig': 14,
        'splits_els': 3,
        'border_nan': 'copy',
        'nonneg_movie': False
    }

    folder_list = [
        Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data_old/"
             "2019-03-07/2/_003_tsub05/stk")
                   ]

    for stk_dir in folder_list:
        print(stk_dir)
        main(Path(stk_dir), mc_dict=mc_opts_dict)
