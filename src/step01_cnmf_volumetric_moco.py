"""
Run motion correction
"""
from pathlib import Path
from src import cnmf_3d_pipeline, motion_metrics


def main(folder):
    print(folder)
    try:
        moco_dir = cnmf_3d_pipeline.main(folder)
        moco_metrics = motion_metrics.main(moco_dir)
    except Exception as e:
        print("motion correction metrics did not run sucessfully.")
        print(e)


if __name__ == '__main__':
    NAS_PROC_DIR = Path('/local/matrix/Remy-Data/projects/odor_space_collab/processed_data')
    stk_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/"
                   "2022-10-11/1/megamat0_001/stk")
    print(stk_dir)
    main(stk_dir)

