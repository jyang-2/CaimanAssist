from pathlib import Path
import caiman as cm
import utils2p
import json

movie_base_savename = 'Image_001_001'
default_save_dir = 'fast_dir' # [fast_dir, parent_dir]
fast_disk = Path("/home/remy/Documents/caiman-fast-disk")


def have_same_parent(file_list, n=0):
    """
    Check if nth parent directory is

    Args:
        file_list (List[Path]):
        n ():

    Returns:

    """
    parent_dirs = [Path(item).parents[n] for item in file_list]
    parent_dirs = list(set(parent_dirs))

    if len(parent_dirs) == 1:
        return True
    elif len(parent_dirs) > 1:
        return False


def mmap_to_hdf5(mmap_file, h5_savefile=None, input_dim_ord='txyz', output_dim_ord='tzyx'):
    if isinstance(mmap_file, str):
        mmap_file = Path(mmap_file)

    # load movie
    Y = cm.load(mmap_file, is3D=True)

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = Y.transpose(dim_ord)

    if h5_savefile is None:
        h5_savefile = mmap_file.with_suffix('.h5')

    Y.save(h5_savefile)
    return h5_savefile


def single_tiff_to_hdf5(tiff_file, h5_savefile=None, input_dim_ord='tzyx', output_dim_ord='txyz'):
    if isinstance(tiff_file, str):
        tiff_file = Path(tiff_file)

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = cm.load_movie(tiff_file, is3D=True)
    Y = Y.transpose(dim_ord)

    if h5_savefile is None:
        h5_savefile = tiff_file.with_suffix('.h5')
    return h5_savefile


def tiff_list_to_hdf5(tiff_files, h5_savefile=None, input_dim_ord='tzyx', output_dim_ord='txyz'):
    """Loads a list of tiff stack files, and saves to hdf5 file.
        `h5_savefile` is required, if `tiff_files` are not in the same parent folder."""

    if any([isinstance(item, str) for item in tiff_files]):
        tiff_files = [Path(item) for item in tiff_files]

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = cm.load_movie_chain(tiff_files, is3D=True)
    Y = Y.transpose(dim_ord)

    if h5_savefile is None:
        parent_dirs = list(set(item.parent for item in tiff_files))
        if len(parent_dirs) > 0:
            raise(ValueError, "h5_savefile is required (tiff_files don't have the same parent directory.")
        elif len(parent_dirs) == 0:
            h5_savefile = parent_dirs[0].joinpath(f"{movie_base_savename}.h5")
    Y.save(h5_savefile)
    return h5_savefile


def stk_dir_to_hdf5(stk_dir, h5_savefile=None, input_dim_ord='tzyx', output_dim_ord='txyz'):
    """ Loads split tiff stacks from `{movie dir}/stk`, fixes dimension order, and saves as .hdf5"""
    if isinstance(stk_dir, str):
        stk_dir = Path(stk_dir)

    tiff_files = sorted(list(stk_dir.glob("*.tif")))

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = cm.load_movie_chain(tiff_files, is3D=True)
    Y = Y.transpose(dim_ord)

    if h5_savefile is None:
        h5_savefile = stk_dir.joinpath(f'{movie_base_savename}.h5')

    Y.save(h5_savefile)
    return h5_savefile


def copy_to_suite2p(moco_dir, movie_type='m_els'):
    source_extract_s2p_dir = moco_dir.with_name('source_extraction_s2p')

    if not source_extract_s2p_dir.is_dir():
        source_extract_s2p_dir.mkdir()

    if movie_type == 'm_els':
        fname_mmap = list(moco_dir.glob("*_els_*.mmap"))[0]
    elif movie_type == 'm_rig':
        fname_mmap = list(moco_dir.glob("*_rig_*.mmap"))[0]

    fname_h5 = source_extract_s2p_dir.joinpath(fname_mmap.name).with_suffix('.h5')

    print(f"\ncopying:")
    print(f"\t- {fname_mmap}")
    print(f"\t- {fname_h5}")
    # convert pw_rigid corrected .mmap file to .h5 dataset in suite2p order ('tzyz')
    saved_file = mmap_to_hdf5(fname_mmap, h5_savefile=fname_h5)
    print(f"\tdone")
    return saved_file


def copy_to_suite2p_from_moco_dir(moco_dir):
    """Reads moco_metrics.json in moco_dir, and moves the specified .mmap file to
    a new folder, source_extraction_s2p, based on moco_metrics['which_movie'].

    """
    with moco_dir.joinpath('moco_metrics.json').open('r') as f:
        moco_metrics = json.load(f)
    print(moco_metrics)
    saved_file = copy_to_suite2p(moco_dir, moco_metrics['which_movie'])
    return saved_file
