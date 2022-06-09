import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from pathlib import Path
import numpy as np
import logging
logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

fast_disk = Path("/home/remy/Documents/caiman-fast-disk")
prj = 'narrow_odors'

if prj == 'natural_mixtures':
    NAS_PRJ_DIR = Path('/local/storage/Remy/natural_mixtures')
    NAS_PROC_DIR = NAS_PRJ_DIR.joinpath('processed_data')
    CAIMAN_FAST_DISK = Path("/home/remy/Documents/caiman-fast-disk")
elif prj == 'narrow_odors':
    NAS_PRJ_DIR = Path('/local/storage/Remy/narrow_odors')
    NAS_PROC_DIR = NAS_PRJ_DIR.joinpath('processed_data')
    CAIMAN_FAST_DISK = Path("/home/remy/Documents/caiman-fast-disk")

# %%
if 'dview' in locals():
    cm.stop_server(dview=locals()['dview'])
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


#%%
fname = Path("/local/storage/Remy/narrow_odors/processed_data/2022-04-19/2/anat_001__tsub_3/0/caiman_mc" \
             "/Image_001_00_rig__d1_256_d2_256_d3_16_order_F_frames_952_.mmap")

mc = np.load(fname.with_name('mc_rig.npy'), allow_pickle=True).item()
border_to_0 = 0 if mc['border_nan'] == 'copy' else mc['border_to_0']
print(Path(mc['mmap_file'][0]).is_file())

m_rig = cm.load(mc['mmap_file'], is3D=True)
fname = Path(mc['mmap_file'][0])
m_rig.save(fname.with_name('Image_001_00_rig.mmap'), order='C')

# reload mmap file (C_order)
# NOTE: MAKE SURE `fname_new` is a str, not a Path!
fname_new = Path("/home/remy/Documents/caiman-fast-disk/Image_001_00_rig__d1_256_d2_256_d3_16_order_C_frames_952_.mmap")
Yr, dims, T = cm.load_memmap(str(fname_new))
images = np.reshape(Yr.T, [T] + list(dims), order='F')

# %%
opts_dict = dict(               # parameters for source extraction and deconvolution
    strides=(48, 48, 16),
    overlaps=(24, 24, 2),
    max_shifts=(6, 6),
    p=2,                       # order of the autoregressive system
    nb=2,                     # number of global background components
    merge_thr=0.85,            # merging threshold, max correlation allowed
    rf=25,                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride=10,             # amount of overlap between the patches in pixels
    K=50,                       # number of components per patch
    gSig=[3, 3, 2],               # expected half size of neurons in pixels
    method_init='greedy_roi',  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub=1,                    # spatial subsampling during initialization
    tsub=1,                    # temporal subsampling during intialization
    # parameters for component evaluation
    min_SNR=0.5,               # signal to noise ratio for accepting a component
    rval_thr=0.85,              # space correlation threshold for accepting a component
    use_cnn=False)
opts = params.CNMFParams(params_dict=opts_dict)

#%%
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% run patches on 3D movie
# set parameters
rf = 25  # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 10  # amount of overlap between the patches in pixels
K = 20  # number of neurons expected per patch
gSig = [3, 3, 2]  # expected half size of neurons
merge_thresh = 0.9  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system

cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview,
                rf=rf, stride=stride, only_init_patch=True)
cnm.params.set('spatial', {'se': np.ones((3, 3, 1), dtype=np.uint8)})
cnm = cnm.fit(images,
              # indices=(slice(None), slice(None), slice(None))
              )

print(('Number of components:' + str(cnm.estimates.A.shape[-1])))
#%%
Cn = cm.local_correlations(images)  # compute local correlation image
cnm.save(str(CAIMAN_FAST_DISK.joinpath('analysis_results.hdf5')))

# view
cnm.estimates.nb_view_components_3d(image_type='mean', dims=dims, axis=2)
#%%
import time
start = time.time()

# load memmapped file
Yr, dims, T = cm.load_memmap(mc['mmap_file'][0])
print(f'Yr in F-order: {np.isfortran(Yr)}')

fname_new = cm.save_memmap(mc['mmap_file'],
                           base_name="/local/storage/Remy/narrow_odors/processed_data/2022-04-19/2/anat_001__tsub_3/0"
                                     "/caiman_mc/Image_001_00_rig_",
                           order='C', is_3D=True,
                           border_to_0=border_to_0, dview=dview)

base_name = "/home/remy/Documents/caiman-fast-disk/Image_001_00_rig__d1_256_d2_256_d3_16_order_F_frames_952_.mmap"

m_rig = cm.load(mc['mmap_file'], is3D=True)
fname_new = cm.save_memmap([m_rig],
                           base_name='"/home/remy/Documents/caiman-fast-disk/memmap_',
                           order='C',
                           border_to_0=border_to_0,
                           is_3D=True, dview=dview)
#%%
fname_new = cm.save_memmap(mc['mmap_file'],
                           base_name='"/home/remy/Documents/caiman-fast-disk/Image_001_001_rig_',
                           order='C',
                           is_3D=True,
                           border_to_0=border_to_0, dview=dview)    # exclude borders
end = time.time()
print(f"Time elapsed: {end-start}")
#%% now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')

#%% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
