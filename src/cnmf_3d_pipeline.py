"""Run 3D motion correction for a given movie.
01. load tiff stacks, save to temporary hdf5 file in `caiman-fast-disk`.
    - files generated:
        - "Image_001_001.h5"
02. set motion correction options, and save the following files to caiman-fast-disk:
    - `mc_opts0.json`
    - `mc_opts_rig.npy`
03. run rigid motion correction w/ save_movie=True. save
    - 'Image_001_001_rig_(...).mmap'
    - 'mc_rig.npy'
04. run non-rigid (pw_rigid=True) motion correction w/ save_movie=True.
    - 'Image_001_001_els_(...).mmap'
    - 'mc_els.npy'
05. plot shifts and save plots
    - cnmf_rig_shifts.png
    - cnmf_els_shifts.png
06. Compute motion correction metrics
    - bord_px_rig, bord_px_els
    - correlations, flows, norms, crispness

07. Copy results to movie folder (should be folder above tiff_folder)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import json
import shutil
import caiman as cm
import caiman.source_extraction.cnmf as cnmf
import utils2p
from caiman.motion_correction import MotionCorrect
from itertools import chain


CAIMAN_FAST_DISK = Path("/home/remy/Documents/caiman-fast-disk")


# %%
def compute_smoothness_2d(m):
    smoothness = np.sum(np.sum(np.array(np.gradient(m)) ** 2, 0))
    # smoothness = np.sqrt(
    #     np.sum(np.sum(np.array(np.gradient(np.mean(m, 0))) ** 2, 0)))
    return smoothness


def compute_crispness(m):
    if m.ndim == 3:
        grad = np.array(np.gradient(m.data))
    elif m.ndim == 2:
        grad = np.array(np.gradient(m.data))
    elif m.ndim == 4:
        grad = np.array(np.gradient(np.mean(m, 0)))

    crispness = np.sqrt(np.sum(np.sum(grad ** 2, 0)))
    return crispness


def mmap_to_hdf5(mmap_file, h5_savefile=None, input_dim_ord='txyz', output_dim_ord='tzyx'):
    if isinstance(mmap_file, str):
        mmap_file = Path(mmap_file)

    Y = cm.load(mmap_file, is3D=True)

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = Y.transpose(dim_ord)

    if h5_savefile is None:
        h5_savefile = mmap_file.with_suffix('.h5')

    Y.save(h5_savefile)

    return h5_savefile


def batched_tiffs_to_hdf5(stk_dir, input_dim_ord='tzyx', output_dim_ord='txyz'):
    """ Loads split tiff stacks from `{movie dir}/stk`, fixes dimension order, and saves as .hdf5"""
    if isinstance(stk_dir, str):
        stk_dir = Path(stk_dir)

    tiff_files = sorted(list(stk_dir.glob("*.tif")))

    dim_ord = [output_dim_ord.index(d) for d in input_dim_ord]
    Y = cm.load_movie_chain(tiff_files, is3D=True)
    Y = Y.transpose(dim_ord)

    h5_fname = CAIMAN_FAST_DISK.joinpath('Image_001_001.h5')
    Y.save(h5_fname)
    return h5_fname


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


def plot_rig_shifts(rig_shifts):
    """plot rigid shifts"""
    x_shifts_rig, y_shifts_rig, z_shifts_rig = zip(*mc.shifts_rig)
    fig_rig_shifts, axarr = plt.subplots(figsize=(20, 10), nrows=3, ncols=1)

    axarr[0].plot(x_shifts_rig)
    axarr[0].set_ylabel('x shifts (pixels)')

    axarr[1].plot(y_shifts_rig)
    axarr[1].set_ylabel('y_shifts (pixels)')

    axarr[2].plot(z_shifts_rig)
    axarr[2].set_ylabel('z_shifts (pixels)')

    axarr[-1].set_xlabel('frames')

    fig_rig_shifts.suptitle('caiman: rig shifts')
    return fig_rig_shifts, axarr


def plot_els_shifts(x_shifts_els, y_shifts_els, z_shifts_els=None):
    """ Plot piecewise rigid shifts"""
    fig_els_shifts, axarr = plt.subplots(figsize=(20, 10), nrows=3, ncols=1)

    axarr[0].plot(x_shifts_els)
    axarr[0].set_ylabel('x shifts (pixels)')

    axarr[1].plot(y_shifts_els)
    axarr[1].set_ylabel('y_shifts (pixels)')

    axarr[2].plot(z_shifts_els)
    axarr[2].set_ylabel('z_shifts (pixels)')

    axarr[-1].set_xlabel('frames')

    fig_els_shifts.suptitle('caiman: els shifts')
    return fig_els_shifts, axarr


def save_summary_images(moco_dir):
    fname_rig = list(moco_dir.glob("*_rig_*.mmap"))[0]
    fname_els = list(moco_dir.glob("*_els_*.mmap"))[0]

    # rigid
    print(f'\nm_rig:')
    m_rig = cm.load(fname_rig, is3D=True)

    print(f'\t- movie loaded, computing mean')
    Ym_rig = np.nanmean(m_rig, axis=0)

    utils2p.save_img(moco_dir.joinpath('Ym_rig.tif'), np.array(Ym_rig.T))
    print(f'\t- time-averaged movie saved')

    # nonrigid
    print(f'\nm_els:')
    m_els = cm.load(fname_els, is3D=True)

    print(f'\t- movie loaded, computing mean')
    Ym_els = np.nanmean(m_els, axis=0)

    utils2p.save_img(moco_dir.joinpath('Ym_els.tif'), np.array(Ym_els.T))
    print(f'\t- time-averaged movie saved')

    return Ym_rig, Ym_els


def save_grouped_mean_images(moco_dir, chunk_size=500):
    fname_rig = list(moco_dir.glob("*_rig_*.mmap"))[0]
    fname_els = list(moco_dir.glob("*_els_*.mmap"))[0]

    # rigid
    m_rig = cm.load(fname_rig, is3D=True)
    split_rig = np.split(np.array(m_rig),
                         np.arange(chunk_size, m_rig.shape[0], chunk_size)
                         )
    grouped_rig_mean = np.stack([x.mean(axis=0).T for x in split_rig])
    utils2p.save_img(moco_dir.joinpath(f'Ym_rig_{chunk_size}.tif'), grouped_rig_mean)

    # pw_rigid
    m_els = cm.load(fname_els, is3D=True)
    split_els = np.split(np.array(m_els),
                         np.arange(chunk_size, m_els.shape[0], chunk_size)
                         )
    grouped_els_mean = np.stack([x.mean(axis=0).T for x in split_els])
    utils2p.save_img(moco_dir.joinpath(f'Ym_els_{chunk_size}.tif'), grouped_els_mean)
    return grouped_rig_mean, grouped_els_mean


def save_template_images(moco_dir):
    mc_rig = np.load(moco_dir.joinpath('mc_rig.npy'), allow_pickle=True).item()

    total_template_rig = mc_rig['total_template_rig']
    utils2p.save_img(moco_dir.joinpath('total_template_rig.tif'), total_template_rig.T)

    template_rig = np.stack(mc_rig['templates_rig'], axis=-1)
    utils2p.save_img(moco_dir.joinpath('template_rig.tif'),
                     template_rig.T)

    mc_els = np.load(moco_dir.joinpath('mc_els.npy'), allow_pickle=True).item()
    template_els = np.stack(mc_els['templates_els'], axis=-1)
    utils2p.save_img(moco_dir.joinpath('template_els.tif'), template_els.T)
    #
    # cm_dim_ord = 'txyz'
    # s2p_dim_ord = 'tzyx'

    return template_rig, template_els


# def save_grouped_mean_images(moco_dir, batch_size=500):


def compute_cm_grouped_mean(Y, batch_size=500):
    Ym_list = [Y[t:t + batch_size, ...].mean(axis=0) for t in range(0, Y.shape[0], batch_size)]
    Ym_grouped = cm.concatenate(Ym_list, axis=0)
    return Ym_grouped


# def save_batched_mean_images(moco_dir, batch_size=500):
#     Y = np.load(moco_dir.joinpath('mc_rig.npy'), allow_pickle=True).item()
#     Ym_list = []
#     for t in range(0, Y.shape[0], batch_size):
#         stk_idx = t//batch_size
#         Ym = Y[t:t + batch_size].mean(axis=0).transpose(0, 3, 2, 1)
#         Ym_list.append(np.array(Ym))
#     Ym_grouped =
#         substack = np.array(Y[t:t+batch_size].transpose(dim_ord))
#         print(substack.shape)
#         print(f"...Saving {filestr}")
#         utils2p.save_img(tiff_dir.joinpath(filestr), np.array(substack))

# %%
def main(tiff_folder, mc_dict=None, interactive=False, run_nonrigid=True):
    print(f'\n\nRunning caiman motion correction:')
    print(f'---------------------------------')
    print(f"tiff_folder = {tiff_folder}")
    print(f"\nCopying to caiman-fast-disk:")
    h5_file = batched_tiffs_to_hdf5(tiff_folder)
    print(f"\t{h5_file} created.")

    # folder in natural_mixtures/processed_data to save 3D motion correction results
    mov_mc_savedir = tiff_folder.parent.joinpath('caiman_mc')

    # =============================================================
    # Section 01. Initialize motion correction parameters
    # =============================================================
    if mc_dict is None:
        mc_dict = {
            'strides': (64 - 8, 64 - 8, 3),  # start a new patch for pw-rigid motion correction every x
            # # pixels
            'overlaps': (8, 8, 2),  # overlap between pathes (size of patch strides+overlaps)
            # 'strides': (96, 96, 4),  # start a new patch for pw-rigid motion correction every x pixels
            # 'overlaps': (32, 32, 2),  # overlap between pathes (size of patch strides+overlaps)
            'max_shifts': (32, 32, 5),  # maximum allowed rigid shifts (in pixels)
            'max_deviation_rigid': 5,  # maximum shifts deviation allowed for patch with respect to
            # rigid shifts
            'pw_rigid': False,  # flag for performing non-rigid motion correction
            'is3D': True,
            'niter_rig': 2,
            'splits_rig': 14,
            'splits_els': 3,
            'border_nan': 'copy',
            'nonneg_movie': True
        }
    print(f"\nInitializing moco parameters.")

    # save motion correction initial parameters to json file
    with open('/home/remy/Documents/caiman-fast-disk/mc_opts0.json', 'w') as f:
        json.dump(mc_dict, f, indent=4)
    print(f"\tmc_opts0.json saved.")

    # create options obj and save initial parameters
    opts = cnmf.params.CNMFParams(params_dict=mc_dict)
    np.save(CAIMAN_FAST_DISK.joinpath('mc_opts_rig.npy'), opts)
    print(f"\tmc_opts_rig.npy saved.")

    # =============================================================
    # Section 02b. Run rigid motion correction, and save results
    # =============================================================
    # start parallel cluster
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

    print(f'\nRunning rigid motion correction.')
    mc = cm.motion_correction.MotionCorrect(h5_file, dview=dview, **opts.get_group('motion'))
    start = time.time()
    mc.motion_correct(save_movie=True)
    end = time.time()
    dtime = end - start
    print(f"\texecution time for rigid motion correction: {dtime}")

    bord_px_rig = np.max(np.abs(np.array(mc.shifts_rig)), axis=0)
    bord_px_rig = np.ceil(bord_px_rig).astype('int')
    print(f"\tbord_px_rig: {bord_px_rig}")

    # save rigid results as dict
    mc_as_dict = {k: v for k, v in mc.__dict__.items() if k != 'dview'}
    np.save(CAIMAN_FAST_DISK.joinpath('mc_rig.npy'), mc_as_dict)
    print(f"\tmc_rig.npy saved.")

    # =============================================================
    # Section 02c. Run piecewise motion correction
    # =============================================================
    print(f'\nRunning non-rigid motion correction.')
    mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction

    # use the template obtained before to save in computation (optional)
    # mc.template = mc.mmap_file

    # run non-rigid correction (may take a while)
    start = time.time()
    mc.motion_correct(save_movie=True,
                      template=mc.total_template_rig
                      )
    end = time.time()
    dtime = end - start
    print(f"\texecution time for non-rigid motion correction: {dtime}")

    # compute border pixels
    bord_px_els = np.array([np.max(np.abs(mc.x_shifts_els)),
                            np.max(np.abs(mc.y_shifts_els)),
                            np.max(np.abs(mc.z_shifts_els))])
    bord_px_els = np.ceil(bord_px_els).astype('int')
    print(f"\tbord_px_els: {bord_px_els}")

    # save as dict
    mc_as_dict = {k: v for k, v in mc.__dict__.items() if k != 'dview'}
    np.save(CAIMAN_FAST_DISK.joinpath('mc_els.npy'), mc_as_dict)
    print(f"\tmc_els.npy saved.")

    # stop the server
    cm.stop_server(dview=dview)

    # =============================================================
    # Section 03. Compute metrics and save to json
    # =============================================================
    metrics = dict(bord_px_rig=bord_px_rig.tolist(),
                   bord_px_els=bord_px_els.tolist())

    with open(CAIMAN_FAST_DISK.joinpath('metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # =============================================================
    # Section 04. move motion correction result files to "{MOV_DIR}/caiman_mc"
    # =============================================================
    print(f"\n- Copying files to: {mov_mc_savedir}")

    shutil.copytree(CAIMAN_FAST_DISK, mov_mc_savedir,
                    ignore=shutil.ignore_patterns("Image_001_001.h5"))
    print(f"\t- all caiman motion correction files copied.")

    # delete files from fask disk
    print('\t- deleting CAIMAN_FAST_DISK contents:')
    for file in CAIMAN_FAST_DISK.iterdir():
        print(f"\t\t{file.name}")
        file.unlink()
        print(f"\t\tdeleted")

    print(f"\ncaiman motion correction complete.")

    Ym_rig, Ym_els = save_summary_images(mov_mc_savedir)
    Ym_rig_chunked, Ym_els_chunked = save_grouped_mean_images(mov_mc_savedir, 500)
    template_rig, template_els = save_template_images(mov_mc_savedir)

    return mov_mc_savedir


# %% copy selected movie from 'caiman_mc' --> 'source_extraction_2p'


# %%

# Yr, dims, T = cm.load_memmap(mc_files[4])
# m_els = Yr.T.reshape((T,) + dims)
# %% load and play rigid motion correction results
# if interactive:
#     if 'm_rig' not in locals():
#         m_rig = cm.load(mc.fname_tot_rig, is3D=True)
#
#     downsample_ratio = 0.2
#     m_rig[:, :, :, 0].resize(1, 1, downsample_ratio) \
#         .play(q_max=99.5, fr=30, magnification=2)
# %%
from skimage.util import montage

# mov_mc_savedir = Path("/local/storage/Remy/natural_mixtures/processed_data/"
#                       "2022-04-09/2/kiwi/caiman_mc")
# mov_mc_savedir = Path("/local/matrix/Remy-Data/projects/natural_mixtures/processed_data/"
#                       "2022-07-12/3/"
#                       "kiwi_components_again_with_partial_and_probes/caiman_mc")
# fname_rig = list(mov_mc_savedir.glob("*_rig_*.mmap"))[0]
# fname_els = list(mov_mc_savedir.glob("*_els_*.mmap"))[0]
#

# # Ym_rig, Ym_rig_isnan, Ym_els, Ym_els_isnan = save_summary_images(mov_mc_savedir)
# # %%
# Ym_rig_isnan = np.array(Ym_rig_isnan).T
# Ym_els_isnan = np.array(Ym_els_isnan).T
# # %%
# from skimage.util import montage
#
# fig, (ax_rig, ax_els) = plt.subplots(1, 2)
# ax_rig.imshow(montage(np.array(~Ym_rig_isnan * 1), grid_shape=(4, 4)), aspect='equal')
# ax_rig.set_title('Ym_rig_isgood')
# ax_els.imshow(montage(np.array(~Ym_els_isnan * 1), grid_shape=(4, 4)), aspect='equal')
# ax_els.set_title('Ym_els_isgood')
# fig.suptitle(f'mean moco results: good pixels\n{mov_mc_savedir}')
# plt.show()
# # %%
# mov_mc_savedir = Path("/local/storage/Remy/natural_mixtures/processed_data/"
#                       "2022-04-09/2/kiwi_ea_eb_only/caiman_mc")
# %%
# print(mov_mc_savedir)
#
# downsample_ratio = 0.1
# z = 13
# movie_type = 'm_rig'
#
# # cm.concatenate(m_rig[:, :, :, z].resize(1, 1, downsample_ratio),
# #                m_els[:, :, :, z].resize(1, 1, downsample_ratio))\
#
# if movie_type == 'm_rig':
#     print(f'\tm_rig, plane {z}')
#     m_rig[:, :, :, z].resize(1, 1, downsample_ratio) \
#         .play(q_max=99.5, fr=30, magnification=2)
#
# if movie_type == 'm_els':
#     print(f'\tm_els, plane {z}')
#     m_els[:, :, :, z].resize(1, 1, downsample_ratio) \
#         .play(q_max=99.5, fr=60, magnification=2)
# # %%
# bad_planes_els = [0, 1, 14, 15]  # done
# bad_planes_rig = [0, 1, 15, 15]
# # %%
# interactive = True
# if interactive:
#     if 'm_rig' not in locals():
#         m_rig = cm.load(fname_rig, is3D=True)
#
# downsample_ratio = 0.1
# m_els[:, :, :, 1].resize(1, 1, downsample_ratio) \
#     .play(q_max=99.5, fr=60, magnification=2)
# # %% check non-rigid planes
# interactive = True
# if interactive:
#     if 'm_els' not in locals():
#         m_els = cm.load(fname_els, is3D=True)
#
#     downsample_ratio = 0.05
#     m_els[:, :, :, 2].resize(1, 1, downsample_ratio) \
#         .play(q_max=99.5, fr=60, magnification=2)

# %%
# # compute max correlation image
# Cn_max_els = cm.summary_images.max_correlation_image(m_els, swap_dim=False)
#
#
# # %% load and play piecewise motion correction results
# if 'm_els' not in locals():
#     m_els = cm.load(mc.fname_tot_els, is3D=True)
#
# m_els[:, :, :, 1].resize(1, 1, downsample_ratio)\
#     .play(q_max=99.5, fr=30, magnification=2)   # play
# #%% motion metrics
# final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
# winsize = 100
# swap_dim = False
# resize_fact_flow = .2    # downsample for computing ROF
#
# tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
#     mc.fname_tot_els[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
# #%%
# final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els) # remove pixels in the boundaries
# winsize = 100
# swap_dim = False
# resize_fact_flow = .2    # downsample for computing ROF
#
# tmpl_els, \
# correlations_els, \
# flows_els, \
# norms_els, \
# crispness_els = cm.motion_correction.compute_metrics_motion_correction(mc.fname_tot_els[0],
#                                                                        final_size[0],
#                                                                        final_size[1],
#                                                                        swap_dim,
#                                                                        winsize=winsize,
#                                                                        play_flow=False,
#                                                                        resize_fact_flow=resize_fact_flow)
# #%% Run CNMF using patch approach
#
# # set parameters
# rf = 25  # half-size of the patches in pixels. rf=25, patches are 50x50
# stride = 10  # amount of overlap between the patches in pixels
# K = 500  # number of neurons expected per patch
# gSig = [2, 2, 1]  # expected half size of neurons
# merge_thresh = 0.95  # merging threshold, max correlation allowed
# p = 0  # order of the autoregressive system
# rval_thr = 0.2
#
# cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview,
#                 rf=rf, stride=stride, only_init_patch=True, rval_thr=rval_thr)
# cnm.params.set('spatial', {'se': np.ones((2,2,1), dtype=np.uint8)})
# cnm = cnm.fit(images)
#
# print(('Number of components:' + str(cnm.estimates.A.shape[-1])))
#
# #%% COMPONENT EVALUATION
# # the components are evaluated in two ways:
# #   a) the shape of each component must be correlated with the data
# #   b) a minimum peak SNR is required over the length of a transient
#
# fr = 1. # approx final rate  (after eventual downsampling )
# decay_time = 10.  # length of typical transient in seconds
# use_cnn = False  # CNN classifier is designed for 2d (real) data
# min_SNR = 0.2     # accept components with that peak-SNR or higher
# rval_thr = 0.2   # accept components with space correlation threshold or higher
# cnm.params.change_params(params_dict={'fr': fr,
#                                       'decay_time': decay_time,
#                                       'min_SNR': min_SNR,
#                                       'rval_thr': rval_thr,
#                                       'use_cnn': use_cnn})
#
# cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
# #%% Run motion correction using NoRMCorre
# mc.motion_correct(save_movie=True)
# m_rig = cm.load(mc.fname_tot_rig, is3D=True)
#
#
# #%%
# fname = "stk_0001_Image_001_001.tif"
# Y = cm.load(fname)
# Y = Y[:, :16, :, :]
# dims = Y.shape[1::]
# Cn = cm.local_correlations(Y, swap_dim=False)
# d1, d2, d3 = dims
# x, y = (int(1.2 * (d1 + d3)), int(1.2 * (d2 + d3)))
# scale = 10/x
# fig = plt.figure(figsize=(scale*x, scale*y))
# axz = fig.add_axes([1-d1/x, 1-d2/y, d1/x, d2/y])
# plt.imshow(Cn.max(2).T, cmap='gray')
# plt.title('Max.proj. z')
# plt.xlabel('x')
# plt.ylabel('y')
# axy = fig.add_axes([0, 1-d2/y, d3/x, d2/y])
# plt.imshow(Cn.max(0), cmap='gray')
# plt.title('Max.proj. x')
# plt.xlabel('z')
# plt.ylabel('y')
# axx = fig.add_axes([1-d1/x, 0, d1/x, d3/y])
# plt.imshow(Cn.max(1).T, cmap='gray')
# plt.title('Max.proj. y')
# plt.xlabel('x')
# plt.ylabel('z');
# plt.show()
# #%%
# #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
# if 'dview' in locals():
#     cm.stop_server(dview=dview)
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='local', n_processes=None, single_thread=False)
#
# #%%
# opts_dict = {'fnames': fname,
#             'strides': (128, 128, 5),    # start a new patch for pw-rigid motion correction every x pixels
#             'overlaps': (12, 12, 2),   # overlap between pathes (size of patch strides+overlaps)
#             'max_shifts': (4, 4, 2),   # maximum allowed rigid shifts (in pixels)
#             'max_deviation_rigid': 5,  # maximum shifts deviation allowed for patch with respect to rigid shifts
#             'pw_rigid': False,         # flag for performing non-rigid motion correction
#             'is3D': True}
#
#
# opts = cnmf.params.CNMFParams(params_dict=opts_dict)
#
# # first we create a motion correction object with the parameters specified
# mc = cm.motion_correction.MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
# # note that the file is not loaded in memory
#
# mc.motion_correct(save_movie=True)
# m_rig = cm.load(mc.fname_tot_rig, is3D=True)
# #%%
# plt.figure(figsize=(12, 3))
# plt.plot(np.array(mc.shifts_rig))    for k in (0,1,2):
#         plt.plot(np.array(s)[:,k], label=('x','y','z')[k])
# for k in (0, 1, 2):
#             plt.plot(np.array(s)[:, k], label=('x', 'y', 'z')[k])
# for i, s in enumerate((mc.shifts_rig, shifts)):
#     plt.subplot(1,2,i+1)
#     for k in (0,1,2):
#         plt.plot(np.array(s)[:,k], label=('x','y','z')[k])
#     plt.legend()
#     plt.title(('inferred shifts', 'true shifts')[i])
#     plt.xlabel('frames')
#     plt.ylabel('pixels')
# plt.show()
# #%%
# # set parameters
# init_params = dict(gSig = [2, 2, 2],
#                    min_corr = 0.3,min_pnr=5, ssub=4, tsub=4)
#
#
#
# # set parameters
# rf = 15  # half-size of the patches in pixels. rf=25, patches are 50x50
# stride = 10  # amount of overlap between the patches in pixels
# K = 50  # number of neurons expected per patch
# gSig = [2, 2, 2]  # expected half size of neurons
# merge_thresh = 0.8  # merging threshold, max correlation allowed
# p = 2  # order of the autoregressive system
#
# img = utils2p.load_img(fname, memmap=True)
# cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview,
#                 rf=rf, stride=stride, only_init_patch=True, c)
# cnm = cnm.fit(img)
#
# print(('Number of components:' + str(cnm.estimates.A.shape[-1])))
# #%%
# # set parameters
# rf = 15  # half-size of the patches in pixels. rf=25, patches are 50x50
# stride = 10  # amount of overlap between the patches in pixels
# K = 10  # number of neurons expected per patch
# gSig = [4, 4, 2]  # expected half size of neurons
# merge_thresh = 0.8  # merging threshold, max correlation allowed
# p = 2  # order of the autoregressive system
#
# #%%
# fname_new = cm.save_memmap(fnames, base_name='memmap_',
#                                    order='C', border_to_0=0, dview=dview)
#  Yr, dims, T = cm.load_memmap(fname_new)
#  images = Yr.T.reshape((T,) + dims, order='F')
# #%% CNMF parameters
# # dataset dependent parameters
#
#
# #%% MEMORY MAPPING
# # memory map the file in order 'C'
# fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
#                            border_to_0=0, dview=dview) # exclude borders
#
# # now load the file
# Yr, dims, T = cm.load_memmap(fname_new)
# images = np.reshape(Yr.T, [T] + list(dims), order='F')
#     #load frames in python format (T x X x Y)
#
#
# #%%
# # set parameters
# if 'dview' in locals():
#     cm.stop_server(dview=dview)
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='local', n_processes=None, single_thread=False)
# #%%
#
# rf = 96  # half-size of the patches in pixels. rf=25, patches are 50x50
# stride = 8  # amount of overlap between the patches in pixels
# K = 500  # number of neurons expected per patch
# gSig = [2, 2, 1]  # expected half size of neurons
# merge_thresh = 0.95  # merging threshold, max correlation allowed
# p = 0  # order of the autoregressive system
#
# # RUN ALGORITHM ON PATCHES
# cnm = cnmf.CNMF(n_processes,
#                 k=500,
#                 gSig=[2, 2, 1],
#                 merge_thresh=0.95,
#                 p=0,
#                 dview=dview,
#                 rf=96,
#                 stride=8,
#                 normalize_init=False,
#                 only_init_patch=True,)
# cnm.params.set('spatial', {'se': np.ones((2, 2,1), dtype=np.uint8)})
# cnm = cnm.fit(images)
#
# print(('Number of components:' + str(cnm.estimates.A.shape[-1])))
# #%%
# fr = 30                             # imaging rate in frames per second
# decay_time = 0.4                    # length of a typical transient in seconds
#
# # motion correction parameters
# strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
# overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
# max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
# max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
# pw_rigid = True             # flag for performing non-rigid motion correction
#
# # parameters for source extraction and deconvolution
# p = 1                       # order of the autoregressive system
# gnb = 2                     # number of global background components
# merge_thr = 0.85            # merging threshold, max correlation allowed
# rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
# stride_cnmf = 6             # amount of overlap between the patches in pixels
# K = 4                       # number of components per patch
# gSig = [4, 4]               # expected half size of neurons in pixels
# method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
# ssub = 1                    # spatial subsampling during initialization
# tsub = 1                    # temporal subsampling during intialization
#
# # parameters for component evaluation
# min_SNR = 2.0               # signal to noise ratio for accepting a component
# rval_thr = 0.85              # space correlation threshold for accepting a component
# cnn_thr = 0.99              # threshold for CNN based classifier
# cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
# #%%
#
# opts_grps = ['data',
# 'spatial_params', 'temporal_params', 'init_params', 'preprocess_params',
# 'patch_params', 'online', 'quality', 'merging', 'motion', 'ring_CNN']
#
# opts_dict = {'fnames': fnames,
#             'fr': 1.0,
#             'decay_time': 10,
#              #
#             'strides': strides,
#             'overlaps': overlaps,
#             'max_shifts': max_shifts,
#             'max_deviation_rigid': max_deviation_rigid,
#             'pw_rigid': pw_rigid,
#             'p': p,
#             'nb': gnb,
#             'rf': rf,
#             'K': K,
#             'stride': stride_cnmf,
#             'method_init': method_init,
#             'rolling_sum': True,
#             'only_init': True,
#             'ssub': ssub,
#             'tsub': tsub,
#             'merge_thr': merge_thr,
#             'min_SNR': min_SNR,
#             'rval_thr': rval_thr,
#             'use_cnn': True,
#             'min_cnn_thr': cnn_thr,
#             'cnn_lowest': cnn_lowest}
#
# opts = params.CNMFParams(params_dict=opts_dict)
#
#
