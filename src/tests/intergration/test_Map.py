import sys
import traceback
import unittest

from util.datamodel import *
from util.analysis import *
from util.preparation import *
from util.processing import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import seaborn as sns
import util.ScientificColourMaps5 as SCMaps

# File paths  and files needed for tests
dir_tests = str(Path.cwd().parent)
dir_integration = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
marker1, marker2, marker3, marker4, marker5 = [25, 20, 10, 5, 3]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_vm, color_ca, color_ec = ['#FF8888', '#8888FF', '#842926']
cmap_vm, cmap_ca = [plt.get_cmap('YlOrRd').reversed(), plt.get_cmap('GnBu').reversed()]
color_snr = '#C3E7B0'
colors_times = {'Start': '#C07B60',
                'Activation': '#842926',
                'Peak': '#4B133D',
                'Downstroke': '#436894',
                'End': '#94B0C3',
                'Baseline': '#C5C3C2'}  # SCMapsViko, circular colormap
# Colormap for SNR maps
cmap_snr = SCMaps.tokyo
cmap_snr.set_bad(color=gray_light, alpha=0)

# Colormaps for analysis maps
cmap_activation = SCMaps.lajolla
cmap_activation.set_bad(color=gray_light, alpha=0)
ACT_MAX_PIG_WHOLE = 250
# ACT_RAW_MAX_PIG_WHOLE = 250
ACT_MAX_PIG_LV = 100
cmap_duration = SCMaps.oslo.reversed()
cmap_duration.set_bad(color=gray_light, alpha=0)
DUR_MIN_PIG_WHOLE = 60
DUR_MIN_PIG_LV = 120
DUR_MAX_PIG = DUR_MAX

cmap_ec = SCMaps.bilbao
cmap_ec.set_bad(color=gray_light, alpha=0)


# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # yellow -> orange -> purple
# colors_times = [SCMaps.vik0, ..., ..., ..., ..., ...]  # redish -> purple -> blue
# colors_times = ['#003EDC', '#FB2595', '#FF6172', '#FFD067', '#FFF92', '#000000']  # redish -> purple -> blue?
# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # redish -> purple -> blue?


def image_colorbar(axis, image):
    # add colorbar (lower right of frame)
    ax_ins_img = inset_axes(axis, width="5%", height="100%", loc=5,
                            bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=axis.transAxes,
                            borderpad=0)
    cb_img = plt.colorbar(image, cax=ax_ins_img, orientation="vertical")
    cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
    cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
    cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
    cb_img.ax.tick_params(labelsize=fontsize3)


def add_map_colorbar_stats(axis, img, map_data, map_range, bins=100, unit='unit', stat_color=gray_heavy):
    ax_ins_cbar = inset_axes(axis, width="5%", height="100%", loc='center left',
                             bbox_to_anchor=(1.3, 0, 1, 1), bbox_transform=axis.transAxes,
                             borderpad=0)
    cbar = plt.colorbar(img, cax=ax_ins_cbar, orientation="vertical")
    cbar.ax.set_xlabel(unit, fontsize=fontsize3)
    # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
    if map_range[1] <= 100:
        maj_tics = map_range[1] // 2
    else:
        maj_tics = 50
    min_tics = int(maj_tics / 5)
    cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(maj_tics))
    cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(min_tics))
    cbar.ax.tick_params(labelsize=fontsize3)

    # Histogram/Violin plot of analysis values (along left side of colorbar)
    # use 2 histograms to (quickly) create a "violin" plot
    map_flat = map_data.flat
    ax_map_hist_l = inset_axes(axis, width="25%", height="100%", loc='center left',
                               bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=axis.transAxes,
                               borderpad=0)
    # ax_map_hist_r = inset_axes(ax_map, width="200%", height="100%", loc='center left',
    #                            bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
    #                            borderpad=0)
    [s.set_visible(False) for s in ax_map_hist_l.spines.values()]
    # [s.set_visible(False) for s in ax_map_hist_r.spines.values()]
    # ax_map_hist_l.hist(map_flat, bins=bins,
    #                    histtype='stepfilled', orientation='horizontal', color='gray')
    # ax_map_hist_r.hist(map_flat, bins=bins,
    #                    histtype='stepfilled', orientation='horizontal', color='gray')
    # ax_map_hist_l.invert_xaxis()

    # print('Generating swarmplot ... ')
    sns.violinplot(ax=ax_map_hist_l, data=map_flat, cut=0,
                   color=stat_color, inner="stick")
    for ax in [ax_map_hist_l]:
        ax.set_ylim([map_range[0], map_range[1]])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])


# noinspection PyTypeChecker
class TestMapAnalysisPig(unittest.TestCase):
    def setUp(self):
        # Load data to test with
        fps = 500.0
        self.exp_name = '2-wk'
        # exp_name = 'MEHP: Baseline'
        file_XY = (770, 1048)
        self.scale_px_cm = 101.4362
        # file_path_local = '/20200228-piga/baseline/04-450_Vm(941-1190).tif'
        # strict = (2, 5)
        # file_path_local = '/20200228-piga/baseline/04-450_Ca(941-1190).tif'
        # strict = (4, 7)
        # file_frames = (941, 1190)
        # file_path_local = '/20200228-piga/baseline/05-400_Vm(1031-1280).tif'
        # strict = (2, 5)
        file_path_local = '/20200228-piga/baseline/05-400_Ca(1031-1280).tif'
        strict = (4, 7)
        file_frames = (1031, 1280)

        # file_path_local = '/20200228-piga/baseline/06-350_Vm(941-1190).tif'
        # strict = (2, 5)
        # file_path_local = '/20200228-piga/baseline/06-350_Ca(941-1190).tif'
        # strict = (4, 7)
        # file_frames = (941, 1190)
        # file_X0Y0_Vm = (190, 200)
        # file_X0Y0_Ca = (1140, 200)

        # exp_name = 'MEHP: 60 uM'
        # file_X0Y0 = (1060, 160)
        # file_path = dir_tests + '/data/20200228-piga/MEHP 60 uM/09-400_Ca(871-1120).tif'
        # file_name_pig = '2020/02/28 piga-09, Vm, ' + exp_name + ' PCL 400ms'
        # file_frames = (871, 1120)

        # self.exp_name = '6-wk'
        # # file_path_local = '/20190322-pigb/01-350_Ca_transient.tif'
        # # self.scale_px_cm = 67.0000
        #
        # file_XY = (900, 1200)
        # self.scale_px_cm = 158.7823
        # # file_path_local = '/20190517-piga/02-400_Vm(501-700).tif'
        # # strict = (1, 3)
        # # file_X0Y0_Vm = (950, 150)
        # # file_path_local = '/20190517-piga/02-400_Ca(501-700).tif'
        # file_path_local = '/20190517-piga/03-350_Ca(251-500).tif'
        # strict = (5, 8)
        # file_X0Y0_Ca = (70, 150)
        # # file_XY = (900, 1440)
        # # self.scale_px_cm = 143.3298
        # # file_path = dir_tests + '/data/20191004-piga/01-350_Vm(880-1060).tif'
        # # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Vm, PCL 350ms'
        # # file_X0Y0_Vm = (1010, 250)
        # # file_frames = (880, 1060)
        # # file_path = dir_tests + '/data/20191004-piga/02-300_Ca(480-660).tif'
        # # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Ca, PCL 300ms'
        # # file_X0Y0_Ca = (0, 40)
        # # file_frames = (480, 660)

        # #
        study_name = file_path_local.split(sep='/')[1]  # e.g. 20200828-pigd
        self.file_path = dir_tests + '/data/' + file_path_local
        self.file_name = file_path_local.split(sep='/')[-1].split(sep='.')[0].split(sep='(')[0]  # e.g. 08-228_Vm
        self.test_name = '{}, {}, {}'.format(self.exp_name, study_name, self.file_name)

        self.scale_cm_px = 1 / self.scale_px_cm
        self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_path)
        self.stack_real_frame = self.stack_real_full[0, :, :]  # frame from stack

        stack_out = self.stack_real_full.copy()

        # *** Preparation ***
        # # Crop (to size of _ X _) based on original resolution
        # new_width, new_height = int(500), int(500)
        # new_top_left_x, new_top_left_y = 160, 0  # coordinates of top left corner
        #
        # d_x, d_y = int(-new_top_left_x), \
        #            int(-new_top_left_y)  # cropped from top left
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        #
        # d_x, d_y = stack_out.shape[2] - new_width, stack_out.shape[1] - new_height  # cropped from bottom right
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        # print('Cropped stack from {}x{} to {}x{}...'
        #       .format(self.stack_real_frame.shape[0], self.stack_real_frame.shape[1],
        #               stack_out.shape[1], stack_out.shape[2]))

        # Reduce
        self.reduction = 7  # set to XX (to min ~200 X 200 pixels)
        self.scale_px_cm = int(self.scale_px_cm / self.reduction)
        self.scale_cm_px = self.scale_cm_px * self.reduction
        stack_out = reduce_stack(stack_out, self.reduction)
        print('\nDONE Reducing stack')
        # Mask
        print('Generating Masking ...')
        self.frame_bright = np.zeros_like(stack_out[0])  # use brightest frame to generate mask
        frame_bright_idx = 0
        for idx, frame in enumerate(stack_out):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(self.frame_bright):
                frame_bright_idx = idx
                self.frame_bright = frame.copy()
        print('Brightest frame: {}'.format(frame_bright_idx))
        mask_type = 'Random_walk'
        _, self.mask_out, _ = mask_generate(self.frame_bright, mask_type, strict)
        print('\nDONE generating Mask')
        # stack_out = mask_apply(stack_out, self.mask_out)

        self.prep = 'Reduced x{}, Mask'.format(self.reduction)
        # #

        # *** Processing ***
        # # Drift
        # map_shape = stack_out.shape[1:]
        # poly_ideal_order = 3
        # print('Removing drift from stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     signal_filtered, drift = filter_drift(stack_out[:, iy, ix], drift_order=poly_ideal_order)
        #     stack_out[:, iy, ix] = signal_filtered
        # print('\nDONE Removing drift from stack')

        # Normalize
        if 'Vm' in file_path_local:
            print('\t * Hello Voltage!')
            print('Inverting stack with {} frames, size W {} X H {} ...'
                  .format(stack_out.shape[0], stack_out.shape[2], stack_out.shape[1]))
            stack_out = invert_stack(stack_out)
            print('\nDONE Inverting stack')
        elif 'Ca' in file_path_local:
            print('\t * Hello Calcium!')
        else:
            print('\t * Hello Signal?')
        print('Normalizing stack with {} frames, size W {} X H {} ...'
              .format(stack_out.shape[0], stack_out.shape[2], stack_out.shape[1]))
        stack_out = normalize_stack(stack_out)
        print('\nDONE Normalized stack')

        # Filter
        # spatial
        kernel_cm = 0.3  # set to X.X cm (~0.3)
        self.kernel = floor(kernel_cm / self.scale_cm_px)
        if self.kernel > 3:
            if self.kernel % 2 == 0:
                self.kernel = self.kernel - 1
        self.kernel_marker_size = self.kernel
        print('Filtering (spatial) with kernel: {} px ...'.format(self.kernel))
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            frame_filtered = filter_spatial(frame, kernel=self.kernel)
            # f_filtered = np.ma.masked_where(f_filtered == 0, f_filtered)
            stack_out[idx, :, :] = frame_filtered
        print('\nDONE Filtering (spatial) stack')
        # Re-apply mask to avoid smudged edges
        stack_out = mask_apply(stack_out, self.mask_out)
        # # temporal
        # freq_cutoff = 100.0
        # map_shape = stack_out.shape[1:]
        # print('Filtering (temporal) stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     filter_order = 'auto'
        #     signal_filtered = filter_temporal(stack_out[:, iy, ix], fps, freq_cutoff=freq_cutoff,
        #                                       filter_order=filter_order)
        #     stack_out[:, iy, ix] = signal_filtered
        # print('\nDONE Filtering (temporal) stack')

        self.process = 'Gaussian: {} cm ({} px)'.format(kernel_cm, self.kernel)
        # self.process = 'Gaussian: {} px, LP {} Hz'.format(freq_cutoff, self.kernel)
        # #

        # Generate array of timestamps
        fpms = fps / 1000
        t_final = floor(stack_out.shape[0] / fpms)

        time_real = np.linspace(start=0, stop=t_final, num=stack_out.shape[0])

        # self.x_lv_apex, self.y_lv_apex = (int(stack.shape[2] * (1/2)), int(stack.shape[1] * (2/3)))     # LV Apex
        x_lv_base, y_lv_base = (int(stack_out.shape[2] * (2 / 3)), int(stack_out.shape[1] * (1 / 2)))  # LV Base

        self.signal_x, self.signal_y = x_lv_base, y_lv_base

        left, bottom, width, height = (self.signal_x - (self.kernel_marker_size / 2),
                                       self.signal_y - (self.kernel_marker_size / 2),
                                       self.kernel_marker_size, self.kernel_marker_size)
        self.roi_lbwh = (left, bottom, width, height)

        self.time_pig, self.stack_pig = time_real, stack_out

    def test_map_snr_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig

        # Plot a frame from the stack, the map of that stack, and signals of interest
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        ax_frame.set_title('{}\n{}, {}'
                           .format(self.test_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns
        gs_min = gs_signals[0].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_xy = gs_signals[1].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_max = gs_signals[2].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column

        ax_signal_min = fig_map.add_subplot(gs_min[0])
        ax_signal_xy = fig_map.add_subplot(gs_xy[0])
        ax_signal_max = fig_map.add_subplot(gs_max[0])
        # Derivatives
        ax_df_min = fig_map.add_subplot(gs_min[1], sharex=ax_signal_min)
        ax_df_xy = fig_map.add_subplot(gs_xy[1], sharex=ax_signal_xy)
        ax_df_max = fig_map.add_subplot(gs_max[1], sharex=ax_signal_max)

        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(50))
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Common between all derivative axes
        for ax in [ax_df_min, ax_df_xy, ax_df_max]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('F (arb. u.)')
        ax_df_min.set_ylabel('dF/dt')
        ax_df_xy.set_xlabel('Time (ms)')

        # Calculate the SNR map, returns timestamps
        analysis_map = map_snr(stack)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max + 5.1, -1))
        # map_max_display = ceil(round(map_max, 2))
        map_max_display = SNR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('SNR Map\n{} - {} ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        # mask_frame = np.ma.masked_where(self.frame_bright == 0, self.frame_bright)
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.frame_bright.min(), vmax=self.frame_bright.max())
        img_frame = ax_frame.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        heart_scale = [self.scale_px_cm, self.scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Signal traces and location on frame
        # plot trace with the chosen ROI pixel
        signal_roi = stack[:, self.signal_x, self.signal_y]
        ax_signal_xy.plot(stack_time, signal_roi, color=gray_heavy, linestyle='None', marker='+')
        ax_frame.plot(self.signal_x, self.signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=color_snr,
                      markersize=1, transform=ax_frame.transData)
        roi_sym = plt.Rectangle((self.roi_lbwh[0], self.roi_lbwh[1]), self.roi_lbwh[2], self.roi_lbwh[3],
                                fc='None', ec=color_snr, transform=ax_frame.transData)
        ax_frame.add_patch(roi_sym)

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=color_snr, markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=color_snr, markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, ax_df, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max],
                                  [ax_df_min, ax_df_xy, ax_df_max],
                                  [signal_min, signal_roi, signal_max]):
            print('* Plotting a signal *')
            # Signal of interest (and underlying calculations)
            # ax_data.set_xticklabels([])
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(sig)
            snr_display = round(snr, 2)
            peak_peak_display = round(peak_peak, 2)
            sd_noise_display = round(sd_noise, 2)
            try:
                ax.plot(stack_time[ir_noise], sig[ir_noise],
                        "x", color=color_snr, markersize=marker4)
                ax.plot(stack_time[i_peak], sig[i_peak],
                        "x", color=colors_times['Peak'], markersize=marker3)

                # df/dt
                time_df = np.linspace(stack_time[0], stack_time[-2], len(sig - 1) * SPLINE_FIDELITY)
                x_df, df_signal = spline_deriv(sig)

                ax_df.plot(time_df, df_signal, color=gray_med,
                           linestyle='-', label='dF/dt')

                # Text: Conditions
                ax.text(0.7, 0.9, '{}/{}'.format(peak_peak_display, sd_noise_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
                ax.text(0.7, 0.8, '{} SNR'.format(snr_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            except Exception:
                traceback.print_exc(file=sys.stdout)

        # SNR Map
        ax_map.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        cmap_norm_snr = colors.Normalize(vmin=map_min_display,
                                         vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_snr, cmap=cmap_snr)
        map_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                        label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                        fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(map_scale_bar)

        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='SNR', bins=hist_bins, stat_color=color_snr)

        fig_map.savefig(dir_integration + '/results/MapPig_SNR_{}_{}.png'.
                        format(self.exp_name, self.file_name))
        fig_map.show()

    def test_map_activation_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig
        d_stack_time = stack_time[-1] - stack_time[-2]

        # Plot a frame from the stack, the map of that stack, and signals of interest
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        ax_frame.set_title('{}\n{}, {}'
                           .format(self.test_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns
        gs_min = gs_signals[0].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_xy = gs_signals[1].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_max = gs_signals[2].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column

        ax_signal_min = fig_map.add_subplot(gs_min[0])
        ax_signal_xy = fig_map.add_subplot(gs_xy[0])
        ax_signal_max = fig_map.add_subplot(gs_max[0])
        # Derivatives
        ax_df_min = fig_map.add_subplot(gs_min[1], sharex=ax_signal_min)
        ax_df_xy = fig_map.add_subplot(gs_xy[1], sharex=ax_signal_xy)
        ax_df_max = fig_map.add_subplot(gs_max[1], sharex=ax_signal_max)

        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(50))
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Common between all derivative axes
        for ax in [ax_df_min, ax_df_xy, ax_df_max]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('F (arb. u.)')
        ax_df_min.set_ylabel('dF/dt')
        ax_df_xy.set_xlabel('Time (ms)')

        # Calculate the Activation map
        analysis_map = map_tran_analysis(stack, find_tran_act, stack_time)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        map_max_display = ACT_MAX_PIG_WHOLE
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Activation Map\n{} - {} ms ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        # mask_frame = np.ma.masked_where(self.frame_bright == 0, self.frame_bright)
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.frame_bright.min(), vmax=self.frame_bright.max())
        img_frame = ax_frame.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        heart_scale = [self.scale_px_cm, self.scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Signal traces and location on frame
        # plot trace with the chosen ROI pixel
        signal_roi = stack[:, self.signal_x, self.signal_y]
        ax_signal_xy.plot(stack_time, signal_roi, color=gray_heavy, linestyle='None', marker='+')
        ax_frame.plot(self.signal_x, self.signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=colors_times['Activation'],
                      markersize=1, transform=ax_frame.transData)
        roi_sym = plt.Rectangle((self.roi_lbwh[0], self.roi_lbwh[1]), self.roi_lbwh[2], self.roi_lbwh[3],
                                fc='None', ec=colors_times['Activation'], transform=ax_frame.transData)
        ax_frame.add_patch(roi_sym)

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Activation'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        map_min_raw = find_tran_act(signal_min) * d_stack_time  # calculate unadjusted activation time (ms)
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Activation'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, ax_df, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max],
                                  [ax_df_min, ax_df_xy, ax_df_max],
                                  [signal_min, signal_roi, signal_max]):
            print('* Plotting a signal *')
            # Signal of interest (and underlying calculations)
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(sig)
            snr_display = round(snr, 2)
            i_peak = find_tran_peak(sig)  # max of signal, Peak
            i_activation = find_tran_act(sig)  # 1st df max, Activation
            try:
                ax.plot(stack_time[ir_noise], sig[ir_noise],
                        "x", color=color_snr, markersize=marker4)
                ax.plot(stack_time[i_peak], sig[i_peak],
                        "x", color=colors_times['Peak'], markersize=marker3)
                ax.plot(stack_time[i_activation], sig[i_activation],
                        "x", color=colors_times['Activation'], markersize=marker3)

                # df/dt
                time_df = np.linspace(stack_time[0], stack_time[-2], len(sig - 1) * SPLINE_FIDELITY)
                x_df, df_signal = spline_deriv(sig)
                ax_df.plot(time_df, df_signal, color=gray_med,
                           linestyle='-', label='dF/dt')

                noise_rms = rms_bounds[0]
                # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
                # duration = calc_tran_duration(sig, percent=dur_percent)
                # ax.plot(stack_time[i_activation + duration], sig[i_activation + duration],
                #         "x", color=colors_times['Downstroke'], markersize=marker3)

                ax.axhline(y=noise_rms,
                           # xmin=stack_time[i_activation],
                           # xmax=stack_time[i_activation + duration],
                           color=gray_light, linestyle='-.',
                           label='Baseline')
                # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
                #                color=colors_times['Downstroke'], label='Downstroke')
                ax.vlines(x=stack_time[i_activation],
                          ymin=np.min(sig),
                          ymax=sig[i_activation],
                          color=colors_times['Activation'], linestyle=':',
                          label='Activation')
                # ax.vlines(x=stack_time[i_peak],
                #           ymin=sig[i_activation + duration],
                #           ymax=sig[i_peak],
                #           color=colors_times['Peak'], linestyle=':',
                #           label='{}% of Peak-Peak'.format(dur_percent))
                # ax.vlines(x=stack_time[i_peak],
                #           ymin=noise_rms,
                #           ymax=sig[i_activation + duration],
                #           color=gray_light, linestyle=':',
                #           label='{}% of Peak-Peak'.format(dur_percent))

                # ax.hlines(y=sig[i_activation + duration],
                #           xmin=stack_time[i_activation],
                #           xmax=stack_time[i_activation + duration],
                #           color=colors_times['Downstroke'], linewidth=2,
                #           label='Downstroke')
                # Symbol on the masked frame showing this signal's source
                # ax.plot(min_sym, transform=ax.transAxes)

                # Text: Activation and SNR
                # When mapping activation, align time with the "first" aka lowest activation time
                activation_ms = (i_activation * d_stack_time) - map_min_raw
                activation_ms_display = round(activation_ms, 2)
                ax.text(0.73, 0.9, '{} ms'.format(activation_ms_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
                ax.text(0.73, 0.8, '{} SNR'.format(snr_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            except Exception:
                traceback.print_exc(file=sys.stdout)

        # Activation Map
        ax_map.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_activation = colors.Normalize(vmin=map_min_display,
                                                vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        map_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                        label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                        fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(map_scale_bar)

        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='ms', bins=hist_bins, stat_color=colors_times['Activation'])

        fig_map.savefig(dir_integration + '/results/MapPig_Activation_{}_{}.png'.
                        format(self.exp_name, self.file_name))
        fig_map.show()

    def test_map_duration_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig
        d_stack_time = stack_time[-1] - stack_time[-2]

        # Chosen duration %
        dur_percent = 80

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n{}, {}'
                           .format(self.test_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns
        gs_min = gs_signals[0].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_xy = gs_signals[1].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_max = gs_signals[2].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column

        ax_signal_min = fig_map.add_subplot(gs_min[0])
        ax_signal_xy = fig_map.add_subplot(gs_xy[0])
        ax_signal_max = fig_map.add_subplot(gs_max[0])
        # Derivatives
        ax_df_min = fig_map.add_subplot(gs_min[1], sharex=ax_signal_min)
        ax_df_xy = fig_map.add_subplot(gs_xy[1], sharex=ax_signal_xy)
        ax_df_max = fig_map.add_subplot(gs_max[1], sharex=ax_signal_max)

        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(50))
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Common between all derivative axes
        for ax in [ax_df_min, ax_df_xy, ax_df_max]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('F (arb. u.)')
        ax_df_min.set_ylabel('dF/dt')
        ax_df_xy.set_xlabel('Time (ms)')

        # Calculate the Duration map
        analysis_map = map_tran_analysis(stack, calc_tran_duration, stack_time, percent=dur_percent)
        # Exclusion criteria for pigs
        # analysis_map[analysis_map < DUR_MIN_PIG_WHOLE] = np.nan

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        map_max_display = DUR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Duration-{}% Map\n{} - {} ms ({} pixels)'
                         .format(dur_percent,
                                 round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        # mask_frame = np.ma.masked_where(self.frame_bright == 0, self.frame_bright)
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.frame_bright.min(), vmax=self.frame_bright.max())
        img_frame = ax_frame.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        heart_scale = [self.scale_px_cm, self.scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Signal traces and location on frame
        # plot trace with the chosen ROI pixel
        signal_roi = stack[:, self.signal_x, self.signal_y]
        ax_signal_xy.plot(stack_time, signal_roi, color=gray_heavy, linestyle='None', marker='+')
        ax_frame.plot(self.signal_x, self.signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=colors_times['Downstroke'],
                      markersize=1, transform=ax_frame.transData)
        roi_sym = plt.Rectangle((self.roi_lbwh[0], self.roi_lbwh[1]), self.roi_lbwh[2], self.roi_lbwh[3],
                                fc='None', ec=colors_times['Downstroke'], transform=ax_frame.transData)
        ax_frame.add_patch(roi_sym)

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, ax_df, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max],
                                  [ax_df_min, ax_df_xy, ax_df_max],
                                  [signal_min, signal_roi, signal_max]):
            print('* Plotting a signal *')
            # for ax, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_roi, signal_max]):
            # Signal of interest (and underlying calculations)
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(sig)
            snr_display = round(snr, 2)
            i_peak = find_tran_peak(sig)  # max of signal, Peak
            i_activation = find_tran_act(sig)  # 1st df max, Activation
            try:
                ax.plot(stack_time[ir_noise], sig[ir_noise],
                        "x", color=color_snr, markersize=marker4)
                ax.plot(stack_time[i_peak], sig[i_peak],
                        "x", color=colors_times['Peak'], markersize=marker3)
                ax.plot(stack_time[i_activation], sig[i_activation],
                        "x", color=colors_times['Activation'], markersize=marker3)

                # df/dt
                time_df = np.linspace(stack_time[0], stack_time[-2], len(sig - 1) * SPLINE_FIDELITY)
                x_df, df_signal = spline_deriv(sig)
                ax_df.plot(time_df, df_signal, color=gray_med,
                           linestyle='-', label='dF/dt')

                noise_rms = rms_bounds[0]
                # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
                duration = calc_tran_duration(sig, percent=dur_percent)
                ax.plot(stack_time[i_activation + duration], sig[i_activation + duration],
                        "x", color=colors_times['Downstroke'], markersize=marker3)

                ax.axhline(y=noise_rms,
                           # xmin=stack_time[i_activation],
                           # xmax=stack_time[i_activation + duration],
                           color=gray_light, linestyle='-.',
                           label='Baseline')
                # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
                #                color=colors_times['Downstroke'], label='Downstroke')
                ax.vlines(x=stack_time[i_activation],
                          ymin=sig[i_activation + duration],
                          ymax=sig[i_activation],
                          color=colors_times['Activation'], linestyle=':',
                          label='Activation')
                ax.vlines(x=stack_time[i_peak],
                          ymin=noise_rms,
                          ymax=sig[i_peak],
                          color=colors_times['Peak'], linestyle=':',
                          label='{}% of Peak-Peak'.format(dur_percent))
                ax.vlines(x=stack_time[i_peak],
                          ymin=noise_rms,
                          ymax=sig[i_activation + duration],
                          color=gray_light, linestyle=':',
                          label='{}% of Peak-Peak'.format(dur_percent))

                # color_duration = map_img.cmap(duration)
                ax.hlines(y=sig[i_activation + duration],
                          xmin=stack_time[i_activation],
                          xmax=stack_time[i_activation + duration],
                          color=colors_times['Downstroke'], linewidth=2,
                          label='Downstroke')
                # Symbol on the masked frame showing this signal's source
                # ax.plot(min_sym, transform=ax.transAxes)

                # Text: Duration and SNR
                duration_ms = duration * d_stack_time
                duration_ms_display = round(duration_ms, 2)
                ax.text(0.73, 0.9, '{} ms'.format(duration_ms_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
                ax.text(0.73, 0.8, '{} SNR'.format(snr_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            except Exception:
                # exctype, exvalue, traceback = sys.exc_info()
                # print("* Failed to calculate/plot a signal:\n\t" + str(exctype) + ' : ' + str(exvalue) +
                #       '\n\t\t' + str(traceback))
                traceback.print_exc(file=sys.stdout)

        # Duration Map
        ax_map.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_duration = colors.Normalize(vmin=map_min_display,
                                              vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_duration, cmap=cmap_duration)
        map_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                        label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                        fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(map_scale_bar)

        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='ms', stat_color=colors_times['Downstroke'])

        fig_map.savefig(dir_integration + '/results/MapPig_Duration_{}_{}.png'.
                        format(self.exp_name, self.file_name))
        fig_map.show()


class TestMapCouplingPig(unittest.TestCase):
    def setUp(self):
        # Load data to test with
        fps = 500.0
        self.exp_name = '2-wk old'
        file_path_local = '/20200228-piga/baseline/05-400(1031-1280).tif'
        self.scale_px_cm = 101.4362
        self.scale_cm_px = 1 / self.scale_px_cm
        shape_out = (770, 1048)
        x0y0_Vm = (190, 200)
        x0y0_Ca = (1140, 200)
        strict_vm = (2, 5)
        strict_ca = (4, 7)

        self.file_path = dir_tests + '/data/' + file_path_local
        study_name = file_path_local.split(sep='/')[1]  # e.g. 20200828-pigd
        self.file_name = file_path_local.split(sep='/')[-1].split(sep='(')[0]  # e.g. 08-228_Vm
        self.test_name = '{}, {}, {}'.format(self.exp_name, study_name, self.file_name)
        # #

        print('Opening stack ...')
        stack1, meta1 = open_stack(source=self.file_path)
        print('DONE Opening stack\n')
        self.stack1_min, self.stack1_max = stack1.min(), stack1.max()
        self.frame1 = stack1[0, :, :]

        # Generate array of timestamps
        fpms = fps / 1000
        t_final = floor(stack1.shape[0] / fpms)
        self.time_real = np.linspace(start=0, stop=t_final, num=stack1.shape[0])

        print('Splitting stacks ...')
        # Crop twice for each: once from the bottom/right, once for top/left
        shape_in = (self.frame1.shape[1], self.frame1.shape[0])

        crop_vm_1 = (shape_in[0] - (shape_out[0] + x0y0_Vm[0]), shape_in[1] - (shape_out[1] + x0y0_Vm[1]))
        crop_vm_2 = (-x0y0_Vm[0], -x0y0_Vm[1])

        crop_ca_1 = (shape_in[0] - (shape_out[0] + x0y0_Ca[0]), shape_in[1] - (shape_out[1] + x0y0_Ca[1]))
        crop_ca_2 = (-x0y0_Ca[0], -x0y0_Ca[1])

        stack_vm_dirty = crop_stack(stack1, d_x=crop_vm_1[0], d_y=crop_vm_1[1])
        self.stack_vm = crop_stack(stack_vm_dirty, d_x=crop_vm_2[0], d_y=crop_vm_2[1])
        # self.stack_vm = self.stack_vm[file_frames[0]:file_frames[1], :, :]

        stack_ca_dirty = crop_stack(stack1, d_x=crop_ca_1[0], d_y=crop_ca_1[1])
        self.stack_ca = crop_stack(stack_ca_dirty, d_x=crop_ca_2[0], d_y=crop_ca_2[1])
        # self.stack_ca = self.stack_ca[file_frames[0]:file_frames[1], :, :]
        print('DONE Splitting stacks\n')

        print('Aligning stacks ...')
        # Align the Voltage stack to the Calcium stack
        self.stack_vm = align_stacks(self.stack_ca, self.stack_vm)
        print('DONE Aligning stacks\n')
        self.frame_vm = self.stack_vm[0, :, :]
        self.frame_ca = self.stack_ca[0, :, :]

        # Prep and Process both stacks
        self.reduction = 9  # set to XX (to min ~200 X 200 pixels)
        self.reduction_factor = 1 / self.reduction
        self.scale_px_cm = int(self.scale_px_cm / self.reduction)
        self.scale_cm_px = self.scale_cm_px * self.reduction
        # mask_type = 'Random_walk'
        self.prep = 'Reduced x{}, Mask'.format(self.reduction)

        self.kernel_cm = 0.5  # set to X.X cm (~0.3)
        self.kernel = floor(self.kernel_cm / self.scale_cm_px)
        self.process = 'Gaussian: {} cm ({} px)'.format(self.kernel_cm, self.kernel)
        self.stack_processed_vm, self.stack_processed_ca = None, None
        self.mask_vm, self.mask_ca = None, None

        # *** Voltage ***
        # *** Preparation ***
        # Reduce
        test_frame = rescale(self.stack_vm[0], self.reduction_factor, multichannel=False)
        print('Reducing stack from W {} X H {} ... to size W {} X H {} ...'
              .format(self.stack_vm.shape[2], self.stack_vm.shape[1], test_frame.shape[1], test_frame.shape[0]))
        stack_reduced_shape = (self.stack_vm.shape[0], test_frame.shape[0], test_frame.shape[1])
        stack_reduced = np.empty(stack_reduced_shape, dtype=self.stack_vm.dtype)  # empty stack
        for idx, frame in enumerate(self.stack_vm):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, self.stack_vm.shape[0]), end='', flush=True)
            #     f_filtered = filter_spatial(frame, kernel=self.kernel)
            frame_reduced = img_as_uint(rescale(frame, self.reduction_factor, anti_aliasing=True, multichannel=False))
            stack_reduced[idx, :, :] = frame_reduced
        self.stack_processed_vm = stack_reduced
        print('\nDONE Reducing stack')
        # Mask
        print('Generating Masking ...')
        self.frame_bright = np.zeros_like(self.stack_processed_vm[0])  # use brightest frame to generate mask
        frame_bright_idx = 0
        for idx, frame in enumerate(self.stack_processed_vm):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(self.frame_bright):
                frame_bright_idx = idx
                self.frame_bright = frame.copy()
        print('Brightest frame: {}'.format(frame_bright_idx))
        mask_type = 'Random_walk'
        _, mask, _ = mask_generate(self.frame_bright, mask_type, strict_vm)
        print('\nDONE generating Mask')
        # *** Processing ***
        # Invert
        print('\t * Hello Voltage!')
        print('Inverting stack with {} frames, size W {} X H {} ...'
              .format(self.stack_processed_vm.shape[0],
                      self.stack_processed_vm.shape[2], self.stack_processed_vm.shape[1]))
        self.stack_processed_vm = invert_stack(self.stack_processed_vm)
        print('\nDONE Inverting stack')
        # Normalize
        self.stack_processed_vm = normalize_stack(self.stack_processed_vm)
        # Filter
        # spatial
        self.kernel = floor(self.kernel_cm / self.scale_cm_px)
        if self.kernel > 3:
            if self.kernel % 2 == 0:
                self.kernel = self.kernel - 1
        self.kernel_marker_size = self.kernel
        print('Filtering (spatial) with kernel: {} px ...'.format(self.kernel))
        for idx, frame in enumerate(self.stack_processed_vm):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, self.stack_processed_vm.shape[0]), end='', flush=True)
            frame_filtered = filter_spatial(frame, kernel=self.kernel)
            # f_filtered = np.ma.masked_where(f_filtered == 0, f_filtered)
            self.stack_processed_vm[idx, :, :] = frame_filtered
        print('\nDONE Filtering (spatial) stack')
        # Re-apply mask to avoid smudged edges
        self.stack_processed_vm = mask_apply(self.stack_processed_vm, mask)

        # *** Calcium ***
        # *** Preparation ***
        # Reduce
        test_frame = rescale(self.stack_ca[0], self.reduction_factor, multichannel=False)
        print('Reducing stack from W {} X H {} ... to size W {} X H {} ...'
              .format(self.stack_ca.shape[2], self.stack_ca.shape[1], test_frame.shape[1], test_frame.shape[0]))
        stack_reduced_shape = (self.stack_ca.shape[0], test_frame.shape[0], test_frame.shape[1])
        stack_reduced = np.empty(stack_reduced_shape, dtype=self.stack_ca.dtype)  # empty stack
        for idx, frame in enumerate(self.stack_ca):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, self.stack_ca.shape[0]), end='', flush=True)
            #     f_filtered = filter_spatial(frame, kernel=self.kernel)
            frame_reduced = img_as_uint(rescale(frame, self.reduction_factor, anti_aliasing=True, multichannel=False))
            stack_reduced[idx, :, :] = frame_reduced
        self.stack_processed_ca = stack_reduced
        print('\nDONE Reducing stack')
        # Mask
        print('Generating Masking ...')
        self.frame_bright = np.zeros_like(self.stack_processed_ca[0])  # use brightest frame to generate mask
        frame_bright_idx = 0
        for idx, frame in enumerate(self.stack_processed_ca):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(self.frame_bright):
                frame_bright_idx = idx
                self.frame_bright = frame.copy()
        print('Brightest frame: {}'.format(frame_bright_idx))
        mask_type = 'Random_walk'
        _, mask, _ = mask_generate(self.frame_bright, mask_type, strict_ca)
        print('\nDONE generating Mask')
        # *** Processing ***
        # Normalize
        self.stack_processed_ca = normalize_stack(self.stack_processed_ca)
        # Filter
        # spatial
        kernel_cm = 0.3  # set to X.X cm (~0.3)
        self.kernel = floor(kernel_cm / self.scale_cm_px)
        if self.kernel > 3:
            if self.kernel % 2 == 0:
                self.kernel = self.kernel - 1
        self.kernel_marker_size = self.kernel
        print('Filtering (spatial) with kernel: {} px ...'.format(self.kernel))
        for idx, frame in enumerate(self.stack_processed_ca):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, self.stack_processed_ca.shape[0]), end='', flush=True)
            frame_filtered = filter_spatial(frame, kernel=self.kernel)
            # f_filtered = np.ma.masked_where(f_filtered == 0, f_filtered)
            self.stack_processed_ca[idx, :, :] = frame_filtered
        print('\nDONE Filtering (spatial) stack')
        # Re-apply mask to avoid smudged edges
        self.stack_processed_ca = mask_apply(self.stack_processed_ca, mask)

        # Regions of Interest: pixels from the LV Apex and LV Base
        self.lv_apex_x, self.lv_apex_y = (int(self.stack_processed_vm.shape[2] * (1 / 2)),
                                          int(self.stack_processed_vm.shape[1] * (2 / 3)))  # LV Apex
        self.apex_lbwh = (self.lv_apex_x - (self.kernel_marker_size / 2),
                          self.lv_apex_y - (self.kernel_marker_size / 2),
                          self.kernel_marker_size, self.kernel_marker_size)

        self.lv_base_x, self.lv_base_y = (int(self.stack_processed_vm.shape[2] * (2 / 3)),
                                          int(self.stack_processed_vm.shape[1] * (1 / 2)))  # LV Base
        self.base_lbwh = (self.lv_base_x - (self.kernel_marker_size / 2),
                          self.lv_base_y - (self.kernel_marker_size / 2),
                          self.kernel_marker_size, self.kernel_marker_size)

    def test_map_coupling(self):
        # Make sure EC coupling map looks correct
        stack_time = self.time_real
        stack_processed_vm = self.stack_processed_vm
        # frame_processed_vm = stack_processed_vm[0, :, :]
        stack_processed_ca = self.stack_processed_ca
        # frame_processed_ca = stack_processed_ca[0, :, :]

        # Plot a frame from the stack, the map of that stack, and signals of interest
        fig_coupling = plt.figure(figsize=(16, 8))  # _ x _ inch page
        gs0 = fig_coupling.add_gridspec(1, 3, width_ratios=[0.25, 0.25, 0.5])  # 1 row, 3 columns
        gs_frames = gs0[0].subgridspec(2, 1)
        ax_vm = fig_coupling.add_subplot(gs_frames[0])
        ax_ca = fig_coupling.add_subplot(gs_frames[1])
        ax_vm.set_title('{}, Vm\n{}, {}'
                        .format(self.test_name, self.prep, self.process))
        ax_ca.set_title('{}, Ca\n{}, {}'
                        .format(self.test_name, self.prep, self.process))

        gs_maps = gs0[1].subgridspec(2, 1)
        ax_map_vm = fig_coupling.add_subplot(gs_maps[0])
        ax_map_ca = fig_coupling.add_subplot(gs_maps[1])

        gs_coupling = gs0[2].subgridspec(2, 1)
        ax_map = fig_coupling.add_subplot(gs_coupling[0])
        gs_signals = gs_coupling[1].subgridspec(1, 2)
        ax_signals = fig_coupling.add_subplot(gs_signals[0])
        ax_signal_rises = fig_coupling.add_subplot(gs_signals[1])

        # Common between axes
        for ax in [ax_vm, ax_ca, ax_map_vm, ax_map_ca, ax_map]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])

        for ax in [ax_signals, ax_signal_rises]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xlabel('Time (ms)')

        # ax_signals.set_ylabel('Amplitude\n(Normalized)')
        ax_signals.xaxis.set_major_locator(plticker.MultipleLocator(100))
        ax_signals.xaxis.set_minor_locator(plticker.MultipleLocator(50))
        ax_signal_rises.xaxis.set_major_locator(plticker.MultipleLocator(50))
        ax_signal_rises.xaxis.set_minor_locator(plticker.MultipleLocator(10))

        # Vm frames
        cmap_vm_norm = colors.Normalize(vmin=self.stack_vm.min(), vmax=self.stack_vm.max())
        img_vm = ax_vm.imshow(self.frame_vm, norm=cmap_vm_norm, cmap=cmap_vm)
        image_colorbar(ax_vm, img_vm)
        ax_map_vm.imshow(self.frame_vm, norm=cmap_vm_norm, cmap=cmap_vm)
        # image_colorbar(ax_map_vm, img_vm)
        # Ca frames
        cmap_ca_norm = colors.Normalize(vmin=self.stack_ca.min(), vmax=self.stack_ca.max())
        img_ca = ax_ca.imshow(self.frame_ca, norm=cmap_ca_norm, cmap=cmap_ca)
        image_colorbar(ax_ca, img_ca)
        ax_map_ca.imshow(self.frame_ca, norm=cmap_ca_norm, cmap=cmap_ca)
        # image_colorbar(ax_map_ca, img_ca)

        # Activation maps
        map_act_vm = map_tran_analysis(stack_processed_vm, find_tran_act, stack_time,
                                       raw_data=True)
        map_act_ca = map_tran_analysis(stack_processed_ca, find_tran_act, stack_time,
                                       raw_data=True)

        act_map_min = np.nanmin([map_act_vm, map_act_ca])
        act_map_max = np.nanmax([map_act_vm, map_act_ca])
        map_n_vm = np.count_nonzero(~np.isnan(map_act_vm))
        map_n_ca = np.count_nonzero(~np.isnan(map_act_ca))
        map_act_min_display = (act_map_min // 50) * 50
        map_act_max_display = map_act_min_display + ACT_MAX_PIG_WHOLE
        print('Activation Map MIN value: ', act_map_min)
        print('Activation Map MAX value: ', act_map_max)
        # hist_bins = map_act_max_display
        map_range = (map_act_min_display, map_act_max_display)
        cmap_norm_activation = colors.Normalize(vmin=map_act_min_display,
                                                vmax=map_act_max_display)
        ax_map_vm.set_title('Vm Activation Map\n{} - {} ms ({} pixels)'
                            .format(round(np.nanmin(map_act_vm), 2), round(np.nanmax(map_act_vm), 2), map_n_vm))
        ax_map_ca.set_title('Ca Activation Map\n{} - {} ms ({} pixels)'
                            .format(round(np.nanmin(map_act_ca), 2), round(np.nanmax(map_act_ca), 2), map_n_ca))
        # Vm activation map
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.frame_bright.min(), vmax=self.frame_bright.max())
        ax_map_vm.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        img_act_vm = ax_map_vm.imshow(map_act_vm, norm=cmap_norm_activation, cmap=cmap_activation)
        add_map_colorbar_stats(ax_map_vm, img_act_vm, map_act_vm, map_range,
                               unit='ms', stat_color=colors_times['Activation'])
        # Ca activation map
        ax_map_ca.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        img_act_ca = ax_map_ca.imshow(map_act_ca, norm=cmap_norm_activation, cmap=cmap_activation)
        add_map_colorbar_stats(ax_map_ca, img_act_ca, map_act_ca, map_range,
                               unit='ms', stat_color=colors_times['Activation'])

        # Coupling map
        map_ec = map_coupling(map_vm=map_act_vm, map_ca=map_act_ca)
        map_ec_min = np.nanmin(map_ec)
        map_ec_max = np.nanmax(map_ec)
        map_n_ec = np.count_nonzero(~np.isnan(map_ec))
        map_coup_min_display = 0
        map_coup_max_display = EC_MAX
        print('Coupling Map MIN value: ', map_ec_min)
        print('Coupling Map MAX value: ', map_ec_max)
        map_range = (map_coup_min_display, map_coup_max_display)
        cmap_norm_coupling = colors.Normalize(vmin=map_coup_min_display, vmax=map_coup_max_display)
        ax_map.set_title('EC Coupling Map\n{} - {} ms ({} pixels)'
                         .format(round(map_ec_min, 2), round(map_ec_max, 2), map_n_ec))
        ax_map.imshow(self.frame_bright, norm=cmap_norm_frame, cmap=cmap_frame)
        img_ec_map = ax_map.imshow(map_ec, norm=cmap_norm_coupling, cmap=cmap_ec)
        # Add colorbar (right of map)
        add_map_colorbar_stats(ax_map, img_ec_map, map_ec, map_range,
                               unit='ms', stat_color=color_snr)

        # Signal traces and locations on frame
        # Apex
        signal_apex_vm = stack_processed_vm[:, self.lv_apex_y, self.lv_apex_x]
        ax_signals.plot(stack_time, signal_apex_vm, color=color_vm, linestyle='dashed')
        signal_apex_ca = stack_processed_ca[:, self.lv_apex_y, self.lv_apex_x]
        ax_signals.plot(stack_time, signal_apex_ca, color=color_ca, linestyle='dashed')
        ax_map.plot(self.lv_apex_x, self.lv_apex_y, marker='s', markerfacecolor='None',
                    markeredgecolor=color_ec,
                    markersize=1, transform=ax_map.transData)
        roi_sym_coupling = plt.Rectangle((self.apex_lbwh[0], self.apex_lbwh[1]), self.apex_lbwh[2], self.apex_lbwh[3],
                                         linestyle='dashed', fc='None', ec=color_ec, transform=ax_map.transData)
        ax_map.add_patch(roi_sym_coupling)
        # Base
        signal_base_vm = stack_processed_vm[:, self.lv_base_y, self.lv_base_x]
        ax_signals.plot(stack_time, signal_base_vm, color=color_vm)
        signal_base_ca = stack_processed_ca[:, self.lv_base_y, self.lv_base_y]
        ax_signals.plot(stack_time, signal_base_ca, color=color_ca)
        ax_map.plot(self.lv_base_x, self.lv_base_y, marker='s', markerfacecolor='None',
                    markeredgecolor=color_ec,
                    markersize=1, transform=ax_map.transData)
        roi_sym_coupling = plt.Rectangle((self.base_lbwh[0], self.base_lbwh[1]), self.base_lbwh[2], self.base_lbwh[3],
                                         fc='None', ec=color_ec, transform=ax_map.transData)
        ax_map.add_patch(roi_sym_coupling)

        # Signal traces zoomed in to show rises, activations, and EC coupling variation
        ax_signal_rises.plot(stack_time, signal_apex_vm, color=color_vm, linestyle='dashed')
        ax_signal_rises.plot(stack_time, signal_apex_ca, color=color_ca, linestyle='dashed')
        ax_signal_rises.plot(stack_time, signal_base_vm, color=color_vm)
        ax_signal_rises.plot(stack_time, signal_base_ca, color=color_ca)
        ax_signal_rises.set_xlim(75, 170)
        # Activation timepoints
        for sig in [signal_apex_vm, signal_apex_ca, signal_base_vm, signal_base_ca]:
            i_activation = find_tran_act(sig)  # 1st df max, Activation
            ax_signal_rises.plot(stack_time[i_activation],
                                 sig[i_activation],
                                 ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                                 color=colors_times['Activation'], label='Activation')

        # EC Coupling timespans
        i_act_apex_ca = find_tran_act(signal_apex_ca)
        ec_coupling_apex = calc_coupling(signal_vm=signal_apex_vm, signal_ca=signal_apex_ca)
        ax_signal_rises.hlines(y=signal_apex_ca[i_act_apex_ca],
                               xmin=stack_time[i_act_apex_ca - ec_coupling_apex],
                               xmax=stack_time[i_act_apex_ca],
                               color=cmap_ec(ec_coupling_apex), linewidth=marker5)
        i_act_base_ca = find_tran_act(signal_base_ca)
        ec_coupling_base = calc_coupling(signal_vm=signal_base_vm, signal_ca=signal_base_ca)
        ax_signal_rises.hlines(y=signal_base_ca[i_act_base_ca],
                               xmin=stack_time[i_act_base_ca - ec_coupling_apex],
                               xmax=stack_time[i_act_base_ca],
                               color=cmap_ec(ec_coupling_base), linewidth=marker5)

        fig_coupling.savefig(dir_integration + '/results/MapPig_Coupling_{}_{}.png'.
                             format(self.exp_name, self.file_name))
        fig_coupling.show()


if __name__ == '__main__':
    unittest.main()
