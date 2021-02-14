import unittest

from matplotlib.patches import Circle, ConnectionPatch

from util.datamodel import *
from util.preparation import *
from util.processing import *
from util.analysis import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps
import util.vikO as SCMapsViko

# File paths needed for tests
dir_tests = str(Path.cwd().parent)
dir_unit = str(Path.cwd())
dir_integration = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
marker1, marker2, marker3, marker4, marker5 = [25, 20, 10, 5, 3]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_clear = (0, 0, 0, 0)
color_vm, color_ca = ['#FF9999', '#9999FF']
colors_times = {'Start': '#C07B60',
                'Activation': '#842926',
                'Peak': '#4B133D',
                'Downstroke': '#436894',
                'End': '#94B0C3',
                'Baseline': gray_med}  # SCMapsViko, circular colormap
# 'Baseline': '#C5C3C2'}  # SCMapsViko, circular colormap
# colors_times = {'Start': '#FFD649',
#                 'Activation': '#FFA253',
#                 'Peak': '#F6756B',
#                 'Downstroke': '#CB587F',
#                 'End': '#8E4B84',
#                 'Baseline': '#4C4076'}  # yellow -> orange -> purple
# colors_times = [SCMapsViko[0], SCMapsViko[0], SCMapsViko[0],
#                 SCMapsViko[0], SCMapsViko[0], SCMapsViko[0]]  # redish -> purple -> blue

TRAN_MAX = 200
# Colormap and normalization range for activation maps
ACT_MAX = 150
cmap_activation = SCMaps.lajolla
cmap_activation.set_bad(color=gray_light, alpha=0)
# Colormap and normalization range for Duration maps
cmap_duration = SCMaps.oslo.reversed()
cmap_duration.set_bad(color=gray_light, alpha=0)


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.tick_params(axis='x', which='minor', length=3, bottom=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True)
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def plot_map():
    # Setup a figure to show a frame and a map generated from that frame
    fig = plt.figure(figsize=(8, 4))  # _ x _ inch page
    axis_img = fig.add_subplot(121)
    axis_map = fig.add_subplot(122)
    # Common between the two
    for ax in [axis_img, axis_map]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

    return fig, axis_img, axis_map


class TestStart(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = 40

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_start, signal_in=True)
        self.assertRaises(TypeError, find_tran_start, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_start : np.int64
        i_start = find_tran_start(self.signal)
        self.assertIsInstance(i_start, np.int64)  # index of start start
        self.assertAlmostEqual(self.time[i_start], self.signal_t0, delta=5)  # start time

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and any analysis points
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Point: Start')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')

        ax_points.plot(self.time, self.signal, color=gray_heavy,
                       linestyle='-', marker='x', label='Vm (Model)')
        ax_points.set_xlim(0, self.zoom_t)

        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.set_ylabel('dF/dt, d2F/dt2')  # we already handled the x-label with ax1

        # df/dt
        # d2f/dt2
        # ax_dfs.set_ylim([-df_max, df_max])

        # Start
        i_start = find_tran_start(self.signal)  # 1st df2 max, Start
        ax_points.axvline(self.time[i_start], color=colors_times['Start'],
                          label='Start')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.show()


class TestActivation(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 500
        self.signal_noise = 5

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = 40

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_act, signal_in=True)
        self.assertRaises(TypeError, find_tran_act, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_activation : np.int64
        i_activation = find_tran_act(self.signal)
        self.assertIsInstance(i_activation, np.int64)  # index of activation time
        self.assertGreater(self.time[i_activation], self.signal_t0)  # activation time
        # self.assertLess(self.time[i_activation], self.signal_t0, delta=5)  # activation time

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and any analysis points
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Point: Activation')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (frame #)')

        ax_points.plot(self.time, self.signal, color=gray_heavy,
                       linestyle='-', marker='x', label='Vm (Model)')
        ax_points.set_xlim(0, self.zoom_t)

        # show the workings of this analysis
        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.set_ylabel('dF/dt')

        # Activation
        i_act = find_tran_act(self.signal)  # 1st df max, Activation
        ax_points.axvline(self.time[i_act], color=colors_times['Activation'],
                          label='Activation')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.show()


class TestPeak(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_noise = 5  # as a % of the signal amplitude

        time_vm, signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                              noise=self.signal_noise)
        time_ca, signal_ca = model_transients(model_type='Ca', t=self.signal_t, t0=self.signal_t0,
                                              noise=self.signal_noise)
        self.time, self.signal = time_vm, invert_signal(signal_vm)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_peak : np.int64
        i_peak = find_tran_peak(self.signal)
        self.assertIsInstance(i_peak, np.int64)  # index of peak time


# class TestDownstroke(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : uint16 or float
#         self.assertRaises(TypeError, find_tran_downstroke, signal_in=True)
#         self.assertRaises(TypeError, find_tran_downstroke, signal_in=signal_bad_type)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  i_downstroke : int
#         i_downstroke = find_tran_downstroke(self.signal_ca)
#         self.assertIsInstance(i_downstroke, np.int64)
#
#         self.assertAlmostEqual(i_downstroke, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT
#
#
# class TestEnd(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : uint16 or float
#         self.assertRaises(TypeError, find_tran_end, signal_in=True)
#         self.assertRaises(TypeError, find_tran_end, signal_in=signal_bad_type)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  i_end : int
#         i_end = find_tran_end(self.signal_ca)
#         self.assertIsInstance(i_end, np.int64)


class TestAnalysisPoints(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = [0, 55]

        # # Import real data
        # real trace
        file_signal_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_30x30-LV-198x324.csv'
        file_name_pig = '2019/03/22 pigb-01-Ca'
        self.file_name, file_signal = file_name_pig, file_signal_pig
        self.signal_cl = '350'
        self.time_real, self.signal_real = open_signal(source=file_signal, fps=404)

    def test_detections(self):
        # Build a figure to plot the signal, it's derivatives, and the analysis points
        # General layout
        fig_analysis = plt.figure(figsize=(6, 6))  # _ x _ inch page
        gs0 = fig_analysis.add_gridspec(3, 1, height_ratios=[0.5, 0.25, 0.25])  # 3 row, 1 columns

        # Data plot
        ax_data = fig_analysis.add_subplot(gs0[0])
        # ax_data.set_title('Analysis Points')
        ax_data.set_ylabel('Fluorescence (arb. u.)')
        # Derivatives
        ax_df1 = fig_analysis.add_subplot(gs0[1])
        ax_df1.set_ylabel('dF/dt')
        ax_df2 = fig_analysis.add_subplot(gs0[2])
        ax_df2.set_xlabel('Time (ms)')
        ax_df2.set_ylabel('d2F/dt2')
        points_lw = 3
        # Set axes z orders so connecting lines are shows
        ax_data.set_zorder(3)
        # ax_data.set_xlim(self.zoom_t)

        ax_df1.set_zorder(2)
        # df_zoom_t = [SPLINE_FIDELITY * x for x in self.zoom_t]
        # ax_df1.set_xlim(df_zoom_t)

        ax_df2.set_zorder(1)
        # df2_zoom_t = [SPLINE_FIDELITY * x for x in df_zoom_t]
        # ax_df2.set_xlim(df2_zoom_t)

        for ax in [ax_data, ax_df1]:
            ax.set_xticklabels([])

        # Common between all axes
        for ax in [ax_data, ax_df1, ax_df2]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlim(self.zoom_t)
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Plot signals and points
        ax_data.plot(self.time, self.signal, color=gray_heavy,
                     linestyle='-', marker='.', markersize=points_lw, label='Vm (Model)')
        # spline
        x_sp, spline = spline_signal(self.signal)
        # ax_data.plot(x_sp, spline(x_sp), color=color_filtered,
        #              linestyle='-', label='LSQ spline')

        # df/dt
        time_df = np.linspace(self.time[0], self.time[-2], len(self.time - 1) * SPLINE_FIDELITY)
        xx_d1f, d1f_smooth = spline_deriv(self.signal)
        ax_df1.plot(time_df, d1f_smooth, color=gray_med,
                    linestyle='--')
        ax_df1.hlines(0, xmin=0, xmax=time_df[-1], color=gray_light, linewidth=1)
        # d1f_max = round(abs(max(d1f_smooth, key=abs)) + 0.5, -1)
        # ax_df1.set_ylim([-d2f_smooth.max(), d2f_smooth.max()])

        # d2f/dt2
        time_df2 = np.linspace(time_df[0], time_df[-2], len(time_df - 1) * SPLINE_FIDELITY)
        xx_d2f, d2f_smooth = spline_deriv(d1f_smooth)
        ax_df2.plot(time_df2, d2f_smooth, color=gray_med,
                    linestyle=':')
        ax_df2.hlines(0, xmin=0, xmax=time_df2[-1], color=gray_light, linewidth=1)
        # d2f_max = round(abs(max(d2f_smooth, key=abs)) + 0.5, -1)
        # ax_df2.set_ylim([-d2f_smooth.max(), d2f_smooth.max()])

        # Start
        i_start = find_tran_start(self.signal)  # 1st df2 max, Start
        ax_data.vlines(x=self.time[i_start],
                       ymin=0,
                       ymax=self.signal[i_start],
                       color=colors_times['Start'], linestyle=':', linewidth=points_lw,
                       label='Start')
        ax_data.plot(self.time[i_start],
                     self.signal[i_start],
                     "x", color=colors_times['Start'], markersize=marker3)
        ax_df2.plot(time_df2[i_start * SPLINE_FIDELITY * SPLINE_FIDELITY],
                    d2f_smooth[i_start * SPLINE_FIDELITY * SPLINE_FIDELITY],
                    "x", color=colors_times['Start'], markersize=marker3)

        # Activation
        i_activation = find_tran_act(self.signal)  # 1st df max, Activation
        ax_data.vlines(x=self.time[i_activation],
                       ymin=0,
                       ymax=self.signal[i_activation],
                       color=colors_times['Activation'], linestyle=':', linewidth=points_lw,
                       label='Activation')
        ax_data.plot(self.time[i_activation],
                     self.signal[i_activation],
                     "x", color=colors_times['Activation'], markersize=marker3)
        ax_df1.plot(time_df[i_activation * SPLINE_FIDELITY],
                    d1f_smooth[i_activation * SPLINE_FIDELITY],
                    "x", color=colors_times['Activation'], markersize=marker3)

        # Peak
        i_peak = find_tran_peak(self.signal)  # max of signal, Peak
        peak_frac = (self.signal[i_peak] - ax_data.get_ylim()[0]) / \
                    (ax_data.get_ylim()[1] - ax_data.get_ylim()[0])
        ax_data.vlines(x=self.time[i_peak],
                       ymin=0,
                       ymax=self.signal[i_peak],
                       color=colors_times['Peak'], linestyle=':', linewidth=points_lw,
                       label='Peak')
        ax_data.plot(self.time[i_peak],
                     self.signal[i_peak],
                     "x", color=colors_times['Peak'], markersize=marker3)

        # Downstroke
        i_downstroke = find_tran_downstroke(self.signal)  # df min, Downstroke
        ax_data.vlines(x=self.time[i_downstroke],
                       ymin=0,
                       ymax=self.signal[i_downstroke],
                       color=colors_times['Downstroke'], linestyle=':', linewidth=points_lw,
                       label='Downstroke')
        ax_data.plot(self.time[i_downstroke],
                     self.signal[i_downstroke],
                     "x", color=colors_times['Downstroke'], markersize=marker3)
        ax_df1.plot(time_df[i_downstroke * SPLINE_FIDELITY],
                    d1f_smooth[i_downstroke * SPLINE_FIDELITY],
                    "x", color=colors_times['Downstroke'], markersize=marker3)

        # End
        i_end = find_tran_end(self.signal)  # 2st df2 max, End
        ax_data.vlines(x=self.time[i_end],
                       ymin=0,
                       ymax=self.signal[i_end],
                       color=colors_times['End'], linestyle=':', linewidth=points_lw,
                       label='End')
        ax_data.plot(self.time[i_end],
                     self.signal[i_end],
                     "x", color=colors_times['End'], markersize=marker3)
        ax_df2.plot(time_df2[i_end * SPLINE_FIDELITY * SPLINE_FIDELITY],
                    d2f_smooth[i_end * SPLINE_FIDELITY * SPLINE_FIDELITY],
                    "x", color=colors_times['End'], markersize=marker3)

        ax_data.legend(loc='upper right', ncol=1, prop={'size': fontsize3}, numpoints=1, frameon=True)

        # fig_analysis.savefig(dir_unit + '/results/analysis_AnalysisPoints.png')
        fig_analysis.show()

    def test_features(self):
        # Build a figure to plot an isolated transient and major analysis points/durations
        transients, cycle = isolate_transients(self.signal_real)
        transient_signal = transients[1]
        transient_time = self.time_real[0:len(transient_signal)]

        # General layout
        fig_transient = plt.figure(figsize=(12, 8))  # _ x _ inch page
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)
        gs0 = fig_transient.add_gridspec(3, 1, height_ratios=[0.15, 0.75, 0.1], hspace=0.1)  # 3 rows, 1 column
        ax_signal = fig_transient.add_subplot(gs0[0])
        ax_features = fig_transient.add_subplot(gs0[1])
        ax_blank = fig_transient.add_subplot(gs0[2])
        ax_signal.set_zorder(2)
        ax_features.set_zorder(1)

        for ax in [ax_signal, ax_features, ax_blank]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(50))

        ax_signal.set_ylabel('Amplitude\n(arb. u.)')
        ax_signal.set_xticklabels([])

        ax_features.set_ylabel('Amplitude\n(Normalized)')
        ax_features.yaxis.set_major_locator(plticker.MultipleLocator(1))
        ax_features.set_xticks([])
        ax_features.set_xticklabels([])

        ax_blank.set_xlabel('Time (ms)')
        ax_blank.set_yticks([])
        ax_blank.set_yticklabels([])

        ax_signal.plot(self.time_real, self.signal_real, color=gray_light,
                       linestyle='None', marker='.', label='Ca pixel data')
        ax_features.plot(transient_time, transient_signal, color=gray_med, marker='.')
        ax_blank.set_xlim(ax_features.get_xlim())
        ax_blank.set_ylim([0, 1])

        # Transient Features
        # Start
        i_start = find_tran_start(transient_signal)  # 1st df2 max, Start
        ax_features.plot(transient_time[i_start],
                         transient_signal[i_start],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Start'], label='Start')
        # Activation timepoint
        i_activation = find_tran_act(transient_signal)  # 1st df max, Activation
        ax_features.plot(transient_time[i_activation],
                         transient_signal[i_activation],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Activation'], label='Activation')
        # Peak
        i_peak = find_tran_peak(transient_signal)  # max of signal, Peak
        ax_features.plot(transient_time[i_peak],
                         transient_signal[i_peak],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Peak'], label='Peak')
        # Depolarization timespan
        depolarization = i_peak - i_start
        ax_blank.hlines(y=0.9,
                        xmin=transient_time[i_start],
                        xmax=transient_time[i_peak],
                        color=colors_times['Activation'], linewidth=marker5)
        # Duration timepoint- 90%
        duration = calc_tran_duration(transient_signal, percent=90)
        i_duration = i_activation + duration
        ax_features.plot(transient_time[i_duration],
                         transient_signal[i_duration],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Downstroke'], label='Duration 90%')
        # Duration timespan
        ax_blank.hlines(y=0.75,
                        xmin=transient_time[i_activation],
                        xmax=transient_time[i_activation + duration],
                        color=colors_times['Downstroke'], linewidth=marker5)
        # DI (Diastolic Interval) timespan
        diastolic = cycle - duration
        i_diastolic = i_duration + diastolic
        ax_blank.hlines(y=0.6,
                        xmin=transient_time[i_duration],
                        xmax=transient_time[i_diastolic],
                        color=colors_times['End'], linewidth=marker5)
        # # Downstroke
        # i_downstroke = find_tran_downstroke(self.signal)  # df min, Downstroke
        # ax_features.plot(self.time_real[i_downstroke],
        #              self.signal_real[i_downstroke],
        #              "x", color=colors_times['Downstroke'], markersize=marker3)
        # # End
        # i_end = find_tran_end(transient_signal)  # 2st df2 max, End
        # ax_features.plot(transient_time[i_end],
        #                  transient_signal[i_end],
        #                  "x", color=colors_times['End'], markersize=marker3)

        ax_features.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_transient.savefig(dir_unit + '/results/analysis_TransientFeatures.png')
        fig_transient.show()


class TestCoupling(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3
        self.model_coupling = 10

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.signal_vm = invert_signal(self.signal_vm)
        self.time_ca, self.signal_ca = model_transients(model_type='Ca', t=self.signal_t,
                                                        t0=self.signal_t0 + self.model_coupling,
                                                        fps=self.signal_fps, noise=self.signal_noise)

        self.zoom_t = [0, 55]

    def test_plot(self):
        # Build a figure to plot the original signals and the analysis
        # transients, cycle = isolate_transients(self.signal_real)
        # transient_signal = transients[1]
        # transient_time = self.time_real[0:len(transient_signal)]
        self.signal_vm = normalize_signal(self.signal_vm)
        self.signal_ca = normalize_signal(self.signal_ca)

        # General layout
        fig_transient = plt.figure(figsize=(12, 8))  # _ x _ inch page
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)
        gs0 = fig_transient.add_gridspec(3, 1, height_ratios=[0.15, 0.15, 0.7], hspace=0.1)  # 3 rows, 1 column
        ax_vm = fig_transient.add_subplot(gs0[0])
        ax_ca = fig_transient.add_subplot(gs0[1])
        ax_coupling = fig_transient.add_subplot(gs0[2])
        ax_coupling.set_zorder(1)
        ax_ca.set_zorder(2)
        ax_vm.set_zorder(3)

        for ax in [ax_vm, ax_ca, ax_coupling]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(50))

        ax_vm.set_ylabel('Amplitude\n(arb. u.)')
        ax_vm.set_xticklabels([])
        ax_ca.set_ylabel('Amplitude\n(arb. u.)')
        ax_ca.set_xticklabels([])

        ax_coupling.set_ylabel('Amplitude\n(Normalized)')
        ax_coupling.yaxis.set_major_locator(plticker.MultipleLocator(1))
        ax_coupling.set_xticks([])
        ax_coupling.set_xticklabels([])
        ax_coupling.set_xlabel('Time (ms)')
        ax_coupling.set_xlim(self.zoom_t)

        ax_vm.plot(self.time_vm, self.signal_vm, color=color_vm,
                   linestyle='None', marker='.', label='Vm pixel data')
        ax_ca.plot(self.time_ca, self.signal_ca, color=color_ca,
                   linestyle='None', marker='.', label='Ca pixel data')
        ax_coupling.plot(self.time_vm, self.signal_vm, color=color_vm, marker='.')
        ax_coupling.plot(self.time_ca, self.signal_ca, color=color_ca, marker='.')

        # Activation times
        i_activation_vm = find_tran_act(self.signal_vm)
        ax_coupling.plot(self.time_vm[i_activation_vm],
                         self.signal_vm[i_activation_vm],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Activation'], label='Activation')
        i_activation_ca = find_tran_act(self.signal_ca)
        ax_coupling.plot(self.time_ca[i_activation_ca],
                         self.signal_ca[i_activation_ca],
                         ".", fillstyle='none', markersize=marker2, markeredgewidth=marker5,
                         color=colors_times['Activation'], label='Activation')

        # Coupling timespan
        coupling = calc_coupling(self.signal_vm, self.signal_ca)  # difference between activation times
        ax_coupling.hlines(y=self.signal_ca[i_activation_ca],
                           xmin=self.time_ca[i_activation_ca - coupling],
                           xmax=self.time_ca[i_activation_ca],
                           color=colors_times['Activation'], linewidth=marker5)

        ax_coupling.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_transient.savefig(dir_unit + '/results/analysis_Coupling.png')
        fig_transient.show()


# class TestDuration(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         #  percent : int
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=True, percent=True)
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=signal_bad_type, percent='500')
#         self.assertRaises(TypeError, calc_tran_duration, signal_in='word', percent=3j + 7)
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         # percent : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         percent_bad_value = -1
#         self.assertRaises(ValueError, calc_tran_duration, signal_in=signal_bad_value, percent=percent_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  duration : int
#         duration = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(duration, np.int32)
#
#         self.assertAlmostEqual(duration, self.signal_t0)


# class TestTau(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_tran_di, signal_in=True)
#         self.assertRaises(TypeError, calc_tran_di, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_tran_di, signal_in='word')
#         self.assertRaises(TypeError, calc_tran_di, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)
#
#         # should not be applied to signal data containing at least one transient
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  di : float
#         di = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(di, np.float32)
#
#         self.assertAlmostEqual(di, self.signal_t0)


# class TestDI(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=True)
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_tran_tau, signal_in='word')
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  tau : float
#         tau = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(tau, np.float32)
#
#         self.assertAlmostEqual(tau, self.signal_t0)

#  class TestDFreq(unittest.TestCase):

class TestEnsemble(unittest.TestCase):
    def setUp(self):
        # # Create data to test with
        # self.signal_t = 2000
        # self.signal_t0 = 50
        # self.signal_f0 = 1000
        # self.signal_famp = 200
        # self.signal_fps = 500
        # self.signal_num = 'full'
        # self.signal_cl = 150
        # self.signal_noise = 5  # as a % of the signal amplitude
        # # trace
        # self.time_vm, self.signal_vm = \
        #     model_transients(t=self.signal_t, t0=self.signal_t0, fps=self.signal_fps,
        #                      f0=self.signal_f0, famp=self.signal_famp, noise=self.signal_noise,
        #                      num=self.signal_num, cl=self.signal_cl)
        # self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)
        # # stack
        # self.stack_size = (20, 20)
        # self.signal_t = 1000
        # self.d_noise = 20  # as a % of the signal amplitude
        # self.snr_range = (int((self.signal_famp / (self.signal_noise + self.d_noise))),
        #                   int(self.signal_famp / self.signal_noise))
        # self.time_stack, self.stack = \
        #     model_stack_heart(model_type='Ca', size=self.stack_size, d_noise=self.d_noise,
        #                       t=self.signal_t, t0=self.signal_t0, fps=self.signal_fps,
        #                       famp=self.signal_famp, noise=self.signal_noise,
        #                       num=self.signal_num, cl=self.signal_cl)
        # # # Crop the model stack (remove bottom half)
        # # d_x = int(self.stack_size[0] / 2)
        # # d_y = int(self.stack_size[0] / 3)
        # # self.stack = crop_stack(self.stack, d_x=d_x, d_y=d_y)

        # # import model data
        # self.file = 'ModelStackHeart_ca'
        # extension = '.tif'
        # fps = 500
        # # file_stack_rat = dir_tests + '/data/20200109-rata/baseline/' + self.file + extension
        # file_stack_model = dir_unit + '/results/' + self.file + extension
        # print('Opening stack ...')
        # self.stack_import, self.stack_meta = open_stack(source=file_stack_model)
        # print('DONE Opening stack\n')

        # Import real data
        # real trace
        file_signal_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_30x30-LV-198x324.csv'
        file_name_pig = '2019/03/22 pigb-01-Ca'
        self.file_name, file_signal = file_name_pig, file_signal_pig
        self.signal_cl = '350'
        self.time, self.signal = open_signal(source=file_signal, fps=404)
        # # real stack
        # extension = '.tif'
        # fps = 500
        # # self.file = '02-350_ca'
        # # file_stack_rat = dir_tests + '/data/20200109-rata/baseline/' + self.file + extension
        # self.file = '02-300_Ca'
        # file_stack_pig = dir_tests + '/data/20191004-piga/' + self.file + extension
        # self.file_path = file_stack_pig
        # print('Opening stack ...')
        # self.stack_real, self.stack_meta = open_stack(source=self.file_path)
        # print('DONE Opening stack\n')
        # self.stack_frame = self.stack_real[0, :, :]  # frame from stack
        # # Generate array of timestamps
        # FRAMES = self.stack_real.shape[0]
        # FPMS = fps / 1000
        # FINAL_T = floor(FRAMES / FPMS)
        # self.time_real = np.linspace(start=0, stop=FINAL_T, num=FRAMES)
        #
        # # real stack trace
        # self.stack_real_trace_X, self.stack_real_trace_Y = 400, 300
        # self.stack_real_trace = self.stack_real[:, self.stack_real_trace_Y, self.stack_real_trace_X]

    def test_params(self):
        time_bad_type = np.full(100, True)
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # time_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_ensemble, time_in=True, signal_in=self.signal)
        self.assertRaises(TypeError, calc_ensemble, time_in=time_bad_type, signal_in=self.signal)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, calc_ensemble, time_in=self.time, signal_in=True)
        self.assertRaises(TypeError, calc_ensemble, time_in=self.time, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : must have more than one peak
        time_short, signal_shot = model_transients(model_type='Ca')
        self.assertRaises(ValueError, calc_ensemble, time_in=time_short, signal_in=signal_shot)

    def test_results(self):
        # Make sure spatial filter results are correct
        time_out, signal_out, signals, i_peaks, i_acts, est_cycle = calc_ensemble(self.time, self.signal)
        # time_out : ndarray
        self.assertIsInstance(time_out, np.ndarray)  # ensembled signal
        self.assertAlmostEqual(len(time_out), est_cycle * (self.signal_fps / 1000), delta=10)  #

        # signal_out : ndarray
        self.assertIsInstance(signal_out, np.ndarray)  # ensembled signal
        self.assertEqual(len(signal_out), len(signal_out))  #

        # signals : list
        self.assertIsInstance(signals, list)  # ensembled signal
        self.assertEqual(len(signals), self.signal_num)  #

        # i_peaks : ndarray
        self.assertIsInstance(i_peaks, np.ndarray)  # indicies of peaks
        self.assertEqual(len(i_peaks), self.signal_num)

        # i_acts : ndarray
        self.assertIsInstance(i_acts, np.ndarray)  # indicies of activations
        self.assertEqual(len(i_acts), self.signal_num)

        # est_cycle : float
        self.assertIsInstance(est_cycle, float)  # estimated cycle length (ms) of ensemble
        self.assertAlmostEqual(est_cycle, self.signal_cl, delta=5)  #

    def test_trace(self):
        # Make sure ensembled transient looks correct
        ensemble_crop = 'center'
        # ensemble_crop = (50, 150)
        time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
            = calc_ensemble(self.time, self.signal, crop=ensemble_crop)

        # snr_model = round(self.signal_famp / self.signal_noise, 3)
        # last_baselines = find_tran_baselines(signals[-1])

        # fig_snr, ax_snr = plot_test()
        fig_ensemble = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_ensemble.add_gridspec(2, 1, height_ratios=[0.2, 0.8])  # 3 rows, 1 column
        ax_signal = fig_ensemble.add_subplot(gs0[0])
        ax_ensemble = fig_ensemble.add_subplot(gs0[1])
        # ax_ensemble_crop = fig_ensemble.add_subplot(gs0[2])

        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', which='minor', length=3, bottom=True)
        ax_signal.tick_params(axis='x', which='major', length=8, bottom=True)
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)
        signal_markersize = 8

        ax_signal.set_ylabel('arb. u.')
        # ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])
        ax_signal.plot(self.signal, color=gray_light,
                       linestyle='None', marker='+', label='Ca pixel data')
        # ax_signal.plot(self.time[last_baselines], self.signal[],
        #                "x", color=colors_times['Activation'], label='Activations')
        ax_signal.plot(signal_peaks, self.signal[signal_peaks],
                       "+", color=colors_times['Peak'], markersize=signal_markersize, label='Peaks')
        ax_signal.plot(signal_acts, self.signal[signal_acts],
                       ".", color=colors_times['Activation'], markersize=signal_markersize, label='Activations')

        # # Common between the two
        # for ax in [ax_ensemble, ax_ensemble_crop]:
        ax_ensemble.spines['right'].set_visible(False)
        ax_ensemble.spines['top'].set_visible(False)
        ax_ensemble.set_ylabel('Fluorescence (arb. u.)')
        ax_ensemble.set_xlabel('Time (frame #)')

        # Ensembled and original signals
        signal_snrs = []

        for num, sig in enumerate(signals):
            ax_ensemble.plot(sig, color=gray_light, linestyle='-')
        ax_ensemble.plot(signal_ensemble, color=gray_heavy,
                         linestyle='-', marker='+', label='Ensemble signal')

        for num, sig in enumerate(signals):
            # ax_ensemble.plot(signal, color=gray_light, linestyle='-')
            # # Start
            # i_start = find_tran_start(signal)  # 1st df2 max, Start
            # ax_ensemble.plot(time_ensemble[i_start], signal[i_start],
            #                  "x", color=colors_times['Start'], markersize=10)
            # Activation
            i_activation = find_tran_act(sig)  # 1st df max, Activation
            ax_ensemble.plot(i_activation, sig[i_activation],
                             ".", color=colors_times['Activation'], markersize=signal_markersize)
            # Peak
            i_peak = find_tran_peak(sig)  # max of signal, Peak
            ax_ensemble.plot(i_peak, sig[i_peak],
                             "+", color=colors_times['Peak'], markersize=signal_markersize)
            # # Downstroke
            # i_downstroke = find_tran_downstroke(signal)  # df min, Downstroke
            # ax_ensemble.plot(time_ensemble[i_downstroke], signal[i_downstroke],
            #                  ".", color=colors_times['Downstroke'], markersize=10)
            # # End
            # i_end = find_tran_end(signal)  # 2st df2 max, End
            # ax_ensemble.plot(time_ensemble[i_end], signal[i_end],
            #                  "x", color=colors_times['End'], markersize=10)

            snr_results = calculate_snr(sig)
            snr = snr_results[0]
            ir_noise = snr_results[-2]
            signal_snrs.append(snr)
            # ax_ensemble.plot(ir_noise, sig[ir_noise],
            #                  "x", color=gray_med, markersize=signal_markersize / 2)

        # Stats: SNRs
        snr_old = round(np.mean(signal_snrs), 3)
        snr_results = calculate_snr(signal_ensemble)
        snr_new = round(snr_results[0], 3)
        ir_noise_new = snr_results[-2]
        ax_ensemble.plot(ir_noise_new, signal_ensemble[ir_noise_new],
                         ".", color=gray_heavy, markersize=signal_markersize, label='Noise')

        # Ensemble signal points
        ens_signal_markersize = 25
        # # Start
        # i_start = find_tran_start(signal_ensemble)  # 1st df2 max, Start
        # ax_ensemble.plot(time_ensemble[i_start], signal_ensemble[i_start],
        #                  ".", color=colors_times['Start'], markersize=15, label='Start')
        # Activation
        i_activation = find_tran_act(signal_ensemble)  # 1st df max, Activation
        ax_ensemble.plot(i_activation, signal_ensemble[i_activation],
                         ".", color=colors_times['Activation'], fillstyle='none', markersize=ens_signal_markersize,
                         label='Activation')
        # Peak
        i_peak = find_tran_peak(signal_ensemble)  # max of signal, Peak
        ax_ensemble.plot(i_peak, signal_ensemble[i_peak],
                         ".", color=colors_times['Peak'], fillstyle='none', markersize=ens_signal_markersize,
                         label='Peak')
        # # Downstroke
        # i_downstroke = find_tran_downstroke(signal_ensemble)  # df min, Downstroke
        # ax_ensemble.plot(time_ensemble[i_downstroke], signal_ensemble[i_downstroke],
        #                  ".", color=colors_times['Downstroke'], markersize=15, label='Downstroke')
        # # End
        # i_end = find_tran_end(signal_ensemble)  # 2st df2 max, End
        # ax_ensemble.plot(time_ensemble[i_end], signal_ensemble[i_end],
        #                  ".", color=colors_times['End'], markersize=15, label='End')
        ax_ensemble.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        # Text: Conditions
        ax_ensemble.text(0.72, 0.65, 'PCL actual (ms): {}'.format(self.signal_cl),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        ax_ensemble.text(0.72, 0.6, 'File: {}'.format(self.file_name),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        # ax_ensemble.text(0.72, 0.6, 'SNR actual: {}'.format(snr_model),
        #                  color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        # Text: Cycles
        ax_ensemble.text(0.72, 0.5, 'PCL detected (ms): {}'.format(round(np.mean(est_cycle_length), 3)),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        ax_ensemble.text(0.72, 0.45, '# Peaks detected : {}'.format(len(signal_peaks)),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)

        # Text: SNRs
        ax_ensemble.text(0.72, 0.35, 'SNR detected: {}'.format(snr_old),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        ax_ensemble.text(0.72, 0.3, 'SNR ensembled: {}'.format(snr_new),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)

        # # Activation error bar
        # error_act = np.mean(signals_activations).astype(int)
        # ax_ensemble.errorbar(time_ensemble[error_act],
        #                      signal_ensemble[error_act],
        #                      xerr=statistics.stdev(signals_activations), fmt="x",
        #                      color=colors_times['Activation'], lw=3,
        #                      capsize=4, capthick=1.0)

        # fig_ensemble.savefig(dir_unit + '/results/analysis_Ensemble.png')
        fig_ensemble.savefig(dir_unit + '/results/analysis_Ensemble_Pig.png')
        fig_ensemble.show()

    def test_stack(self):
        stack_ens, ensemble_crop, ensemble_yx = calc_ensemble_stack(self.time_stack, self.stack)

        # Make sure filtered stack signals looks correct
        height, width = self.stack.shape[1], self.stack.shape[2]
        signal_x, signal_y = (int(width / 2), int(height / 2))
        # signal_x, signal_y = ensemble_yx[1], ensemble_yx[0]
        signal_saved = []
        signal_r = 1
        signal_markersize = 8
        points_lw = 3
        frame_num = 100
        frame_noisy = self.stack[frame_num]
        frame_ens = stack_ens[frame_num]
        # # find brightest frames
        # frame_bright = np.zeros_like(self.stack[0])
        # frame_bright_idx = 0
        # for idx, frame in enumerate(self.stack):
        #     frame_brightness = np.nanmean(frame)
        #     if frame_brightness > np.nanmean(frame_bright):
        #         frame_bright_idx = idx
        #         frame_noisy = frame
        # for idx, frame in enumerate(stack_ens):
        #     frame_brightness = np.nanmean(frame)
        #     if frame_brightness > np.nanmean(frame_bright):
        #         frame_bright_idx = idx
        #         frame_ens = frame

        # General layout
        fig_ens_stack = plt.figure(figsize=(8, 6))  # _ x _ inch page
        gs0 = fig_ens_stack.add_gridspec(1, 2)  # 1 row, 3 columns
        titles = ['Model Data (SNR: {}-{})'.format(self.snr_range[0], self.snr_range[1]),
                  'Ensembled Data']

        # Plot the frame and a trace from the stack
        for idx, stack in enumerate([self.stack, stack_ens]):
            frame = stack[frame_num]
            signal = stack[:, signal_y, signal_x]

            gs_frame_signal = gs0[idx].subgridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 rows, 1 columns
            ax_frame = fig_ens_stack.add_subplot(gs_frame_signal[0])
            # Frame image
            ax_frame.set_title(titles[idx], fontsize=fontsize2)
            # Create normalization colormap range based on all frames (round up to nearest 10)
            cmap_frames = SCMaps.grayC.reversed()
            if idx is 0:
                frame_min = round(np.nanmin(frame), -1)
                frame_max = round(np.nanmax(frame) + 5.1, -1)
            else:
                frame_min, frame_max = 0, 1
            cmap_norm = colors.Normalize(vmin=frame_min,
                                         vmax=frame_max)
            img_frame = ax_frame.imshow(frame, cmap=cmap_frames, norm=cmap_norm)
            ax_frame.set_yticks([])
            ax_frame.set_yticklabels([])
            ax_frame.set_xticks([])
            ax_frame.set_xticklabels([])
            frame_signal_rect = Rectangle((signal_x - signal_r / 2, signal_y - signal_r / 2),
                                          width=signal_r, height=signal_r,
                                          fc=color_clear, ec=color_raw, lw=points_lw)
            ax_frame.add_artist(frame_signal_rect)
            # if idx is len(titles) - 1:
            # Add colorbar (right of frame)
            ax_ins_filtered = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                         bbox_to_anchor=(0.08, 0, 1, 1),
                                         bbox_transform=ax_frame.transAxes, borderpad=0)
            cb_filtered = plt.colorbar(img_frame, cax=ax_ins_filtered, orientation="vertical")
            cb_filtered.ax.set_xlabel('a.u.', fontsize=fontsize3)
            cb_filtered.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
            cb_filtered.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
            cb_filtered.ax.tick_params(labelsize=fontsize3)
            # Signal trace
            ax_signal = fig_ens_stack.add_subplot(gs_frame_signal[1])
            ax_signal.set_ylabel('Fluorescence (arb. u.)')
            ax_signal.set_xlabel('Time (frame #)')
            # ax_signal.spines['right'].set_visible(False)
            # ax_signal.spines['top'].set_visible(False)
            ax_signal.tick_params(axis='x', which='minor', length=3, bottom=True)
            ax_signal.tick_params(axis='x', which='major', length=8, bottom=True)
            plt.rc('xtick', labelsize=fontsize2)
            plt.rc('ytick', labelsize=fontsize2)

            # Common between the two
            for ax in [ax_frame, ax_signal]:
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            ax_signal.plot(signal, color=gray_heavy, linestyle='None', marker='+')

            # if this is the original signal
            if idx is 0:
                # plot where ensemble crop was applied
                ax_signal.axvline(ensemble_crop[0], color=color_raw)
                ax_signal.axvline(ensemble_crop[1], color=color_raw)
                signal_saved = signal
                # calculate the ensemble
                time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
                    = calc_ensemble(self.time_stack, signal)
                # Activation
                i_activation = find_tran_act(signal)  # 1st df max, Activation
                ax_signal.plot(signal_acts, signal[signal_acts],
                               ".", color=colors_times['Activation'], markersize=signal_markersize)
                # Peak
                i_peak = find_tran_peak(signal)  # max of signal, Peak
                ax_signal.plot(signal_peaks, signal[signal_peaks],
                               "+", color=colors_times['Peak'], markersize=signal_markersize)
            else:
                # calculate the ensemble
                time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
                    = calc_ensemble(self.time_stack, signal_saved)
                for num, signal in enumerate(signals):
                    ax_signal.plot(signal, color=gray_light, linestyle='-')
                    # # Start
                    # i_start = find_tran_start(signal)  # 1st df2 max, Start
                    # ax_ensemble.plot(time_ensemble[i_start], signal[i_start],
                    #                  "x", color=colors_times['Start'], markersize=10)
                    # Activation
                    i_activation = find_tran_act(signal)  # 1st df max, Activation
                    ax_signal.plot(i_activation, signal[i_activation],
                                   ".", color=colors_times['Activation'], markersize=signal_markersize)
                    # Peak
                    i_peak = find_tran_peak(signal)  # max of signal, Peak
                    ax_signal.plot(i_peak, signal[i_peak],
                                   "+", color=colors_times['Peak'], markersize=signal_markersize)

                    snr_results = calculate_snr(signal)

        fig_ens_stack.savefig(dir_unit + '/results/analysis_EnsembleStack.png')
        fig_ens_stack.show()

    def test_export(self):
        # Save ensemble stack
        directory_ens = dir_unit + '/results/EnsembleModelStack_NEW.tif'

        stack_ens, ensemble_crop, ensemble_yx = calc_ensemble_stack(self.time_stack, self.stack)
        print('Saving ensemble stack ...')
        volwrite(directory_ens, stack_ens)

    def test_import_export(self):
        # Save ensemble stack
        directory_ens = dir_unit + '/results/' + self.file + '_Ensemble.tif'
        stack_ens, ensemble_crop, ensemble_yx = calc_ensemble_stack(self.time_stack, self.stack_import)
        print('Saving ensemble stack ...')
        volwrite(directory_ens, stack_ens)

    def test_import_export_real(self):
        # Save ensemble stack
        directory_ens = dir_unit + '/results/' + self.file + '_Ensemble.tif'
        stack_ens, ensemble_crop, ensemble_yx = calc_ensemble_stack(self.time_real, self.stack_real)
        print('Saving ensemble stack ...')
        volwrite(directory_ens, stack_ens)

    #
    # def test_real_trace(self):
    #     # Make sure ensemble of a trace looks correct
    #     time_ensemble, signal_ensemble, signals, signal_peaks, est_cycle_length = \
    #         calc_ensemble(self.stack_time_real, self.stack_real_trace)
    #         # calc_ensemble(self.time_real, self.signal_real)
    #
    #     last_baselines = find_tran_baselines(signals[-1])
    #
    #     # Build a figure to plot SNR results
    #     # fig_snr, ax_snr = plot_test()
    #     fig_ensemble = plt.figure(figsize=(12, 8))  # _ x _ inch page
    #     gs0 = fig_ensemble.add_gridspec(2, 1, height_ratios=[0.2, 0.8])  # 2 rows, 1 column
    #     ax_signal = fig_ensemble.add_subplot(gs0[0])
    #     ax_ensemble = fig_ensemble.add_subplot(gs0[1])
    #
    #     ax_signal.spines['right'].set_visible(False)
    #     ax_signal.spines['top'].set_visible(False)
    #     ax_signal.tick_params(axis='x', which='minor', length=3, bottom=True)
    #     ax_signal.tick_params(axis='x', which='major', length=8, bottom=True)
    #     plt.rc('xtick', labelsize=fontsize2)
    #     plt.rc('ytick', labelsize=fontsize2)
    #
    #     ax_signal.set_ylabel('arb. u.')
    #     # ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])
    #     ax_signal.plot(self.time_real, self.signal_real, color=gray_light,
    #                    linestyle='None', marker='+', label='Ca pixel data')
    #     ax_signal.plot(self.time_real[signal_peaks], self.signal_real[signal_peaks],
    #                    "x", color=colors_times['Peak'], markersize=10, label='Peaks')
    #     ax_signal.plot(self.time_real[last_baselines], self.signal_real[last_baselines],
    #                    "x", color=colors_times['Baseline'], label='Baselines')
    #
    #     ax_ensemble.spines['right'].set_visible(False)
    #     ax_ensemble.spines['top'].set_visible(False)
    #     ax_ensemble.set_ylabel('Fluorescence (arb. u.)')
    #     ax_ensemble.set_xlabel('Time (ms)')
    #
    #     signal_snrs = []
    #     for signal in signals:
    #         ax_ensemble.plot(time_ensemble, signal, color=gray_light, linestyle='-')
    #         # Start
    #         i_start = find_tran_start(signal)  # 1st df2 max, Start
    #         ax_ensemble.plot(time_ensemble[i_start], signal[i_start],
    #                          "x", color=colors_times['Start'], markersize=10)
    #         # Activation
    #         i_activation = find_tran_act(signal)  # 1st df max, Activation
    #         ax_ensemble.plot(time_ensemble[i_activation], signal[i_activation],
    #                          "x", color=colors_times['Activation'], markersize=10)
    #         # Peak
    #         i_peak = find_tran_peak(signal)  # max of signal, Peak
    #         ax_ensemble.plot(time_ensemble[i_peak], signal[i_peak],
    #                          "x", color=colors_times['Peak'], markersize=10)
    #         # # Downstroke
    #         # i_downstroke = find_tran_downstroke(signal)  # df min, Downstroke
    #         # ax_ensemble.plot(time_ensemble[i_downstroke], signal[i_downstroke],
    #         #                  "x", color=colors_times['Downstroke'], markersize=10, label='Downstroke')
    #         # # End
    #         # i_end = find_tran_end(signal)  # 2st df2 max, End
    #         # ax_ensemble.plot(time_ensemble[i_end], signal[i_end],
    #         #                  "x", color=colors_times['End'], markersize=10, label='End')
    #
    #         snr_results = calculate_snr(signal)
    #         snr = snr_results[0]
    #         ir_noise = snr_results[-2]
    #         signal_snrs.append(snr)
    #         ax_ensemble.plot(time_ensemble[ir_noise], signal[ir_noise],
    #                          "x", color=gray_med, markersize=10)
    #
    #     ax_ensemble.plot(time_ensemble, signal_ensemble, color=gray_heavy,
    #                      linestyle='-', marker='+', label='Ensemble signal')
    #     # Start
    #     i_start = find_tran_start(signal_ensemble)  # 1st df2 max, Start
    #     ax_ensemble.plot(time_ensemble[i_start], signal_ensemble[i_start],
    #                      ".", color=colors_times['Start'], markersize=15, label='Start')
    #
    #     # Activation
    #     i_activation = find_tran_act(signal_ensemble)  # 1st df max, Activation
    #     ax_ensemble.plot(time_ensemble[i_activation], signal_ensemble[i_activation],
    #                      ".", color=colors_times['Activation'], markersize=15, label='Activation')
    #     # Peak
    #     i_peak = find_tran_peak(signal_ensemble)  # max of signal, Peak
    #     ax_ensemble.plot(time_ensemble[i_peak], signal_ensemble[i_peak],
    #                      ".", color=colors_times['Peak'], markersize=15, label='Peak')
    #     ax_ensemble.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
    #
    #     # # Downstroke
    #     # i_downstroke = find_tran_downstroke(signal_ensemble)  # df min, Downstroke
    #     # ax_ensemble.plot(time_ensemble[i_downstroke], signal_ensemble[i_downstroke],
    #     #                  ".", color=colors_times['Downstroke'], markersize=15, label='Downstroke')
    #     # # End
    #     # i_end = find_tran_end(signal_ensemble)  # 2st df2 max, End
    #     # ax_ensemble.plot(time_ensemble[i_end], signal_ensemble[i_end],
    #     #                  ".", color=colors_times['End'], markersize=15, label='End')
    #
    #     # Text: Conditions
    #     ax_ensemble.text(0.75, 0.65, 'PCL actual: {}'.format(self.file_cl),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #     ax_ensemble.text(0.75, 0.6, 'File: {}'.format(self.file_name),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #     # Text: Cycles
    #     ax_ensemble.text(0.75, 0.5, 'PCL detected (ms): {}'.format(round(np.mean(est_cycle_length), 3)),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #     ax_ensemble.text(0.75, 0.45, '# Peaks detected : {}'.format(len(signal_peaks)),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #     # Stats: SNRs
    #     snr_old = round(np.mean(signal_snrs), 1)
    #     snr_results = calculate_snr(signal_ensemble)
    #     snr_new = round(snr_results[0], 1)
    #     ir_noise_new = snr_results[-2]
    #
    #     # Text
    #     ax_ensemble.plot(time_ensemble[ir_noise_new], signal_ensemble[ir_noise_new],
    #                      ".", color=gray_heavy, markersize=15, label='Noise')
    #     ax_ensemble.text(0.75, 0.35, 'SNR detected: {}'.format(snr_old),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #     ax_ensemble.text(0.75, 0.3, 'SNR ensembled: {}'.format(snr_new),
    #                      color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
    #
    #     # # Activation error bar
    #     # error_act = np.mean(signals_activations).astype(int)
    #     # ax_ensemble.errorbar(time_ensemble[error_act],
    #     #                      signal_ensemble[error_act],
    #     #                      xerr=statistics.stdev(signals_activations), fmt="x",
    #     #                      color=colors_times['Activation'], lw=3,
    #     #                      capsize=4, capthick=1.0)
    #
    #     # fig_ensemble.savefig(dir_unit + '/results/analysis_Ensemble_Real.png')
    #     fig_ensemble.show()

    # def test_stack(self):
    #     # Make sure ensemble of a model stack looks correct

    # def test_real_stack(self):
    #     # Make sure ensemble of a real stack looks correct
    # #     file_name_pig = '2019/03/22 pigb-01-Ca, PCL 150ms'
    # #     file_signal_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_30x30-LV-198x324.csv'
    # #     file_name, file_signal = file_name_pig, file_signal_pig
    # #     time, signal = open_signal(source=file_signal)


# class TestPhase(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 50
#     signal_time = 1000
#     signal_num = 5
#     noise = 2  # as a % of the signal amplitude
#     noise_count = 100
#     time_vm, signal_vm = model_transients(t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp,
#                                           noise=noise, num=signal_num)
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0 + 15, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp,
#                                           noise=noise, num=signal_num)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_phase, signal_in=True)
#         self.assertRaises(TypeError, calc_phase, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_phase, signal_in='word')
#         self.assertRaises(TypeError, calc_phase, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_phase, signal_in=signal_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         signal_vm_phase = calc_phase(self.signal_vm)
#         signal_ca_phase = calc_phase(self.signal_ca)
#         # signal_FF0 : ndarray, dtyoe : float
#         self.assertIsInstance(signal_ca_phase, np.ndarray)  # The array of phase
#         self.assertIsInstance(signal_ca_phase[0], float)  # dtyoe : float
#
#         # Make sure result values are valid
#         self.assertAlmostEqual(signal_ca_phase.min(), signal_vm_phase.max(), delta=0.01)


if __name__ == '__main__':
    unittest.main()
