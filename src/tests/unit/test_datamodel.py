import unittest
from util.datamodel import *
from pathlib import Path
import time
from math import pi
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from imageio import volwrite
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# File paths needed for tests
dir_tests = str(Path.cwd().parent)
dir_unit = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_vm, color_ca = ['#FF9999', '#99FF99']


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True, top=True)
    axis.xaxis.set_major_locator(plticker.MultipleLocator(25))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(5))
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


class TestModelTransients(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_transients, model_type=True)
        self.assertRaises(TypeError, model_transients, t=3 + 5j)
        self.assertRaises(TypeError, model_transients, t=True)
        self.assertRaises(TypeError, model_transients, t='radius')
        # Some parameters must be an int
        self.assertRaises(TypeError, model_transients, t=250.5, t0=20.5, fps=50.3, f_0=1.5, f_amp=10.4)
        self.assertRaises(TypeError, model_transients, num=True)  # num must be an int or 'full'

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_transients, model_type='voltage')  # proper model type
        self.assertRaises(ValueError, model_transients, t=-2)  # no negative total times
        self.assertRaises(ValueError, model_transients, t=99)  # total time at least 100 ms long
        self.assertRaises(ValueError, model_transients, t=150, t0=150)  # start time < than total time
        self.assertRaises(ValueError, model_transients, fps=150)  # no fps < 200
        self.assertRaises(ValueError, model_transients, fps=1001)  # no fps > 1000
        self.assertRaises(ValueError, model_transients, f_0=2 ** 16)  # no baseline > 16-bit max
        self.assertRaises(ValueError, model_transients, f_amp=2 ** 16)  # no amplitude > 16-bit max
        self.assertRaises(ValueError, model_transients, f_amp=-2)  # no amplitude < 0
        self.assertRaises(ValueError, model_transients, f_0=0, f_amp=20)  # no amplitude < 0, especially Vm
        # Multiple transients
        self.assertRaises(ValueError, model_transients, num=-1)  # no negative transients
        self.assertRaises(ValueError, model_transients, num='5')  # if a string, must be 'full'
        self.assertRaises(ValueError, model_transients, t=300, t0=50, num=3)  # not too many transients
        self.assertRaises(ValueError, model_transients, t=300, t0=50, num=2, cl=49)  # minimum Cycle Length
        # Clipping of fluorescence
        self.assertRaises(ValueError, model_transients, model_type='Ca',
                          f_0=2 ** 16 - 50)  # Would cause clipping due to an overflow of uint16
        self.assertRaises(ValueError, model_transients, model_type='Ca',
                          f_0=2 ** 16 - 50, f_amp=48, noise=5)  # Noise would cause clipping

    def test_results(self):
        # Make sure model results are valid
        self.assertIsInstance(model_transients(), tuple)  # results returned as a tuple
        self.assertEqual(len(model_transients()), 2)  # time and data arrays returned
        self.assertEqual(model_transients(t=1000)[0].size, model_transients(t=1000)[1].size)  # time and data same size
        # Test the returned time array
        self.assertEqual(model_transients(t=150)[0].size, 150)  # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[0].size, 223)  # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[0].all(), 0)  # no negative times
        self.assertLess(model_transients(t=200)[0].all(), 200)  # no times >= total time parameter
        # Test the returned data array
        self.assertIsInstance(model_transients(noise=5)[1][1], np.uint16)  # data values returned as uint16
        self.assertEqual(model_transients(t=150)[1].size, 150)  # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[1].size, 223)  # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[1].all(), 0)  # no negative values
        self.assertLess(model_transients(model_type='Ca')[1].all(), 2 ** 16 - 1)  # no values >= 16-bit max
        self.assertGreaterEqual(2000,
                                model_transients(model_type='Vm', f0=2000)[1].max())  # Vm amplitude handled properly
        self.assertLessEqual(2000,
                             model_transients(model_type='Ca', f0=2000)[1].min())  # Ca amplitude handled properly

        # Test multiple transient generation
        f_amp = 250
        num = 4
        peak_min_height = f_amp / 2

        time_vm, data_vm = model_transients(t=500, f0=2000, famp=f_amp, num=num)
        data_vm_inv = (-(data_vm - 2000)) + 2000
        peaks_vm, _ = find_peaks(data_vm_inv, height=peak_min_height, prominence=f_amp / 2)
        self.assertEqual(num, peaks_vm.size)  # detected peaks matches number of generated transients

        time_ca, data_ca = model_transients(model_type='Ca', t=500, f0=1000, famp=f_amp, num=num)
        # peaks_ca, _ = find_peaks(data_ca, height=1000 + peak_min_height, distance=len(data_ca)/num)
        peaks_ca, _ = find_peaks(data_ca, height=1000 + peak_min_height, prominence=f_amp / 2)
        self.assertEqual(num, peaks_ca.size)  # detected peaks matches number of generated transients

        # time_ca_full, data_ca_full = model_transients(model_type='Ca', t=500, f_0=1000, f_amp=250, num='full')
        num_full = 5000 / 100
        time_ca_full, data_ca_full = model_transients(model_type='Ca', t=5000, f0=1000, famp=f_amp, num='full')
        peaks_ca, _ = find_peaks(data_ca_full, height=peak_min_height, prominence=f_amp / 2)
        self.assertEqual(num_full, peaks_ca.size)  # detected peaks matches calculated transients for 'full'

    def test_plot_single_rat(self):
        time_vm, data_vm = model_transients()
        # time_ca, data_ca = model_transients(model_type='Ca', t=100, f_0=2 ** 16 - 50)

        # Build a figure to plot model data
        fig_single, ax_single = plot_test()
        ax_single.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_single.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        # data_vm_align = -(data_vm - data_vm.max())
        # data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_single.plot(time_vm, data_vm, marker='+', color=color_vm, label='Vm')
        # plot_ca, = ax_single.plot(time_ca, data_ca, marker='+', color=color_ca, label='Ca')
        # plot_baseline = ax_single.axhline(color='gray', linestyle='--', label='baseline')
        ax_single.legend(title='A Model Transient (Defaults)',
                         loc='right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_single.show()

    def test_plot_single_pig(self):
        time_vm, data_vm = model_transients_pig(fps=500)
        # time_ca, data_ca = model_transients(model_type='Ca', t=100, f_0=2 ** 16 - 50)

        # Build a figure to plot model data
        fig_single, ax_single = plot_test()
        ax_single.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_single.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        # data_vm_align = -(data_vm - data_vm.max())
        # data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_single.plot(time_vm, data_vm, marker='+', color=color_vm, label='Vm')
        # plot_ca, = ax_single.plot(time_ca, data_ca, marker='+', color=color_ca, label='Ca')
        # plot_baseline = ax_single.axhline(color='gray', linestyle='--', label='baseline')
        ax_single.legend(title='A Model Pig Transient (Defaults)',
                         loc='right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_single.show()

    def test_plot_single_bspline(self):
        # See a B-spline representation of model data
        x = np.array([0., 1.2, 1.9, 3.2, 4., 6.5])
        y = np.array([0., 2.3, 3., 4.3, 2.9, 3.1])

        t, c, k = interpolate.splrep(x, y, k=2)
        # print('''\
        # t: {}
        # c: {}
        # k: {}
        # '''.format(t, c, k))
        N = 100
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        # Build a figure to plot model data
        fig_bspline, ax_bspline = plot_test()
        ax_bspline.set_ylim([-5, 5])

        ax_bspline.plot(x, y, 'bo', label='Original points')
        ax_bspline.plot(xx, spline(xx), 'r', marker='+', linestyle='', label='B-Spline points')
        ax_bspline.grid()
        ax_bspline.legend(loc='best')

        fig_bspline.show()

    def test_plot_fps(self):
        # Test model Ca single transient data, at different fps
        time_ca_1, data_ca_1 = model_transients(model_type='Ca', fps=250, f0=1000, famp=250)
        time_ca_2, data_ca_2 = model_transients(model_type='Ca', fps=500, f0=1000, famp=250)
        time_ca_3, data_ca_3 = model_transients(model_type='Ca', fps=1000, f0=1000, famp=250)

        # Build a figure to plot model data
        fig_dual_fps, ax_dual_fps = plot_test()
        ax_dual_fps.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_dual_fps.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_ca_1, = ax_dual_fps.plot(time_ca_1, data_ca_1 - 1000, gray_light, marker='1', label='Ca, fps: 250')
        plot_ca_2, = ax_dual_fps.plot(time_ca_2, data_ca_2 - 1000, gray_med, marker='+', label='Ca, fps: 500')
        plot_ca_3, = ax_dual_fps.plot(time_ca_3, data_ca_3 - 1000, gray_heavy, marker='2', label='Ca, fps: 1000')
        plot_baseline = ax_dual_fps.axhline(color='gray', linestyle='--', label='baseline')
        ax_dual_fps.legend(title='FPS Variations',
                           loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        # ax_dual_fps.set_title('FPS Variations')
        fig_dual_fps.show()

    def test_plot_duration(self):
        # Test model Ca single transient data, at different CADs
        time_vm_1, data_vm_1 = model_transients(apd={'20': MIN_APD_20})
        time_vm_2, data_vm_2 = model_transients(apd={'20': 15})
        time_vm_3, data_vm_3 = model_transients(apd={'20': 25})
        time_ca_1, data_ca_1 = model_transients(model_type='Ca', cad={'80': MIN_CAD_80})
        time_ca_2, data_ca_2 = model_transients(model_type='Ca', cad={'80': 60})
        time_ca_3, data_ca_3 = model_transients(model_type='Ca', cad={'80': 80})

        # Build a figure to plot model data
        fig_duration, ax_duration = plot_test()
        ax_duration.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_duration.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        # plot_vm_1, = ax_duration.plot(time_vm_1, -(data_vm_1 - 100), color_vm, marker='1', label='Vm, APD20: 5')
        # plot_vm_2, = ax_duration.plot(time_vm_2, -(data_vm_2 - 100), color_vm, marker='+', label='Vm, APD20: 15')
        # plot_vm_3, = ax_duration.plot(time_vm_3, -(data_vm_3 - 100), color_vm, marker='2', label='Vm, APD20: 25')
        plot_ca_1, = ax_duration.plot(time_ca_1, data_ca_1 - 100, color_ca, marker='1', label='Ca, CAD80: 50}')
        plot_ca_2, = ax_duration.plot(time_ca_2, data_ca_2 - 100, color_ca, marker='+', label='Ca, CAD80: 60')
        plot_ca_3, = ax_duration.plot(time_ca_3, data_ca_3 - 100, color_ca, marker='2', label='Ca, CAD80: 80')
        plot_baseline = ax_duration.axhline(color='gray', linestyle='--', label='baseline')
        ax_duration.legend(title='APD20 and CAD80 Variations',
                           loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_duration.show()

    def test_plot_cyclelength(self):
        # Test model Vm multi transient data, at different Cycle Lengths
        num = 4
        time_vm1, data_vm1 = model_transients(t=500, f0=2000, famp=250, num=num, cl=50)
        time_vm2, data_vm2 = model_transients(t=500, f0=2000, famp=250, num=num)
        time_vm3, data_vm3 = model_transients(t=500, f0=2000, famp=250, num=num, cl=150)

        # Build a figure to plot model data
        fig_cyclelength, ax_old = plot_test()
        ax_cyclelength1 = plt.subplot(3, 1, 1)
        plt.setp(ax_cyclelength1.get_xticklabels(), visible=False)
        ax_cyclelength2 = plt.subplot(3, 1, 2, sharex=ax_cyclelength1)
        ax_cyclelength2.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        plt.setp(ax_cyclelength2.get_xticklabels(), visible=False)
        ax_cyclelength3 = plt.subplot(3, 1, 3, sharex=ax_cyclelength1)
        ax_cyclelength3.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        data_vm1_align, data_vm2_align, data_vm3_align = \
            -(data_vm1 - data_vm1.max()), -(data_vm2 - data_vm2.max()), -(data_vm3 - data_vm3.max())
        plot_vm_1, = ax_cyclelength1.plot(time_vm1, data_vm1_align, gray_light, marker='1', label='Vm, CL: 50')
        plot_vm_2, = ax_cyclelength2.plot(time_vm2, data_vm2_align, gray_med, marker='+', label='Vm, CL: 100')
        plot_vm_3, = ax_cyclelength3.plot(time_vm3, data_vm3_align, gray_heavy, marker='2', label='Vm, CL: 150')
        # plot_baseline = ax_cyclelength.axhline(color='gray', linestyle='--', label='baseline')
        fig_cyclelength.legend(title='Cycle Length Variations',
                               loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_cyclelength.show()

    def test_plot_dual(self):
        # Test model Vm and Ca single transient data
        time_vm, data_vm = model_transients(f0=2000, famp=250)
        time_ca, data_ca = model_transients(model_type='Ca', f0=1000, famp=250)
        self.assertEqual(time_vm.size, data_vm.size)  # data and time arrays returned as a tuple

        # Build a figure to plot model data
        fig_dual, ax_dual = plot_test()
        # ax_dual.set_title('Dual Vm and Ca transient')
        ax_dual.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_dual.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        data_vm_align = -(data_vm - data_vm.max())
        data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_dual.plot(time_vm, data_vm_align, 'r+', label='Vm')
        plot_ca, = ax_dual.plot(time_ca, data_ca_align, 'y+', label='Ca')
        plot_baseline = ax_dual.axhline(color='gray', linestyle='--', label='baseline')
        ax_dual.legend(title='Dual Vm and Ca transient',
                       loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_dual.show()

    def test_plot_dual_multi(self):
        # Test model Vm and Ca multi-transient data, with noise
        num = 4
        time_vm, data_vm = model_transients(t=500, t0=25, f0=2000, famp=250, num=num)
        time_ca, data_ca = model_transients(model_type='Ca', t=500, t0=25, f0=1000, famp=250, num=num)

        # Build a figure to plot model data
        fig_dual_multi, ax_dual_multi = plot_test()
        ax_dual_multi.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_dual_multi.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        data_vm_align = -(data_vm - data_vm.max())
        data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_dual_multi.plot(time_vm, data_vm_align, 'r+', label='Vm')
        plot_ca, = ax_dual_multi.plot(time_ca, data_ca_align, 'y+', label='Ca')
        plot_baseline = ax_dual_multi.axhline(color='gray', linestyle='--', label='baseline')
        ax_dual_multi.legend(title='Dual Vm and Ca transients',
                             loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_dual_multi.show()


class TestModelStack(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_stack, size=20)  # size must be a tuple, e.g. (100, 50)
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_stack, size=(20, 5))  # no size > (10, 10)
        self.assertRaises(ValueError, model_stack, size=(5, 20))  # no size > (10, 10)

    def test_results(self):
        # Make sure model stack results are valid
        self.assertIsInstance(model_stack(), tuple)  # results returned as a tuple
        self.assertEqual(len(model_stack()), 2)  # time and data arrays returned
        stack_time, stack_data = model_stack()
        self.assertEqual(stack_time.size, stack_data.shape[0])  # time and data same size

        # Test the returned time array
        self.assertEqual(stack_time.size, 100)  # length is correct
        self.assertGreaterEqual(stack_time.all(), 0)  # no negative times

        # Test the returned data array
        self.assertEqual(stack_data.shape, (100, 100, 50))  # default dimensions (T, Y, X)
        self.assertGreaterEqual(stack_data.all(), 0)  # no negative values
        self.assertLess(stack_data.all(), 2 ** 16)  # no values >= 16-bit max
        stackSize_time, stackSize_data = model_stack(t=1000, size=(100, 100))
        self.assertEqual(stackSize_data.shape, (1000, 100, 100))  # dimensions (T, Y, X)

    def test_tiff(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        time_vm, data_vm = model_stack(t=1000)
        volwrite(dir_unit + '/results/ModelStack_vm.tif', data_vm)

        time_ca, data_ca = model_stack(model_type='Ca', t=1000)
        volwrite(dir_unit + '/results/ModelStack_ca.tif', data_ca)


class TestModelStackPropagation(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_stack_propagation, size=20)  # size must be a tuple, e.g. (100, 50)
        self.assertRaises(TypeError, model_stack_propagation, velocity='50')
        self.assertRaises(TypeError, model_stack_propagation, snr='10')
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_stack_propagation, size=(20, 5))  # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_propagation, size=(5, 20))  # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_propagation, velocity=4)  # no velocity > 5
        self.assertRaises(ValueError, model_stack_propagation, t=90)  # no t < 100

    def test_results(self):
        # Make sure model stack results are valid
        self.assertIsInstance(model_stack_propagation(), tuple)  # results returned as a tuple
        self.assertEqual(len(model_stack_propagation()), 2)  # time and data arrays returned
        stack_time, stack_data = model_stack_propagation()
        self.assertEqual(stack_time.size, stack_data.shape[0])

        # Test the returned time array
        self.assertEqual(stack_time.size, 150)  # default velocity of 20 -> t of 150
        self.assertGreaterEqual(stack_time.all(), 0)  # no negative times

        # Test the returned data array
        self.assertEqual(stack_data.shape, (150, 100, 50))  # default dimensions (T, Y, X)
        self.assertGreaterEqual(stack_data.all(), 0)  # no negative values
        self.assertLess(stack_data.all(), 2 ** 16)  # no values >= 16-bit max
        stackSize_time, stackSize_data = model_stack_propagation(size=(100, 100))
        self.assertEqual(stackSize_data.shape, (150, 100, 100))  # dimensions (T, Y, X)

    def test_tiff(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        start = time.process_time()
        time_vm, data_vm = model_stack_propagation()
        end = time.process_time()
        print('Timing, test_tiff, Vm : ', end - start)
        volwrite(dir_tests + '/results/ModelStackPropagation_vm.tif', data_vm)
        time_ca, data_ca = model_stack_propagation(model_type='Ca')
        volwrite(dir_tests + '/results/ModelStackPropagation_ca.tif', data_ca)

    def test_tiff_noise(self):
        # Make sure this stack is similar to a noisy 16-bit .tif/.tiff
        start = time.process_time()
        time_vm, data_vm = model_stack_propagation(noise=5, velocity=50, t0=50)
        end = time.process_time()
        print('Timing, test_tiff_noise, Vm : ', end - start)
        volwrite(dir_unit + '/results/ModelStackPropagationNoise_vm.tif', data_vm)
        time_ca, data_ca = model_stack_propagation(noise=5, model_type='Ca', velocity=50, t0=50)
        volwrite(dir_unit + '/results/ModelStackPropagationNoise_ca.tif', data_ca)

    def test_tiff_snr(self):
        # Make sure this stack is similar to a variably noisy 16-bit .tif/.tiff
        start = time.process_time()
        time_vm, data_vm = model_stack_propagation(d_noise=10, noise=5, num='full')
        end = time.process_time()
        print('Timing, test_tiff_snr, Vm : ', end - start)
        volwrite(dir_unit + '/results/ModelStackPropagationSNR_vm.tif', data_vm)
        time_ca, data_ca = model_stack_propagation(model_type='Ca', d_noise=10, noise=5, num='full')
        volwrite(dir_unit + '/results/ModelStackPropagationSNR_ca.tif', data_ca)


class TestModelStackHeart(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack of varying SNR (highest in the center)
        self.size = (100, 100)
        self.d_noise = 45  # as a % of the signal amplitude
        self.signal_t = 500
        self.signal_t0 = 100
        self.signal_f0 = 1000
        self.signal_famp = 500
        self.signal_num = 'full'
        self.signal_cl = 100
        self.signal_noise = 5  # as a % of the signal amplitude

        self.time_vm, self.stack_vm = \
            model_stack_heart(size=self.size, d_noise=self.d_noise,
                              t=self.signal_t, t0=self.signal_t0,
                              f0=self.signal_f0, famp=self.signal_famp, noise=self.signal_noise,
                              num=self.signal_num, cl=self.signal_cl)
        self.time_ca, self.stack_ca = \
            model_stack_heart(model_type='Ca', size=self.size, d_noise=self.d_noise,
                              t=self.signal_t, t0=self.signal_t0,
                              f0=self.signal_f0, famp=self.signal_famp, noise=self.signal_noise,
                              num=self.signal_num, cl=self.signal_cl)

    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_stack_heart, size=20)  # size must be a tuple, e.g. (100, 50)
        self.assertRaises(TypeError, model_stack_heart, velocity='50')
        self.assertRaises(TypeError, model_stack_heart, snr='10')
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_stack_heart, size=(20, 5))  # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_heart, size=(5, 20))  # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_heart, velocity=4)  # no velocity > 5
        self.assertRaises(ValueError, model_stack_heart, t=90)  # no t < 100

    def test_results(self):
        # Make sure model stack results are valid
        self.assertIsInstance(model_stack_heart(), tuple)  # results returned as a tuple
        self.assertEqual(len(model_stack_heart()), 2)  # time and data arrays returned
        stack_time, stack_data = model_stack_heart()
        self.assertEqual(stack_time.size, stack_data.shape[0])

        # Test the returned time array
        self.assertEqual(stack_time.size, 150)  # default velocity of 20 -> t of 150
        self.assertGreaterEqual(stack_time.all(), 0)  # no negative times

        # Test the returned data array
        self.assertEqual(stack_data.shape, (150, 100, 50))  # default dimensions (T, Y, X)
        self.assertGreaterEqual(stack_data.all(), 0)  # no negative values
        self.assertLess(stack_data.all(), 2 ** 16)  # no values >= 16-bit max
        stackSize_time, stackSize_data = model_stack_heart(size=(100, 100))
        self.assertEqual(stackSize_data.shape, (150, 100, 100))  # dimensions (T, Y, X)

    def test_tiff(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        volwrite(dir_unit + '/results/ModelStackHeart_vm.tif', self.stack_vm)
        volwrite(dir_unit + '/results/ModelStackHeart_ca.tif', self.stack_ca)


# Example tests
class TestModelCircle(unittest.TestCase):
    def test_area(self):
        # Test areas when radius >= 0
        self.assertAlmostEqual(circle_area(1), pi)
        self.assertAlmostEqual(circle_area(0), 0)
        self.assertAlmostEqual(circle_area(2.1), pi * 2.1 * 2.1)

    def test_values(self):
        # Make sure valid errors are raised when necessary
        self.assertRaises(ValueError, circle_area, -2)

    def test_type(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, circle_area, 3 + 5j)
        self.assertRaises(TypeError, circle_area, True)
        self.assertRaises(TypeError, circle_area, 'radius')


if __name__ == '__main__':
    unittest.main()
