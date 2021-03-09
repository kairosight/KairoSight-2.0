#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
import traceback
import time
import math
import cv2
import numpy as np
from pathlib import Path, PurePath
from random import random
from matplotlib.animation import FuncAnimation

from util.preparation import (open_stack, reduce_stack, mask_generate,
                              mask_apply, img_as_uint, rescale)
from util.processing import (normalize_stack, filter_spatial, map_snr,
                             find_tran_act, filter_spatial_stack,
                             filter_temporal, invert_signal, filter_drift)
from util.analysis import (calc_tran_duration, map_tran_analysis, DUR_MAX,
                           calc_tran_activation, oap_peak_calc, diast_ind_calc,
                           act_ind_calc, apd_ind_calc, tau_calc,
                           ensemble_xlsx_print)
        
'from ui.KairoSight_WindowMDI import Ui_WindowMDI'
from ui.KairoSight_WindowMain_Retro import Ui_MainWindow
from PyQt5.QtCore import (QObject, pyqtSignal, Qt, QTimer, QRunnable,
                          pyqtSlot, QThreadPool)
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, 
                             QFileDialog, QListWidget, QMessageBox)
from PyQt5.QtGui import QColor, QPalette
import pyqtgraph as pg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import util.ScientificColourMaps5 as SCMaps
from tests.intergration.test_Map import (fontsize3, fontsize4, marker1,
                                         marker3, gray_heavy, color_snr,
                                         cmap_snr, cmap_activation,
                                         ACT_MAX_PIG_LV, ACT_MAX_PIG_WHOLE,
                                         cmap_duration, add_map_colorbar_stats)


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=391, height=391, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class JobRunner(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and
    wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and kwargs will be passed through to the
                     runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(JobRunner, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            # Done
            self.signals.finished.emit()


class MainWindow(QWidget, Ui_MainWindow):
    """Customization for Ui_WindowMain"""

    def __init__(self, parent=None, file_purepath=None):
        # Initialization of the superclass
        super(MainWindow, self).__init__(parent)
        '''self.WindowMDI = parent'''
        # Setup the UI
        self.setupUi(self)
        # Save the background color for future reference
        self.bkgd_color = [240/255, 240/255, 240/255]
        # Setup the image window
        self.mpl_canvas = MplCanvas(self)
        self.mpl_vl_window.addWidget(self.mpl_canvas)
        # Match the matplotlib figure background color to the GUI
        self.mpl_canvas.fig.patch.set_facecolor(self.bkgd_color)
        # Setup the signal windows
        self.mpl_canvas_sig1 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig1)
        self.mpl_canvas_sig1.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig2 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig2)
        self.mpl_canvas_sig2.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig3 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig3)
        self.mpl_canvas_sig3.fig.patch.set_facecolor(self.bkgd_color)
        self.mpl_canvas_sig4 = MplCanvas(self)
        self.mpl_sigvl_window.addWidget(self.mpl_canvas_sig4)
        self.mpl_canvas_sig4.fig.patch.set_facecolor(self.bkgd_color)
        # Setup button functionality
        self.sel_dir_button.clicked.connect(self.sel_dir)
        self.load_button.clicked.connect(self.load_data)
        self.refresh_button.clicked.connect(self.refresh_data)
        self.data_prop_button.clicked.connect(self.data_properties)
        self.signal_select_button.clicked.connect(self.signal_select)
        self.prep_button.clicked.connect(self.run_prep)
        self.analysis_drop.currentIndexChanged.connect(self.analysis_select)
        self.map_pushbutton.clicked.connect(self.map_analysis)
        self.axes_start_time_edit.editingFinished.connect(self.update_win)
        self.axes_end_time_edit.editingFinished.connect(self.update_win)
        self.start_time_edit.editingFinished.connect(self.update_analysis_win)
        self.end_time_edit.editingFinished.connect(self.update_analysis_win)
        self.max_val_edit.editingFinished.connect(self.update_analysis_win)
        self.movie_scroll_obj.valueChanged.connect(self.update_axes)
        self.play_movie_button.clicked.connect(self.play_movie)
        self.pause_button.clicked.connect(self.pause_movie)
        self.export_movie_button.clicked.connect(self.export_movie)
        # Thread runner
        self.threadpool = QThreadPool()
        # Create a timer for regulating the movie while loop
        # self.timer = QTimer()
        # self.timer.timeout.connect()
        # Set up variable for tracking which signal is being selected
        self.data = []
        self.data_filt = []
        self.signal_time = []
        self.analysis_bot_lim = False
        self.analysis_top_lim = False
        self.analysis_y_lim = False
        self.cnames = ['cornflowerblue', 'gold', 'springgreen', 'lightcoral']
        # Designate that dividing by zero will not generate an error
        np.seterr(divide='ignore', invalid='ignore')

    # Button Functions
    def sel_dir(self):
        # Open dialogue box for selecting the data directory
        self.file_path = QFileDialog.getExistingDirectory(
            self, "Open Directory", os.getcwd(), QFileDialog.ShowDirsOnly)
        # Update list widget with the contents of the selected directory
        self.refresh_data()

    def load_data(self):
        # Grab the selected items name
        self.file_name = self.file_list.currentItem().text()
        # Load the data stack into the UI
        self.video_data_raw = open_stack(
            source=(self.file_path + "/" + self.file_name))
        # Extract the optical data from the stack
        self.data = self.video_data_raw[0]
        # Populate the axes start and end indices
        self.axes_start_ind = 0
        self.axes_end_ind = self.data.shape[0]-1
        # Populate the mask variable
        self.mask = np.ones([self.data.shape[1], self.data.shape[2]],
                            dtype=bool)
        # Reset the signal selection variables
        self.signal_ind = 0
        self.signal_coord = np.zeros((4, 2))
        self.signal_toggle = np.zeros((4, 1))
        self.norm_flag = 0
        # Update the movie window tools with the appropriate values
        self.movie_scroll_obj.setMaximum(self.data.shape[0])
        self.play_bool = 0
        # Reset text edit values
        self.frame_rate_edit.setText('')
        self.image_scale_edit.setText('')
        self.start_time_edit.setText('')
        self.end_time_edit.setText('')
        self.max_apd_edit.setText('')
        self.perc_apd_edit.setText('')
        self.axes_end_time_edit.setText('')
        self.axes_start_time_edit.setText('')
        # Update the axes
        self.update_analysis_win()
        # self.update_axes()
        # Activate Properties Interface
        self.frame_rate_label.setEnabled(True)
        self.frame_rate_edit.setEnabled(True)
        self.image_scale_label.setEnabled(True)
        self.image_scale_edit.setEnabled(True)
        self.data_prop_button.setEnabled(True)
        self.image_type_label.setEnabled(True)
        self.image_type_drop.setEnabled(True)
        # Disable Preparation Tools
        self.rm_bkgd_checkbox.setEnabled(False)
        self.rm_bkgd_method_label.setEnabled(False)
        self.rm_bkgd_method_drop.setEnabled(False)
        self.bkgd_dark_label.setEnabled(False)
        self.bkgd_dark_edit.setEnabled(False)
        self.bkgd_light_label.setEnabled(False)
        self.bkgd_light_edit.setEnabled(False)
        self.bin_checkbox.setEnabled(False)
        self.bin_drop.setEnabled(False)
        self.filter_checkbox.setEnabled(False)
        self.filter_label_separator.setEnabled(False)
        self.filter_upper_label.setEnabled(False)
        self.filter_upper_edit.setEnabled(False)
        self.drift_checkbox.setEnabled(False)
        self.drift_drop.setEnabled(False)
        self.normalize_checkbox.setEnabled(False)
        self.prep_button.setEnabled(False)
        # Change the button string
        self.data_prop_button.setText('Start Preparation')
        # Disable Analysis Tools
        self.analysis_drop.setEnabled(False)
        self.analysis_drop.setCurrentIndex(0)
        self.start_time_label.setEnabled(False)
        self.start_time_edit.setEnabled(False)
        self.end_time_label.setEnabled(False)
        self.end_time_edit.setEnabled(False)
        self.map_pushbutton.setEnabled(False)
        self.max_apd_label.setEnabled(False)
        self.max_apd_edit.setEnabled(False)
        self.max_val_label.setEnabled(False)
        self.max_val_edit.setEnabled(False)
        self.perc_apd_label.setEnabled(False)
        self.perc_apd_edit.setEnabled(False)
        # Check the check box
        for n in np.arange(1, len(self.signal_toggle)):
            checkboxname = 'ensemble_cb_0{}'.format(n)
            checkbox = getattr(self, checkboxname)
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
        # Disable Movie and Signal Tools
        self.signal_select_button.setEnabled(False)
        self.movie_scroll_obj.setEnabled(False)
        self.play_movie_button.setEnabled(False)
        self.export_movie_button.setEnabled(False)
        # self.optical_toggle_button.setEnabled(False)
        # Disable axes controls
        self.axes_start_time_label.setEnabled(False)
        self.axes_start_time_edit.setEnabled(False)
        self.axes_end_time_label.setEnabled(False)
        self.axes_end_time_edit.setEnabled(False)

    def refresh_data(self):
        # Grab the applicable file names of the directory and display
        self.data_files = []
        for file in os.listdir(self.file_path):
            if file.endswith(".tif"):
                self.data_files.append(file)
        # If tif files were identified update the list and button availability
        if len(self.data_files) > 0:
            # Clear any potential items from the list widget
            self.file_list.clear()
            # Populate the list widget with the file names
            self.file_list.addItems(self.data_files)
            # Set the current row to the first (i.e., index = 0)
            self.file_list.setCurrentRow(0)
            # Enable the load and refresh buttons
            self.load_button.setEnabled(True)
            self.refresh_button.setEnabled(True)
        else:
            # Clear any potential items from the list widget
            self.file_list.clear()
            # Disable the load button
            self.load_button.setEnabled(False)
            # Create a message box to communicate the absence of data
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No *.tif files in selected directory.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def data_properties(self):
        if self.data_prop_button.text() == 'Start Preparation':
            # Populate global variables with frame rate and scale values
            self.data_fps = float(self.frame_rate_edit.text())
            self.data_scale = float(self.image_scale_edit.text())
            # Check data type and flip if necessary
            if self.image_type_drop.currentIndex() == 0:
                # Membrane potential, flip the data
                self.data_filt = self.data.astype(float)*-1
            else:
                # Calcium transient, don't flip the data
                self.data_filt = self.data.astype(float)
            # Create time vector
            self.signal_time = np.arange(self.data.shape[0])*1/self.data_fps
            # Populate the axes start and end edit boxes
            self.axes_start_time_edit.setText(
                str(self.signal_time[self.axes_start_ind]))
            self.axes_end_time_edit.setText(
                str(self.signal_time[self.axes_end_ind-1]))
            # Adjust the x-axis labeling for the signal windows
            for n in np.arange(1, len(self.signal_coord)+1):
                canvasname = 'mpl_canvas_sig{}'.format(n)
                canvas = getattr(self, canvasname)
                canvas.axes.set_xlim(self.signal_time[0], self.signal_time[-1])
                '''print([self.signal_time[0], self.signal_time[-1]])
                if n != len(self.signal_coord):
                    canvas.axes.tick_params(labelbottom=False)'''
                canvas.fig.tight_layout()
                canvas.draw()
            # Activate Movie and Signal Tools
            self.signal_select_button.setEnabled(True)
            # Activate Preparation Tools
            self.rm_bkgd_checkbox.setEnabled(True)
            self.rm_bkgd_method_label.setEnabled(True)
            self.rm_bkgd_method_drop.setEnabled(True)
            self.bkgd_dark_label.setEnabled(True)
            self.bkgd_dark_edit.setEnabled(True)
            self.bkgd_light_label.setEnabled(True)
            self.bkgd_light_edit.setEnabled(True)
            self.bin_checkbox.setEnabled(True)
            self.bin_drop.setEnabled(True)
            self.filter_checkbox.setEnabled(True)
            self.filter_label_separator.setEnabled(True)
            self.filter_upper_label.setEnabled(True)
            self.filter_upper_edit.setEnabled(True)
            self.drift_checkbox.setEnabled(True)
            self.drift_drop.setEnabled(True)
            self.normalize_checkbox.setEnabled(True)
            self.prep_button.setEnabled(True)
            # Activate Analysis Tools
            self.analysis_drop.setEnabled(True)
            self.start_time_label.setEnabled(True)
            self.start_time_edit.setEnabled(True)
            self.end_time_label.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            self.map_pushbutton.setEnabled(True)
            if self.analysis_drop.currentIndex() == 1:
                self.max_apd_label.setEnabled(True)
                self.max_apd_edit.setEnabled(True)
                self.max_val_label.setEnabled(True)
                self.max_val_edit.setEnabled(True)
                self.perc_apd_label.setEnabled(True)
                self.perc_apd_edit.setEnabled(True)
            else:
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.perc_apd_label.setEnabled(False)
                self.perc_apd_edit.setEnabled(False)
            # Activate axes controls
            self.axes_start_time_label.setEnabled(True)
            self.axes_start_time_edit.setEnabled(True)
            self.axes_end_time_label.setEnabled(True)
            self.axes_end_time_edit.setEnabled(True)
            # Disable Properties Tools
            self.frame_rate_label.setEnabled(False)
            self.frame_rate_edit.setEnabled(False)
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_type_label.setEnabled(False)
            self.image_type_drop.setEnabled(False)
            # Change the button string
            self.data_prop_button.setText('Update Properties')
            # Update the axes
            self.update_axes()
            print(f'Image DPI: {self.mpl_canvas.fig.dpi}')
        else:
            # Disable Preparation Tools
            self.rm_bkgd_checkbox.setEnabled(False)
            self.rm_bkgd_method_label.setEnabled(False)
            self.rm_bkgd_method_drop.setEnabled(False)
            self.bkgd_dark_label.setEnabled(False)
            self.bkgd_dark_edit.setEnabled(False)
            self.bkgd_light_label.setEnabled(False)
            self.bkgd_light_edit.setEnabled(False)
            self.bin_checkbox.setEnabled(False)
            self.bin_drop.setEnabled(False)
            self.filter_checkbox.setEnabled(False)
            self.filter_label_separator.setEnabled(False)
            self.filter_upper_label.setEnabled(False)
            self.filter_upper_edit.setEnabled(False)
            self.drift_checkbox.setEnabled(False)
            self.drift_drop.setEnabled(False)
            self.normalize_checkbox.setEnabled(False)
            self.prep_button.setEnabled(False)
            # Disable Analysis Tools
            self.analysis_drop.setEnabled(False)
            self.start_time_label.setEnabled(False)
            self.start_time_edit.setEnabled(False)
            self.start_time_edit.setText('')
            self.end_time_label.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.end_time_edit.setText('')
            self.map_pushbutton.setEnabled(False)
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.perc_apd_label.setEnabled(False)
            self.perc_apd_edit.setEnabled(False)
            self.perc_apd_edit.setText('')
            # Disable Movie and Signal Tools
            self.signal_select_button.setEnabled(False)
            self.movie_scroll_obj.setEnabled(False)
            self.play_movie_button.setEnabled(False)
            self.export_movie_button.setEnabled(False)
            # self.optical_toggle_button.setEnabled(False)
            # Disable axes controls
            self.axes_start_time_label.setEnabled(False)
            self.axes_start_time_edit.setEnabled(False)
            self.axes_end_time_label.setEnabled(False)
            self.axes_end_time_edit.setEnabled(False)
            # Activate Properties Tools
            self.frame_rate_label.setEnabled(True)
            self.frame_rate_edit.setEnabled(True)
            self.image_scale_label.setEnabled(True)
            self.image_scale_edit.setEnabled(True)
            self.image_type_label.setEnabled(True)
            self.image_type_drop.setEnabled(True)
            # Change the button string
            self.data_prop_button.setText('Start Preparation')

    def run_prep(self):
        # Pass the function to execute
        runner = JobRunner(self.prep_data)
        # Execute
        self.threadpool.start(runner)

    def prep_data(self, progress_callback):
        # Designate that dividing by zero will not generate an error
        np.seterr(divide='ignore', invalid='ignore')
        # Grab unprepped data and check data type to flip if necessary
        if self.image_type_drop.currentIndex() == 0:
            # Membrane potential, flip the data
            self.data_filt = self.data.astype(float)*-1
        else:
            # Calcium transient, don't flip the data
            self.data_filt = self.data.astype(float)
        # Remove background
        if self.rm_bkgd_checkbox.isChecked():
            # Grab the background removal inputs
            rm_method = self.rm_bkgd_method_drop.currentText()
            if rm_method == 'Otsu Global':
                rm_method = 'Otsu_global'
            elif rm_method == 'Random Walk':
                rm_method = 'Random_walk'
            rm_dark = int(self.bkgd_dark_edit.text())
            rm_light = int(self.bkgd_light_edit.text())
            # Generate the mask for background removal using original data
            frame_out, self.mask, markers = mask_generate(
                self.data[0], rm_method, (rm_dark, rm_light))
            # Apply the mask for background removal
            self.data_filt = mask_apply(self.data_filt, self.mask)
        if self.bin_checkbox.isChecked():
            # Grab the kernel size
            bin_kernel = self.bin_drop.currentText()
            if bin_kernel == '3x3':
                bin_kernel = 3
            elif bin_kernel == '5x5':
                bin_kernel = 5
            elif bin_kernel == '7x7':
                bin_kernel = 7
            else:
                bin_kernel = 9
            # Execute spatial filter with selected kernel size
            self.data_filt = filter_spatial_stack(self.data_filt, bin_kernel)
        if self.filter_checkbox.isChecked():
            # Apply the low pass filter
            self.data_filt = filter_temporal(
                self.data_filt, self.data_fps, self.mask, filter_order=100)
        if self.drift_checkbox.isChecked():
            # Grab drift order from dropdown
            drift_order = self.drift_drop.currentIndex()+1
            # Apply drift removal
            self.data_filt = filter_drift(
                self.data_filt, self.mask, drift_order)
        if self.normalize_checkbox.isChecked():
            # Find index of the minimum of each signal
            data_min_ind = np.argmin(self.data_filt, axis=0)
            # Preallocate a variable for collecting minimum values
            data_min = np.zeros(data_min_ind.shape)
            # Grab the number of indices in the time axis (axis=0)
            last_ind = self.data_filt.shape[0]
            # Step through the data
            for n in np.arange(0, self.data_filt.shape[1]):
                for m in np.arange(0, self.data_filt.shape[2]):
                    # Ignore pixels that have been masked out
                    if not self.mask[n, m]:
                        # Check for the leading edge case
                        if data_min_ind[n, m]-10 < 0:
                            data_min[n, m] = np.mean(
                                self.data_filt[0:22, n, m])
                        # Check for the trailing edge case
                        elif data_min_ind[n, m]+11 > last_ind:
                            data_min[n, m] = np.mean(
                                self.data_filt[last_ind-21:last_ind, n, m])
                        # Run assuming all indices are within time indices
                        else:
                            data_min[n, m] = np.mean(
                                self.data_filt[
                                    data_min_ind[n, m]-10:
                                        data_min_ind[n, m]+11,
                                        n, m])
            # Find max amplitude of each signal
            data_diff = np.amax(self.data_filt, axis=0) - data_min
            # Baseline the data
            self.data_filt = self.data_filt-data_min
            # Normalize via broadcasting
            self.data_filt = self.data_filt/data_diff
            # Set normalization flag
            self.norm_flag = 1
        else:
            # Reset normalization flag
            self.norm_flag = 0
        # Update axes
        self.update_axes()
        # Make the movie screen controls available if normalization occurred
        if self.normalize_checkbox.isChecked():
            self.movie_scroll_obj.setEnabled(True)
            self.play_movie_button.setEnabled(True)
            self.export_movie_button.setEnabled(True)
            # self.optical_toggle_button.setEnabled(True)

    def analysis_select(self):
        if self.analysis_drop.currentIndex() == 0:
            # Disable the APD tools
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.perc_apd_label.setEnabled(False)
            self.perc_apd_edit.setEnabled(False)
            self.perc_apd_edit.setText('')
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
        elif self.analysis_drop.currentIndex() == 1:
            # Enable the APD tools
            self.max_apd_label.setEnabled(True)
            self.max_apd_edit.setEnabled(True)
            self.perc_apd_label.setEnabled(True)
            self.perc_apd_edit.setEnabled(True)
            # Disable amplitude and checkboxes
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
        elif self.analysis_drop.currentIndex() == 2:
            self.max_apd_label.setEnabled(False)
            self.max_apd_edit.setEnabled(False)
            self.max_apd_edit.setText('')
            self.max_val_label.setEnabled(True)
            self.max_val_edit.setEnabled(True)
            self.perc_apd_label.setEnabled(False)
            self.perc_apd_edit.setEnabled(False)
            self.perc_apd_edit.setText('')
            # Enable the checkboxes next to populated signal axes
            for cnt, n in enumerate(self.signal_toggle):
                if n == 1:
                    checkboxname = 'ensemble_cb_0{}'.format(cnt+1)
                    checkbox = getattr(self, checkboxname)
                    checkbox.setEnabled(True)

    def run_map(self):
        # Pass the function to execute
        runner = JobRunner(self.map_analysis)
        # Execute
        self.threadpool.start(runner)

    def map_analysis(self, progress_callback):
        # Grab analysis type
        analysis_type = self.analysis_drop.currentIndex()
        # Grab the start and end times
        start_time = float(self.start_time_edit.text())
        end_time = float(self.end_time_edit.text())
        # Find the time index value to which the start entry is closest
        start_ind = abs(self.signal_time-start_time)
        start_ind = np.argmin(start_ind)
        # Find the time index value to which the top entry is closest
        end_ind = abs(self.signal_time-end_time)
        end_ind = np.argmin(end_ind)
        # Calculate activation
        self.act_ind = calc_tran_activation(
            self.data_filt, start_ind, end_ind)
        self.act_val = self.act_ind*(1/self.data_fps)
        max_val = (end_ind-start_ind)*(1/self.data_fps)
        # Generate activation map
        if analysis_type == 0:
            # Generate a map of the activation times
            self.act_map = plt.figure()
            axes_act_map = self.act_map.add_axes([0.05, 0.1, 0.8, 0.8])
            transp = ~self.mask
            transp = transp.astype(float)
            axes_act_map.imshow(self.data[0], cmap='gray')
            axes_act_map.imshow(self.act_val, alpha=transp, vmin=0,
                                vmax=max_val, cmap='jet')
            cax = plt.axes([0.87, 0.12, 0.05, 0.76])
            self.act_map.colorbar(
                cm.ScalarMappable(
                    colors.Normalize(0, max_val),
                    cmap='jet'),
                cax=cax, format='%.3f')
        # Generate action potential duration (APD) map
        if analysis_type == 1:
            # Grab the maximum APD value
            final_apd = float(self.max_apd_edit.text())
            # Find the associated time index
            max_apd_ind = abs(self.signal_time-final_apd)
            max_apd_ind = np.argmin(max_apd_ind)
            # Grab the percent APD
            percent_apd = float(self.perc_apd_edit.text())
            # Find the maximum amplitude of the action potential
            max_amp_ind = np.argmax(
                self.data_filt[
                    start_ind:start_ind+max_apd_ind, :, :], axis=0
                )+start_ind
            # Preallocate variable for percent apd index and value
            apd_ind = np.zeros(max_amp_ind.shape)
            self.apd_val = apd_ind
            # Step through the data
            for n in np.arange(0, self.data_filt.shape[1]):
                for m in np.arange(0, self.data_filt.shape[2]):
                    # Ignore pixels that have been masked out
                    if not self.mask[n, m]:
                        # Grab the data segment between max amp and end
                        tmp = self.data_filt[
                            max_amp_ind[n, m]:start_ind +
                            max_apd_ind, n, m]
                        # Find the minimum to find the index closest to
                        # desired apd percent
                        apd_ind[n, m] = np.argmin(
                            abs(
                                tmp-self.data_filt[max_amp_ind[n, m], n, m] *
                                (1-percent_apd))
                            )+max_amp_ind[n, m]-start_ind
                        # Subtract activation time to get apd
                        self.apd_val[n, m] = (apd_ind[n, m] -
                                              self.act_ind[n, m]
                                              )*(1/self.data_fps)
            # Generate a map of the action potential durations
            self.apd_map = plt.figure()
            axes_apd_map = self.apd_map.add_axes([0.05, 0.1, 0.8, 0.8])
            transp = ~self.mask
            transp = transp.astype(float)
            top = max_apd_ind*(1/self.data_fps)
            axes_apd_map.imshow(self.data[0], cmap='gray')
            axes_apd_map.imshow(self.apd_val, alpha=transp, vmin=0,
                                vmax=top, cmap='jet')
            cax = plt.axes([0.87, 0.1, 0.05, 0.8])
            self.apd_map.colorbar(
                cm.ScalarMappable(
                    colors.Normalize(0, top), cmap='jet'),
                cax=cax, format='%.3f')
        # Generate data for succession of APDs
        if analysis_type == 2:
            # Grab the start and end time, amplitude threshold, and signals
            amp_thresh = float(self.max_val_edit.text())
            # Identify which signals have been selected for calculation
            ensemble_list = [self.ensemble_cb_01.isChecked(),
                             self.ensemble_cb_02.isChecked(),
                             self.ensemble_cb_03.isChecked(),
                             self.ensemble_cb_04.isChecked()]
            ind_analyze = self.signal_coord[ensemble_list, :]
            data_oap = []
            peak_ind = []
            peak_amp = []
            diast_ind = []
            act_ind = []
            apd_val_30 = []
            apd_val_80 = []
            apd_val_tri = []
            tau_fall = []
            f1_f0 = []
            d_f0 = []
            # Iterate through the code
            for idx in np.arange(len(ind_analyze)):
                data_oap.append(
                    self.data_filt[:, self.signal_coord[idx][1],
                                   self.signal_coord[idx][0]])
                # Calculate peak indices
                peak_ind.append(oap_peak_calc(data_oap[idx], start_ind,
                                              end_ind, amp_thresh,
                                              self.data_fps))
                # Calculate peak amplitudes
                peak_amp.append(data_oap[idx][peak_ind[idx]])
                # Calculate end-diastole indices
                diast_ind.append(diast_ind_calc(data_oap[idx],
                                                peak_ind[idx]))
                # Calculate the activation
                act_ind.append(act_ind_calc(data_oap[idx], diast_ind[idx],
                                            peak_ind[idx]))
                # Calculate the APD30
                apd_ind_30 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          0.3)
                apd_val_30.append(self.signal_time[apd_ind_30] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD80
                apd_ind_80 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          0.8)
                apd_val_80.append(self.signal_time[apd_ind_80] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD triangulation
                apd_val_tri.append(apd_val_80[idx]-apd_val_30[idx])
                # Calculate Tau Fall
                tau_fall.append(tau_calc(data_oap[idx], self.data_fps,
                                         peak_ind[idx], diast_ind[idx],
                                         end_ind))
                # Grab raw data, checking the data type to flip if necessary
                if self.image_type_drop.currentIndex() == 0:
                    # Membrane potential, flip the data
                    data = self.data*-1
                else:
                    # Calcium transient, don't flip the data
                    data = self.data
                # Calculate the baseline fluorescence as the average of the
                # first 10 points
                f0 = np.average(data[:11,
                                     self.signal_coord[idx][1],
                                     self.signal_coord[idx][0]])
                # Calculate F1/F0 fluorescent ratio
                f1_f0.append(data[peak_ind[idx],
                                  self.signal_coord[idx][1],
                                  self.signal_coord[idx][0]]/f0)
                # Calculate D/F0 fluorescent ratio
                d_f0.append(data[diast_ind[idx],
                                 self.signal_coord[idx][1],
                                 self.signal_coord[idx][0]]/f0)
            # Open dialogue box for selecting the data directory
            save_fname = QFileDialog.getSaveFileName(
                self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
            # Write results to a spreadsheet
            ensemble_xlsx_print(save_fname[0], self.signal_time, ind_analyze,
                                data_oap, act_ind, peak_ind, tau_fall,
                                apd_val_30, apd_val_80, apd_val_tri, d_f0,
                                f1_f0)

    def signal_select(self):
        # Create placeholders for the x and y coordinates
        self.x = []
        self.y = []
        # Create a button press event
        self.cid = self.mpl_canvas.mpl_connect(
            'button_press_event', self.on_click)

    def update_win(self):
        bot_val = float(self.axes_start_time_edit.text())
        top_val = float(self.axes_end_time_edit.text())
        # Find the time index value to which the bot entry is closest
        bot_ind = abs(self.signal_time-bot_val)
        self.axes_start_ind = np.argmin(bot_ind)
        # Adjust the start time string accordingly
        self.axes_start_time_edit.setText(
            str(self.signal_time[self.axes_start_ind]))
        # Find the time index value to which the top entry is closest
        top_ind = abs(self.signal_time-top_val)
        self.axes_end_ind = np.argmin(top_ind)
        # Adjust the end time string accordingly
        self.axes_end_time_edit.setText(
            str(self.signal_time[self.axes_end_ind]))
        # Update the signal axes
        self.update_axes()

    def update_analysis_win(self):
        # Grab new start time value index and update entry to actual value
        if self.start_time_edit.text():
            bot_val = float(self.start_time_edit.text())
            # Find the time index value to which the bot entry is closest
            bot_ind = abs(self.signal_time-bot_val)
            self.anal_start_ind = np.argmin(bot_ind)
            # Adjust the start time string accordingly
            self.start_time_edit.setText(
                str(self.signal_time[self.anal_start_ind]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_bot_lim = True
        else:
            # Set the start time variable to empty
            self.anal_start_ind = []
            # Set boolean to false so it no longer updates
            self.analysis_bot_lim = False
        # Grab new end time value index and update entry to actual value
        if self.end_time_edit.text():
            top_val = float(self.end_time_edit.text())
            # Find the time index value to which the top entry is closest
            top_ind = abs(self.signal_time-top_val)
            self.anal_end_ind = np.argmin(top_ind)
            # Adjust the end time string accordingly
            self.end_time_edit.setText(
                str(self.signal_time[self.anal_end_ind]))
            # Set boolean to true to signal axes updates accordingly
            self.analysis_top_lim = True
        else:
            # Set the start time variable to empty
            self.anal_end_ind = []
            # Set boolean to false so it no longer updates
            self.analysis_top_lim = False
        if self.max_val_edit.text():
            # Set boolean to true to signal axes updates accordingly
            self.analysis_y_lim = True
        else:
            # Set boolean to false so it no longer updates
            self.analysis_y_lim = False
        # Update the axes accordingly
        self.update_axes()

    def play_movie(self):
        # Grab the current value of the movie scroll bar
        cur_val = self.movie_scroll_obj.value()
        # Grab the maximum value of the movie scroll bar
        max_val = self.movie_scroll_obj.maximum()
        # Pass the function to execute
        self.runner = JobRunner(self.update_frame, (cur_val, max_val))
        self.runner.signals.progress.connect(self.movie_progress)
        # Set or reset the pause variable and activate the pause button
        self.is_paused = False
        self.pause_button.setEnabled(True)
        # Execute
        self.threadpool.start(self.runner)

    def update_frame(self, vals, progress_callback):
        # Start at the current frame and proceed to the end of the file
        for n in np.arange(vals[0]+5, vals[1], 5):
            # Create a minor delay so change is noticeable
            time.sleep(0.5)
            # Emit a signal that will trigger the movie_progress function
            progress_callback.emit(n)
            # If the pause button is hit, break the loop
            if self.is_paused:
                self.pause_button.setEnabled(False)
                break
        # At the end deactivate the pause button
        self.pause_button.setEnabled(False)

    def movie_progress(self, n):
        # Update the scroll bar value, thereby updating the movie screen
        self.movie_scroll_obj.setValue(n)
        # Return the movie screen
        return self.mpl_canvas.fig

    # Function for pausing the movie once the play button has been hit
    def pause_movie(self):
        self.is_paused = True

    # Export movie of ovelayed optical data
    def export_movie(self):
        '''# Set the scroll bar index back to the beginning
        self.movie_scroll_obj.setValue(0)'''
        # Open dialogue box for selecting the file name
        save_fname = QFileDialog.getSaveFileName(
            self, "Save File", os.getcwd(), "mp4 Files (*.mp4)")
        print(f'Filename: {save_fname[0]}')
        # The function for grabbing the video frames
        animation = FuncAnimation(self.mpl_canvas.fig, self.movie_progress,
                                  np.arange(0, self.data.shape[0], 5),
                                  fargs=[], interval=self.data_fps)
        # Execute the function
        animation.save(save_fname[0],
                       dpi=self.mpl_canvas.fig.dpi)

    # ASSIST (I.E., NON-BUTTON) FUNCTIONS
    # Function for grabbing the x and y coordinates of a button click
    def on_click(self, event):
        # Grab the axis coordinates of the click event
        self.signal_coord[self.signal_ind] = [round(event.xdata),
                                              round(event.ydata)]
        self.signal_coord = self.signal_coord.astype(int)
        # Update the toggle variable to indicate points should be plotted
        self.signal_toggle[self.signal_ind] = 1
        # Update the plots accordingly
        self.update_axes()
        # Check the associated check box
        checkboxname = 'ensemble_cb_0{}'.format(self.signal_ind+1)
        checkbox = getattr(self, checkboxname)
        checkbox.setChecked(True)
        # Update the index of the signal for next selection
        if self.signal_ind == 3:
            self.signal_ind = 0
        else:
            self.signal_ind += 1
        # End the button press event
        self.mpl_canvas.mpl_disconnect(self.cid)

    # Function for entering out-of-range values for signal window view
    def sig_win_warn(self, ind):
        # Create a message box to communicate the absence of data
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        if ind == 0:
            msg.setText(
                "Entry must be a numeric value between 0 and {.2f}!".format(
                    self.signal_time[-1]))
        elif ind == 1:
            msg.setText("The Start Time must be less than the End Time!")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # Function for updating the axes
    def update_axes(self):
        # Determine if data is prepped or unprepped
        data = self.data_filt
        # UPDATE THE OPTICAL IMAGE AXIS
        # Clear axis for update
        self.mpl_canvas.axes.cla()
        # Update the UI with an image off the top of the stack
        self.mpl_canvas.axes.imshow(self.data[0], cmap='gray')
        # Match the matplotlib figure background color to the GUI
        self.mpl_canvas.fig.patch.set_facecolor(self.bkgd_color)
        # If normalized, overlay the potential values
        if self.norm_flag == 1:
            # Get the current value of the movie slider
            sig_id = self.movie_scroll_obj.value()
            # Create the transparency mask
            mask = ~self.mask
            thresh = self.data_filt[sig_id, :, :] > 0.3
            transp = mask == thresh
            transp = transp.astype(float)
            # Overlay the voltage on the background image
            self.mpl_canvas.axes.imshow(self.data_filt[sig_id, :, :],
                                        alpha=transp, vmin=0, vmax=1,
                                        cmap='jet')
        # Plot the select signal points
        for cnt, ind in enumerate(self.signal_coord):
            if self.signal_toggle[cnt] == 0:
                continue
            else:
                self.mpl_canvas.axes.scatter(
                    ind[0], ind[1], color=self.cnames[cnt])
        # Tighten the border on the figure
        self.mpl_canvas.fig.tight_layout()
        self.mpl_canvas.draw()
        # UPDATE THE SIGNAL AXES
        # Grab the start and end indices
        start_i = self.axes_start_ind
        end_i = self.axes_end_ind+1
        for cnt, ind in enumerate(self.signal_coord):
            # Grab the canvas's attribute name
            canvasname = 'mpl_canvas_sig{}'.format(cnt+1)
            canvas = getattr(self, canvasname)
            # Clear axis for update
            canvas.axes.cla()
            canvas.draw()
            # Check to see if a signal has been selected
            if int(self.signal_toggle[cnt]) == 1:
                # Plot the signal
                canvas.axes.plot(self.signal_time[start_i:end_i],
                                 data[start_i:end_i, ind[1], ind[0]],
                                 color=self.cnames[cnt])
                # Check to see if normalization has occurred
                if self.normalize_checkbox.isChecked():
                    # Grab the min and max in the y-axis
                    y0 = np.min(data[start_i:end_i, ind[1], ind[0]])-0.05
                    y1 = np.max(data[start_i:end_i, ind[1], ind[0]])+0.05
                    # Get the position of the movie frame
                    x = self.signal_time[self.movie_scroll_obj.value()]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'lime')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                    # Check to see if limits have been established for analysis
                if self.analysis_bot_lim:
                    # Grab the min and max in the y-axis
                    y0 = np.min(data[start_i:end_i, ind[1], ind[0]])-0.05
                    y1 = np.max(data[start_i:end_i, ind[1], ind[0]])+0.05
                    # Get the position of the lower limit marker
                    x = self.signal_time[self.anal_start_ind]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'red')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                if self.analysis_top_lim:
                    # Grab the min and max in the y-axis
                    y0 = np.min(data[start_i:end_i, ind[1], ind[0]])-0.05
                    y1 = np.max(data[start_i:end_i, ind[1], ind[0]])+0.05
                    # Get the position of the lower limit marker
                    x = self.signal_time[self.anal_end_ind]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'red')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                if self.analysis_y_lim:
                    # X-axis bounds
                    x0 = self.signal_time[start_i]
                    x1 = self.signal_time[end_i-1]
                    # Y-axis value
                    y = float(self.max_val_edit.text())
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x0, x1], [y, y], 'red')
                    # Set the x-axis limits
                    canvas.axes.set_xlim(self.signal_time[start_i],
                                         self.signal_time[end_i-1])
                    # Remove the x-axis tick marks
                    canvas.axes.tick_params(labelbottom=False)
                # Set the x-axis limits
                canvas.axes.set_xlim(self.signal_time[start_i],
                                     self.signal_time[end_i-1])
                # If normalized, set the y-axis limits and tick labels
                if self.norm_flag == 1:
                    canvas.axes.set_ylim(-0.3, 1)
                # Tighten the layout
                canvas.fig.tight_layout()
                # Draw the figure
                canvas.draw()


if __name__ == '__main__':
    fig1 = Figure()
    ax1f1 = fig1.add_subplot(111)
    ax1f1.plot(np.random.rand(5))
    # create the GUI application
    app = QApplication(sys.argv)
    # instantiate and show the main window
    ks_main = MainWindow()
    # ks_main.addmpl(fig1)
    ks_main.show()
    # start the Qt main loop execution, exiting from this script
    # with the same return code as the Qt application
    sys.exit(app.exec_())
