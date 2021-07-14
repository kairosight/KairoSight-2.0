import os
import sys
import traceback
import time
import numpy as np
from matplotlib.animation import FuncAnimation

from util.preparation import (open_stack, mask_generate, mask_apply)
from util.processing import (filter_spatial_stack, filter_temporal,
                             filter_drift)
from util.analysis import (calc_tran_activation, oap_peak_calc, diast_ind_calc,
                           act_ind_calc, apd_ind_calc, tau_calc,
                           ensemble_xlsx_print, signal_data_xlsx_print)

from ui.KairoSight_WindowMain_Retro import Ui_MainWindow
from PyQt5.QtCore import (QObject, pyqtSignal, QRunnable,
                          pyqtSlot, QThreadPool)
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QMessageBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


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
        self.crop_cb.stateChanged.connect(self.crop_enable)
        self.signal_select_button.clicked.connect(self.signal_select)
        self.prep_button.clicked.connect(self.run_prep)
        self.analysis_drop.currentIndexChanged.connect(self.analysis_select)
        self.map_pushbutton.clicked.connect(self.map_analysis)
        self.axes_start_time_edit.editingFinished.connect(self.update_win)
        self.axes_end_time_edit.editingFinished.connect(self.update_win)
        self.export_data_button.clicked.connect(self.export_data_numeric)
        self.start_time_edit.editingFinished.connect(self.update_analysis_win)
        self.end_time_edit.editingFinished.connect(self.update_analysis_win)
        self.max_val_edit.editingFinished.connect(self.update_analysis_win)
        self.movie_scroll_obj.valueChanged.connect(self.update_axes)
        self.play_movie_button.clicked.connect(self.play_movie)
        self.pause_button.clicked.connect(self.pause_movie)
        self.sig1_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig1_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig2_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig2_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig3_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig3_y_edit.editingFinished.connect(self.signal_select_edit)
        self.sig4_x_edit.editingFinished.connect(self.signal_select_edit)
        self.sig4_y_edit.editingFinished.connect(self.signal_select_edit)
        self.export_movie_button.clicked.connect(self.export_movie)
        self.rotate_ccw90_button.clicked.connect(self.rotate_image_ccw90)
        self.rotate_cw90_button.clicked.connect(self.rotate_image_cw90)
        self.crop_xlower_edit.editingFinished.connect(self.crop_update)
        self.crop_xupper_edit.editingFinished.connect(self.crop_update)
        self.crop_ylower_edit.editingFinished.connect(self.crop_update)
        self.crop_yupper_edit.editingFinished.connect(self.crop_update)
        self.rm_bkgd_method_drop.currentIndexChanged.connect(
            self.rm_bkgd_options)

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
        self.sig_disp_bools = [[False, False], [False, False],
                               [False, False], [False, False]]
        self.signal_emit_done = 1
        self.rotate_tracker = 0
        self.preparation_tracker = 0
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
        self.data_prop = self.data
        self.im_bkgd = self.data[0]
        # Populate the axes start and end indices
        self.axes_start_ind = 0
        self.axes_end_ind = self.data.shape[0]-1
        # Populate the mask variable
        self.mask = np.ones([self.data.shape[1], self.data.shape[2]],
                            dtype=bool)
        # Reset the signal selection variables
        self.signal_ind = 0
        self.signal_coord = np.zeros((4, 2)).astype(int)
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
        self.perc_apd_edit_01.setText('')
        self.perc_apd_edit_02.setText('')
        self.axes_end_time_edit.setText('')
        self.axes_start_time_edit.setText('')
        # Update the axes
        self.update_analysis_win()
        # self.update_axes()
        # Enbable Properties Interface
        self.frame_rate_label.setEnabled(True)
        self.frame_rate_edit.setEnabled(True)
        self.image_scale_label.setEnabled(True)
        self.image_scale_edit.setEnabled(True)
        self.data_prop_button.setEnabled(True)
        self.image_type_label.setEnabled(True)
        self.image_type_drop.setEnabled(True)
        self.rotate_label.setEnabled(True)
        self.rotate_ccw90_button.setEnabled(True)
        self.rotate_cw90_button.setEnabled(True)
        self.crop_cb.setEnabled(True)
        self.crop_cb.setChecked(False)
        self.crop_xlower_edit.setText('0')
        self.crop_xupper_edit.setText(str(self.data.shape[2]-1))
        self.crop_xbound = [0, self.data.shape[2]-1]
        self.crop_ylower_edit.setText('0')
        self.crop_yupper_edit.setText(str(self.data.shape[1]-1))
        self.crop_ybound = [0, self.data.shape[1]-1]
        # Enable signal coordinate tools and clear edit boxes
        self.sig1_x_edit.setEnabled(False)
        self.sig1_x_edit.setText('')
        self.sig2_x_edit.setEnabled(False)
        self.sig2_x_edit.setText('')
        self.sig3_x_edit.setEnabled(False)
        self.sig3_x_edit.setText('')
        self.sig4_x_edit.setEnabled(False)
        self.sig4_x_edit.setText('')
        self.sig1_y_edit.setEnabled(False)
        self.sig1_y_edit.setText('')
        self.sig2_y_edit.setEnabled(False)
        self.sig2_y_edit.setText('')
        self.sig3_y_edit.setEnabled(False)
        self.sig3_y_edit.setText('')
        self.sig4_y_edit.setEnabled(False)
        self.sig4_y_edit.setText('')
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
        self.data_prop_button.setText('Save Properties')
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
        self.perc_apd_label_01.setEnabled(False)
        self.perc_apd_edit_01.setEnabled(False)
        self.perc_apd_label_02.setEnabled(False)
        self.perc_apd_edit_02.setEnabled(False)
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
        # Disable axes controls and export buttons
        self.axes_start_time_label.setEnabled(False)
        self.axes_start_time_edit.setEnabled(False)
        self.axes_end_time_label.setEnabled(False)
        self.axes_end_time_edit.setEnabled(False)
        self.export_data_button.setEnabled(False)
        self.export_tracings_button.setEnabled(False)

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
        if self.data_prop_button.text() == 'Save Properties':
            # Populate global variables with frame rate and scale values
            self.data_fps = float(self.frame_rate_edit.text())
            self.data_scale = float(self.image_scale_edit.text())
            # Check data type and flip if necessary
            if self.image_type_drop.currentIndex() == 0:
                # Membrane potential, flip the data
                self.data_prop = self.data.astype(float)*-1
            else:
                # Calcium transient, don't flip the data
                self.data_prop = self.data.astype(float)
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
                canvas.fig.tight_layout()
                canvas.draw()
            # Activate Movie and Signal Tools
            self.signal_select_button.setEnabled(True)
            # Activate Processing Tools
            self.rm_bkgd_checkbox.setEnabled(True)
            self.rm_bkgd_method_label.setEnabled(True)
            self.rm_bkgd_method_drop.setEnabled(True)
            if self.rm_bkgd_method_drop.currentIndex() == 2:
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
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.max_apd_label.setEnabled(True)
                self.max_apd_edit.setEnabled(True)
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
            elif self.analysis_drop.currentIndex() == 2:
                self.max_val_label.setEnabled(True)
                self.max_val_edit.setEnabled(True)
                self.max_apd_label.setEnabled(True)
                self.max_apd_edit.setEnabled(True)
                self.perc_apd_label_01.setEnabled(True)
                self.perc_apd_edit_01.setEnabled(True)
                self.perc_apd_label_02.setEnabled(True)
                self.perc_apd_edit_02.setEnabled(True)
            else:
                self.max_apd_label.setEnabled(False)
                self.max_apd_edit.setEnabled(False)
                self.max_val_label.setEnabled(False)
                self.max_val_edit.setEnabled(False)
                self.perc_apd_label_01.setEnabled(False)
                self.perc_apd_edit_01.setEnabled(False)
                self.perc_apd_label_02.setEnabled(False)
                self.perc_apd_edit_02.setEnabled(False)
            # Activate axes controls
            self.axes_start_time_label.setEnabled(True)
            self.axes_start_time_edit.setEnabled(True)
            self.axes_end_time_label.setEnabled(True)
            self.axes_end_time_edit.setEnabled(True)
            # Activate axes signal selection edit boxes
            axes_on = int(sum(self.signal_toggle)+1)
            for cnt in np.arange(axes_on):
                if cnt == 4:
                    continue
                else:
                    xname = 'sig{}_x_edit'.format(cnt+1)
                    x = getattr(self, xname)
                    x.setEnabled(True)
                    yname = 'sig{}_y_edit'.format(cnt+1)
                    y = getattr(self, yname)
                    y.setEnabled(True)
            # Disable Properties Tools
            self.frame_rate_label.setEnabled(False)
            self.frame_rate_edit.setEnabled(False)
            self.image_scale_label.setEnabled(False)
            self.image_scale_edit.setEnabled(False)
            self.image_type_label.setEnabled(False)
            self.image_type_drop.setEnabled(False)
            self.rotate_label.setEnabled(False)
            self.rotate_ccw90_button.setEnabled(False)
            self.rotate_cw90_button.setEnabled(False)
            # Check for image crop
            if self.crop_cb.isChecked():
                self.data_prop = self.data_prop[:,
                                                self.crop_ybound[0]:
                                                    self.crop_ybound[1]+1,
                                                self.crop_xbound[0]:
                                                    self.crop_xbound[1]+1]
                self.im_bkgd = self.im_bkgd[self.crop_ybound[0]:
                                            self.crop_ybound[1]+1,
                                            self.crop_xbound[0]:
                                            self.crop_xbound[1]+1]
            self.crop_cb.setEnabled(False)
            self.crop_xlower_edit.setEnabled(False)
            self.crop_xupper_edit.setEnabled(False)
            self.crop_ylower_edit.setEnabled(False)
            self.crop_yupper_edit.setEnabled(False)
            # Update preparation tracker
            self.preparation_tracker = 0
            # Change the button string
            self.data_prop_button.setText('Update Properties')
            # Update the axes
            self.update_axes()

        else:
            # Disable Processing Tools
            self.rm_bkgd_checkbox.setChecked(False)
            self.rm_bkgd_checkbox.setEnabled(False)
            self.rm_bkgd_method_label.setEnabled(False)
            self.rm_bkgd_method_drop.setEnabled(False)
            self.rm_bkgd_method_drop.setCurrentIndex(0)
            self.bkgd_dark_label.setEnabled(False)
            self.bkgd_dark_edit.setEnabled(False)
            self.bkgd_light_label.setEnabled(False)
            self.bkgd_light_edit.setEnabled(False)
            self.bin_checkbox.setChecked(False)
            self.bin_checkbox.setEnabled(False)
            self.bin_drop.setCurrentIndex(0)
            self.bin_drop.setEnabled(False)
            self.filter_checkbox.setChecked(False)
            self.filter_checkbox.setEnabled(False)
            self.filter_label_separator.setEnabled(False)
            self.filter_upper_label.setEnabled(False)
            self.filter_upper_edit.setEnabled(False)
            self.drift_checkbox.setChecked(False)
            self.drift_checkbox.setEnabled(False)
            self.drift_drop.setCurrentIndex(0)
            self.drift_drop.setEnabled(False)
            self.normalize_checkbox.setChecked(False)
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
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
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
            self.sig1_x_edit.setEnabled(False)
            self.sig1_y_edit.setEnabled(False)
            self.sig2_x_edit.setEnabled(False)
            self.sig2_y_edit.setEnabled(False)
            self.sig3_x_edit.setEnabled(False)
            self.sig3_y_edit.setEnabled(False)
            self.sig4_x_edit.setEnabled(False)
            self.sig4_y_edit.setEnabled(False)
            # Reset signal variables
            self.signal_ind = 0
            self.signal_coord = np.zeros((4, 2)).astype(int)
            self.signal_toggle = np.zeros((4, 1))
            self.norm_flag = 0
            self.play_bool = 0
            # Reset analysis tools
            self.start_time_edit.setText('')
            self.end_time_edit.setText('')
            self.max_apd_edit.setText('')
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.axes_end_time_edit.setText('')
            self.axes_start_time_edit.setText('')
            # Activate Properties Tools
            self.frame_rate_label.setEnabled(True)
            self.frame_rate_edit.setEnabled(True)
            self.image_scale_label.setEnabled(True)
            self.image_scale_edit.setEnabled(True)
            self.image_type_label.setEnabled(True)
            self.image_type_drop.setEnabled(True)
            self.rotate_label.setEnabled(True)
            self.rotate_ccw90_button.setEnabled(True)
            self.rotate_cw90_button.setEnabled(True)
            # Check for image crop
            if self.crop_cb.isChecked():
                self.data = self.video_data_raw[0]
                self.im_bkgd = self.data[0]
            if self.rotate_tracker != 0:
                self.data = np.rot90(self.data,
                                     k=self.rotate_tracker,
                                     axes=(1, 2))
                self.im_bkgd = np.rot90(self.im_bkgd,
                                        k=self.rotate_tracker,
                                        axes=(1, 2))
            self.crop_cb.setEnabled(True)
            self.crop_xlower_edit.setEnabled(True)
            self.crop_xupper_edit.setEnabled(True)
            self.crop_ylower_edit.setEnabled(True)
            self.crop_yupper_edit.setEnabled(True)
            # Change the button string
            self.data_prop_button.setText('Save Properties')
            # Update the axes
            self.update_analysis_win()
            self.update_axes()

    def run_prep(self):
        # Pass the function to execute
        runner = JobRunner(self.prep_data)
        # Execute
        self.threadpool.start(runner)

    def prep_data(self, progress_callback):
        # Designate that dividing by zero will not generate an error
        np.seterr(divide='ignore', invalid='ignore')
        # Grab unprepped data and check data type to flip if necessary
        self.data_filt = self.data_prop

        # Remove background
        if self.rm_bkgd_checkbox.isChecked():
            rm_bkgd_timestart = time.process_time()
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
                self.data_filt[0], rm_method, (rm_dark, rm_light))
            # Flip if the signal is voltage
            if self.image_type_drop.currentIndex() == 0:
                self.mask = ~self.mask
            # Look for saturated signals by comparing the first 50 indices to
            # the first index, if they are all equal the signal is saturated
            is_sat = sum(
                self.data_filt[0:50, :, :] == self.data_filt[0, :, :]) == 50
            # Remove saturated signals from the mask
            print(self.mask)
            self.mask[is_sat] = False
            # Apply the mask for background removal
            self.data_filt = mask_apply(self.data_filt,
                                        self.mask,
                                        1)
            rm_bkgd_timeend = time.process_time()
            print(
                f'Remove Background Time: {rm_bkgd_timeend-rm_bkgd_timestart}')

        # Spatial filter
        if self.bin_checkbox.isChecked():
            bin_timestart = time.process_time()
            # Grab the kernel size
            bin_kernel = self.bin_drop.currentText()
            if bin_kernel == '3x3':
                bin_kernel = 3
            elif bin_kernel == '5x5':
                bin_kernel = 5
            elif bin_kernel == '7x7':
                bin_kernel = 7
            elif bin_kernel == '9x9':
                bin_kernel = 9
            elif bin_kernel == '15x15':
                bin_kernel = 15
            elif bin_kernel == '21x21':
                bin_kernel = 21
            else:
                bin_kernel = 31
            # Execute spatial filter with selected kernel size
            self.data_filt = filter_spatial_stack(self.data_filt, bin_kernel)
            bin_timeend = time.process_time()
            print(
                f'Binning Time: {bin_timeend-bin_timestart}')

        # Temporal filter
        if self.filter_checkbox.isChecked():
            filter_timestart = time.process_time()
            # Apply the low pass filter
            # print(self.filter_upper_edit.text)
            self.data_filt = filter_temporal(
                self.data_filt, self.data_fps, self.mask, filter_order=100,
                freq_cutoff=float(self.filter_upper_edit.text()))
            filter_timeend = time.process_time()
            print(
                f'Filter Time: {filter_timeend-filter_timestart}')

        # Drift Removal
        if self.drift_checkbox.isChecked():
            drift_timestart = time.process_time()
            # Grab drift order from dropdown
            drift_order = self.drift_drop.currentIndex()+1
            # Apply drift removal
            self.data_filt = filter_drift(
                self.data_filt, self.mask, drift_order)
            drift_timeend = time.process_time()
            print(f'Drift Time: {drift_timeend-drift_timestart}')

        # Normalization
        if self.normalize_checkbox.isChecked():
            normalize_timestart = time.process_time()
            # Find index of the minimum of each signal
            data_min_ind = np.argmin(self.data_filt, axis=0)
            # Preallocate a variable for collecting minimum values
            data_min = np.zeros(data_min_ind.shape)
            # Grab the number of indices in the time axis (axis=0)
            last_ind = self.data_filt.shape[0]
            # Ignore pixels that have been masked out
            # if self.image_type_drop.currentIndex() == 0:
            mask = self.mask
            # else:
            #    mask = ~self.mask
            # Step through the data
            for n in np.arange(0, self.data_filt.shape[1]):
                for m in np.arange(0, self.data_filt.shape[2]):
                    # Grab a region of indices around the signal minimum
                    if mask[n, m]:
                        data_min[n, m] = self.data_filt[
                            data_min_ind[n, m], n, m]
                        # Check for the leading edge case
                        if data_min_ind[n, m]-2 < 0:
                            data_seg = self.data_filt[0:6, n, m]
                        # Check for the trailing edge case
                        elif data_min_ind[n, m]+2 > last_ind:
                            data_seg = self.data_filt[
                                last_ind-4:last_ind+1, n, m]
                        # Run assuming all indices are within time indices
                        else:
                            data_seg = self.data_filt[
                                data_min_ind[n, m]-2:data_min_ind[n, m]+2,
                                n,
                                m]
                        # Grab all values less than the maximum of this segment
                        seg_bool = self.data_filt[:, n, m] < max(data_seg)
                        # Create a vector of the index values of the signal
                        ind = np.arange(0, self.data_filt.shape[0])
                        # Pull out all of the low value chunks
                        ind = ind[seg_bool]
                        # Find the points of separation in the chunks
                        # (i.e., index steps > 1)
                        ind_seg = ind[1:] - ind[0:len(ind)-1]
                        ind_seg = np.append(ind_seg, 1)
                        # Create a vector for the gaps
                        gap = ind_seg != 1
                        # Create a vector for the last index in each segment
                        last = ind[gap]
                        last = np.append(last, ind[-1])
                        # Create a vector for the first index in each segment
                        first = ind[np.roll(gap, 1)]
                        first = np.insert(first, 0, 0)
                        # Preallocate variable for the min values of each chunk
                        test_min_all = np.zeros(len(first))
                        # Step through each chunk extracting the minimum value
                        for x in np.arange(0, len(test_min_all)):
                            seg = self.data_filt[first[x]:last[x]+1, n, m]
                            if seg.size == 0:
                                test_min_all[x] = np.nan
                            else:
                                test_min_all[x] = np.min(seg)
                        test_min_all = np.delete(
                            test_min_all,
                            [i for i, x in enumerate(np.isnan(test_min_all)) if x])
                        # Calculate the average minimum
                        if test_min_all.size == 0:
                            continue
                        else:
                            data_min[n, m] = np.mean(test_min_all)
            # Baseline the signals to near zero using the average minimum
            self.data_filt = self.data_filt - data_min
            # Normalize using the maximum value of the signal
            self.data_filt = self.data_filt/np.amax(self.data_filt, axis=0)
            # Set normalization flag
            self.norm_flag = 1
            normalize_timeend = time.process_time()
            print(
                f'Normalize Time: {normalize_timeend-normalize_timestart}')
        else:
            # Reset normalization flag
            self.norm_flag = 0
        # Update preparation tracker
        self.preparation_tracker = 1
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
            self.perc_apd_label_01.setEnabled(False)
            self.perc_apd_edit_01.setEnabled(False)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            self.analysis_y_lim = False
            self.ensemble_cb_01.setEnabled(False)
            self.ensemble_cb_02.setEnabled(False)
            self.ensemble_cb_03.setEnabled(False)
            self.ensemble_cb_04.setEnabled(False)
        elif self.analysis_drop.currentIndex() == 1:
            # Enable the APD tools
            self.max_apd_label.setEnabled(True)
            self.max_apd_edit.setEnabled(True)
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            self.perc_apd_label_02.setEnabled(False)
            self.perc_apd_edit_02.setEnabled(False)
            # Disable amplitude and checkboxes
            self.max_val_label.setEnabled(False)
            self.max_val_edit.setEnabled(False)
            self.max_val_edit.setText('')
            self.analysis_y_lim = False
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
            self.perc_apd_label_01.setEnabled(True)
            self.perc_apd_edit_01.setEnabled(True)
            self.perc_apd_label_02.setEnabled(True)
            self.perc_apd_edit_02.setEnabled(True)
            self.perc_apd_edit_01.setText('')
            self.perc_apd_edit_02.setText('')
            # Enable the checkboxes next to populated signal axes
            for cnt, n in enumerate(self.signal_toggle):
                if n == 1:
                    checkboxname = 'ensemble_cb_0{}'.format(cnt+1)
                    checkbox = getattr(self, checkboxname)
                    checkbox.setEnabled(True)
        # Update the axes accordingly
        self.update_axes()

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
        # Grab masking information
        '''if self.image_type_drop.currentIndex() == 0:
            transp = ~self.mask
        else:'''
        transp = self.mask
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
            '''if self.image_type_drop.currentIndex() == 0:
                transp = ~self.mask
            else:
                transp = self.mask'''
            transp = transp.astype(float)
            axes_act_map.imshow(self.data[0,
                                          self.crop_ybound[0]:
                                              self.crop_ybound[1]+1,
                                          self.crop_xbound[0]:
                                              self.crop_xbound[1]+1],
                                cmap='gray')
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
            max_apd_ind = abs(self.signal_time-(final_apd+start_time))
            max_apd_ind = np.argmin(max_apd_ind)
            # Grab the percent APD
            percent_apd = float(self.perc_apd_edit_01.text())
            # Find the maximum amplitude of the action potential
            max_amp_ind = np.argmax(
                self.data_filt[
                    start_ind:max_apd_ind, :, :], axis=0
                )+start_ind
            # Preallocate variable for percent apd index and value
            apd_ind = np.zeros(max_amp_ind.shape)
            self.apd_val = np.zeros(max_amp_ind.shape)
            # Step through the data
            for n in np.arange(0, self.data_filt.shape[1]):
                for m in np.arange(0, self.data_filt.shape[2]):
                    # Ignore pixels that have been masked out
                    if transp[n, m]:
                        # Grab the data segment between max amp and end
                        tmp = self.data_filt[
                            max_amp_ind[n, m]:max_apd_ind, n, m]
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
            '''transp = ~self.mask'''
            transp = transp.astype(float)
            top = self.signal_time[max_apd_ind]-self.signal_time[start_ind]
            axes_apd_map.imshow(self.data[0,
                                          self.crop_ybound[0]:
                                              self.crop_ybound[1]+1,
                                          self.crop_xbound[0]:
                                              self.crop_xbound[1]+1],
                                cmap='gray')
            axes_apd_map.imshow(self.apd_val, alpha=transp, vmin=0,
                                vmax=top, cmap='jet')
            cax = plt.axes([0.87, 0.1, 0.05, 0.8])
            self.apd_map.colorbar(
                cm.ScalarMappable(
                    colors.Normalize(0, top), cmap='jet'),
                cax=cax, format='%.3f')
        # Generate data for succession of APDs
        if analysis_type == 2:
            # Grab the amplitude threshold, apd values and signals
            amp_thresh = float(self.max_val_edit.text())
            apd_input_01 = float(self.perc_apd_edit_01.text())
            apd_input_02 = float(self.perc_apd_edit_02.text())
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
            apd_val_01 = []
            apd_val_02 = []
            apd_val_tri = []
            tau_fall = []
            f1_f0 = []
            d_f0 = []
            # Iterate through the code
            for idx in np.arange(len(ind_analyze)):
                data_oap.append(
                    self.data_filt[:, ind_analyze[idx][1],
                                   ind_analyze[idx][0]])
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
                apd_ind_01 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_01)
                apd_val_01.append(self.signal_time[apd_ind_01] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD80
                apd_ind_02 = apd_ind_calc(data_oap[idx], end_ind,
                                          diast_ind[idx], peak_ind[idx],
                                          apd_input_02)
                apd_val_02.append(self.signal_time[apd_ind_02] -
                                  self.signal_time[act_ind[idx]])
                # Calculate APD triangulation
                apd_val_tri.append(apd_val_02[idx]-apd_val_01[idx])
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
                                     ind_analyze[idx][1],
                                     ind_analyze[idx][0]])
                # Calculate F1/F0 fluorescent ratio
                f1_f0.append(data[peak_ind[idx],
                                  ind_analyze[idx][1],
                                  ind_analyze[idx][0]]/f0)
                # Calculate D/F0 fluorescent ratio
                d_f0.append(data[diast_ind[idx],
                                 ind_analyze[idx][1],
                                 ind_analyze[idx][0]]/f0)
            # Open dialogue box for selecting the data directory
            save_fname = QFileDialog.getSaveFileName(
                self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
            # Write results to a spreadsheet
            ensemble_xlsx_print(save_fname[0], self.signal_time, ind_analyze,
                                data_oap, act_ind, peak_ind, tau_fall,
                                apd_input_01, apd_val_01, apd_input_02,
                                apd_val_02, apd_val_tri, d_f0, f1_f0,
                                self.image_type_drop.currentIndex())

    def export_data_numeric(self):
        # Determine if data is prepped or unprepped
        if self.preparation_tracker == 0:
            data = self.data_prop
        else:
            data = self.data_filt
        # Grab oaps
        data_oap = []
        for idx in np.arange(0, 4):
            if self.signal_toggle[idx] == 1:
                data_oap.append(
                    data[:, self.signal_coord[idx, 1],
                         self.signal_coord[idx, 0]])
        # Open dialogue box for selecting the data directory
        save_fname = QFileDialog.getSaveFileName(
            self, "Save File", os.getcwd(), "Excel Files (*.xlsx)")
        # Write results to a spreadsheet
        signal_data_xlsx_print(save_fname[0], self.signal_time, data_oap,
                               self.signal_coord, self.data_fps)

    def export_data_tracing(self):
        pass

    def signal_select(self):
        # Create placeholders for the x and y coordinates
        self.x = []
        self.y = []
        # Create a button press event
        self.cid = self.mpl_canvas.mpl_connect(
            'button_press_event', self.on_click)

    def signal_select_edit(self):
        if self.signal_emit_done == 1:
            # Update the tracker to negative (i.e., 0) and continue
            self.signal_emit_done = 0
            # Grab all of the values and make sure they are integer values
            for n in np.arange(4):
                # Create iteration names for the x and y structures
                xname = 'sig{}_x_edit'.format(n+1)
                x = getattr(self, xname)
                yname = 'sig{}_y_edit'.format(n+1)
                y = getattr(self, yname)
                # Check to see if there is an empty edit box in the pair
                if x.text() == '' or y.text() == '':
                    continue
                else:
                    # Make sure the entered values are numeric
                    try:
                        new_x = int(x.text())
                    except ValueError:
                        self.sig_win_warn(3)
                        x.setText(str(self.signal_coord[n][0]))
                        self.signal_emit_done = 1
                        break
                    try:
                        new_y = int(y.text())
                    except ValueError:
                        self.sig_win_warn(3)
                        y.setText(str(self.signal_coord[n][1]))
                        self.signal_emit_done = 1
                        break
                    # Grab the current string values and convert to integers
                    coord_ints = [new_x, new_y]
                    # Check to make sure the coordinates are within range
                    if coord_ints[0] < 0 or (
                            coord_ints[0] >= self.data.shape[2]):
                        self.sig_win_warn(2)
                        x.setText(str(self.signal_coord[n][0]))
                        self.signal_emit_done = 1
                        break
                    elif coord_ints[1] < 0 or (
                            coord_ints[1] >= self.data.shape[1]):
                        self.sig_win_warn(2)
                        y.setText(str(self.signal_coord[n][1]))
                        self.signal_emit_done = 1
                        break
                    # Place integer values in global signal coordinate variable
                    self.signal_coord[n] = coord_ints
                    # Convert integers to strings and update the edit boxes
                    x.setText(str(coord_ints[0]))
                    y.setText(str(coord_ints[1]))
                    # Make sure the axes is toggled on for plotting
                    self.signal_toggle[n] = 1
                    # Make sure the APD Ensemble check box is enabled
                    cb_name = 'ensemble_cb_0{}'.format(n+1)
                    cb = getattr(self, cb_name)
                    cb.setChecked(True)
                    # Check to see if the next edit boxes should be toggled on
                    if sum(self.signal_toggle) < 4:
                        # Grab the number of active axes
                        act_ax = int(sum(self.signal_toggle))
                        # Activate the next set of edit boxes
                        xname = 'sig{}_x_edit'.format(act_ax+1)
                        x = getattr(self, xname)
                        x.setEnabled(True)
                        yname = 'sig{}_y_edit'.format(act_ax+1)
                        y = getattr(self, yname)
                        y.setEnabled(True)
                        # Update the select signal button index
                        self.signal_ind = int(sum(self.signal_toggle))
            # Update the axes
            self.update_axes()
            self.signal_emit_done = 1

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
        # Grab new max amplitude value and update entry to actual value
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
        # The function for grabbing the video frames
        animation = FuncAnimation(self.mpl_canvas.fig, self.movie_progress,
                                  np.arange(0, self.data.shape[0], 5),
                                  fargs=[], interval=self.data_fps)
        # Execute the function
        animation.save(save_fname[0],
                       dpi=self.mpl_canvas.fig.dpi)

    # Rotate image 90 degrees counterclockwise function
    def rotate_image_ccw90(self):
        # Rotate the data 90 degress counterclockwise
        self.data = np.rot90(self.data, k=1, axes=(1, 2))
        # Rotate the bacground image 90 degrees counterclockwise
        self.im_bkgd = np.rot90(self.im_bkgd, k=1)
        # Update variable for tracking rotation
        if self.rotate_tracker < 3:
            self.rotate_tracker += 1
        else:
            self.rotate_tracker = 0
        # Swap crop box values
        self.crop_bound_rot()
        # Update cropping strings according to new image dimensions
        self.crop_update()

    # Rotate image 90 degress clockwise function
    def rotate_image_cw90(self):
        # Rotate the data 90 degress clockwise
        self.data = np.rot90(self.data, k=-1, axes=(1, 2))
        # Rotate the bacground image 90 degrees clockwise
        self.im_bkgd = np.rot90(self.im_bkgd, k=-1)
        # Update variable for tracking rotation
        if self.rotate_tracker == 0:
            self.rotate_tracker = 3
        else:
            self.rotate_tracker -= 1
        # Swap crop box values
        self.crop_bound_rot()
        # Update cropping strings according to new image dimensions
        self.crop_update()

    # Rotate cropping bounding box
    def crop_bound_rot(self):
        # Grab the current values
        new_x = [int(self.crop_ylower_edit.text()),
                 int(self.crop_yupper_edit.text())]
        new_y = [int(self.crop_xlower_edit.text()),
                 int(self.crop_xupper_edit.text())]
        # Replace x values
        self.crop_xlower_edit.setText(str(new_x[0]))
        self.crop_xupper_edit.setText(str(new_x[1]))
        # Replace y values
        self.crop_ylower_edit.setText(str(new_y[0]))
        self.crop_yupper_edit.setText(str(new_y[1]))

    # Enable the crop limit boxes
    def crop_enable(self):
        if self.crop_cb.isChecked():
            # Enable the labels and edit boxes for cropping
            self.crop_xlabel.setEnabled(True)
            self.crop_xlower_edit.setEnabled(True)
            self.crop_xupper_edit.setEnabled(True)
            self.crop_ylabel.setEnabled(True)
            self.crop_ylower_edit.setEnabled(True)
            self.crop_yupper_edit.setEnabled(True)
            # Update the axes
            self.update_axes()
        else:
            # Disable the labesl and edit boxes for cropping
            self.crop_xlabel.setEnabled(False)
            self.crop_xlower_edit.setEnabled(False)
            self.crop_xupper_edit.setEnabled(False)
            self.crop_ylabel.setEnabled(False)
            self.crop_ylower_edit.setEnabled(False)
            self.crop_yupper_edit.setEnabled(False)

    def crop_update(self):
        if self.signal_emit_done == 1:
            # Create variable for stopping double tap
            self.signal_emit_done = 0
            # Check to make sure the x coordinates are within the image bounds
            try:
                new_x = [int(self.crop_xlower_edit.text()),
                         int(self.crop_xupper_edit.text())]
            except ValueError:
                self.sig_win_warn(3)
                self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
            else:
                # Update the bounds of the crop box
                if (new_x[0] < 0 or new_x[0] > self.data.shape[2] or
                        new_x[1] < 0 or new_x[1] > self.data.shape[2]):
                    self.sig_win_warn(2)
                    self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                    self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
                elif new_x[0] >= new_x[1]:
                    self.sig_win_warn(4)
                    self.crop_xlower_edit.setText(str(self.crop_xbound[0]))
                    self.crop_xupper_edit.setText(str(self.crop_xbound[1]))
                else:
                    self.crop_xbound = [new_x[0], new_x[1]]
            # Check to make sure the y coordinates are within the image bounds
            try:
                new_y = [int(self.crop_ylower_edit.text()),
                         int(self.crop_yupper_edit.text())]
            except ValueError:
                self.sig_win_warn(3)
                self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
            else:
                # Update the bounds of the crop box
                if (new_y[0] < 0 or new_y[0] > self.data.shape[1] or
                        new_y[1] < 0 or new_y[1] > self.data.shape[1]):
                    self.sig_win_warn(2)
                    self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                    self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
                elif new_y[0] >= new_y[1]:
                    self.sig_win_warn(4)
                    self.crop_ylower_edit.setText(str(self.crop_ybound[0]))
                    self.crop_yupper_edit.setText(str(self.crop_ybound[1]))
                else:
                    self.crop_ybound = [new_y[0], new_y[1]]
            # Update the image axis
            self.update_axes()
            # Indicate function has ended
            self.signal_emit_done = 1

    def rm_bkgd_options(self):
        if self.rm_bkgd_method_drop.currentIndex() == 2:
            self.bkgd_dark_label.setEnabled(True)
            self.bkgd_dark_edit.setEnabled(True)
            self.bkgd_light_label.setEnabled(True)
            self.bkgd_light_edit.setEnabled(True)
        else:
            self.bkgd_dark_label.setEnabled(False)
            self.bkgd_dark_edit.setEnabled(False)
            self.bkgd_light_label.setEnabled(False)
            self.bkgd_light_edit.setEnabled(False)

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
        # Populate the signal coordinate edit boxes
        sigx_name = 'sig{}_x_edit'.format(self.signal_ind+1)
        sigx = getattr(self, sigx_name)
        sigx.setText(str(self.signal_coord[self.signal_ind][0]))
        sigy_name = 'sig{}_y_edit'.format(self.signal_ind+1)
        sigy = getattr(self, sigy_name)
        sigy.setText(str(self.signal_coord[self.signal_ind][1]))
        # Check to see if the next edit boxes should be toggled on
        if sum(self.signal_toggle) < 4:
            # Grab the number of active axes
            act_ax = int(sum(self.signal_toggle))
            # Activate the next set of edit boxes
            xname = 'sig{}_x_edit'.format(act_ax+1)
            x = getattr(self, xname)
            x.setEnabled(True)
            yname = 'sig{}_y_edit'.format(act_ax+1)
            y = getattr(self, yname)
            y.setEnabled(True)
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
        elif ind == 2:
            msg.setText("Entered coordinates outside image dimensions!")
        elif ind == 3:
            msg.setText("Entered value must be numeric!")
        elif ind == 4:
            msg.setText("Lower limit must be less than upper limit!")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # Function for updating the axes
    def update_axes(self):
        # UPDATE THE IMAGE AXIS
        # Determine if data is prepped or unprepped
        if self.preparation_tracker == 0:
            data = self.data_prop
        else:
            data = self.data_filt
        # UPDATE THE OPTICAL IMAGE AXIS
        # Clear axis for update
        self.mpl_canvas.axes.cla()
        # Update the UI with an image off the top of the stack
        self.mpl_canvas.axes.imshow(self.im_bkgd, cmap='gray')
        # Match the matplotlib figure background color to the GUI
        self.mpl_canvas.fig.patch.set_facecolor(self.bkgd_color)
        # If normalized, overlay the potential values
        if self.norm_flag == 1:
            # Get the current value of the movie slider
            sig_id = self.movie_scroll_obj.value()
            # Create the transparency mask
            mask = self.mask
            thresh = self.data_filt[sig_id, :, :] > 0.3
            transp = mask == thresh
            transp = transp.astype(float)
            # Overlay the voltage on the background image
            self.mpl_canvas.axes.imshow(self.data_filt[sig_id, :, :],
                                        alpha=transp, vmin=0, vmax=1,
                                        cmap='jet')
        # Check to see if signals have been selected and activate export tools
        if self.signal_ind != 1:
            self.export_data_button.setEnabled(True)
            self.export_tracings_button.setEnabled(True)
        # Plot the select signal points
        for cnt, ind in enumerate(self.signal_coord):
            if self.signal_toggle[cnt] == 0:
                continue
            else:
                self.mpl_canvas.axes.scatter(
                    ind[0], ind[1], color=self.cnames[cnt])
        # Check to see if crop is being utilized
        if self.crop_cb.isChecked():
            if self.data_prop_button.text() == 'Save Properties':
                # Plot vertical sides of bounding box
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[0]],
                    [self.crop_ybound[0], self.crop_ybound[1]],
                    color='orange')
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[1], self.crop_xbound[1]],
                    [self.crop_ybound[0], self.crop_ybound[1]],
                    color='orange')
                # Plot horizontal sides of bounding box
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[1]],
                    [self.crop_ybound[0], self.crop_ybound[0]],
                    color='orange')
                self.mpl_canvas.axes.plot(
                    [self.crop_xbound[0], self.crop_xbound[1]],
                    [self.crop_ybound[1], self.crop_ybound[1]],
                    color='orange')
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
                # Grab the min and max in the y-axis
                y0 = np.min(data[start_i:end_i, ind[1], ind[0]])-0.05
                y1 = np.max(data[start_i:end_i, ind[1], ind[0]])+0.05
                # Check for NAN values
                if np.isnan(y0) or np.isnan(y1):
                    y0 = -1.0
                    y1 = 1.0
                # Set y-axis limits
                canvas.axes.set_ylim(y0, y1)
                # Check to see if normalization has occurred
                if self.normalize_checkbox.isChecked():
                    # Get the position of the movie frame
                    x = self.signal_time[self.movie_scroll_obj.value()]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'lime')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                # Check to see if limits have been established for analysis
                if self.analysis_bot_lim:
                    # Get the position of the lower limit marker
                    x = self.signal_time[self.anal_start_ind]
                    # Overlay the frame location of the play feature
                    canvas.axes.plot([x, x], [y0, y1], 'red')
                    # Set the y-axis limits
                    canvas.axes.set_ylim(y0, y1)
                if self.analysis_top_lim:
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
                # Set the x-axis limits
                canvas.axes.set_xlim(self.signal_time[start_i],
                                     self.signal_time[end_i-1])
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
