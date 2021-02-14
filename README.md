# KairoSight
Python 3.7 software to analyze scientific images across time. 

This project started as a python port of Camat (cardiac mapping analysis tool) and is inspired by design cues and algorithms from RHYTHM and ImageJ.  
 

# Setup (Windows and Pip)
* Clone or download repository.
* From your directory, use pip to install required packages (ensure pip uses python3, or use pip3 if necessary):

	```pip install -r requirements.txt```	

# Setup (Linux/Mac/Win with conda)
* The following should work on all platforms with conda.
* Please check the linux branch for now. The changes have not yet been merged into master.  

```conda install --file requirements[linux_conda].txt```

# Use
* From /src/ start the GUI with:  

    ```python kairosight.py```

# Edit
## User Interface (UI)
* The UI is built with Qt Designer (Version 5.13.0) which, once packages are installed, can be found at:
	
	```venv\Lib\site-packages\pyqt5_tools\Qt\bin\designer.exe```

* The resulting ```.ui``` file must be converted into a ```.py``` file by using the ```puic5``` command from the project's directory. For example:

	```pyuic5 KairoSight_WindowMain.ui > KairoSight_WindowMain.py```

* The primary UI file, ```KairoSight_WindowMain.py```, contains all of the analysis components. ```KairoSight_WindowMDI.py``` is a Multiple-document Interface (MDI) which can contain multiple primary UI windows.