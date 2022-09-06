# KairoSight 2.0
Python 3.8 software to analyze time series physiological data of optical action potentials.

This project started as a python port of Camat (cardiac mapping analysis tool, PMCID: [PMC4935510](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4935510/)) and is inspired by design cues and algorithms from RHYTHM (PMCID: [PMC5811559](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811559/)) and ImageJ (PMCID: [PMC5554542](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5554542/)). It is important to note that Kairosight 2.0 is currently only available on Windows machines.
 
In order to get up and running with Kairosight 2.0 you will need to complete the following set up steps:
1. First you will need to install Anaconda, which can be found [here](https://docs.anaconda.com/anaconda/install/windows/).
2. Clone or download the repository
3. On a Windows PC follow steps 4-6, for Mac skip to steps 7-10.
4. Open the Anaconda Prompt and navigate to the directory where you cloned/downloaded the repository (e.g., "cd OneDrive\Documents\GitHub\kairosight-2.0")
5. Enter the following command to setup the Anaconda environment: `conda env create -f kairosight_env.yml`
6. Close the Anaconda Prompt
7. Open terminal
8. Enter the following command: 'conda create -n kairosight_env.yml'
9. Proceed with 'y'
10. Enter the following command: 'conda activate kairosight_env.yml'
12. Launch Anaconda Navigator and switch to the newly created environment
 --On a Mac, once Anaconda Navigator is open, navigate to environments<select 'installed' drop down and change it to 'all'<search for 'pyqt'<check pyqtwebengine and follow prompts for this change to be applied.
13. Launch Spyder
14. In the top menu select: Tools -> Preferences
15. Select IPython console on the left hand menu
16. Select the Graphics tab and make sure the Graphics backend is set to Qt5
17. Open kairosight_retro.py and hit the play button 
18. If an error occurs that states "No module named 'insertmodulenamehere', simply type "conda install insertmodulenamehere", this should update the packages and allow the program to run.
  --numpy
  --matplotlib
  --imagio
  --xlsxwriter
  --for skimage use (pip install scikit-image)
  --for cv2 use (conda install opencv) 
