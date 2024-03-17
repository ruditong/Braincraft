# Retina in a box
Tools for interfacing with the "Retina in a box" (Braincraft workshop).

## Installation
1. Clone the repository `git clone https://github.com/ruditong/Braincraft`
2. Extract the files and move to /home/pi/Projects folder (or any folder, but remember to edit the run_experiment.sh file in that case for easy access).
3. Install dependencies by running `pip install -r --no-deps requirements.txt` (do not use pip to install numpy or matplotlib since it will cause an error with the preinstalled version. To fix this problem, run `pip uninstall numpy matplotlib`)
4. Move the run_experiment.sh file to Desktop for shortcut.
5. Run the run_experiment.sh file or directly in shell `python ./gui.py`

## Quick guide
!(https://raw.githubusercontent.com/ruditong/Braincraft/main/Outreach_Handout.png)
