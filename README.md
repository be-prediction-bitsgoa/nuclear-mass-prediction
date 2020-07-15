# __**IMPORTANT NOTE: **__ If you are not able to clone this repository, please download the files from [this Google drive link](https://drive.google.com/drive/folders/1EWUot993Ci-BxP99V8DFrNbzKTuLLLaf?usp=sharing) and proceed with the further steps.

# nuclear-mass-prediction

## Installation


### Since these files are large, [GIT LFS](https://git-lfs.github.com/) must be initialized before cloning.

1. You must have Python3 (64-bit) running on your system. ([link](https://www.python.org/downloads/) to download python) - Make sure that 64-bit Python is installed. The program will not run on 32-bit python.

2. You must have pip downloaded ([link](https://pip.pypa.io/en/stable/installing/) to download pip)

3. Clone this repository

`git clone https://github.com/be-prediction-bitsgoa/nuclear-mass-prediction.git`

4. Go to the folder on your local machine and open terminal (for ubuntu) or command prompt (for windows) in that directory.

5. Install all dependencies for the program using beow command

`pip install -r requirements.txt`

6. Create a .csv file in the folder, and put your test data into the csv file (csv file should be Comma delimited ONLY).

7. Make sure the file is present in the same folder as the driver.py script

8. Make sure file does not contain header row. File must only contain 2 columns. First column should contain Z values. Second column should contain N values.

9. run the driver script using command `py driver.py` for windows and `python driver.py` for ubuntu.

10. Enter the name of your .csv file after the prompt and Hit Enter

11. Predictions will be displayed on the command line in tabular format.
