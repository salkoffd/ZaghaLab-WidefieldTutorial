# ZaghaLab-WidefieldTutorial
This tutorial showcases the tools used for pre-processing widefield videos, projecting into low dimension, creating a design matrix, and running multiple regression on neural/behavioral data.

- Ridge regression functions can be downloaded from https://github.com/churchlandlab/ridgeModel
- Example widefield/behavioral data can be downloaded from [here](https://drive.google.com/drive/folders/1pkWcg_YW7DrdqnO32IvYPy1QJSaZwqYg?usp=sharing)
    - The script will look for experimental data inside a folder "190204" inside another folder "GSS+03".
- Before you run the script, open it and enter the file path for your data (variable is dataFolder, first line of code). It should be the parent folder of GSS+03
- Make sure to add any new functions to the MATLAB path.
- Script works on MATLAB 2018a and later.


**Usage**
-----

You can run this script by entering:

    WidefieldTutorial

into the MATLAB command window.

-----
![Analysis Pipeline](https://i.imgur.com/HovpEhs.png)

**Figures Preview**
-----
Hit trial activity
![Hit Trial Activity](https://i.imgur.com/YAdpT1g.png)
Regression Model Comparison
![Regression Model Comparison](https://i.imgur.com/QYClUYU.png)
