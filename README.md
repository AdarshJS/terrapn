## Code Usage

This is the main working code and model for TerraPN, the paper found here: https://ieeexplore.ieee.org/document/9981942

# Steps to follow:

0. Install ROS Melodic.

1. Setup the conda environment using terrapn.yaml inside the conda folder.

2. Setup RGB image stream from a camera. outdoor_dwa contains the ros callback function to subscribe to images, and velocities from the robot's odometry and  output a "surface costmap".

3. Run outdoor_dwa.py within the terrapn conda environment.

# Training

* The training code, and associated txt files needed to read the dataset can be found in the model folder along with the network model.

* Our dataset to train a new model can be found [here](https://drive.google.com/file/d/1_FLILRz9FmYFFfTTeis_VSnhAXzIEHV3/view?usp=drive_link).
