## Code Usage

This is the main working code and model for TerraPn, the paper found here: https://ieeexplore.ieee.org/document/9981942

# Steps to follow:

0. Install ROS Melodic.

1. Setup the conda environment using terrapn.yaml.

2. Setup RGB image stream from a camera. outdoor_dwa_v4 contains the ros callback function to subscribe to images, and output a 
"surface costmap".

3. Run outdoor_dwa_v4.py within the terrapn conda environment. 