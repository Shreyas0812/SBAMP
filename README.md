# SBAMP
Sampling Based Adaptive Motion Planning

### Steps to run:

Note: Make sure you are at the correct location as given by the text before `$`

```bash
git clone --recurse-submodules https://github.com/Shreyas0812/SBAMP.git
```

```bash
/SBAMP/src/f1tenth_gym_ros/f1tenth_gym$ pip install -e .
```

```
$ cd SBAMP
```

#### To Launch the Roboracer Simulator

```bash
$ source /opt/ros/humble/setup.bash
$ colcon build
$ source install/local_setup.bash
$ ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

```bash
/SBAMP/src/sbamp/scripts$ chmod +x sbamp_node.py 
```

#### To Launch SBAMP:

```bash
/SBAMP$ colcon build
/SBAMP$ source install/local_setup.bash 
/SBAMP$ source install/setup.bash 
/SBAMP$ ros2 launch sbamp sbamp.launch.py 
```

If there is an error running any of the above commands, run this and try again:

```bash
/SBAMP$ rosdep install --from-paths src -y --ignore-src
```

Visualization Node:

```bash
/SBAMP/src/sbamp/scripts$ chmod +x visualization_node.py 
```

#### Other Launch Options:

To run with visualization:

```bash
/SBAMP$ ros2 launch sbamp sbamp_visualization.launch.py 
```

To run without visualization:

```bash
/SBAMP$ ros2 launch sbamp sbamp.launch.py 
```

To run only rrt:

```bash
/SBAMP$ ros2 launch sbamp only_rrt.launch.py 
```

### Video - Simulation:

https://github.com/user-attachments/assets/24c63235-09f2-482f-8d48-276d0a401b4b


### Poster:

<img src="SBAMP-Official.png" alt="SBAMP Poster" width="800"/>




