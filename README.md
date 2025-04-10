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

```bash
$ source /opt/ros/humble/setup.bash
$ colcon build
$ source install/local_setup.bash
$ ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

```bash
/SBAMP/src/sbamp/scripts$ chmod +x sbamp_node.py 
```

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

To pull with submodule changes:

```bash
git pull --recurse-submodules
```