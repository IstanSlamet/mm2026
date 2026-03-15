import os
import numpy as np
import mengine as m
from ik_ros_utils_sim import IK

# Create environment and ground plane
env = m.Env(gravity=[0, 0, -1])
ground = m.Ground()

# Create table
table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=[-1.3, 0, 0], orientation=[0, 0, 0, 1])

# Create mustard bottle
mustard = m.Shape(m.Mesh(filename=os.path.join(m.directory, 'ycb', 'mustard.obj'), scale=[1, 1, 1]), static=False, mass=1.0, position=[-0.6, 0, 0.85], orientation=[0, 0, 0, 1], rgba=None, visual=True, collision=True)

robot = m.Robot.Stretch3(position=[0, 0, 0], orientation=[0, 0, 0])
# robot.print_joint_info()

ik = IK(m, env, robot)

robot.set_joint_angles(angles=[0.9], joints=[4])

# Let the object settle on the table
m.step_simulation(steps=100, realtime=False)

# ee_position, ee_orientation = robot.get_link_pos_orient(robot.end_effector)
object_position, object_orientation = mustard.get_base_pos_orient()
m.Points([object_position])

euler = [-np.pi/2, 0, 0]

ik.move_to_grasp_goal(object_position, euler, max_rotation_steps=300, max_translation_steps=1000)
m.step_simulation(steps=1000, realtime=True)

