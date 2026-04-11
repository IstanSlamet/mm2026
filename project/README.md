Course Project:
Motivation
Retrieving objects from the floor can be challenging for individuals with limited mobility, such as
elderly people or pregnant individuals. Simple tasks like picking up a dropped bottle may require
significant effort or assistance. This project aims to develop an autonomous robotic system that
can locate, pick up, and deliver objects to a designated location, improving accessibility and
independence in everyday environments.
Objective
We propose to use the Hello Robot Stretch to autonomously retrieve objects from the floor and
deliver them to a predefined location (e.g., a table). The system will integrate navigation,
perception, and manipulation to complete this task with minimal user intervention.
Methodology
The system will operate in three main stages:
1. User Interaction
The robot receives commands via text or voice input (e.g., “Bring me the water bottle”).
This input will be parsed to identify the target object.
2. Autonomous Navigation & Object Search
The robot navigates a known indoor environment using waypoint-based navigation or
patrol behavior. It will use onboard sensors and vision models to detect the target object
during exploration.
3. Object Detection and Grasping
Once the object is detected, the robot will approach it and perform grasping using visual
feedback. We plan to use a visual servoing strategy to refine the robot’s alignment and
ensure successful pickup.
4. Object Delivery
After grasping, the robot navigates back to a predefined drop-off location (e.g., a table)
and releases the object.
Evaluation Plan
We will evaluate the system based on:
● Success rate of object detection and grasping
● Navigation accuracy and reliability
● End-to-end task completion rate
