---
title: Getting Started
sidebar_position: 4
---

# Getting Started

Welcome to the Physical AI & Humanoid Robotics course! This guide will help you set up your environment and begin your journey into embodied intelligence.

## Initial Setup

### 1. Verify Prerequisites
Before starting, ensure you have:
- Ubuntu 22.04 LTS installed (native or dual-boot)
- Basic command-line familiarity
- Git installed and configured
- Python 3.8+ available

### 2. Install ROS 2 Humble Hawksbill
The course uses ROS 2 Humble Hawksbill (LTS version) for stability:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2 python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 3. Set up Environment
```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 4. Install Additional Tools
```bash
# For simulation
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-dev

# For navigation
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# For perception
sudo apt install ros-humble-vision-opencv ros-humble-cv-bridge
```

## Course Navigation

### Module-Based Learning
The course is structured in four progressive modules:
1. Start with [Module 1: The Robotic Nervous System (ROS 2)](/docs/module1-ros2/index) to build foundational knowledge
2. Progress to [Module 2: The Digital Twin](/docs/module2-digital-twin/index) for simulation skills
3. Advance to [Module 3: The AI-Robot Brain](/docs/module3-ai-brain/index) for perception and navigation
4. Complete with [Module 4: Vision-Language-Action](/docs/module4-vla/index) for conversational robotics

### Weekly Path
Follow the structured weekly path if you prefer a guided learning experience:
- [Week 1-3](/docs/weekly-path/week-01): ROS 2 fundamentals
- [Week 4-6](/docs/weekly-path/week-04): Simulation and digital twins
- [Week 7-9](/docs/weekly-path/week-07): AI and perception
- [Week 10-13](/docs/weekly-path/week-10): Integration and capstone

## Recommended Learning Sequence

### For Beginners
1. Complete the prerequisites review
2. Work through Module 1 at a comfortable pace
3. Practice with the labs before moving to the next module
4. Use the weekly path as a guide

### For Experienced Developers
1. Quickly review Module 1 if familiar with ROS 2
2. Focus on Modules 2-4 for new concepts
3. Skip ahead to specific topics of interest
4. Jump directly to the [Capstone Project](/docs/capstone/index)

## First Steps

1. **Start with Module 1** - Even if you're experienced with ROS, review our approach to humanoid-specific applications
2. **Set up your development environment** - Follow the installation steps above
3. **Try the first lab** - [Lab 1: ROS 2 Node Creation](/docs/labs/lab-01-ros2-basics) to verify your setup
4. **Join the community** - Check the resources in the footer for forums and support

## Troubleshooting Common Issues

### ROS 2 Installation Issues
- Ensure you're using Ubuntu 22.04 (not 20.04 or 24.04)
- Check that your locale is set to UTF-8: `locale` should show `LANG=en_US.UTF-8`
- If packages fail to install, try: `sudo apt update && sudo apt upgrade`

### Simulation Performance
- For Gazebo simulation, ensure your GPU drivers are properly installed
- If simulation runs slowly, reduce the physics update rate in world files
- Consider using lightweight models during development

## Next Steps

Once you've completed the setup:
1. Read the [Course Overview](/docs/index) if you haven't already
2. Review the [Prerequisites](/docs/prerequisites) to ensure you're prepared
3. Begin with [Module 1: The Robotic Nervous System (ROS 2)](/docs/module1-ros2/index)

Ready to begin? Continue to the [Module 1 overview](/docs/module1-ros2/index) to start building your robotic nervous system!