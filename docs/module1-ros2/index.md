---
title: "Module 1: The Robotic Nervous System (ROS 2)"
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

Welcome to the first module of the Physical AI & Humanoid Robotics course! In this module, you'll learn how ROS 2 serves as the "nervous system" of a robot, enabling reliable communication between software components in real time.

## Module Overview

ROS 2 (Robot Operating System 2) is the middleware that connects all components of a robotic system. Just as the nervous system connects different parts of a biological organism, ROS 2 connects sensors, controllers, planners, and actuators in a robot.

### What You'll Learn

In this module, you'll master:
- **ROS 2 Architecture**: Understanding the fundamental concepts that make distributed robotics possible
- **Communication Patterns**: Nodes, topics, services, and actions - when to use each
- **Practical Development**: Creating ROS 2 packages with Python and rclpy
- **Launch Systems**: Coordinating multiple nodes with launch files and parameters
- **Humanoid Applications**: How ROS 2 concepts apply specifically to humanoid robotics
- **URDF Integration**: Understanding robot description in the context of ROS 2

### Learning Objectives

By the end of this module, you will be able to:
1. Explain the ROS 2 architecture and its role in robotic systems
2. Distinguish between topics, services, and actions and choose the appropriate pattern
3. Create and run basic ROS 2 nodes using Python
4. Configure complex systems using launch files and parameters
5. Understand how ROS 2 enables humanoid robot subsystems to communicate
6. Work with URDF files in the context of ROS 2

## Module Structure

This module is organized into several interconnected lessons:

### Core Concepts
- [ROS 2 Architecture](./architecture.md): The foundational concepts of ROS 2
- [Nodes, Topics, Services, Actions](./nodes-topics-services-actions.md): Communication patterns in detail
- [URDF Primer for Humanoids](./urdf-primer.md): Robot description for humanoid systems

### Practical Skills
- [Practical ROS 2 Development](./practical-development.md): Hands-on Python development
- [Launch Files & Parameters](./launch-files-parameters.md): System coordination and configuration

### Application Context
- [Humanoid Context Applications](./humanoid-context.md): How ROS 2 applies to humanoid systems

### Practice & Troubleshooting
- [Module 1 Labs](./labs.md): Hands-on exercises to reinforce learning
- [Module 1 Troubleshooting](./troubleshooting.md): Common issues and solutions

## Prerequisites

Before starting this module, ensure you have:
- Basic Python programming skills (functions, classes, modules)
- Understanding of distributed systems concepts (optional but helpful)
- Completed the [Getting Started](/docs/getting-started.md) guide
- ROS 2 Humble Hawksbill installed on your system

## Why This Module Matters

ROS 2 is the backbone of most modern robotic systems. Understanding how to use it effectively is crucial for:
- Building reliable robotic applications
- Integrating different software components
- Creating maintainable and scalable robot systems
- Working with the broader robotics community

In the context of humanoid robotics, ROS 2 becomes even more critical as it must coordinate numerous sensors, actuators, and complex behaviors simultaneously.

## Getting Started

Begin with the [ROS 2 Architecture](./architecture.md) lesson to understand the foundational concepts, then progress through the practical development sections to build hands-on skills.

## Next Steps

After completing this module, you'll have a solid foundation in ROS 2 that will serve you well in:
- [Module 2: The Digital Twin (Gazebo & Unity)](/docs/module2-digital-twin/index): Where you'll see ROS 2 in simulation
- [Module 3: The AI-Robot Brain (NVIDIA Isaac)](/docs/module3-ai-brain/index): Where ROS 2 connects to AI systems
- [Module 4: Vision-Language-Action (VLA) & Conversational Robotics](/docs/module4-vla/index): Where ROS 2 enables natural human-robot interaction

Ready to dive in? Start with the [ROS 2 Architecture](./architecture.md) lesson to build your foundational understanding.