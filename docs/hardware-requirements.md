---
title: Hardware Requirements
sidebar_position: 3
---

# Hardware Requirements

This course accommodates different learning approaches with varying hardware requirements. Choose the approach that best fits your situation and resources.

## On-Premise Lab Approach

For the full hands-on experience, we recommend the following hardware setup:

### Minimum Specifications
- **CPU**: 8-core processor (Intel i7 or AMD Ryzen 7)
- **RAM**: 32 GB (64 GB recommended)
- **GPU**: NVIDIA RTX 3070 or better (RTX 4070+ recommended)
- **Storage**: 1 TB SSD (NVMe preferred)
- **OS**: Ubuntu 22.04 LTS

### Recommended Specifications
- **CPU**: 16-core processor (Intel i9 or AMD Ryzen 9)
- **RAM**: 64 GB or more
- **GPU**: NVIDIA RTX 4080/4090 or RTX 6000 Ada (for advanced Isaac Sim)
- **Storage**: 2 TB NVMe SSD
- **Network**: Gigabit Ethernet or better

### Robotics Hardware (Optional but Recommended)
- **Robot Platform**: Any ROS 2 compatible humanoid or mobile robot
- **Sensors**: RGB-D camera, LiDAR, IMU
- **Computing**: Jetson Orin AGX or equivalent for edge deployment
- **Peripherals**: USB-to-serial adapters, power banks, etc.

## Cloud-Native "Ether Lab" Approach

For those without access to high-end hardware, cloud-based alternatives are available:

### Simulation Cloud Services
- **AWS RoboMaker**: Managed service for robotics simulation
- **Azure Digital Twins**: Cloud-based digital twin capabilities
- **NVIDIA Omniverse**: For advanced Isaac Sim capabilities
- **Google Cloud**: For perception and AI workloads

### Considerations
- **Latency**: Critical for real-time robotics applications
- **Bandwidth**: High-speed internet required for remote simulation
- **Cost**: Ongoing operational expenses vs upfront hardware investment
- **Performance**: May vary based on cloud provider and region

## The "Latency Trap" Warning

When using cloud-based robotics, be aware of the **latency trap**:

- **Real-time Control**: Most robotics applications require sub-100ms response times
- **Cloud Limitations**: Even with optimized connections, cloud-based control may introduce unacceptable delays
- **Simulation vs Reality**: Latency effects may be masked in simulation but appear in real hardware
- **Edge Computing**: For real-time applications, edge computing (Jetson, etc.) is often necessary

## Hardware Ranges (2025)

### Workstation Options
- **Budget**: $3,000-$5,000 for RTX 4070-based workstation
- **Mid-range**: $6,000-$10,000 for RTX 4080/4090 workstation
- **High-end**: $15,000+ for RTX 6000 Ada or A6000 workstation

### Robotics Platforms
- **Simulation Only**: No physical robot required
- **Educational**: $5,000-$15,000 for basic humanoid platforms
- **Research**: $20,000-$50,000+ for advanced humanoid robots

### Cloud Budgeting
- **Development**: $200-$500/month for basic cloud simulation
- **Advanced**: $1,000-$3,000/month for Isaac Sim and high-end GPU instances
- **Production**: Variable based on usage patterns

## Getting Started Options

### Option 1: Simulation-First
Start with simulation using minimum hardware requirements, then add physical hardware later.

### Option 2: Cloud-First
Begin with cloud-based simulation, then transition to on-premise or hybrid approaches as needed.

### Option 3: Hybrid Approach
Combine local development with cloud-based simulation and testing.

Choose the approach that best fits your budget, learning goals, and intended applications.