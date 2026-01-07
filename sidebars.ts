import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics textbook
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Overview',
      items: ['intro', 'prerequisites', 'hardware-requirements', 'getting-started'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1-ros2/index',
        'module1-ros2/architecture',
        'module1-ros2/nodes-topics-services-actions',
        'module1-ros2/practical-development',
        'module1-ros2/launch-files-parameters',
        'module1-ros2/humanoid-context',
        'module1-ros2/urdf-primer',
        'module1-ros2/labs',
        'module1-ros2/troubleshooting'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2-digital-twin/index',
        'module2-digital-twin/digital-twin-concepts',
        'module2-digital-twin/gazebo-fundamentals',
        'module2-digital-twin/worlds-physics',
        'module2-digital-twin/urdf-vs-sdf',
        'module2-digital-twin/sensor-simulation',
        'module2-digital-twin/unity-overview',
        'module2-digital-twin/labs',
        'module2-digital-twin/troubleshooting'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module3-ai-brain/index',
        'module3-ai-brain/isaac-sim-overview',
        'module3-ai-brain/isaac-ros-overview',
        'module3-ai-brain/vslam-explained',
        'module3-ai-brain/nav2-path-planning',
        'module3-ai-brain/sim-to-real',
        'module3-ai-brain/hardware-requirements',
        'module3-ai-brain/labs',
        'module3-ai-brain/troubleshooting'
      ],
      collapsed: false,
    },
    // Commenting out sections that don't have existing files
    // {
    //   type: 'category',
    //   label: 'Module 4: Vision-Language-Action (VLA) & Conversational Robotics',
    //   items: [
    //     'module4-vla/index',
    //     'module4-vla/vla-concepts',
    //     'module4-vla/voice-to-action',
    //     'module4-vla/cognitive-planning',
    //     'module4-vla/ros2-actions-integration',
    //     'module4-vla/multimodal-interaction',
    //     'module4-vla/capstone-bridge',
    //     'module4-vla/labs',
    //     'module4-vla/troubleshooting'
    //   ],
    //   collapsed: false,
    // },
    // {
    //   type: 'category',
    //   label: 'Weekly Path (Weeks 1-13)',
    //   items: [
    //     'weekly-path/index',
    //     'weekly-path/week-01',
    //     'weekly-path/week-02',
    //     'weekly-path/week-03',
    //     'weekly-path/week-04',
    //     'weekly-path/week-05',
    //     'weekly-path/week-06',
    //     'weekly-path/week-07',
    //     'weekly-path/week-08',
    //     'weekly-path/week-09',
    //     'weekly-path/week-10',
    //     'weekly-path/week-11',
    //     'weekly-path/week-12',
    //     'weekly-path/week-13'
    //   ],
    //   collapsed: false,
    // },
    // {
    //   type: 'category',
    //   label: 'Capstone: Autonomous Humanoid',
    //   items: [
    //     'capstone/index',
    //     'capstone/voice-command-to-action',
    //     'capstone/deliverables-checklist',
    //     'capstone/integration-guide',
    //     'capstone/troubleshooting'
    //   ],
    //   collapsed: false,
    // },
    // {
    //   type: 'category',
    //   label: 'Labs',
    //   items: [
    //     'labs/index',
    //     'labs/lab-01-ros2-basics',
    //     'labs/lab-02-simulation-setup',
    //     'labs/lab-03-vslam-implementation',
    //     'labs/lab-04-navigation-pipeline',
    //     'labs/lab-05-vla-integration',
    //     'labs/troubleshooting'
    //   ],
    //   collapsed: false,
    // },
    // {
    //   type: 'category',
    //   label: 'Glossary',
    //   items: [
    //     'glossary/index',
    //     'glossary/ros2-terms',
    //     'glossary/simulation-terms',
    //     'glossary/ai-perception-terms',
    //     'glossary/vla-terms',
    //     'glossary/robotics-terms'
    //   ],
    //   collapsed: false,
    // },
    // {
    //   type: 'category',
    //   label: 'Hardware & Lab Setup',
    //   items: [
    //     'hardware-lab/index',
    //     'hardware-lab/on-prem-approach',
    //     'hardware-lab/cloud-ether-lab',
    //     'hardware-lab/latency-trap-warning',
    //     'hardware-lab/hardware-ranges'
    //   ],
    //   collapsed: false,
    // },
  ],
};

export default sidebars;