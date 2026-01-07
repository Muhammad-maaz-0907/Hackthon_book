# Implementation Plan: Physical AI & Humanoid Robotics — Living Textbook

## Architecture & Information Architecture

### Content Architecture (Docusaurus IA)
```
Physical AI & Humanoid Robotics Textbook
├── Overview
│   ├── Course Prerequisites
│   ├── Hardware Requirements
│   └── Getting Started Guide
├── Module 1: The Robotic Nervous System (ROS 2)
│   ├── ROS 2 Architecture
│   ├── Nodes, Topics, Services, Actions
│   ├── Practical ROS 2 Development
│   ├── Launch Files & Parameters
│   ├── Humanoid Context Applications
│   ├── URDF Primer for Humanoids
│   ├── Module 1 Labs
│   └── Module 1 Troubleshooting
├── Module 2: The Digital Twin (Gazebo & Unity)
│   ├── Digital Twin Concepts
│   ├── Gazebo Fundamentals
│   ├── Worlds & Physics Settings
│   ├── URDF vs SDF for Simulation
│   ├── Sensor Simulation
│   ├── Unity Overview
│   ├── Module 2 Labs
│   └── Module 2 Troubleshooting
├── Module 3: The AI-Robot Brain (NVIDIA Isaac)
│   ├── Isaac Sim Overview
│   ├── Isaac ROS Overview
│   ├── VSLAM Explained
│   ├── Nav2 Path Planning
│   ├── Sim-to-Real Concepts
│   ├── Hardware Requirements & Alternatives
│   ├── Module 3 Labs
│   └── Module 3 Troubleshooting
├── Module 4: Vision-Language-Action (VLA) & Conversational Robotics
│   ├── VLA Pipeline Concepts
│   ├── Voice-to-Action Workflow
│   ├── Cognitive Planning
│   ├── ROS 2 Actions Integration
│   ├── Multimodal Interaction
│   ├── Capstone Bridge
│   ├── Module 4 Labs
│   └── Module 4 Troubleshooting
├── Weekly Path (Weeks 1-13)
│   ├── Week 1: Introduction & ROS 2 Basics
│   ├── Week 2: ROS 2 Architecture Deep Dive
│   ├── Week 3: ROS 2 Practical Development
│   ├── Week 4: Simulation Introduction
│   ├── Week 5: Gazebo Fundamentals
│   ├── Week 6: Sensor Simulation
│   ├── Week 7: Isaac Perception Systems
│   ├── Week 8: Navigation Systems
│   ├── Week 9: AI Integration
│   ├── Week 10: VLA Concepts
│   ├── Week 11: VLA Implementation
│   ├── Week 12: Capstone Project Phase 1
│   └── Week 13: Capstone Project Phase 2
├── Capstone: Autonomous Humanoid
│   ├── Complete Pipeline Narrative
│   ├── Voice Command → Planning → Navigation → Perception → Manipulation
│   ├── Deliverables Checklist
│   ├── Integration Guide
│   └── Troubleshooting
├── Labs Section
│   ├── Lab 1: ROS 2 Node Creation
│   ├── Lab 2: Gazebo Simulation Setup
│   ├── Lab 3: VSLAM Implementation
│   ├── Lab 4: Navigation Pipeline
│   ├── Lab 5: VLA Integration
│   └── Lab Troubleshooting Guide
├── Glossary
│   ├── ROS 2 Terms (Node, Topic, Service, Action)
│   ├── Simulation Terms (URDF, SDF, Digital Twin)
│   ├── AI/Perception Terms (VSLAM, Nav2, Isaac)
│   ├── VLA Terms (Multimodal, Cognitive Planning)
│   └── Robotics Terms (SLAM, VSLAM, Nav2, sim-to-real)
└── Hardware & Lab Setup
    ├── On-Prem Lab Approach (RTX workstation + Jetson + sensors)
    ├── Cloud-Native "Ether Lab" Approach
    ├── "Latency Trap" Warning & Guidance
    └── Recommended Hardware Ranges
```

### System/Data-Flow Architecture (Course Concept)

#### Primary Flow: Physical Robotics Pipeline
```
Physical Robot/Simulation → Sensors → Perception/VSLAM → Nav2 → Control → Robot/Sim Feedback
     ↑                                                                 ↓
     └───────────────────────── ROS 2 Communication Layer ←─────────────┘
```

#### Secondary Flow: VLA (Vision-Language-Action) Loop
```
Voice Command → ASR (Whisper-style) → Text → Intent → LLM Planning → ROS 2 Actions → Execution → Feedback
     ↑                                                                                             ↓
     └─────────────────────────────────────── ROS 2 Action Feedback ←───────────────────────────────┘
```

### Delivery Architecture
```
GitHub Repository (main branch)
    ↓ (push triggers)
GitHub Actions CI
    ↓ (build process)
Docusaurus Static Build
    ↓ (deploy to)
GitHub Pages (gh-pages branch)
    ↓ (served at)
https://[username].github.io/humanoid-robotics-book
```

**Branch Strategy:**
- `main`: Production-ready content (what gets deployed)
- `dev`: Active development branch
- `feature/###-topic-name`: Individual feature branches for modules
- `release/vX.X`: Release branches for major versions

**Required Secrets:**
- None needed for basic deployment (GitHub Pages is public)

## Section Structure

### Module 1: The Robotic Nervous System (ROS 2)
- **docs/module1-ros2/**
  - `index.md` - Module overview
  - `architecture.md` - ROS 2 architecture concepts
  - `nodes-topics-services-actions.md` - Communication patterns
  - `practical-development.md` - rclpy, packages, practical work
  - `launch-files-parameters.md` - Launch files and parameters
  - `humanoid-context.md` - ROS 2 in humanoid systems
  - `urdf-primer.md` - URDF for humanoids
  - `labs.md` - Module 1 labs with troubleshooting
  - `troubleshooting.md` - Module 1 specific issues

### Module 2: The Digital Twin (Gazebo & Unity)
- **docs/module2-digital-twin/**
  - `index.md` - Module overview
  - `digital-twin-concepts.md` - What is a digital twin
  - `gazebo-fundamentals.md` - Gazebo basics
  - `worlds-physics.md` - Worlds and physics settings
  - `urdf-vs-sdf.md` - URDF vs SDF for simulation
  - `sensor-simulation.md` - Sensor simulation concepts
  - `unity-overview.md` - Unity for visualization
  - `labs.md` - Module 2 labs with troubleshooting
  - `troubleshooting.md` - Module 2 specific issues

### Module 3: The AI-Robot Brain (NVIDIA Isaac)
- **docs/module3-ai-brain/**
  - `index.md` - Module overview
  - `isaac-sim-overview.md` - Isaac Sim features
  - `isaac-ros-overview.md` - Isaac ROS integration
  - `vslam-explained.md` - VSLAM concepts and implementation
  - `nav2-path-planning.md` - Navigation systems
  - `sim-to-real.md` - Transitioning from simulation to reality
  - `hardware-requirements.md` - RTX and alternatives
  - `labs.md` - Module 3 labs with troubleshooting
  - `troubleshooting.md` - Module 3 specific issues

### Module 4: Vision-Language-Action (VLA) & Conversational Robotics
- **docs/module4-vla/**
  - `index.md` - Module overview
  - `vla-concepts.md` - VLA pipeline concepts
  - `voice-to-action.md` - Voice processing workflow
  - `cognitive-planning.md` - LLM-based planning
  - `ros2-actions-integration.md` - ROS 2 actions mapping
  - `multimodal-interaction.md` - Multimodal systems
  - `capstone-bridge.md` - Connection to capstone project
  - `labs.md` - Module 4 labs with troubleshooting
  - `troubleshooting.md` - Module 4 specific issues

### Weekly Path Structure
- **docs/weekly-path/**
  - `index.md` - Overview of weekly path
  - `week-01.md` - Week 1 content and readings
  - `week-02.md` - Week 2 content and readings
  - ...
  - `week-13.md` - Week 13 content and readings

### Capstone Structure
- **docs/capstone/**
  - `index.md` - Capstone overview and pipeline narrative
  - `voice-command-to-action.md` - Complete pipeline description
  - `deliverables-checklist.md` - Checklist of required outputs
  - `integration-guide.md` - How to connect all modules
  - `troubleshooting.md` - Capstone-specific issues

### Labs Structure
- **docs/labs/**
  - `index.md` - Labs overview
  - `lab-01-ros2-basics.md` - ROS 2 fundamentals lab
  - `lab-02-simulation-setup.md` - Simulation setup lab
  - `lab-03-vslam-implementation.md` - VSLAM lab
  - `lab-04-navigation-pipeline.md` - Navigation lab
  - `lab-05-vla-integration.md` - VLA integration lab
  - `troubleshooting.md` - General lab troubleshooting

### Glossary Structure
- **docs/glossary/**
  - `index.md` - Glossary overview
  - `ros2-terms.md` - ROS 2 specific terms
  - `simulation-terms.md` - Simulation specific terms
  - `ai-perception-terms.md` - AI/perception terms
  - `vla-terms.md` - VLA specific terms
  - `robotics-terms.md` - General robotics terms

### Hardware & Lab Setup Structure
- **docs/hardware-lab/**
  - `index.md` - Hardware overview
  - `on-prem-approach.md` - On-premise setup guide
  - `cloud-ether-lab.md` - Cloud-based approach
  - `latency-trap-warning.md` - Latency considerations
  - `hardware-ranges.md` - Recommended hardware guidance

## Research Approach

### Research-Concurrent Writing Strategy
- **Verify Claims While Drafting**: For each technical claim, immediately verify with primary documentation
- **"To-Verify" Queue**: Maintain a running list of claims that need verification
- **Primary Documentation Sources**:
  - ROS 2: docs.ros.org, official tutorials
  - Gazebo: gazebosim.org, tutorials
  - Unity: docs.unity3d.com (for robotics applications)
  - NVIDIA Isaac: docs.nvidia.com/isaac/
  - Nav2: navigation.ros.org
- **Citation Approach**: Include source links close to each technical claim
- **Running Bibliography**: Maintain per-module reference lists

## Quality Validation

### Per-Page Checklist Template
- [ ] Clear learning objectives stated
- [ ] Prerequisites clearly listed
- [ ] Steps are clear and actionable
- [ ] Expected outputs clearly described
- [ ] Troubleshooting section included
- [ ] Glossary terms linked at first mention
- [ ] Code examples are complete and tested
- [ ] Images have appropriate alt text
- [ ] Internal links work correctly

### Per-Release Checklist
- [ ] Docusaurus build passes locally (npm run build)
- [ ] Docusaurus build passes in CI
- [ ] Broken-link checks pass (all internal links work)
- [ ] Image/asset checks pass (all images load)
- [ ] Sidebar integrity verified (no orphan pages)
- [ ] GitHub Pages deploy succeeds
- [ ] Site loads correctly and reflects latest changes

## Decisions Log

### Information Architecture Choice: Modules-first vs Weeks-first vs Hybrid
- **Decision**: Hybrid approach (Modules + Week-path)
- **Options**:
  - Modules-first: Organize primarily by technical topics
  - Weeks-first: Organize primarily by time progression
  - Hybrid: Combine both approaches
- **Tradeoffs**:
  - Modules-first: Good for reference, harder for structured learning
  - Weeks-first: Good for structured learning, harder for reference
  - Hybrid: Best of both, more complex navigation
- **Decision Rule**: Choose hybrid to serve both structured learners and reference users

### Code Policy: Snippets Only vs Runnable Companion Repo(s)
- **Decision**: Snippets only in documentation
- **Options**:
  - Full runnable code in docs: Complete, but harder to maintain
  - Snippets only: Easier to maintain, less comprehensive
- **Tradeoffs**:
  - Full code: Students can run immediately, but docs become bloated
  - Snippets: Clean docs, students need to integrate code themselves
- **Decision Rule**: Choose snippets to keep focus on concepts rather than implementation details

### ROS 2 Baseline: Humble vs Iron
- **Decision**: ROS 2 Humble Hawksbill (LTS)
- **Options**: Humble (LTS, long-term support) vs Iron (latest features)
- **Tradeoffs**:
  - Humble: More stable, longer support, fewer breaking changes
  - Iron: Latest features, but shorter support window
- **Decision Rule**: Choose LTS for educational stability

### Simulation Depth: Gazebo Deep + Unity Overview vs Equal Depth
- **Decision**: Gazebo deep + Unity overview
- **Options**: Equal depth vs Gazebo focus + Unity overview
- **Tradeoffs**:
  - Equal depth: More comprehensive but complex
  - Gazebo focus: Practical for ROS integration, Unity as enhancement
- **Decision Rule**: Focus on Gazebo as primary simulation tool for ROS ecosystem

### Isaac Accessibility: Local RTX vs Cloud Workflow
- **Decision**: Include both with clear guidance on tradeoffs
- **Options**: Local RTX requirement vs Cloud-first approach
- **Tradeoffs**:
  - Local RTX: Better performance, higher barrier to entry
  - Cloud: More accessible, potential latency issues
- **Decision Rule**: Document both approaches with clear pros/cons

### Capstone Structure: Single Path vs Multiple Variants
- **Decision**: Single "golden path" with optional extensions
- **Options**: Single path vs Multiple variants (sim-only, sim-to-real, edge-first)
- **Tradeoffs**:
  - Single path: Clearer, more focused
  - Multiple variants: More flexible but complex
- **Decision Rule**: Single path to avoid overwhelming students

### Hardware Guidance Style: Prices vs Ranges vs No Prices
- **Decision**: Ranges with date stamps
- **Options**: Exact prices, ranges, or no prices
- **Tradeoffs**:
  - Exact prices: Specific but quickly outdated
  - Ranges: Flexible but less specific
  - No prices: Never outdated but not actionable
- **Decision Rule**: Ranges with date stamps for balance of actionability and longevity

### Diagram Tooling: Mermaid vs SVG/PNG
- **Decision**: Mermaid for simple diagrams, SVG/PNG for complex ones
- **Options**: Pure Mermaid, pure SVG/PNG, or hybrid
- **Tradeoffs**:
  - Mermaid: Maintainable in text, limited complexity
  - SVG/PNG: Full control, harder to maintain
- **Decision Rule**: Use Mermaid for simple flow diagrams, SVG/PNG for complex technical diagrams

## Testing Strategy

### Content Acceptance Tests
- [ ] Module coverage: 4 modules, ≥4 lessons each, labs included
- [ ] Week path coverage: Weeks 1–13 pages exist and link correctly (no dead ends)
- [ ] Capstone completeness: voice→plan→navigate→perceive→manipulate pipeline described + deliverables checklist
- [ ] Hardware section: includes On-Prem + Cloud "Ether Lab" + "latency trap" warning

### Technical Acceptance Tests
- [ ] Docusaurus build passes locally and in CI (no errors; warnings triaged)
- [ ] Broken-link checks and image/asset checks pass
- [ ] GitHub Pages deploy succeeds; site loads and reflects latest release

### Editorial Acceptance Tests
- [ ] Consistent templates (Concept → Example → Lab → Troubleshooting → Checkpoint)
- [ ] Glossary terms linked at first mention; consistent frontmatter and headings

## Milestone Schedule

### Phase 1: Foundation (Week 1)
- [ ] Set up Docusaurus project structure
- [ ] Create basic navigation and sidebar
- [ ] Write Module 1: The Robotic Nervous System (ROS 2) - Days 1-3
- [ ] Write Module 1 Labs and Troubleshooting - Day 4
- [ ] Review and refine Module 1 content - Day 5

### Phase 2: Simulation (Week 2)
- [ ] Write Module 2: The Digital Twin (Gazebo & Unity) - Days 6-8
- [ ] Write Module 2 Labs and Troubleshooting - Day 9
- [ ] Create weekly path for Weeks 1-5 - Day 10

### Phase 3: AI/Perception (Week 3)
- [ ] Write Module 3: The AI-Robot Brain (NVIDIA Isaac) - Days 11-13
- [ ] Write Module 3 Labs and Troubleshooting - Day 14
- [ ] Create weekly path for Weeks 6-9 - Day 15

### Phase 4: VLA & Integration (Week 4)
- [ ] Write Module 4: Vision-Language-Action (VLA) - Days 16-18
- [ ] Write Module 4 Labs and Troubleshooting - Day 19
- [ ] Create weekly path for Weeks 10-13 - Day 20

### Phase 5: Capstone & Completion (Week 5)
- [ ] Write Capstone section - Days 21-22
- [ ] Write Labs section - Day 23
- [ ] Write Glossary - Day 24
- [ ] Write Hardware & Lab Setup - Day 25

### Phase 6: Quality & Deployment (Week 6)
- [ ] Content review and editing - Days 26-28
- [ ] Technical testing and validation - Days 29-30
- [ ] Deployment setup and final testing - Days 31-32
- [ ] Final review and documentation - Days 33-35