---
description: "Task list for Physical AI & Humanoid Robotics textbook implementation"
---

# Tasks: Physical AI & Humanoid Robotics — Living Textbook

**Input**: Design documents from `/specs/1-humanoid-robotics/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Docusaurus project**: `docs/`, `src/`, `static/`, `blog/` at repository root
- **Content files**: `docs/` directory with subdirectories for modules, weekly path, etc.
- **Navigation**: `sidebars.js` for sidebar configuration
- **Configuration**: `docusaurus.config.js` for site configuration

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Docusaurus project initialization and basic structure

- [ ] T001 Create Docusaurus project structure in repository root
- [ ] T002 Configure docusaurus.config.js with site metadata and navigation
- [ ] T003 [P] Set up basic docs directory structure per plan.md
- [ ] T004 [P] Configure sidebar.js with initial navigation structure
- [ ] T005 Initialize git repository with proper .gitignore for Docusaurus

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Foundational tasks for Docusaurus textbook:

- [ ] T006 Create overview/index page with course introduction in docs/index.md
- [ ] T007 [P] Create basic module directories: docs/module1-ros2/, docs/module2-digital-twin/, docs/module3-ai-brain/, docs/module4-vla/
- [ ] T008 [P] Create weekly path directory: docs/weekly-path/
- [ ] T009 [P] Create capstone directory: docs/capstone/
- [ ] T010 [P] Create labs directory: docs/labs/
- [ ] T011 [P] Create glossary directory: docs/glossary/
- [ ] T012 [P] Create hardware-lab directory: docs/hardware-lab/
- [ ] T013 Configure basic sidebar navigation structure in sidebars.js
- [ ] T014 Set up basic Docusaurus styling and theme configuration

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Course Overview and Navigation (Priority: P1) 🎯 MVP

**Goal**: Student can visit the landing page, understand prerequisites and hardware requirements, and navigate to the first week's content without confusion.

**Independent Test**: Student can visit the homepage, understand prerequisites and hardware requirements, and navigate to the first week's content without confusion.

### Implementation for User Story 1

- [ ] T015 [P] [US1] Create course overview page in docs/index.md
- [ ] T016 [P] [US1] Create prerequisites page in docs/prerequisites.md
- [ ] T017 [P] [US1] Create hardware requirements page in docs/hardware-requirements.md
- [ ] T018 [P] [US1] Create getting started guide in docs/getting-started.md
- [ ] T019 [US1] Update sidebar.js to include overview section navigation
- [ ] T020 [US1] Create landing page with course structure summary in docs/overview.md
- [ ] T021 [US1] Add navigation links from overview to modules and weekly path
- [ ] T022 [US1] Implement basic search and navigation functionality

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Module-Based Learning Path (Priority: P1)

**Goal**: Student can progress through the four modules (ROS 2, Simulation, NVIDIA Isaac, VLA) learning concepts in a structured way, with each module containing multiple lessons that build upon each other.

**Independent Test**: Student can complete Module 1 (ROS 2 fundamentals) and demonstrate understanding of nodes, topics, services, and actions without needing other modules.

### Implementation for User Story 2

- [ ] T023 [P] [US2] Create Module 1 index page in docs/module1-ros2/index.md
- [ ] T024 [P] [US2] Create ROS 2 architecture concepts page in docs/module1-ros2/architecture.md
- [ ] T025 [P] [US2] Create nodes, topics, services, actions page in docs/module1-ros2/nodes-topics-services-actions.md
- [ ] T026 [P] [US2] Create practical development page in docs/module1-ros2/practical-development.md
- [ ] T027 [US2] Create launch files and parameters page in docs/module1-ros2/launch-files-parameters.md
- [ ] T028 [US2] Create humanoid context page in docs/module1-ros2/humanoid-context.md
- [ ] T029 [US2] Create URDF primer page in docs/module1-ros2/urdf-primer.md
- [ ] T030 [P] [US2] Create Module 1 labs page in docs/module1-ros2/labs.md
- [ ] T031 [US2] Create Module 1 troubleshooting page in docs/module1-ros2/troubleshooting.md
- [ ] T032 [US2] Update sidebar.js to include Module 1 navigation structure
- [ ] T033 [P] [US2] Create Module 2 index page in docs/module2-digital-twin/index.md
- [ ] T034 [P] [US2] Create digital twin concepts page in docs/module2-digital-twin/digital-twin-concepts.md
- [ ] T035 [P] [US2] Create Gazebo fundamentals page in docs/module2-digital-twin/gazebo-fundamentals.md
- [ ] T036 [P] [US2] Create worlds and physics page in docs/module2-digital-twin/worlds-physics.md
- [ ] T037 [US2] Create URDF vs SDF page in docs/module2-digital-twin/urdf-vs-sdf.md
- [ ] T038 [US2] Create sensor simulation page in docs/module2-digital-twin/sensor-simulation.md
- [ ] T039 [US2] Create Unity overview page in docs/module2-digital-twin/unity-overview.md
- [ ] T040 [P] [US2] Create Module 2 labs page in docs/module2-digital-twin/labs.md
- [ ] T041 [US2] Create Module 2 troubleshooting page in docs/module2-digital-twin/troubleshooting.md
- [ ] T042 [US2] Update sidebar.js to include Module 2 navigation structure
- [ ] T043 [P] [US2] Create Module 3 index page in docs/module3-ai-brain/index.md
- [ ] T044 [P] [US2] Create Isaac Sim overview page in docs/module3-ai-brain/isaac-sim-overview.md
- [ ] T045 [P] [US2] Create Isaac ROS overview page in docs/module3-ai-brain/isaac-ros-overview.md
- [ ] T046 [P] [US2] Create VSLAM explained page in docs/module3-ai-brain/vslam-explained.md
- [ ] T047 [US2] Create Nav2 path planning page in docs/module3-ai-brain/nav2-path-planning.md
- [ ] T048 [US2] Create sim-to-real page in docs/module3-ai-brain/sim-to-real.md
- [ ] T049 [US2] Create hardware requirements page in docs/module3-ai-brain/hardware-requirements.md
- [ ] T050 [P] [US2] Create Module 3 labs page in docs/module3-ai-brain/labs.md
- [ ] T051 [US2] Create Module 3 troubleshooting page in docs/module3-ai-brain/troubleshooting.md
- [ ] T052 [US2] Update sidebar.js to include Module 3 navigation structure
- [ ] T053 [P] [US2] Create Module 4 index page in docs/module4-vla/index.md
- [ ] T054 [P] [US2] Create VLA concepts page in docs/module4-vla/vla-concepts.md
- [ ] T055 [P] [US2] Create voice-to-action page in docs/module4-vla/voice-to-action.md
- [ ] T056 [P] [US2] Create cognitive planning page in docs/module4-vla/cognitive-planning.md
- [ ] T057 [US2] Create ROS 2 actions integration page in docs/module4-vla/ros2-actions-integration.md
- [ ] T058 [US2] Create multimodal interaction page in docs/module4-vla/multimodal-interaction.md
- [ ] T059 [US2] Create capstone bridge page in docs/module4-vla/capstone-bridge.md
- [ ] T060 [P] [US2] Create Module 4 labs page in docs/module4-vla/labs.md
- [ ] T061 [US2] Create Module 4 troubleshooting page in docs/module4-vla/troubleshooting.md
- [ ] T062 [US2] Update sidebar.js to include Module 4 navigation structure
- [ ] T063 [US2] Ensure each module has at least 4 lessons as specified in requirements

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Capstone Project Guidance (Priority: P2)

**Goal**: Student can access the capstone section to understand the complete "Autonomous Humanoid" pipeline from voice command to action execution, with a clear checklist of deliverables to guide their implementation.

**Independent Test**: Student can read the capstone narrative and understand the complete pipeline from voice command to manipulation, with clear steps they can follow to implement it.

### Implementation for User Story 3

- [ ] T064 [P] [US3] Create capstone index page in docs/capstone/index.md
- [ ] T065 [P] [US3] Create voice command to action pipeline page in docs/capstone/voice-command-to-action.md
- [ ] T066 [P] [US3] Create deliverables checklist page in docs/capstone/deliverables-checklist.md
- [ ] T067 [US3] Create integration guide page in docs/capstone/integration-guide.md
- [ ] T068 [US3] Create capstone troubleshooting page in docs/capstone/troubleshooting.md
- [ ] T069 [US3] Update sidebar.js to include capstone navigation structure
- [ ] T070 [US3] Link capstone section to relevant modules (especially Module 4 VLA)
- [ ] T071 [US3] Ensure capstone includes complete pipeline narrative: voice→plan→navigate→perceive→manipulate

**Checkpoint**: At this point, User Stories 1, 2 AND 3 should all work independently

---

## Phase 6: User Story 4 - Reference and Troubleshooting (Priority: P2)

**Goal**: Student can access the glossary and lab instructions to understand terminology and troubleshoot issues they encounter during practical exercises.

**Independent Test**: Student can look up a ROS 2 term in the glossary and understand its definition and role in the system.

### Implementation for User Story 4

- [ ] T072 [P] [US4] Create glossary index page in docs/glossary/index.md
- [ ] T073 [P] [US4] Create ROS 2 terms page in docs/glossary/ros2-terms.md
- [ ] T074 [P] [US4] Create simulation terms page in docs/glossary/simulation-terms.md
- [ ] T075 [P] [US4] Create AI/perception terms page in docs/glossary/ai-perception-terms.md
- [ ] T076 [US4] Create VLA terms page in docs/glossary/vla-terms.md
- [ ] T077 [US4] Create robotics terms page in docs/glossary/robotics-terms.md
- [ ] T078 [US4] Update sidebar.js to include glossary navigation structure
- [ ] T079 [P] [US4] Create labs index page in docs/labs/index.md
- [ ] T080 [P] [US4] Create Lab 1: ROS 2 basics in docs/labs/lab-01-ros2-basics.md
- [ ] T081 [P] [US4] Create Lab 2: Simulation setup in docs/labs/lab-02-simulation-setup.md
- [ ] T082 [P] [US4] Create Lab 3: VSLAM implementation in docs/labs/lab-03-vslam-implementation.md
- [ ] T083 [US4] Create Lab 4: Navigation pipeline in docs/labs/lab-04-navigation-pipeline.md
- [ ] T084 [US4] Create Lab 5: VLA integration in docs/labs/lab-05-vla-integration.md
- [ ] T085 [US4] Create lab troubleshooting guide in docs/labs/troubleshooting.md
- [ ] T086 [US4] Update sidebar.js to include labs navigation structure
- [ ] T087 [P] [US4] Create hardware/lab setup index in docs/hardware-lab/index.md
- [ ] T088 [P] [US4] Create on-prem approach page in docs/hardware-lab/on-prem-approach.md
- [ ] T089 [P] [US4] Create cloud Ether Lab approach page in docs/hardware-lab/cloud-ether-lab.md
- [ ] T090 [US4] Create latency trap warning page in docs/hardware-lab/latency-trap-warning.md
- [ ] T091 [US4] Create hardware ranges page in docs/hardware-lab/hardware-ranges.md
- [ ] T092 [US4] Update sidebar.js to include hardware/lab setup navigation structure
- [ ] T093 [US4] Ensure all glossary terms are linked from their first mention in content

**Checkpoint**: At this point, User Stories 1, 2, 3 AND 4 should all work independently

---

## Phase 7: Weekly Path Implementation

**Goal**: Create weekly breakdown pages (Weeks 1-13) linking to relevant lessons as specified in requirements.

- [ ] T094 [P] Create weekly path index page in docs/weekly-path/index.md
- [ ] T095 [P] Create Week 1 page in docs/weekly-path/week-01.md
- [ ] T096 [P] Create Week 2 page in docs/weekly-path/week-02.md
- [ ] T097 [P] Create Week 3 page in docs/weekly-path/week-03.md
- [ ] T098 [P] Create Week 4 page in docs/weekly-path/week-04.md
- [ ] T099 [P] Create Week 5 page in docs/weekly-path/week-05.md
- [ ] T100 [P] Create Week 6 page in docs/weekly-path/week-06.md
- [ ] T101 [P] Create Week 7 page in docs/weekly-path/week-07.md
- [ ] T102 [P] Create Week 8 page in docs/weekly-path/week-08.md
- [ ] T103 [P] Create Week 9 page in docs/weekly-path/week-09.md
- [ ] T104 [P] Create Week 10 page in docs/weekly-path/week-10.md
- [ ] T105 [P] Create Week 11 page in docs/weekly-path/week-11.md
- [ ] T106 [P] Create Week 12 page in docs/weekly-path/week-12.md
- [ ] T107 [P] Create Week 13 page in docs/weekly-path/week-13.md
- [ ] T108 Update sidebar.js to include weekly path navigation structure
- [ ] T109 Link each week to relevant lessons in modules as specified in plan

**Checkpoint**: All weekly path pages exist and link correctly to relevant lessons

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T110 [P] Add diagrams to explain ROS graph, VLA pipeline, and other concepts using Mermaid/SVG
- [ ] T111 [P] Add alt text to all images and diagrams for accessibility
- [ ] T112 [P] Add frontmatter to all pages with consistent metadata
- [ ] T113 [P] Create custom Docusaurus styling for consistent look and feel
- [ ] T114 [P] Add code syntax highlighting for all code examples
- [ ] T115 [P] Add table of contents to longer pages
- [ ] T116 [P] Add breadcrumbs navigation for better UX
- [ ] T117 [P] Add search functionality configuration
- [ ] T118 [P] Add previous/next navigation between lessons
- [ ] T119 [P] Add feedback forms or GitHub links for each page
- [ ] T120 Run Docusaurus build locally to verify no errors (npm run build)
- [ ] T121 [P] Run broken-link checker to ensure all internal links work
- [ ] T122 [P] Verify all images load correctly
- [ ] T123 [P] Test site navigation to ensure no orphan pages
- [ ] T124 [P] Verify sidebar integrity and navigation structure
- [ ] T125 [P] Test mobile responsiveness of the site
- [ ] T126 [P] Verify all glossary terms are properly linked
- [ ] T127 [P] Verify all troubleshooting sections are complete
- [ ] T128 [P] Add canonical links and SEO optimization
- [ ] T129 [P] Add analytics configuration if needed
- [ ] T130 [P] Verify constitution compliance: Clarity for target audience
- [ ] T131 [P] Verify constitution compliance: Consistency in presentation
- [ ] T132 [P] Verify constitution compliance: Actionable content with examples
- [ ] T133 [P] Verify constitution compliance: Progressive learning structure
- [ ] T134 [P] Verify constitution compliance: Accessibility requirements
- [ ] T135 [P] Verify constitution compliance: Technical excellence standards
- [ ] T136 Set up GitHub Actions for CI/CD deployment to GitHub Pages
- [ ] T137 [P] Create README.md with project setup and contribution instructions
- [ ] T138 [P] Create CONTRIBUTING.md with contribution guidelines
- [ ] T139 [P] Create CODE_OF_CONDUCT.md for project community
- [ ] T140 Final review of all content for accuracy and completeness

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Weekly Path (Phase 7)**: Depends on all modules being created
- **Polish (Final Phase)**: Depends on all desired content being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Should integrate with modules from US2
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Can reference content from other stories

### Within Each User Story

- Content creation follows logical structure: overview → detailed content → labs → troubleshooting
- Module content should be created in the order specified in plan.md
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All pages within a module marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members
- Glossary and lab creation can happen in parallel with module content

---

## Parallel Example: Module 1 Creation

```bash
# Launch all Module 1 pages together:
Task: "Create Module 1 index page in docs/module1-ros2/index.md"
Task: "Create ROS 2 architecture concepts page in docs/module1-ros2/architecture.md"
Task: "Create nodes, topics, services, actions page in docs/module1-ros2/nodes-topics-services-actions.md"
Task: "Create practical development page in docs/module1-ros2/practical-development.md"
Task: "Create Module 1 labs page in docs/module1-ros2/labs.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test navigation and overview functionality independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Add weekly path → Test navigation → Deploy/Demo
7. Add polish → Final validation → Deploy/Demo
8. Each addition adds value without breaking previous functionality

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Module overview and navigation)
   - Developer B: User Story 2 (Module 1 content)
   - Developer C: User Story 2 (Module 2 content)
   - Developer D: User Story 2 (Module 3 content)
   - Developer E: User Story 2 (Module 4 content)
3. Continue with remaining stories as modules are completed
4. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [US1], [US2], [US3], [US4] labels map task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Ensure all 4 modules have at least 4 lessons each as required
- Ensure all 13 weeks are covered in weekly path
- Ensure capstone includes complete pipeline narrative and deliverables checklist
- Ensure hardware section includes On-Prem + Cloud "Ether Lab" + "latency trap" warning