---
id: 2
title: "book-planning"
stage: "plan"
date_iso: "2025-12-20"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "1-humanoid-robotics"
branch: "1-humanoid-robotics"
user: "user"
command: "/sp.plan"
labels: ["planning", "architecture", "docusaurus", "humanoid-robotics", "education"]
link_spec: "specs/1-humanoid-robotics/spec.md"
link_ticket: "null"
link_adr: "null"
link_pr: "null"
files_yaml: |
-specs/1-humanoid-robotics/plan.md
- specs/1-humanoid-robotics/qa-checklist.md
- specs/1-humanoid-robotics/architectural-decisions.md
tests_yaml: |
-
outcome: "Comprehensive implementation plan created for Physical AI & Humanoid Robotics textbook"
evaluation: "Plan includes architecture, section structure, research approach, quality validation, decisions log, and milestone schedule"
---

# PHR: book-planning

## User Input

```
/sp.plan

Create:
- Architecture sketch:
  - Content architecture (Docusaurus IA): Overview → Modules 1–4 → Weeks 1–13 reading path → Capstone → Labs → Glossary → Hardware/Lab (On-Prem + Cloud).
  - System/data-flow architecture (course concept): Sensors/Sim → Perception/VSLAM → Nav2 → Control → Robot/Sim feedback; plus VLA loop (Voice → Text → Intent → Plan → ROS 2 Actions → Execution → Feedback).
  - Delivery architecture: GitHub repo → CI build → GitHub Pages deploy (branch strategy, environments, required secrets).
- Section structure:
  - Define exact docs folders and page list for each module (lessons + labs + troubleshooting).
  - Define weekly pages (Weeks 1–13) that link to the relevant lessons.
  - Define Capstone pages (end-to-end narrative + deliverables checklist).
- Research approach:
  - Research-concurrent writing: verify claims while drafting; keep a "to-verify" queue; prefer primary docs for ROS 2, Gazebo, Unity, Isaac, Nav2.
  - Citation approach: cite sources close to each claim; maintain a running bibliography per module.
- Quality validation:
  - Per-page checklist (clarity, prerequisites, steps, expected output, troubleshooting, glossary links).
  - Per-release checklist (build, links, images/alt text, sidebar integrity, deploy success).

Decisions needing documentation (record: decision, options, tradeoffs, decision rule):
- Information architecture choice: Modules-first vs Weeks-first vs Hybrid (Modules + Week-path).
- Code policy: snippets only vs runnable companion repo(s).
- ROS 2 baseline: Humble vs Iron; compatibility notes.
- Simulation depth: Gazebo deep + Unity overview vs equal depth.
- Isaac accessibility: local RTX requirement vs cloud workflow; include latency/safety guidance.
- Capstone structure: single "golden path" vs multiple variants (sim-only, sim-to-real, edge-first).
- Hardware guidance style: exact prices vs ranges with date stamps vs no prices.
- Diagram tooling: Mermaid vs SVG/PNG exports; maintainability tradeoffs.

Testing strategy (validation checks tied to acceptance criteria):
- Content acceptance tests:
  - Module coverage: 4 modules, ≥4 lessons each, labs included.
  - Week path coverage: Weeks 1–13 pages exist and link correctly (no dead ends).
  - Capstone completeness: voice→plan→navigate→perceive→manipulate pipeline described + deliverables checklist.
  - Hardware section: includes On-Prem + Cloud "Ether Lab" + "latency trap" warning.
- Technical acceptance tests:
  - Docusaurus build passes locally and in CI (no errors; warnings triaged).
  - Broken-link checks and image/asset checks pass.
  - GitHub Pages deploy succeeds; site loads and reflects latest release.
- Editorial acceptance tests:
  - Consistent templates (Concept → Example → Lab → Troubleshooting → Checkpoint).
  - Glossary terms linked at first mention; consistent frontmatter and headings.

Technical details:
- Use research-concurrent approach (research while writing, not all upfront).
- Follow the project constitution's citation and quality rules.
- Organize work by phases: Research → Foundation → Analysis → Synthesis.
- Produce milestone schedule aligned to the hackathon timeline (tasks per day/week).

Outputs required from this plan:
- Architecture & IA page (with diagrams).
- Decisions log (ADR-style bullets are fine).
- QA checklist for contributors.
- Milestone schedule mapped to pages/lessons/labs.
```

## Response Summary

Created a comprehensive implementation plan for the Physical AI & Humanoid Robotics textbook project with:

1. **Architecture & Information Architecture**:
   - Detailed content architecture with complete Docusaurus IA structure
   - System/data-flow architecture showing both physical robotics pipeline and VLA loop
   - Delivery architecture with GitHub Actions CI/CD workflow

2. **Section Structure**:
   - Exact docs folders and page lists for all 4 modules with lessons, labs, and troubleshooting
   - Weekly path structure for Weeks 1-13
   - Complete capstone, labs, glossary, and hardware/lab setup structures

3. **Research Approach**:
   - Research-concurrent writing strategy with verification queue
   - Primary documentation sources identified for each technology
   - Citation approach with running bibliography guidance

4. **Quality Validation**:
   - Comprehensive per-page checklist template
   - Per-release checklist for builds and deployments
   - Specialized checklists for different content types

5. **Architectural Decisions Log**:
   - 8 major decisions documented with context, options, trade-offs, and rationale
   - Covers IA, code policy, technology choices, and content structure

6. **Testing Strategy**:
   - Content, technical, and editorial acceptance tests aligned with success criteria
   - Validation checks for all specified requirements

7. **Milestone Schedule**:
   - 6-phase schedule aligned to 6-week timeline
   - Specific tasks mapped to days/weeks for structured development

The plan provides a complete roadmap for implementing the textbook project with clear structure, quality standards, and development timeline.

## Files Created

- specs/1-humanoid-robotics/plan.md: Comprehensive implementation plan with architecture, structure, and schedule
- specs/1-humanoid-robotics/qa-checklist.md: Detailed quality assurance checklist for contributors
- specs/1-humanoid-robotics/architectural-decisions.md: Complete log of major architectural decisions with rationale