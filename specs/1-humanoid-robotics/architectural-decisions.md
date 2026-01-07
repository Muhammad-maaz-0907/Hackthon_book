# Architectural Decisions Log

## Decision 1: Information Architecture - Hybrid Approach
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Need to organize content for both structured learning (progressive weeks) and reference use (by technical topic).

**Decision**:
Adopt a hybrid approach that combines modules-first organization with a weekly path overlay.

**Options Considered**:
- Modules-first: Organize primarily by technical topics (ROS 2, Simulation, AI, VLA)
- Weeks-first: Organize primarily by time progression (Week 1, Week 2, etc.)
- Hybrid: Combine both approaches with modules as primary organization and weekly path as learning guide

**Trade-offs**:
- Modules-first: Good for reference and deep learning, harder for structured progression
- Weeks-first: Good for structured learning, harder for reference and jumping between topics
- Hybrid: Serves both use cases but creates more complex navigation

**Rationale**:
The hybrid approach best serves both structured learners who want to follow a weekly progression and reference users who need to access specific technical topics. The modules provide the comprehensive technical coverage while the weekly path provides a guided learning experience.

**Consequences**:
- More complex sidebar navigation
- Need to maintain two pathways (module and weekly)
- Better user experience for both learning styles

## Decision 2: Code Policy - Snippets Only
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Decide whether to include complete runnable code in documentation or focus on conceptual code snippets.

**Decision**:
Include code snippets only in documentation, not complete runnable repositories.

**Options Considered**:
- Full runnable code: Include complete, runnable examples in docs
- Snippets only: Include focused code examples that illustrate concepts
- Companion repo: Separate repository with complete examples

**Trade-offs**:
- Full runnable code: Students can run immediately, but docs become bloated and harder to maintain
- Snippets only: Clean, focused documentation, students need to integrate concepts themselves
- Companion repo: Balanced approach but adds complexity

**Rationale**:
Snippets keep the focus on concepts rather than implementation details, making the educational content clearer and more maintainable. Students will need to practice integrating concepts, which is valuable learning.

**Consequences**:
- Cleaner, more focused documentation
- Students need to practice integration skills
- Easier maintenance of documentation

## Decision 3: ROS 2 Baseline - Humble Hawksbill (LTS)
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Choose which ROS 2 distribution to use as the baseline for the course.

**Decision**:
Use ROS 2 Humble Hawksbill as the baseline distribution.

**Options Considered**:
- Humble Hawksbill (LTS): Long-term support, stable, longer maintenance window
- Iron Irwini: Latest features, shorter support window

**Trade-offs**:
- Humble: More stable, longer support, fewer breaking changes, but fewer new features
- Iron: Latest features and capabilities, but shorter support and potential breaking changes

**Rationale**:
For an educational textbook, stability and long-term support are more important than cutting-edge features. Humble provides the best balance of features and stability for a learning environment.

**Consequences**:
- Longer maintenance window for the course
- More stable learning environment
- May miss some newer features but gains stability

## Decision 4: Simulation Depth - Gazebo Focus + Unity Overview
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Determine the depth of coverage for Gazebo vs Unity simulation tools.

**Decision**:
Provide deep coverage of Gazebo with overview-level coverage of Unity.

**Options Considered**:
- Equal depth: Provide similar depth for both Gazebo and Unity
- Gazebo focus: Deep Gazebo coverage, Unity overview
- Unity focus: Deep Unity coverage, Gazebo overview

**Trade-offs**:
- Equal depth: More comprehensive but complex and time-consuming
- Gazebo focus: Practical for ROS integration, Unity as enhancement
- Unity focus: High-fidelity visualization focus, but less ROS integration

**Rationale**:
Gazebo has deeper integration with the ROS ecosystem and is more commonly used in ROS-based robotics projects. Unity is valuable for high-fidelity visualization but is more of an enhancement to the core ROS simulation capabilities.

**Consequences**:
- Strong focus on ROS-integrated simulation with Gazebo
- Unity covered as enhancement tool
- Better alignment with ROS ecosystem

## Decision 5: Isaac Accessibility - Dual Approach with Guidance
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Address the hardware requirements and accessibility of NVIDIA Isaac tools.

**Decision**:
Document both local RTX requirements and cloud-based alternatives with clear tradeoff guidance.

**Options Considered**:
- Local RTX requirement: Focus on local high-performance setup
- Cloud-first: Focus on cloud-based Isaac services
- Dual approach: Document both with clear guidance

**Trade-offs**:
- Local RTX: Better performance, higher barrier to entry, cost
- Cloud-first: More accessible, potential latency issues, ongoing costs
- Dual approach: Comprehensive but requires more documentation

**Rationale**:
Different students have different resource constraints. Documenting both approaches with clear tradeoffs allows students to choose based on their specific situation while understanding the implications.

**Consequences**:
- More comprehensive documentation
- Accommodates students with different resource levels
- Clear guidance on performance vs accessibility tradeoffs

## Decision 6: Capstone Structure - Single Golden Path
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Determine how to structure the capstone project to accommodate different student capabilities and goals.

**Decision**:
Provide a single "golden path" capstone project with optional extensions.

**Options Considered**:
- Single path: One comprehensive capstone project
- Multiple variants: Different capstone options (sim-only, sim-to-real, edge-first)
- Tiered approach: Basic to advanced capstone options

**Trade-offs**:
- Single path: Clearer, more focused, but may not suit all students
- Multiple variants: More flexible but complex to document
- Tiered approach: Scales to different abilities but more complex

**Rationale**:
A single golden path provides clear direction and reduces confusion for students. Optional extensions can accommodate different interest levels without overwhelming beginners.

**Consequences**:
- Clear, focused capstone project
- Easier for students to follow
- Optional complexity for advanced students

## Decision 7: Hardware Guidance - Ranges with Date Stamps
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Decide how to provide hardware recommendations that remain useful over time.

**Decision**:
Provide hardware ranges with date stamps rather than exact prices.

**Options Considered**:
- Exact prices: Specific dollar amounts for components
- Ranges: Price ranges for different tiers
- No prices: Guidance without specific numbers
- Ranges with dates: Price ranges with timestamp for relevance

**Trade-offs**:
- Exact prices: Specific but quickly outdated
- Ranges: Flexible but less specific
- No prices: Never outdated but not actionable
- Ranges with dates: Balanced approach but requires updating

**Rationale**:
Ranges with dates provide actionable guidance while acknowledging that prices change over time. This gives students realistic expectations while noting the temporal context.

**Consequences**:
- Actionable hardware guidance
- Acknowledges price volatility
- Requires periodic updates to remain relevant

## Decision 8: Diagram Tooling - Mermaid + SVG/PNG Hybrid
**Date**: 2025-12-20
**Status**: Accepted

**Context**:
Choose diagram tooling that balances maintainability with visual quality.

**Decision**:
Use Mermaid for simple diagrams and SVG/PNG for complex technical diagrams.

**Options Considered**:
- Pure Mermaid: All diagrams in Mermaid format
- Pure SVG/PNG: All diagrams as static images
- Hybrid: Use different tools based on complexity needs

**Trade-offs**:
- Pure Mermaid: All diagrams maintainable in text, limited complexity
- Pure SVG/PNG: Full visual control, harder to maintain
- Hybrid: Best of both, requires tool decision per diagram

**Rationale**:
Simple flow diagrams benefit from Mermaid's text-based maintainability, while complex technical diagrams need the visual control of SVG/PNG. The hybrid approach optimizes for both maintainability and visual quality.

**Consequences**:
- Maintainable simple diagrams
- High-quality complex diagrams
- Need to choose appropriate tool per diagram type