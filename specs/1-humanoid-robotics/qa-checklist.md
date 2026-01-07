# Quality Assurance Checklist for Contributors

## Per-Page Content Checklist

### Learning Objectives & Structure
- [ ] Clear learning objectives stated at the top of the page
- [ ] Prerequisites clearly listed (what knowledge/dependencies are needed)
- [ ] Content follows logical progression (concept → example → lab → troubleshooting)
- [ ] Learning checkpoints or summaries included where appropriate

### Technical Content Quality
- [ ] All technical claims verified against primary documentation
- [ ] Code examples are complete, properly formatted, and tested
- [ ] Commands are clearly distinguished from explanations
- [ ] Expected outputs clearly described for all examples
- [ ] Error handling and common failure modes addressed

### Accessibility & Usability
- [ ] Images have appropriate alt text descriptions
- [ ] All internal links work correctly (no broken links)
- [ ] External links open in new tabs when appropriate
- [ ] Glossary terms linked at first mention in the document
- [ ] Consistent formatting and styling throughout

### Troubleshooting & Support
- [ ] Troubleshooting section included with common issues
- [ ] Error messages and their solutions clearly documented
- [ ] Alternative approaches mentioned where relevant
- [ ] Performance considerations and optimizations noted

## Per-Module Quality Checklist

### Content Coverage
- [ ] Module has 4+ lessons/pages as specified
- [ ] All key concepts from the specification are covered
- [ ] Practical examples connect to real-world applications
- [ ] Labs included with clear steps and expected outcomes
- [ ] Troubleshooting guide comprehensive for the module

### Integration & Navigation
- [ ] Module links properly integrated into sidebar navigation
- [ ] Cross-module references work correctly
- [ ] Forward references to later modules are appropriately noted
- [ ] Module summary/recap included

## Per-Release Quality Checklist

### Build & Deployment
- [ ] Docusaurus build passes locally (npm run build)
- [ ] No build warnings or errors
- [ ] All assets (images, diagrams, code) load correctly
- [ ] GitHub Actions CI build passes
- [ ] GitHub Pages deployment succeeds

### Navigation & Structure
- [ ] Sidebar navigation complete and accurate
- [ ] No orphan pages (all pages accessible through navigation)
- [ ] Weekly path links work correctly
- [ ] Capstone section properly connected to modules
- [ ] Glossary terms properly linked throughout

### Content Validation
- [ ] All external links verified and working
- [ ] Broken-link check passes (use Docusaurus link checker)
- [ ] All images load correctly across different screen sizes
- [ ] Code syntax highlighting works properly
- [ ] Mathematical notation renders correctly if present

## Editorial Standards Checklist

### Writing Quality
- [ ] Clear, concise language appropriate for target audience
- [ ] Consistent terminology throughout the module and book
- [ ] Technical terms defined before use or linked to glossary
- [ ] Passive voice minimized, active voice preferred
- [ ] Sentences are well-structured and easy to follow

### Formatting & Style
- [ ] Consistent heading hierarchy (H1 for page title, H2 for sections, etc.)
- [ ] Code blocks properly formatted with appropriate language tags
- [ ] Lists are properly formatted (numbered for steps, bulleted for items)
- [ ] Tables have appropriate headers and are readable
- [ ] Callout blocks (info, warning, tip) used appropriately

### Citations & References
- [ ] All technical claims cite primary documentation sources
- [ ] Links to external resources are current and working
- [ ] Bibliography maintained per module where needed
- [ ] Version numbers specified for software/tools where important
- [ ] OS and environment assumptions clearly stated

## Specialized Module Checklists

### For ROS 2 Content (Module 1)
- [ ] ROS 2 Humble baseline consistently referenced
- [ ] rclpy examples work correctly
- [ ] Launch file examples complete and tested
- [ ] Node, topic, service, action concepts clearly differentiated
- [ ] URDF examples valid and properly structured

### For Simulation Content (Module 2)
- [ ] Gazebo version compatibility noted
- [ ] World file examples complete and tested
- [ ] URDF/SDF conversion concepts clear
- [ ] Sensor simulation examples work
- [ ] Unity integration concepts explained at appropriate level

### For AI/Isaac Content (Module 3)
- [ ] Isaac Sim and Isaac ROS concepts clearly differentiated
- [ ] VSLAM concepts explained with required sensors
- [ ] Nav2 integration concepts clear
- [ ] Hardware requirements clearly stated with alternatives
- [ ] Sim-to-real concepts and limitations addressed

### For VLA Content (Module 4)
- [ ] VLA pipeline clearly explained from voice to action
- [ ] Cognitive planning concepts connected to ROS 2 actions
- [ ] Safety considerations and guardrails addressed
- [ ] Latency considerations noted for cloud vs edge
- [ ] Multimodal interaction concepts covered

## Final Pre-Publication Checklist

### Content Completeness
- [ ] All 4 modules completed with 4+ lessons each
- [ ] All 13 weeks covered in weekly path
- [ ] Capstone section complete with pipeline narrative and checklist
- [ ] All labs included with troubleshooting
- [ ] Glossary comprehensive and linked
- [ ] Hardware/lab setup section complete with on-prem and cloud guidance

### Technical Validation
- [ ] Site builds without errors
- [ ] All internal navigation works
- [ ] All external links verified
- [ ] All images and diagrams display correctly
- [ ] Code examples tested and functional
- [ ] Search functionality works properly

### Quality Assurance
- [ ] Content reviewed for accuracy against current documentation
- [ ] Consistency check completed across all modules
- [ ] Accessibility features verified
- [ ] Mobile responsiveness tested
- [ ] All troubleshooting sections complete and accurate
- [ ] All glossary references properly linked