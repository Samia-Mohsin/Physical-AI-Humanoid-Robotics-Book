---
id: 1
title: Create Curriculum Content Spec
stage: spec
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: curriculum-content
branch: 16-curriculum-content
user: samia
command: /sp.specify
labels: [spec,curriculum,robotics,ai,humanoid,ros,isaac,gazebo,vla]
links:
  spec: specs/16-curriculum-content.yaml
  ticket: null
  adr: null
  pr: null
files:
 - specs/16-curriculum-content.yaml
 - specs/16-curriculum-content/checklists/requirements.md
tests:
---

## Prompt

component: book_content
constitution: true
source: advanced_12_module_curriculum_2025
generate_with_claude_subagent: ContentGeneratorPro
features:
  - mdx_v2
  - live_code_blocks: [python, bash, xml]  # URDF examples
  - interactive_quizzes_per_section
  - embedded_simulation_widgets  # Future: WebAssembly Gazebo
  - translation_ready: [urdu, roman_urdu]
output: frontend/docs/

chapters:
  - number: "01"
    title: "The Robotic Nervous System — ROS 2 Masterclass"
    subtitle: "From Zero to Real-Time Distributed Control"
    duration: 3 weeks
    topics:
      - ROS 2 Architecture Deep Dive (DDS, Real-Time, QoS Profiles, Managed Nodes)
      - Multi-machine deployment (Docker Compose + Micro-ROS on ESP32)
      - Building Python & C++ Nodes with rclpy & rclcpp
      - Advanced Topics: Parameters, Lifecycle Nodes, Bond, Component Manager
      - Bridging OpenAI/Realtime Agents → ROS 2 Actions using ros2-web-bridge
      - URDF → xacro → SDF conversion pipeline for Gazebo & Isaac Sim
      - Hands-on: Deploy a 7-DOF humanoid arm controller using MoveIt 2 + ROS 2 Control

  - number: "02"
    title: "Digital Twin Engineering — Gazebo Fortress & Unity Robotics"
    subtitle: "Photorealistic Simulation at Scale"
    duration: 3 weeks
    topics:
      - Gazebo Ignition Fortress + ROS 2 Integration (ign_ros2_control)
      - Physics tuning: Bullet vs Dart, friction, ERP, CFM parameters
      - Sensor simulation: GPU LiDAR (NVIDIA Ray Tracing), Depth Camera noise models, IMU drift
      - Unity Robotics Hub + ROS-TCP-Connector for high-fidelity HRI rendering
      - Domain Randomization pipelines (textures, lighting, dynamics)
      - Sim-to-Real gap analysis with real Boston Dynamics Spot data
      - Hands-on: Build a full digital twin of Tesla Optimus with accurate mass distribution

  - number: "03"
    title: "NVIDIA Isaac™ Platform — The GPU-Accelerated Brain"
    subtitle: "From Synthetic Data to Deployed Perception"
    duration: 4 weeks
    topics:
      - Isaac Sim 2024.2 + Omniverse Extensions for humanoid training
      - PhysX 5 GPU rigid body + TensorRT-accelerated perception
      - Isaac ROS 2 GEMs: NVBlox (3D Neural Reconstruction), CuVSLAM, DeepMap
      - Synthetic data generation at 10k FPS with Replicator + SDXL
      - Training VLA models using Isaac Lab (Open X-Embodiment dataset)
      - Nav2 + Behavior Trees for bipedal navigation with dynamic obstacle avoidance
      - Hands-on: Train a humanoid to walk using PPO + Domain Randomization in Isaac Gym

  - number: "04"
    title: "Vision-Language-Action Models (VLA) — The Future of Robot Intelligence"
    subtitle: "LLMs as the Universal Robot Brain"
    duration: 4 weeks
    topics:
      - OpenAI RT-2, Google RT-X, Microsoft VLA-2 architecture breakdown
      - Whisper → GPT-4o → ROS 2 Action Sequence (zero-shot voice commands)
      - Prompt engineering for spatial reasoning ("pick the red cup behind the chair")
      - VLA fine-tuning using Open X-Embodiment + RoboNet datasets
      - Memory-Augmented VLA (vector DB of past experiences)
      - Safety layer: Formal verification of generated action sequences
      - Hands-on: Build an end-to-end voice-controlled humanoid using Realtime API + ROS 2

  - number: "05"
    title: "Perception Stack for Humanoids"
    subtitle: "Seeing, Understanding, and Predicting the World"
    duration: 2 weeks
    topics:
      - Foundation models: Segment Anything 2 (SAM-2), Grounding DINO, Depth Anything V2
      - Real-time 3D scene graphs using NVBlox + Kimera
      - Human pose + hand tracking (MediaPipe + FrankMocap)
      - Dynamic object tracking in cluttered environments
      - Multimodal fusion: Vision + Language + Tactile (GelSight + DIGIT sensors)

  - number: "06"
    title: "Bipedal Locomotion & Whole-Body Control"
    subtitle: "From Walking to Parkour"
    duration: 3 weeks
    topics:
      - Model-based: LQR, MPC, Centroidal Dynamics
      - Learning-based: Reinforcement Learning (PPO, SAC) in Isaac Gym
      - Hybrid: Cassio et al. (Google DeepMind) style hierarchical control
      - Zero-moment point (ZMP), Capture Point, Divergent Component of Motion
      - Whole-body operational space control with quadratic programming

  - number: "07"
    title: "Dexterous Manipulation & In-Hand Reorientation"
    subtitle: "Beyond Parallel Grippers"
    duration: 2 weeks
    topics:
      - Advanced manipulation techniques beyond simple grippers

## Response snapshot

Created a comprehensive specification for the Advanced Physical AI & Humanoid Robotics Curriculum with 7 detailed chapters. The specification includes:

1. User Scenarios & Testing section with 3 prioritized user stories:
   - Comprehensive Learning Journey (P1)
   - Interactive Learning Experience (P1)
   - Multilingual Accessibility (P2)

2. Functional Requirements section with 12 specific requirements covering:
   - All 7 chapters with their specific titles, durations, and comprehensive topics
   - Claude ContentGeneratorPro integration for content generation
   - MDX v2 format support
   - Live code blocks for Python, Bash, and XML (URDF examples)
   - Interactive quizzes per section
   - Embedded simulation widgets support
   - Translation readiness for Urdu and Roman Urdu
   - Proper output to frontend/docs directory
   - Technical accuracy maintenance
   - Duration preservation for each chapter
   - Hands-on project inclusion
   - Content quality assurance

3. Success Criteria with 7 measurable outcomes including:
   - Comprehensive chapter creation with specified topics
   - Functional live code examples across all supported languages
   - Interactive quiz presence and functionality
   - Content accuracy verification
   - Proper MDX formatting and Docusaurus integration
   - Translation placeholder availability
   - Hands-on project completion success

4. Quality checklist to validate the specification completeness.

## Outcome

- ✅ Impact: Created comprehensive curriculum specification preserving all 7 chapters with detailed topics
- 🧪 Tests: No automated tests run, but quality checklist completed
- 📁 Files: Created spec file and quality checklist
- 🔁 Next prompts: Ready for /sp.plan or /sp.clarify
- 🧠 Reflection: Specification maintains all requested content while adding interactive features and multilingual support

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Not applicable