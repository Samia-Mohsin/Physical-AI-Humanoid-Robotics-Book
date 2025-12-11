<!--
Sync Impact Report
Version change: 1.0.0 → 1.1.0
Modified principles: Enhanced RAG Chatbot integration and agent skills focus
Added sections: Agent Skills, Subagents, Databases, Authentication, RAG Chatbot features
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ✅ updated
- .specify/templates/spec-template.md: ✅ updated
- .specify/templates/tasks-template.md: ✅ updated
- .specify/templates/commands/*.md: ✅ updated
Follow-up TODOs: None
-->
# AI/Spec-Driven Book on Physical AI & Humanoid Robotics Constitution

## Core Principles

### Educational clarity
Content must be structured for clear progression from beginner to advanced topics, ensuring accessibility and depth for all learners.

### Technical accuracy
All technical content, especially regarding ROS2, Gazebo, Unity, NVIDIA Isaac, VLA, and RAG, must be precise and verifiable.

### Practical outcomes
The book will prioritize hands-on learning with a simulation-first approach, guiding students to practical implementation.

### Ethical responsibility
Content will emphasize safety, reproducible AI practices, and responsible deployment considerations in humanoid robotics.

### Personalization
The platform will offer optional content adaptation based on user profiles and skill levels to enhance individual learning experiences.

### RAG Integration
The platform will provide comprehensive Retrieval-Augmented Generation (RAG) capabilities with OpenAI Agents/ChatKit SDK integration, Neon Postgres for conversation history, and Qdrant Cloud for vector storage.

## Standards and Quality

- Content must be original and traceable to authoritative sources.
- Code examples must be fully runnable on Ubuntu 22.04 with ROS2 Humble/Iron.
- Robotics concepts must align with real ROS2 control and URDF standards.
- Agentic AI content must reflect production practices.
- Inline citations to official documentation or reputable research are mandatory.
- The tone should be consistently mentor-to-student, respectful, and clear.
- RAG responses must include source citations and be context-aware of selected text.
- Authentication must use Better-Auth with background user profiling.

## Structure and Features

- Chapters will start from a specification, followed by objectives, examples, and exercises.
- Modules will include inputs, outputs, architecture, code, failure modes, and safety notes.
- RAG Chatbot integration will allow answering questions using selected text and provide optional personalized guidance.
- Optional buttons per chapter will enable translation to Urdu and content personalization based on user skills.
- Interactive features include live code blocks, interactive quizzes, text selection popup, hover tooltips, accordion sections, and progress tracking.
- Multilingual support for English, Urdu, and Roman Urdu with translation buttons.
- Docusaurus embedding for seamless chatbot integration with bottom-right position and Ask/Explain Selection buttons.

## Agent Skills

- explain_concept: Ability to explain complex concepts in simple terms
- generate_quiz: Create interactive quizzes based on content
- translate_to_urdu: Translate content to Urdu language
- translate_to_roman_urdu: Translate content to Roman Urdu
- simplify_for_beginner: Simplify content for beginner learners
- add_advanced_code: Provide advanced code examples for expert learners
- explain_diagram_vision: Explain visual diagrams and concepts
- retrieve_rag: Retrieve information using RAG capabilities

## Subagents

- ContentGenerator: Generates educational content based on specifications
- Personalizer: Adapts content based on user preferences and skill level
- UrduTranslator: Translates content to Urdu
- RomanUrduConverter: Converts content to Roman Urdu
- QuizMaster: Creates and manages interactive quizzes
- DiagramExplainer: Explains visual diagrams and concepts
- RagIngester: Handles document ingestion for RAG system

## Databases

- Neon Postgres for conversation history and user data
- Qdrant Cloud for vector storage and similarity search

## Authentication

- Better-Auth provider for secure user authentication
- Background questions for user profiling and personalization

## RAG Chatbot Features

- OpenAI Agents/ChatKit SDK integration
- Selected text context awareness
- Ask and Explain Selection buttons
- Bottom-right position in Docusaurus interface
- Docusaurus embedding for seamless integration

## Constraints

- Total length: 20,000–35,000 words.
- Minimum 1 working simulation demo.
- Images/figures include alt text.
- Use Docusaurus Markdown + Mermaid diagrams.
- Open, reproducible workflows only.
- RAG responses must be under 2 seconds response time.
- Support for 100+ concurrent users.

## Success Criteria

- Student can build humanoid pipeline end-to-end.
- Book deploys to GitHub Pages with no build errors.
- Code reproducible; instructions precise.
- Simulation-first approach; optional real-world deployment.
- RAG chatbot successfully answers questions with source citations.
- Authentication system works with Better-Auth.
- Multilingual support for all three languages (English, Urdu, Roman Urdu).

## Governance

This constitution supersedes all other project practices and documentation. Amendments require thorough documentation, explicit approval from project leads, and a clear migration plan for any affected systems or content. All pull requests and code reviews must explicitly verify compliance with the principles and standards outlined herein.

**Version**: 1.1.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-10
