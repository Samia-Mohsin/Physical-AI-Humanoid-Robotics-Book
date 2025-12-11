---
id: 1
title: Create Implementation Plan
stage: plan
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: full-platform
branch: main
user: samia
command: /sp.plan
labels: [plan,architecture,design,full-platform]
links:
  spec: specs/03-rag-chatbot.yaml
  ticket: null
  adr: null
  pr: null
files:
 - specs/plan.md
 - specs/research.md
 - specs/data-model.md
 - specs/contracts/api-contracts.md
 - specs/quickstart.md
tests:
 - specs/plan.md
 - specs/research.md
 - specs/data-model.md
 - specs/contracts/api-contracts.md
---

## Prompt

– Integrated RAG Chatbot
• Embedded in the published Docusaurus book (bottom-right)
• Built with OpenAI Agents SDK + ChatKit SDK
• FastAPI backend
• Neon Serverless Postgres + Qdrant Cloud Free Tier
• Fully working selected-text mode → when user selects text and clicks "Explain Selection", the bot answers using ONLY the selected text (priority_only mode)

– Claude Code Subagents & Reusable Agent Skills
• All subagents listed in constitution are active
• Agent skills (generate_ros2_node, generate_urdf_from_description, debug_gazebo_launch, etc.) are callable inside the RAG chatbot

– Better-Auth with background questions at signup
• Using https://www.better-auth.com/ latest
• At signup ask: experience_level, has_rtx_gpu, has_jetson, preferred_language
• Data stored in Neon

– Personalize button at start of each chapter
• One-click "Personalize for me" button at top of every chapter
• Calls /api/personalize → returns rewritten MDX adapted to user background using GPT-4o

– Urdu + Roman Urdu translation button
• One-click toggle at top of every chapter (English → اردو → Roman Urdu)
• Uses GPT-4o with perfect Nastaliq + robotics terminology
• Results cached in Neon per user per chapter

Strictly respect every clarification answer already given:
• Content generation = Claude + automated fact-checking (no human review)
• Simulation widgets = placeholder only (A)
• Live code blocks = static highlighted code only, no execution (D)
• Translation = machine-only GPT-4o, no human review (C)
• Quizzes = basic interactive forms with immediate feedback + Neon score tracking (A)

Deploy:
• Frontend → GitHub Pages (gh-pages branch)
• Backend → Vercel serverless (auto-detect)
• Include working ingest script that scrapes the final deployed book URL and loads every chapter into Qdrant
• Include .env.example with all keys
• Include README with live URLs and screenshots proving every bonus

summary:


Implement exactly these decisions from clarification phase:
- Content: Claude + automated fact-checking (no human review)
- Simulation widgets: Placeholder only (no WASM, no WebGL, no iframe simulation)
- Live code blocks: Static highlighted code only (no execution, no sandbox, no Node.js/WASM)
- Translation: Machine-only GPT-4o translation (no human review, no quality guarantees)
- Quizzes: Basic interactive forms with immediate feedback and Neon score tracking
- RAG Chatbot: OpenAI Agents + ChatKit SDK, selected_text priority mode fully working, streaming, bottom-right
- Auth: Better-Auth latest with custom background questions stored in Neon
- Personalization: One-click "Personalize for me" button per chapter using user profile from Neon
- Urdu/Roman Urdu: One-click toggle buttons per chapter, GPT-4o, cached in Neon
- Claude Subagents & Agent Skills: Fully activated and callable inside RAG chatbot (especially ROS2/URDF generation and Gazebo debugging)

Deploy:
- Frontend → GitHub Pages (gh-pages branch)
- Backend → Vercel serverless (auto-detected from backend/ folder)

Include:
- Working ingest script that scrapes the final deployed GitHub Pages URL and loads all chapters into Qdrant
- .env.example with all required variables
- README with live book URL, backend URL, and proof screenshots for all bonuses

## Response snapshot

Created comprehensive implementation plan for the Physical AI & Humanoid Robotics educational platform. The plan includes:

1. Complete architecture overview covering frontend (Docusaurus), backend (FastAPI), and deployment (GitHub Pages + Vercel)
2. Detailed research findings on key technologies (OpenAI Agents, Better-Auth, Qdrant, etc.)
3. Complete data model with all entities and relationships
4. Full API contract documentation with request/response examples
5. Quickstart guide for developers to get up and running

All constitution requirements and clarification decisions have been incorporated:
- RAG chatbot with selected-text priority mode
- Claude Code Subagents and agent skills
- Better-Auth with background questions
- Personalization engine
- Urdu/Roman Urdu translation
- Static live code blocks (no execution)
- Basic interactive quizzes with Neon tracking

The plan is structured in phases and includes risk analysis, success criteria, and deployment strategy.

## Outcome

- ✅ Impact: Created comprehensive implementation plan addressing all platform requirements
- 🧪 Tests: Created documentation artifacts for verification
- 📁 Files: Created plan.md, research.md, data-model.md, api-contracts.md, quickstart.md
- 🔁 Next prompts: Ready for /sp.tasks to generate implementation tasks
- 🧠 Reflection: Plan addresses all constitution bonuses and clarification requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Proceed to task generation phase