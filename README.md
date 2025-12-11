# Physical AI & Humanoid Robotics Educational Platform

An interactive educational platform for Physical AI & Humanoid Robotics with multilingual support and integrated RAG chatbot.

## Features

- **Integrated RAG Chatbot**: Built with OpenAI Agents SDK, answers questions about book content with selected-text priority mode
- **Claude Code Subagents**: ContentGenerator, Personalizer, UrduTranslator, RomanUrduConverter, QuizMaster, DiagramExplainer, RagIngester
- **Reusable Agent Skills**: explain_concept, generate_quiz, translate_to_urdu, translate_to_roman_urdu, simplify_for_beginner, add_advanced_robotics_code, explain_diagram_with_vision, retrieve_and_answer_from_selection, generate_ros2_node, generate_urdf_from_description, debug_gazebo_launch
- **Better-Auth**: With background questions (experience_level, has_rtx_gpu, has_jetson, preferred_language)
- **Personalization**: One-click "Personalize for me" button per chapter
- **Multilingual Support**: Urdu and Roman Urdu translation toggle
- **Interactive Elements**: Live code blocks, interactive quizzes with progress tracking
- **Docusaurus Frontend**: With embedded chatbot widget
- **FastAPI Backend**: With Neon Postgres and Qdrant Cloud integration

## Tech Stack

- **Frontend**: Docusaurus v3, React, TypeScript
- **Backend**: FastAPI, Python
- **Database**: Neon Serverless Postgres
- **Vector Store**: Qdrant Cloud Free Tier
- **Authentication**: Better-Auth
- **AI Services**: OpenAI API (GPT-4o, embeddings)

## Setup

### Prerequisites

- Node.js 18+
- Python 3.9+
- Git

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

5. Start the backend server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Architecture

- **Frontend**: Docusaurus with React components, interactive MDX, live code blocks, quizzes, and embedded chatbot
- **Backend**: FastAPI with OpenAI integration, Neon Postgres for user data, and Qdrant Cloud for vector storage
- **Deployment**: GitHub Pages for frontend, Vercel serverless for backend

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# Backend Configuration
PORT=8000
APP_ENV=development
APP_DEBUG=true

# Frontend Configuration
FRONTEND_URL=http://localhost:3000

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Neon Postgres Database Configuration
DATABASE_URL=your_neon_postgres_connection_string_here

# Better-Auth Configuration
BETTER_AUTH_SECRET=your_secure_auth_secret_here
BETTER_AUTH_URL=http://localhost:8000

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_EXPIRES_IN=7d
```

## Deployment

### Frontend (GitHub Pages)
```bash
npm run build
GIT_USER=<Your GitHub username> CURRENT_BRANCH=main npm run deploy
```

### Backend (Vercel)
```bash
vercel --prod
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify your license here]