# Quickstart Guide for Physical AI & Humanoid Robotics Platform

## Prerequisites

- Node.js 18+ (for frontend development)
- Python 3.9+ (for backend development)
- Git
- Access to OpenAI API key
- Access to Qdrant Cloud (free tier)
- Access to Neon Postgres (free tier)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/physical-ai-humanoid-book.git
cd physical-ai-humanoid-book
```

### 2. Backend Setup

#### Create Backend Directory Structure
```bash
mkdir -p backend/app/{routers,services,models}
mkdir -p backend/ingest
```

#### Create Backend Requirements File
```bash
# Create backend/requirements.txt
cat > backend/requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
openai==1.3.7
qdrant-client==1.7.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
better-exceptions==0.3.2
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
cryptography==41.0.8
httpx==0.25.2
EOF
```

#### Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Create Environment File
```bash
# Create backend/.env
cat > backend/.env << EOF
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=your_neon_postgres_connection_string_here
BETTER_AUTH_SECRET=your_secure_auth_secret_here
JWT_SECRET_KEY=your_jwt_secret_key_here
EOF
```

### 3. Frontend Setup

#### Create Frontend Directory Structure
```bash
cd ..
mkdir -p frontend/src/{components,contexts,utils}
mkdir -p frontend/docs
```

#### Initialize Docusaurus
```bash
cd frontend
npm init -y
npm install @docusaurus/core@latest @docusaurus/preset-classic@latest @mdx-js/react@3.0.0 react@18.2.0 react-dom@18.2.0
```

#### Create Docusaurus Config
```bash
# Create frontend/docusaurus.config.js
cat > docusaurus.config.js << EOF
// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Digital Brain to Embodied Intelligence',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-username.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: 'your-username', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-book', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur', 'roman_urdu'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/your-username/physical-ai-humanoid-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Book',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/physical-ai-humanoid-book',
              },
            ],
          },
        ],
        copyright: \`Copyright © \${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.\`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
EOF
```

#### Create Sidebars Configuration
```bash
# Create frontend/sidebars.js
cat > sidebars.js << EOF
/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Chapters',
      items: [
        'chapter-1',
        'chapter-2',
        'chapter-3',
        'chapter-4',
        'chapter-5',
        'chapter-6',
        'chapter-7',
      ],
    },
  ],
};

export default sidebars;
EOF
```

### 4. Run Development Servers

#### Backend Development Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

#### Frontend Development Server
```bash
cd frontend
npm run start
```

## Key Features Implementation

### 1. RAG Chatbot with Selected-Text Mode

The chatbot component will be implemented in `frontend/src/components/Chatbot.tsx` and will include:
- Text selection detection using browser Selection API
- Integration with backend RAG service
- Streaming responses
- Source citations
- Claude agent skills integration

### 2. Authentication with Better-Auth

The authentication system will include:
- Registration with background questions (experience level, hardware, etc.)
- Login/logout functionality
- User profile management
- Integration with Neon Postgres

### 3. Personalization Engine

The personalization feature will:
- Adapt content based on user profile
- Provide simplified or advanced content based on experience level
- Integrate with GPT-4o for content modification

### 4. Translation System

The translation system will:
- Provide Urdu and Roman Urdu translations
- Cache translations in Neon Postgres
- Use GPT-4o for accurate robotics terminology translation

## Environment Variables

### Backend (.env)
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: URL to your Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant Cloud
- `DATABASE_URL`: Connection string for Neon Postgres
- `BETTER_AUTH_SECRET`: Secret for Better-Auth
- `JWT_SECRET_KEY`: Secret for JWT token generation

### Frontend (Environment variables will be passed through backend API)

## Deployment

### Frontend Deployment (GitHub Pages)
```bash
# Build the frontend
cd frontend
npm run build

# Deploy to GitHub Pages
GIT_USER=<Your GitHub username> \
  CURRENT_BRANCH=main \
  USE_SSH=true \
  npm run deploy
```

### Backend Deployment (Vercel)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
cd backend
vercel --prod
```

## Development Workflow

1. **Feature Development**: Create feature branches from main
2. **Testing**: Write unit tests for backend services and component tests for frontend
3. **Code Review**: Submit pull requests for all changes
4. **CI/CD**: Automated testing and deployment pipelines

## Troubleshooting

### Common Issues

1. **Environment Variables Not Loading**
   - Ensure .env file is in the correct directory
   - Check that environment variables are properly named

2. **Database Connection Issues**
   - Verify Neon Postgres connection string
   - Check firewall settings

3. **Qdrant Connection Issues**
   - Verify Qdrant URL and API key
   - Check network connectivity

4. **OpenAI API Issues**
   - Verify API key is valid and has sufficient credits
   - Check API usage limits