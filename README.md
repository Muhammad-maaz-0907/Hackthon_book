# Physical AI & Humanoid Robotics Book

[![Documentation Status](https://img.shields.io/badge/docs-online-brightgreen)](https://your-book-url.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Living textbook for ROS 2, Simulation, NVIDIA Isaac, and Vision-Language-Action Systems

## Overview

This repository contains the source code and content for the "Physical AI & Humanoid Robotics" textbook. The book covers essential topics for building humanoid robots, including ROS 2, simulation environments, NVIDIA Isaac ecosystem, and Vision-Language-Action systems.

## Features

- 📘 **Comprehensive Content**: Modules covering ROS 2, Gazebo/Unity simulation, NVIDIA Isaac, and VLA systems
- 💬 **AI-Powered Chatbot**: RAG-based assistant with citation support for questions about the book
- 📚 **Interactive Learning**: Hands-on labs and practical exercises
- 🌐 **Docusaurus Powered**: Modern, searchable documentation site
- 🔍 **Citation Support**: References to specific sections and pages in responses

## Table of Contents

### Core Modules

1. **Module 1**: The Robotic Nervous System (ROS 2)
2. **Module 2**: The Digital Twin (Gazebo & Unity)
3. **Module 3**: The AI-Robot Brain (NVIDIA Isaac)
4. **Module 4**: Vision-Language-Action (VLA) & Conversational Robotics
5. **Capstone**: Autonomous Humanoid Project

## RAG Chatbot Integration

The textbook includes an AI-powered chatbot that can answer questions about the book content with proper citations.

### Features

- **Normal RAG Mode**: Search entire book for answers with citations
- **Selected Text Mode**: Answer questions based only on selected text
- **Citation Support**: Shows document path, heading, and chunk ID for all sources
- **Safety**: Prevents hallucinations and prompt injection attacks

### Technical Stack

- **Backend**: FastAPI, OpenAI GPT-4o, Qdrant vector database
- **Frontend**: React component integrated into Docusaurus
- **Database**: Neon Postgres for metadata and logging

## Getting Started

### Prerequisites

- Node.js 18+ for Docusaurus
- Python 3.11+ for backend services
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/humanoid-robotics-book.git
cd humanoid-robotics-book
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Start the development server:
```bash
# Frontend
npm start

# Backend (in separate terminal)
cd backend
uvicorn main:app --reload
```

### Setting Up the Chatbot

1. Configure environment variables:
```bash
cd backend
cp .env.example .env
# Edit .env with your API keys
```

2. Index the book content:
```bash
python index_documents.py
```

3. The chatbot will be available on all documentation pages.

## Project Structure

```
humanoid-robotics-book/
├── docs/                    # Book content (markdown files)
│   ├── module1-ros2/        # ROS 2 module content
│   ├── module2-digital-twin/ # Simulation module content
│   ├── module3-ai-brain/    # NVIDIA Isaac module content
│   └── module4-vla/         # VLA systems module content
├── src/                     # Docusaurus custom components
│   └── components/Chatbot/  # AI chatbot React component
├── backend/                 # FastAPI backend services
│   ├── main.py             # Main API application
│   ├── rag_engine.py       # RAG implementation
│   ├── qdrant_client.py    # Vector database client
│   └── postgres_client.py  # Metadata database client
├── docusaurus.config.ts    # Docusaurus configuration
└── package.json            # Frontend dependencies
```

## Development

### Adding New Content

1. Create markdown files in the appropriate module directory under `docs/`
2. Update `sidebars.ts` to include the new content in the navigation
3. The content will automatically be indexed by the chatbot

### Extending the Chatbot

The chatbot is built with extensibility in mind:

- **Backend**: Modify `backend/` for new API endpoints or enhanced RAG logic
- **Frontend**: Update `src/components/Chatbot/` for UI enhancements
- **Safety**: Extend `backend/safety_checker.py` for additional safety measures

## Deployment

### Documentation Site

The Docusaurus site can be deployed to GitHub Pages, Netlify, Vercel, or any static hosting service.

### Backend Services

The backend services (FastAPI, Qdrant, Postgres) can be deployed separately:

- **Containerized**: Using Docker Compose
- **Cloud**: AWS, GCP, Azure, or Heroku
- **Kubernetes**: With provided manifests

See the [backend deployment guide](backend/DEPLOYMENT.md) for detailed instructions.

## Contributing

We welcome contributions to improve the textbook content and chatbot functionality:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Content Contributions

- Add new modules or improve existing content in the `docs/` directory
- Create new practical exercises and labs
- Enhance examples with code and diagrams

### Technical Contributions

- Improve the RAG system and retrieval accuracy
- Add new safety measures and hallucination detection
- Enhance the user interface and experience

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 **Documentation**: [Link to docs]
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/humanoid-robotics-book/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/humanoid-robotics-book/discussions)

## Acknowledgments

- The Docusaurus team for the excellent documentation framework
- The OpenAI team for powerful language models
- The Qdrant team for vector database technology
- The robotics and AI research community for inspiration

## Roadmap

- [ ] Advanced multimodal capabilities
- [ ] Interactive code examples
- [ ] Video integration
- [ ] Mobile-optimized chat interface
- [ ] Offline content access
- [ ] Translation support

---

Built with ❤️ for the humanoid robotics community.
