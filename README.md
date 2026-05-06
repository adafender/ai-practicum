# AI-Powered Customer Service Training Platform

This is an AI-powered customer service training platform that simulates realistic customer interactions using Retrieval-Augmented Generation (RAG) and Text-to-Speech (TTS). Trainees can practice handling various customer scenarios with intelligent AI personas, receive detailed performance feedback, and track their progress.

## Features

- **Interactive Conversations**: Engage in realistic customer service scenarios with AI-generated personas
- **RAG Integration**: Incorporates company documents for context-aware responses
- **Text-to-Speech**: Audio playback for customer responses
- **Performance Reports**: Detailed coaching feedback with scores and improvement suggestions
- **Multiple Personas**: Different customer types with varying difficulty levels
- **Document Upload**: Add custom company documents for RAG

## Prerequisites

Before running the application, ensure you have the following installed:

### Backend Requirements
- **Python 3.8 or higher**
- **OpenAI API Key** (for LLM and TTS functionality)

### Frontend Requirements
- **Node.js 16 or higher**
- **npm** (comes with Node.js)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adafender/ai-practicum.git
cd ai-practicum
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
pip install flask flask-cors langchain-openai openai python-dotenv faiss-cpu numpy PyPDF2
```

#### Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

#### Prepare Company Documents

The application uses Retrieval-Augmented Generation (RAG) to incorporate company knowledge. Place your company documents in the `company_docs/` directory. Supported formats:
- `.txt` files
- `.pdf` files

Example:
```
company_docs/
├── faq.txt
├── policies.pdf
└── products.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Start the Backend Server

```bash
python app.py
```

The backend will start on `http://127.0.0.1:5000`

### Start the Frontend (in a new terminal)

```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000`

### Access the Application

Open your browser and navigate to `http://localhost:3000` to start using the training platform.

## Usage

1. **Configure Session**: Select a persona, enter product/industry details, and scenario
2. **Upload Documents** (optional): Add company-specific documents for better context
3. **Start Conversation**: Begin the training session
4. **Practice**: Respond to customer messages as a customer service agent
5. **End Session**: Receive a detailed performance report with coaching feedback

## Project Structure

```
ai-practicum/
├── app.py                 # Flask backend application
├── main.py                # Core conversation logic and RAG
├── config.py              # Configuration constants
├── prompts.py             # AI prompts and templates
├── document_loader.py     # Document processing and vector storage
├── personas.json          # Customer persona definitions
├── company_docs/          # Directory for company documents (RAG)
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── api.js         # API client functions
│   │   └── App.js         # Main React app
│   └── package.json
├── faiss.index            # FAISS vector index (generated)
├── docs.pkl               # Document chunks (generated)
└── speech.mp3             # TTS audio files (generated)
```

## API Endpoints

- `POST /start` - Initialize a new conversation
- `POST /message` - Send a message in the conversation
- `POST /end` - End conversation and generate report
- `GET /personas` - Get available customer personas
- `POST /upload` - Upload company documents
- `GET /audio` - Serve TTS audio files

## Troubleshooting

### Backend Issues
- Ensure Python 3.8+ is installed: `python --version`
- Check that all dependencies are installed: `pip list`
- Verify OpenAI API key is set in `.env`
- Make sure port 5000 is available

### Frontend Issues
- Ensure Node.js 16+ is installed: `node --version`
- Check npm installation: `npm --version`
- Clear npm cache if needed: `npm cache clean --force`

### Common Errors
- **"Failed to load personas"**: Check that `personas.json` exists and is valid JSON
- **"Audio blocked"**: Browser autoplay policies may prevent audio playback
- **"Request failed"**: Ensure backend is running on port 5000
- **RAG not working**: Check that documents are in `company_docs/` and FAISS index is built

## Development

### Adding New Personas

Edit `personas.json` to add new customer personas with different traits and difficulty levels.

### Customizing Prompts

Modify prompts in `prompts.py` to change AI behavior and evaluation criteria.

### Extending Functionality

- Add new API endpoints in `app.py`
- Create new React components in `frontend/src/components/`
- Implement additional evaluation metrics in `main.py`
