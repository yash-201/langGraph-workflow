# Streamlit LangGraph Chatbot

A stateful, conversational AI chatbot interface built with **Streamlit**, **LangGraph**, and **Google Gemini** (`ChatGoogleGenerativeAI`).

---

## 🚀 Setup & Execution

### 1. Activate Virtual Environment
Ensure your virtual environment is active. From the project root (`d:\langGraph`):

#### On Windows (PowerShell):
```powershell
.\myenv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
source myenv/bin/activate
```

---

### 2. Configure Environment Variables
Make sure you have a `.env` file in the root directory (`d:\langGraph\.env`) containing your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# OR
GOOGLE_API_KEY=your_google_api_key_here
```

---

### 3. Install Dependencies
If not already installed in your virtual environment:

```bash
pip install -U streamlit langgraph langchain-google-genai python-dotenv
```

---

### 4. Run the Streamlit Application

Navigate to the `chatbot` folder and launch Streamlit:

```bash
cd d:\langGraph\chatbot
streamlit run streamlit_frontend.py
```

Alternatively, from the project root:

```bash
streamlit run chatbot/streamlit_frontend.py
```

---

## 📁 Project Structure

- **`langgrap_backend.py`**: Defines the `ChatState` typed schema, `StateGraph`, `InMemorySaver` checkpointer, and Gemini model node.
- **`streamlit_frontend.py`**: Web UI powered by Streamlit that manages chat state session history and invokes the backend graph workflow.
- **`README.md`**: Instructions for setup and running the project.
