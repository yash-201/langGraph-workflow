# LangGraph Workflows

This repository contains Python Jupyter Notebook workflows built with **LangGraph**, **LangChain**, **OpenAI**, and **Google Gemini**.

---

## 🚀 Setup & Execution Guide

Follow these steps to set up and run the project inside a Python virtual environment.

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

---

### 2. Create a Virtual Environment

Open your terminal in the project root directory (`d:/langGraph`) and run:

```bash
python -m venv myenv
```

---

### 3. Activate the Virtual Environment

#### On Windows:
- **PowerShell**:
  ```powershell
  .\myenv\Scripts\Activate.ps1
  ```

  or 
  
   ```powershell
  .\myenv\Scripts\activate 
  ```
- **Command Prompt (cmd)**:
  ```cmd
  myenv\Scripts\activate.bat
  ```

#### On macOS / Linux:
```bash
source myenv/bin/activate
```

*(Once activated, your terminal prompt will display `(myenv)` at the beginning).*

---

### 4. Install Dependencies

Install all required packages inside the active virtual environment:

```bash
pip install -U langgraph langchain-openai langchain-google-genai python-dotenv ipykernel
```

---

### 5. Register Virtual Environment in Jupyter Kernel

To make sure VS Code or Jupyter Notebook uses your `myenv` virtual environment:

```bash
python -m ipykernel install --user --name=myenv --display-name "myenv"
```

---

### 6. Configure Environment Variables

1. Copy the sample environment file to create `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

### 7. Run Notebook Workflows & Chatbot App

Open VS Code or Jupyter Notebook and run the following notebooks:

1. **`1_bmi_workflow.ipynb`**: Deterministic LangGraph workflow for calculating and labeling BMI.
2. **`2_simple_llm_workflow.ipynb`**: LLM-powered LangGraph workflow using Google Gemini / OpenAI.

#### Run Streamlit Chatbot:
To run the interactive Streamlit chatbot application:

```bash
cd chatbot
streamlit run streamlit_frontend.py
```

---

## 🛠️ Deactivating Virtual Environment

When you are finished working:

```bash
deactivate
```

