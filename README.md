# RAG Locally

Here is the project for making a RAG application to run locally using Ollama.

## Setup

1. **Clone the Repository:**
    ```bash
    git pull <link-of-this-repo>
    ```

2. **Set Up Virtual Environment:**
    ```bash
    python<version> -m venv <virtual-environment-name>
    ```

3. **Activate the Virtual Environment:**
    - On macOS/Linux:
      ```bash
      source <virtual-environment-name>/bin/activate
      ```
    - On Windows (CMD):
      ```cmd
      <virtual-environment-name>\Scripts\activate.bat
      ```
    - On Windows (PowerShell):
      ```powershell
      <virtual-environment-name>\Scripts\Activate.ps1
      ```

4. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Download Ollama:**
    - Visit the link below to download Ollama:
      [Ollama Download Link](https://ollama.com/)

## Running the Application

1. **Pull Llama Model:**
    ```bash
    ollama pull llama3
    ```

2. **Run Llama Model:**
    ```bash
    ollama run llama3
    ```

3. **Serve Ollama:**
    ```bash
    ollama serve
    ```

    Now you have your Ollama server running locally. You can check it by visiting:
    [http://localhost:11434/](http://localhost:11434/) where you will see the "Ollama is running" web page.

4. **Run Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

    This will bring you to a Streamlit app where you need to upload a file and process it. Once processed, you can ask questions in the chat input box. Be patient, as it runs locally and may be slow.

## Backend Process

The application performs the following steps:

- Loads a document, splits it into chunks, and converts them into embeddings using Ollama.
- These embeddings are stored in a database.
- Your prompt is also converted into an embedding and the closest neighbor among the chunks is found.
- Ollama processes all of this to provide a response based on the found context and your prompt, which is then displayed as a chat using Streamlit.
