# Running the Docs Assistant from the Terminal

This guide starts the Docs Assistant with an existing persistent Chroma vector store on the MSU HPCC.

## Requirements

- A Python environment containing the Docs Assistant dependencies
- A persistent Chroma directory created with the collection name `icer_docs_collection_full`

The default models in `config.py` are:

```text
Embedding model: mxbai-embed-large:latest
Chat model:      gpt-oss:20b
```

## 1. Start Ollama

Open a terminal in the interactive session and load Ollama:

```bash
module load Ollama/0.12.6
```

Start the server:

```bash
ollama serve
```

Keep this terminal open. The Ollama server and the Docs Assistant must run on the same compute node.


## 3. Activate the Python environment

Load Miniforge and activate the environment containing the Docs Assistant dependencies:

```bash
module load Miniforge3
conda activate /mnt/home/rayarjun/.conda/envs/arjunenv
```

Skip this step if the correct environment name already appears in the terminal prompt.

## 4. Start the Docs Assistant

Move to the Docs Assistant directory:

```bash
cd /path/to/docs-assistant
```

Run the assistant with the persistent Chroma directory created by the notebook:

```bash
python main.py --persist_dir /path/to/chroma_icer_docs_db
```

The assistant will display the default embedding and chat models. Enter `n` to use them, or enter `y` to select different installed models.

## 5. Ask questions

Enter a question at the `User:` prompt:

```text
User: What type of GPUs are on the Data Machine?
```

To request a JSON response, begin the question with `/json`:

```text
User: /json What type of GPUs are on the Data Machine?
```

Press Enter on an empty schema line to use the default JSON schema.

To stop the assistant, enter:

```text
exit
```

## Common errors

- `model not found`: Run `ollama list` and use an installed model name.
- `connection refused`: Confirm that `ollama serve` is running on the same node and port.
- `address already in use`: An Ollama server is already running on port `11434`; use the existing server.
- `runner process has terminated: signal: killed`: Request more system memory for the interactive session.
- Empty collection: Confirm that `--persist_dir` points to the directory containing the `icer_docs_collection_full` collection.
