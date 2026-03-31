# GenAI on the HPCC Workshop
GenAI on the High-Performance Computing Center (HPCC) seminar content: workshop + tutorials designed to provide researchers from a wide variety of disciplines with a hands-on introduction to tools for integrating GenAI into research computing pipelines on the MSU HPCC. This workshop was given at MSU on March 11 and March 18, 2026. 

## Cloning the repository 
Open up a new dev node terminal (via OnDemand or an ssh client) and `git clone` the repository URL to create a copy of the code into your home directory. Note: you can also just copy the code from the workshop folder `/mnt/research/HPCC_AI_Workshop/icer-hpccai` into your home folder like we did during the workshop, but this is a more permanent solution. 

## Accessing Materials 
This repository contains all of the code and slides presented during the workshop. The materials required to run the Python Notebook files in this repo (e.g., ICER docs) can be found on the MSU HPCC research space for the workshop (`/mnt/research/HPCC_AI_Workshop`). All registered participants in the workshop should have access to this research space.   

## Running the code in this repo on the MSU HPCC 
In order to run the code in this repo, we recommend starting an interactive Jupyter session on the MSU HPCC (like we did during the workshop). However, because the dedicated node space is no longer reserved for us like it was during the workshop, you'll need to configure the interactive session slightly differently. In particular, **<ins>you will not be able to use the</ins>** `genai_workshop` **<ins>account; you should use the general account, or a buy-in account if you have one</ins>**.

> You can simply leave the account blank, or not set and it will default to general or buy-in (if you have one)

### Configuring the interactive session 
* Jupyter location: Conda Environment using Miniforge3 module
* Conda Environment name or path: `/mnt/research/HPCC_AI_Workshop/.conda/envs/hpccai` 
* Working directory: /your/space/icer-hpccai (e.g., /mnt/home/$USER/icer-hpccai/ or wherever you cloned this repo)
* Use GPUs: Check and select `1` for number of GPUs and `any` for GPU type.
* Number of tasks: 1

> If you would like to run and test larger llm and/or embedding models, you can request specific GPU types, referencing the [ICER Cluster Resources](https://docs.icer.msu.edu/Cluster_Resources/) docs page. 

### Starting Ollama and running the tutorials
Once your session starts, open Jupyter and go to **File > New > Terminal**
Run the following commands:
```bash 
module load Ollama
export OLLAMA_HOST=127.0.0.1:40063
```
> We suggest you set your port number (roughly) between 40001 and 40840. 

```bash 
ollama serve
```
Note that at this point some text output will appear, and it will look like the commands are still executing (i.e., your shell prompt indicator “~$” will not reappear). This is what you want! It means the Ollama server is running, and you're seeing the server logs.

Now we’ve got Ollama up and running so we should be able to run the tutorials! Just make sure that you change the port number in the Python notebooks to match the port you specified in the terminal. 

On the left side of the Jupyter screen should be the workshop directory you created at the beginning of these instructions. Double click docs_bot_kb.ipynb to open up the docs bot tutorial and you’re off to the races! You'll also see the structured output (interaction extraction) tutorial in the interaction_extraction.ipynb notebook. 

## Questions
If you have any questions about the materials or workflows, please contact Julian Venegas (venegas5@msu.edu) or Kelly Kapsar (kapsarke@msu.edu). 
