from typing import Dict, Any
from pydantic import BaseModel, model_validator
from IPython.display import Markdown, display

def show_table(md_str):
    """display | delim table output as a markdown table."""
    display(Markdown('\n'.join(
        ['| ' + ' | '.join(row.split('|')) + ' |' for row in md_str.splitlines()] \
        if len(md_str.splitlines()) == 1 else \
        ['| ' + ' | '.join(md_str.splitlines()[0].split('|')) + ' |',
         '| ' + ' | '.join(['---']*len(md_str.splitlines()[0].split('|'))) + ' |'] + \
        ['| ' + ' | '.join(row.split('|')) + ' |' for row in md_str.splitlines()[1:]]
    )))

def show_response_stats(response):
    """display LLM response stats in a markdown table."""
    markdown_table = f"""
| Metric | Description | Units | Value |
|--------|-------------|-------|-------|
| `prompt_eval_count` | Number of tokens in the prompt | tokens | {response.get('prompt_eval_count', 'N/A')} |
| `eval_count` | Number of tokens generated in the response | tokens | {response.get('eval_count', 'N/A')} |
| `total_duration` | Total time spent for the request | nanoseconds | {response.get('total_duration', 'N/A')} |
| `prompt_eval_duration` | Time spent evaluating the prompt | nanoseconds | {response.get('prompt_eval_duration', 'N/A')} |
| `eval_duration` | Time spent generating tokens | nanoseconds | {response.get('eval_duration', 'N/A')} |
"""
    display(Markdown(markdown_table))

class LLMOptions(BaseModel):
    """define schema for Ollama llm model options handling."""
    #Ollama supported options
    num_ctx: int = 2048
    temperature: float = 0.8
    repeat_last_n: int = None
    repeat_penalty: float = None
    seed: int = None
    stop: str = None
    num_predict: int = None
    top_k: int = None
    top_p: float = None
    min_p: float = None

    # model_config = {
    #     "extra": "allow"  # invalid keys raise an error
    # }

    @model_validator(mode='before')
    def validate_opts(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """validate opts are in Ollama supported list."""
        invalid = set(params.keys()) - set(cls.model_fields.keys())
        if invalid:
            raise ValueError(f"Invalid Ollama options: {invalid}. Supported: {OLLAMA_SUPPORTED_OPTIONS}")
        return params