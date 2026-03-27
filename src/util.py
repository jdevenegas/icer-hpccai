from IPython.display import Markdown, display

def show_table(md_str):
    """
    Display | delimited table output as a markdown table.
    """
    display(Markdown('\n'.join(
        ['| ' + ' | '.join(row.split('|')) + ' |' for row in md_str.splitlines()] \
        if len(md_str.splitlines()) == 1 else \
        ['| ' + ' | '.join(md_str.splitlines()[0].split('|')) + ' |',
         '| ' + ' | '.join(['---']*len(md_str.splitlines()[0].split('|'))) + ' |'] + \
        ['| ' + ' | '.join(row.split('|')) + ' |' for row in md_str.splitlines()[1:]]
    )))

def show_response_stats(response):
    """
    Display LLM response stats in a markdown table.
    """
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