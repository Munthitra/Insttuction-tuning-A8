import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


final_output_path = './results/final'

model = AutoModelForCausalLM.from_pretrained(
    final_output_path,
    device_map = 'auto')

tokenizer = AutoTokenizer.from_pretrained(
    final_output_path)

text_generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    device_map = 'auto',
    pad_token_id = tokenizer.eos_token_id,
    max_new_tokens = 50
)

def format_input(instruction, prompt_input=None):
	
	if prompt_input:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{prompt_input}

### Response:
""".strip()
			
	else:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
""".strip()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Text Generation Web App"),
    
    dcc.Input(id='user_instruction', type='text', placeholder='Enter instruction...'),
    dcc.Input(id='user_input', type='text', placeholder='Enter input...'),
    
    html.Button('Submit', id='submit_button', n_clicks=0),
    
    html.H2("Answer:"),
    html.P(id='response'),
])

@app.callback(
    Output('response', 'children'),
    [Input('submit_button', 'n_clicks')],
    [dash.dependencies.State('user_instruction', 'value'),
     dash.dependencies.State('user_input', 'value')]
)
def generate_text(n_clicks, user_instruction, user_input):
	
	if n_clicks > 0:
		formatted_input = format_input(user_instruction, user_input)
		output = text_generator(formatted_input)
		response = output[0]['generated_text'].split("### Response:\n")[-1]
		
		return response

if __name__ == '__main__':
    app.run_server(debug=True)