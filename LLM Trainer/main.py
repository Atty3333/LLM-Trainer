# pip install llama-cpp-python
from llama_cpp import Llama

# Path to your downloaded GGUF file
model_path = "./function_gemma_router-Q8_0.gguf"

llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

def get_router_decision(user_query):
    # This must match the instruction used in training
    instruction = (
        "You are a model that can do function calling with the following functions\n"
        "<start_function_declaration>declaration:get_current_time{description:<escape>Get the exact current time<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "<start_function_declaration>declaration:get_current_date{description:<escape>Get today's date<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "<start_function_declaration>declaration:Sleep_Mode{description:<escape>Shut down or enter sleep mode<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "If no tool matches, output 'NONE'."
    )
    
    prompt = (
        f"<start_of_turn>developer\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_query}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    output = llm(prompt, max_tokens=32, stop=["<end_of_turn>", "<eos>"])
    result = output["choices"][0]["text"].strip()
    
    # Simple Parser
    if "<start_function_call>" in result:
        # Extract tool name from <start_function_call>call:tool_name{}<end_function_call>
        decision = result.split("call:")[1].split("{")[0]
        return f"TOOL TRIGGERED: {decision}"
    
    return f"CHAT MODE: {result}"

# Interactive loop
while True:
    query = input("\nUser: ")
    if query.lower() in ["exit", "quit"]: break
    print(get_router_decision(query))