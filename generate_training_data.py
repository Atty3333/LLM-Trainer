import json
import random
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
output_file = DATA_DIR / "training_data.jsonl"

def generate_dataset():
    dataset = []
    
    # Official FunctionGemma Activation Instruction
    instruction = (
        "You are a model that can do function calling with the following functions\n"
        "<start_function_declaration>declaration:get_current_time{description:<escape>Get the exact current time<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "<start_function_declaration>declaration:get_current_date{description:<escape>Get today's date<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "<start_function_declaration>declaration:Sleep_Mode{description:<escape>Shut down or enter sleep mode<escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration>\n"
        "If no tool matches, output 'NONE'."
    )

    variations = {
        "get_current_time": ["What time is it?", "Current time please", "Clock check", "What's the hour?"],
        "get_current_date": ["What's the date?", "What day is it today?", "Current date?", "Check the calendar."],
        "Sleep_Mode": ["Go to sleep", "Shut down", "Stop listening", "Power down.", "sleep"],
        "NONE": ["Hello", "How are you?", "What is 2+2?", "Explain quantum physics.", "Who is the president?"]
    }

    weights = ["get_current_time", "get_current_date", "Sleep_Mode", "NONE", "NONE"]
    
    for _ in range(500):
        tool_choice = random.choice(weights)
        phrase = random.choice(variations[tool_choice])
        
        if tool_choice == "NONE":
            formatted_output = "NONE"
        else:
            formatted_output = f"<start_function_call>call:{tool_choice}{{}}<end_function_call>"
        
        entry = {"instruction": instruction, "input": phrase, "output": formatted_output}
        dataset.append(entry)

    random.shuffle(dataset)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Generated {len(dataset)} examples at {output_file}")

if __name__ == "__main__":
    generate_dataset()