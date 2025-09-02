import os
from openai import OpenAI
import dotenv

from llm_toolchain import Toolchain  , tools, SemanticToolSelector

def main():
    """
    A simple example demonstrating how to use the Toolchain with an OpenAI LLM
    and a custom weather tool.
    """
    dotenv.load_dotenv()

    # 1. Load the OpenAI API key from the .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        print("Please create a .env file and add your OpenAI API key to it.")
        return

    # 2. Initialize the OpenAI LLM client
    try:
        llm_client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    # 3. Create an instance of the OpenAI adapter
    #    This adapter knows how to communicate with the OpenAI API.
    #    You can specify any compatible model, like 'gpt-4o' or 'gpt-3.5-turbo'.
    openai_adapter = OpenAIAdapter()

    # 4. Initialize the Toolchain
    #    We provide it with the LLM client, adapter, and a selector for our tools.
    chain = Toolchain(
        llm_client=llm_client,
        adapter=openai_adapter,
        tools=[],  # Tools are handled by the selector
        selector=SemanticToolSelector(
            all_tools=[
                tools.calculate_compound_interest,
                tools.get_weather,
                tools.write_file,
                tools.read_file,
                tools.list_files,
                tools.show_directory_tree,
                tools.change_directory,
                tools.run_python_code,
                tools.get_address_from_coordinates,
                tools.open_and_read_website,
                tools.create_calendar_event,
                tools.test_regex_pattern,
                tools.convert_units,
                tools.visualize_graph,
                tools.append_to_file,
                tools.delete_file
            ]
        )
    )

    # 5. Define an initial prompt that should trigger a tool
    prompt = "What's the weather in New York City?"
    while prompt:
        print(f"\n-> User Prompt: {prompt}\n")
        try:
            final_response = chain.run(prompt=prompt)
            print(f"<- LLM Response: {final_response}")
        except Exception as e:
            print(f"An error occurred while running the toolchain: {e}")

        prompt = input("\nEnter another prompt (or press Enter to quit): ").strip()

if __name__ == "__main__":
    main()