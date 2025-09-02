import os
from google import generativeai as genai
import dotenv

from llm_toolchain import Toolchain, GenAIAdapter  , tools, SemanticToolSelector


def main():
    """
    A simple example demonstrating how to use the Toolchain with an OpenAI LLM
    and a custom weather tool.
    """


    dotenv.load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: gemini key not found.")
        print("Please create a .env file and add your API key to it.")
        return

    # 2. Initialize the LLM client
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error initializing  client: {e}")
        return

    # 3. Create an instance of the OpenAI adapter
    #    This adapter knows how to communicate with the OpenAI API.
    geminiadapter = GenAIAdapter()
    llm = genai.GenerativeModel('gemini-2.0-flash-001')
    # 4. Initialize the Toolchain
    #    We provide it with a list of tools it can use and the adapter.
    chain = Toolchain(
        llm_client=llm,  # The client will be passed in at runtime
        adapter =geminiadapter,
        tools = [],
        selector = SemanticToolSelector(all_tools=[tools.calculate_compound_interest, tools.get_weather, tools.write_file, tools.read_file, tools.list_files,  tools.show_directory_tree, tools.change_directory, tools.run_python_code, tools.get_address_from_coordinates, tools.open_and_read_website, tools.create_calendar_event, tools.test_regex_pattern, tools.convert_units, tools.visualize_graph, tools.append_to_file, tools.delete_file])
    )

    # 5. Define a prompt that should trigger the get_weather tool
    
    prompt = "Whats the weather in New York City?"
    while prompt:
        print(f"-> User Prompt: {prompt}\n")

        try:
            final_response = chain.run(
                prompt=prompt
            )
            
            print(f"<- LLM Response: {final_response}")

        except Exception as e:
            print(f"An error occurred while running the toolchain: {e}")
            print(f"Exact error message with line number: {e.__traceback__}")
        prompt = input("\nEnter another prompt (or press Enter to quit): ").strip()


if __name__ == "__main__":
    main()
