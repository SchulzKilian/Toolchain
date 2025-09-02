import os
import vertexai
from vertexai.generative_models import GenerativeModel
import dotenv

from llm_toolchain import tools, Toolchain, VertexAIAdapter, SemanticToolSelector




def main():
    """
    A simple example demonstrating how to use the Toolchain with an OpenAI LLM
    and a custom weather tool.
    """


    dotenv.load_dotenv()
    project_id = os.getenv("GEMINI_PROJECT_ID")
    location = "europe-west1"
    vertexai.init(project=project_id, location=location)

 

    # 3. Create an instance of the OpenAI adapter
    #    This adapter knows how to communicate with the OpenAI API.
    geminiadapter = VertexAIAdapter()
    llm = GenerativeModel('gemini-2.0-flash-001')
    # 4. Initialize the Toolchain
    #    We provide it with a list of tools it can use and the adapter.
    chain = Toolchain(
        llm_client=llm, 
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
