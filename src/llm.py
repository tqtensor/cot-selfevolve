import json
import logging
import os
import tempfile
from enum import Enum
from typing import Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from litellm import completion, token_counter
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    prompt_messages: List[Dict[str, str]]
    prompt_token_count: int
    completion_message: str
    completion_token_count: int


class LLMModel(Enum):
    AZURE_GPT_35_TURBO_16K_0613 = "azure-gpt-35-turbo-16k-0613"
    AZURE_GPT_4_32K_0613 = "azure-gpt-4-32k-0613"
    AZURE_GPT_4_TURBO_2024_04_09 = "azure-gpt-4-turbo-2024-04-09"
    OPENAI_GPT_3_5_TURBO_16K_0613 = "openai-gpt-3.5-turbo-16k-0613"
    OPENAI_GPT_4_32K_0613 = "openai-gpt-4-32k-0613"
    OPENAI_GPT_4_TURBO = "openai-gpt-4-turbo"
    OPENAI_GPT_4_O = "openai-gpt-4o"
    BEDROCK_ANTHROPIC_CLAUDE_V2_1 = "bedrock-anthropic.claude-v2:1"
    BEDROCK_ANTHROPIC_CLAUDE_3_SONNET_20240229_V1_0 = (
        "bedrock-anthropic.claude-3-sonnet-20240229-v1:0"
    )
    BEDROCK_LLAMA_3_70B = "bedrock-meta.llama3-70b-instruct-v1:0"
    BEDROCK_MISTRAL_MISTRAL_LARGE_2402_V1_0 = "bedrock-mistral.mistral-large-2402-v1:0"
    BEDROCK_MISTRAL_MISTRAL_8X7B_INSTRUCT_V0_1 = (
        "bedrock-mistral.mistral-8x7b-instruct-v0:1"
    )
    VERTEX_GEMINI_1_5_FLASH_PREVIEW_0514 = "vertex-gemini-1.5-flash-preview-0514"
    VERTEX_GEMINI_1_5_PRO_PREVIEW_0514 = "vertex-gemini-1.5-pro-preview-0514"

    def __str__(self):
        return self.value


class LLM:
    def __init__(
        self,
        model: LLMModel = LLMModel.AZURE_GPT_35_TURBO_16K_0613,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if str(model).startswith("azure"):
            self.configs = {
                "model": "azure/{}".format(str(model).replace("azure-", ""))
            }
        elif str(model).startswith("bedrock"):
            self.configs = {
                "model": "bedrock/{}".format(str(model).replace("bedrock-", ""))
            }
        elif str(model).startswith("openai"):
            self.configs = {"model": "{}".format(str(model).replace("openai-", ""))}
        elif str(model).startswith("vertex"):
            _load_vertex_ai_credentials()
            self.configs = {
                "model": "vertex_ai/{}".format(str(model).replace("vertex-", ""))
            }
        else:
            raise ValueError(f"Unknown model: {str(model)}")

        self.configs["temperature"] = temperature
        self.configs["top_p"] = top_p

    @staticmethod
    def prompt_to_messages(messages: ChatPromptTemplate) -> List[Dict[str, str]]:
        _messages = []
        for message in messages.messages:
            if isinstance(message, SystemMessage):
                _messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                _messages.append({"role": "user", "content": message.content})
            else:
                raise ValueError(f"Unknown message type: {type(message)}")

        return _messages

    def completion_call(self, messages: ChatPromptTemplate) -> Optional[LLMResponse]:
        response = None
        try:
            prompt_messages = self.prompt_to_messages(messages)
            prompt_token_count = token_counter(
                model=self.configs["model"], messages=prompt_messages
            )
            response = completion(**self.configs, messages=prompt_messages)

            response = response.choices[0].message.content
            response_token_count = token_counter(
                model=self.configs["model"], text=response
            )
        except Exception as e:
            logger.exception(e)
        finally:
            if response:
                return LLMResponse(
                    prompt_messages=prompt_messages,
                    prompt_token_count=prompt_token_count,
                    completion_message=response,
                    completion_token_count=response_token_count,
                )
            else:
                return None

    def invoke(self, messages: ChatPromptTemplate) -> Optional[LLMResponse]:
        return self.completion_call(messages=messages)


def _load_vertex_ai_credentials():
    """
    Loads Vertex AI credentials from a JSON file and sets them as an
    environment variable.

    This function reads the `vertex_key.json` file in the same directory as
    this script. If the file does not exist or is not valid JSON, it creates an
    empty dictionary. It then writes this dictionary to a temporary file and
    sets the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path
    of this temporary file.

    If the `vertex_key.json` file does not exist, this function will not raise
    a FileNotFoundError. Instead, it will create an empty dictionary and
    proceed as described above.
    """
    # Define the path to the vertex_key.json file
    logging.info("Loading Vertex AI credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            logging.info("Read Vertex AI file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary file
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)
