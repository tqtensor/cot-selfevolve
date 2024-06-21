import json
import logging
import os
import re
from typing import Optional

import tiktoken as tk
from langchain_core.prompts import ChatPromptTemplate

from src.dataset.stack_overflow import StackOverflowDataset
from src.llm import LLM
from src.prompts import Human, Stage, Strategy, System, Task
from src.utils import syntax_check

# Tiktoken encoding
encoding = tk.encoding_for_model("gpt-3.5-turbo-0613")


logger = logging.getLogger(__name__)


class CoTGenerator:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        demo: bool = False,
    ) -> None:
        self.llm = LLM(model=model, temperature=temperature, top_p=top_p)
        self.demo = demo

    def generate(
        self,
        stage: Stage,
        strategy: Strategy,
        problem: str,
        log_file_path: str,
        generated_code: Optional[str],
        feedback: Optional[str],
    ) -> str:
        # Build the prompts
        system_prompt = System(strategy=strategy).system
        human_prompt = Human(
            stage=stage,
            strategy=strategy,
            task=Task.COT_GENERATION,
            feedback=feedback,
        ).human

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

        # Get StackOverflow post
        if str(stage) == "initial":
            post = StackOverflowDataset().retrieve(query=problem, k=1)
            post = post[0].page_content if post else ""

            if self.demo:
                print("\033[92mStackOverflow post:\033[0m")
                print("\033[92m" + post + "\033[0m")
                input("Press Enter to continue...")
        else:
            post = ""

        # Invoke the LLM
        try:
            response = self.llm.invoke(
                messages=prompt.invoke(
                    {
                        "problem_description": problem,
                        "post": post,
                        "generated_code": generated_code,
                        "feedback": (
                            feedback
                            if len(encoding.encode(feedback)) < 4096
                            else feedback[:4096]
                        ),
                    }
                ),
            )

            if response:
                answer = response.completion_message
                response_json = json.dumps(response.dict(), indent=4)

                os.makedirs(os.path.join(log_file_path, "cot"), exist_ok=True)
                with open(os.path.join(log_file_path, "cot", "log.json"), "w") as f:
                    f.write(response_json)
        except Exception as e:
            logger.exception(e)
        return answer


class CodeGenerator:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        demo: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = LLM(model=model, temperature=temperature, top_p=top_p)
        self.demo = demo

    def extract_code(self, answer: str):
        # Extract code from the answer
        match = re.search(r"```python\n(.*?)\n```", answer, re.DOTALL)
        if match:
            code = match.group(1)
            if syntax_check(code)["status"] == "success":
                return code
            else:
                return ""
        else:
            if syntax_check(answer)["status"] == "success":
                return answer
            else:
                return ""

    def generate(
        self,
        stage: Stage,
        strategy: Strategy,
        problem: str,
        log_file_path: str,
        code_context: str,
        generated_code: Optional[str],
        feedback: Optional[str],
    ) -> str:
        # Build the prompts
        system_prompt = System(strategy=strategy).system
        human_prompt = Human(
            stage=stage,
            strategy=strategy,
            task=Task.CODE_GENERATION,
            feedback=feedback,
        ).human

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

        # Generate CoT suggestion
        if str(strategy) == "cot":
            cot_generator = CoTGenerator(
                model=self.model, temperature=self.temperature, demo=self.demo
            )
            cot_suggestion = cot_generator.generate(
                stage=stage,
                strategy=strategy,
                problem=problem,
                log_file_path=log_file_path,
                generated_code=generated_code,
                feedback=feedback,
            )

            if self.demo:
                print("\033[92mCoT suggestion:\033[0m")
                print("\033[92m" + cot_suggestion + "\033[0m")
                input("Press Enter to continue...")
        else:
            cot_suggestion = ""

        # Invoke the LLM
        try:
            response = self.llm.invoke(
                messages=prompt.invoke(
                    {
                        "problem_description": problem,
                        "code_context": code_context,
                        "generated_code": generated_code,
                        "feedback": (
                            feedback
                            if len(encoding.encode(feedback)) < 4096
                            else feedback[:4096]
                        ),
                        "cot_suggestion": cot_suggestion,
                    }
                ),
            )

            if response:
                answer = response.completion_message
                response_json = json.dumps(response.dict(), indent=4)

                os.makedirs(os.path.join(log_file_path, "code"), exist_ok=True)
                with open(os.path.join(log_file_path, "code", "log.json"), "w") as f:
                    f.write(response_json)
        except Exception as e:
            logger.exception(e)

        generated_code = self.extract_code(answer)

        if self.demo:
            print("\033[92mGenerated code:\033[0m")
            print("\033[92m" + generated_code + "\033[0m")
            input("Press Enter to continue...")
        return generated_code
