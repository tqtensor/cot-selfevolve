from enum import Enum
from typing import Optional


class Stage(Enum):
    INITIAL = "initial"
    CORRECTION = "correction"

    def __str__(self):
        return self.value


class Strategy(Enum):
    ZEROSHOT = "zeroshot"
    COT = "cot"

    def __str__(self):
        return self.value


class Task(Enum):
    CODE_GENERATION = "code_generation"
    COT_GENERATION = "cot_generation"

    def __str__(self):
        return self.value


class System:
    def __init__(self, strategy: Strategy):
        if str(strategy) == "zeroshot":
            self.system = "You are a helpful code debugging expert named as CoT-SelfEvolve that can understand and solve programming problems. You have the ability to analyze and execute code, providing feedback and suggestions to help users debug and improve their code. By leveraging your knowledge and expertise, you can assist users in solving complex programming problems and guide them towards writing correct and efficient code. Your goal is to empower users to become better programmers by providing them with valuable insights and assistance throughout their coding journey."
        elif str(strategy) == "cot":
            self.system = "You are a helpful Chain-of-Thought expert named as CoT-Guru that can understand the reasoning behind programming problems and provide step-by-step guidance to solve them. You have the ability to analyze code and generate a series of suggestions that guide others to reason and solve programming problems effectively. By leveraging your knowledge and expertise, you can assist users in understanding complex programming concepts and help them develop their problem-solving skills. Your goal is to empower users to think critically and logically about programming problems, enabling them to become better programmers."
        else:
            raise ValueError("Invalid strategy")


class Human:
    def __init__(
        self,
        stage: Stage,
        strategy: Strategy,
        task: Task,
        feedback: Optional[str] = None,
    ):
        if str(stage) == "initial":
            if str(task) == "code_generation":
                if str(strategy) == "zeroshot":
                    self.human = """
Given the problem description with the code, you need to fulfill the task by writing the code that solves the problem.
The problem is: {problem_description}.
Your duty is to solve the problem described above by writing the code that solves the problem.
You will replace the code inside the `[insert]` block with your code as following code context:
```python
{code_context}
```
Inside the context, you can see which libraries will be used, the input/output format, and the expected behavior of the code. And most imporantly, how your code will be tested. Please do not import any additional libraries as they have been provided in the context.
Make sure your code is correct and complete to solve the problem.
"""
                elif str(strategy) == "cot":
                    self.human = (
                        Human(
                            stage=Stage.INITIAL,
                            strategy=Strategy.ZEROSHOT,
                            task=Task.CODE_GENERATION,
                        ).human
                        + """
To support you in solving the problem, here are the Chain-of-Thought reasoning suggestions, you should follow these suggestions one by one, to use them as a guide for your internal reasoning process to solve the problem.
{cot_suggestion}
"""
                    )
            elif str(task) == "cot_generation":
                self.human = """
Given the problem description with the code, and one or multiple StackOverflow posts, you need to learn from the comments to generate step-by-step suggestions that help another agent (CoT-SelfEvolve) to solve the problem.
The given problem is: {problem_description}.
The StackOverflow post with supportive comments is: {post}.
Please generate a series of suggestions or questions that guide CoT-SelfEvolve to reason and to solve the problem step-by-step.
Here are some suggestions:
- Suggestion 1: [You should ...]
- Suggestion 2: [, then ...]
- Suggestion 3: [, then ...]
- Final suggestion: [, and finally, ...]
"""
            else:
                raise ValueError("Invalid task")
        elif str(stage) == "correction":
            if str(task) == "code_generation":
                if str(strategy) == "zeroshot":
                    if "traceback" in feedback.lower():
                        self.human = (
                            Human(
                                stage=Stage.INITIAL,
                                strategy=Strategy.ZEROSHOT,
                                task=Task.CODE_GENERATION,
                            ).human
                            + """
In the previous attempt, you generated the following code:
GENERATED_CODE:
```
{generated_code}
```
However, the code has an error. The error message is:
FEEDBACK:
```
{feedback}
```
Please analyze the error message and fix the code accordingly.
"""
                        )
                    elif ("executed" in feedback.lower()) and (
                        "expected" in feedback.lower()
                    ):
                        self.human = (
                            Human(
                                stage=Stage.INITIAL,
                                strategy=Strategy.ZEROSHOT,
                                task=Task.CODE_GENERATION,
                            ).human
                            + """
In the previous attempt, you generated the following code:
GENERATED_CODE:
```
{generated_code}
```
The code executed successfully but failed the test case. Most probably you have generated test samples by yourself, and that is wrong.
You also can analyze the difference between the expected output and the generated output to understand the problem.
FEEDBACK:
```
{feedback}
```
"""
                        )
                    else:
                        self.human = (
                            Human(
                                stage=Stage.INITIAL,
                                strategy=Strategy.ZEROSHOT,
                                task=Task.CODE_GENERATION,
                            ).human
                            + """
In the previous attempt, you generated the following code:
GENERATED_CODE:
```
{generated_code}
```
However, the system has given you the following instruction:
FEEDBACK:
```
{feedback}
```
Please comply with the instruction and generate the code accordingly.
"""
                        )
                elif str(strategy) == "cot":
                    self.human = (
                        Human(
                            stage=Stage.INITIAL,
                            strategy=Strategy.ZEROSHOT,
                            task=Task.CODE_GENERATION,
                        ).human
                        + """
To support you in solving the problem, here are the Chain-of-Thought reasoning suggestions, you should follow these suggestions one by one, to use them as a guide for your internal reasoning process to solve the problem.
{cot_suggestion}
"""
                    )
            elif str(task) == "cot_generation":
                self.human = """
Given the problem description with the code, and the code generated by another agent (CoT-SelfEvolve) together with the feedback from the system, you need to generate step-by-step Chain-of-Thought reasoning to help the CoT-SelfEvolve to solve the problem by himself.
The given problem is: {problem_description}.
In the previous attempt, CoT-SelfEvolve generated the following code:
GENERATED_CODE:
```
{generated_code}
```
And it received the following feedback:
FEEDBACK:
```
{feedback}
```
In general, the steps to reason about the problem are:
- Step 1: understand the problem, what does it require?
- Step 2: analyze the GENERATED_CODE vs problem, what is the problem? does it related with the defined problem?
- Step 3: analyze the FEEDBACK, what is the error message? what is the expected output?
Please help the CoT-SelfEvolve agent by providing step-by-step guidance to solve the problem. DO NOT attempt to solve the problem directly.
Remember that you are helping another agent to solve the problem, not solving the problem directly.
"""
