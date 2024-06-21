# Reset the environment variables
import os

for key in list(os.environ.keys()):
    if ("AZURE" in key) or ("OPENAI" in key):
        del os.environ[key]

import glob
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import List

import click
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()  # load env before importing other modules

from src.dataset import Dataset
from src.generators import CodeGenerator
from src.llm import LLMModel
from src.prompts import Stage, Strategy

LIBRARIES = [
    "Matplotlib",
    "Numpy",
    "Pandas",
    "Pytorch",
    "Scipy",
    "Sklearn",
    "Tensorflow",
]


@click.command()
@click.option("--experiment_name", required=True, help="Name of the experiment.")
@click.option(
    "--libraries",
    multiple=True,
    default=LIBRARIES,
    help="List of libraries to include in the experiment.",
)
@click.option(
    "--sampling_fraction",
    default=1.0,
    type=float,
    help="Fraction of the dataset to sample.",
)
@click.option(
    "--initial_strategy",
    type=click.Choice([e.name for e in Strategy]),
    help="Initial strategy for code generation.",
)
@click.option(
    "--correction_strategy",
    type=click.Choice([e.name for e in Strategy]),
    help="Strategy for code correction.",
)
@click.option(
    "--initial_model",
    type=click.Choice([e.name for e in LLMModel]),
    help="Initial model for code generation.",
)
@click.option(
    "--correction_model",
    type=click.Choice([e.name for e in LLMModel]),
    help="Model for code correction.",
)
@click.option(
    "--temperature",
    default=0.9,
    type=float,
    help="Temperature parameter for the model.",
)
@click.option("--top_p", default=0.9, type=float, help="Top-p parameter for the model.")
@click.option(
    "--self_correction/--no-self_correction",
    default=False,
    help="Enable or disable self correction.",
)
@click.option(
    "--max_self_correction_attempts",
    default=5,
    type=int,
    help="Max self correction attempts.",
)
@click.option(
    "--demo/--no-demo", default=False, help="Run in demo mode with reduced logging."
)
def execute(
    experiment_name: str,
    libraries: List[str],
    sampling_fraction: float,
    initial_strategy: str,
    correction_strategy: str,
    initial_model: str,
    correction_model: str,
    temperature: float,
    top_p: float,
    self_correction: bool = False,
    max_self_correction_attempts: int = 5,
    demo: bool = False,
) -> None:
    # Convert string arguments back to enums
    initial_strategy = Strategy[initial_strategy]
    correction_strategy = Strategy[correction_strategy]
    initial_model = LLMModel[initial_model]
    correction_model = LLMModel[correction_model]

    if demo:
        # Set logging level to ERROR
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    else:
        # Set logging level to INFO
        logging.getLogger("LiteLLM").setLevel(logging.INFO)

    # Initialize datasets
    problem_dataset = Dataset(dataset="ds-1000", kwargs=None).dataset

    # Create directories
    if not os.path.exists(f"artifacts/{experiment_name}"):
        os.makedirs(f"artifacts/{experiment_name}")

    # Save experiment configuration
    with open(f"artifacts/{experiment_name}/config.json", "w") as f:
        json.dump(
            {
                "libraries": libraries,
                "initial_strategy": initial_strategy.name,
                "correction_strategy": correction_strategy.name,
                "initial_model": initial_model.name,
                "correction_model": correction_model.name,
                "temperature": temperature,
                "top_p": top_p,
            },
            f,
        )

    for lib in libraries:
        # Randomly sample problems
        if sampling_fraction == 1.0:
            problem_indices = list(range(len(problem_dataset[lib])))
        else:
            problem_indices = random.choices(
                list(range(len(problem_dataset[lib]))),
                k=max(1, int(len(problem_dataset[lib]) * sampling_fraction)),
            )

        for i in tqdm(
            problem_indices,
            desc=f"Solving problem for {lib}",
            total=len(problem_indices),
        ):
            # Create directories
            os.makedirs(
                f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/logs/initial",
                exist_ok=True,
            )
            os.makedirs(
                f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/logs/correction",
                exist_ok=True,
            )

            # Skip solved problems
            if os.path.exists(
                f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/result.txt"
            ):
                continue

            challenge = problem_dataset[lib][i]
            problem = challenge["prompt"]
            code_context = challenge["code_context"]

            dataset_dir = os.getcwd()  # save current directory

            initial_code_generator = CodeGenerator(
                model=initial_model, temperature=temperature, top_p=top_p, demo=demo
            )

            generated_code = initial_code_generator.generate(
                stage=Stage.INITIAL,
                strategy=initial_strategy,
                problem=problem,
                log_file_path=os.path.join(
                    dataset_dir,
                    f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/logs/initial",
                ),
                code_context=code_context,
                generated_code="",
                feedback="",
            )

            is_correct = challenge.test(generated_code)

            # If not using self correction, save the result and continue
            if not self_correction:
                attempt = float("inf")
            else:
                attempt = 1

            # Handle self correction
            while (attempt < max_self_correction_attempts) and (is_correct != True):
                if isinstance(is_correct, tuple):
                    correction_code_generator = CodeGenerator(
                        model=correction_model,
                        temperature=temperature,
                        top_p=top_p,
                        demo=demo,
                    )

                    os.chdir(dataset_dir)
                    os.makedirs(
                        f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/logs/correction/{str(attempt).zfill(2)}",
                        exist_ok=True,
                    )

                    generated_code = correction_code_generator.generate(
                        stage=Stage.CORRECTION,
                        strategy=correction_strategy,
                        problem=problem,
                        log_file_path=os.path.join(
                            dataset_dir,
                            f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/logs/correction/{str(attempt).zfill(2)}",
                        ),
                        code_context=code_context,
                        generated_code=is_correct[0],
                        feedback=is_correct[1],
                    )

                    is_correct = challenge.test(generated_code)
                elif isinstance(is_correct, bool):
                    break

                attempt += 1

            os.chdir(dataset_dir)
            with open(
                f"artifacts/{experiment_name}/{lib}_{str(i).zfill(3)}/result.txt",
                "w",
            ) as f:
                if isinstance(is_correct, bool) and (is_correct == True):
                    f.write("Correct")
                else:
                    f.write("Incorrect")

            time.sleep(5)

    results = defaultdict(lambda: [0, 0])
    for file in glob.glob(f"artifacts/{experiment_name}/*/result.txt"):
        lib = file.split("/")[-2].split("_")[0]
        is_correct = open(file, "r").read().strip() == "Correct"
        if is_correct:
            results[lib][0] += 1
            results[lib][1] += 1
        else:
            results[lib][1] += 1

    results_dict = {}
    for lib, (correct, total) in results.items():
        accuracy = correct / total if total else 0
        results_dict[lib] = accuracy * 100

    # Save the results to a JSON file
    with open(f"artifacts/{experiment_name}/result.json", "w") as f:
        json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    execute()
