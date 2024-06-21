# CoT-SelfEvolve

## Reproduce the Experiments

### Presequites

#### AWS CLI

You need to install AWS CLI and have an AWS account to run DVC to pull data from S3. You can install AWS CLI by following the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).

#### Python dependencies

Due to the constraints of DS-1000, they expect to run the benchmark tests on specific versions of the libraries, so we have to maintain the Python dependencies at a specific version of Python 3.8. To install the necessary dependencies, run the following command:

```bash
pip install -U poetry
poetry install
```

#### LLMs

Currently, we support running with the following LLMs:
- Azure OpenAI
- OpenAI
- Vertex AI
- AWS Bedrock

Depending on which LLM you have access to, please fill in the `.env` file with the necessary credentials.

```text
AZURE_API_KEY=
AZURE_API_BASE=
AZURE_API_VERSION=

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION_NAME=

OPENAI_API_KEY=
```

Especially for Vertex AI, you need to put the service account key in the `src/vertex_key.json` file.

### Start the Vector DB

To start the Vector DB, you can run the following command to download the embeddings and start the Chroma server:

```bash
poetry run dvc pull
bash scripts/docker_compose.sh
```

### Running the Experiments

You can control the experiment settings through command-line arguments when running the `main.py` file. Here are the available options:

- `--experiment_name`: The name of the experiment.
- `--sampling_fraction`: The fraction of the dataset to sample.
- `--initial_strategy`: The initial strategy to use, either Chain-of-Thought (COT) or Zero-Shot (ZEROSHOT).
- `--correction_strategy`: The correction strategy to use, either Chain-of-Thought (COT) or Zero-Shot (ZEROSHOT).
- `--initial_model`: The initial LLM model to use at the initial stage.
- `--correction_model`: The correction LLM model to use at the correction stage.
- `--temperature`: The temperature setting for LLM.
- `--top_p`: The top-p setting for LLM.
- `--self_correction`: Enable self correction. Use `--no-self_correction` to disable.
- `--max_self_correction_attempts`: Max self correction attempts.
- `--demo`: Run in demo mode with reduced logging. Use `--no-demo` to disable.

To run the experiment with custom settings, execute the following command in your terminal:

```bash
python main.py --experiment_name <experiment_name> \
    --sampling_fraction <fraction> \
    --initial_strategy <strategy> --correction_strategy <strategy> \
    --initial_model <model_name> \
    --correction_model <model_name> \
    --temperature <value> --top_p <value> \
    --self_correction --max_self_correction_attempts <attempts> --demo
```

## Acknowledgements

I would like to express my gratitude to the following individuals for their valuable contributions to this project:
- [Sean](https://github.com/seanphan) from [PixelML](https://pixelml.com/) deserves special recognition for his substantial support in providing LLMs credits.
