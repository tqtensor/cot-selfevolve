import json
import os
import platform
import shutil
import subprocess
import tarfile

import pyzstd
import requests
from xxhash import xxh64


def syntax_check(code: str) -> dict:
    try:
        compile(code, "<string>", "exec")
        return {"status": "success"}
    except SyntaxError as e:
        return {"status": "error", "line": e.lineno, "message": e.msg}


def _subprocess_run(command: list, cwd: str = None) -> None:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    _, stderr = process.communicate()
    print(stderr.decode())


def setup_test_env():
    """
    Sets up the test environment on a Linux system.

    This function checks if the system is Linux and if the `~/.cot-selfevolve`
    directory exists. If the system is not Linux, it raises an OSError. If the
    `~/.cot-selfevolve` directory does not exist, it downloads a Python binary from
    a specified URL, decompresses the downloaded file, and extracts its
    contents to the `~/.cot-selfevolve` directory.

    After setting up the `~/.cot-selfevolve` directory, it sets the Python
    interpreter path based on the metadata in the
    `~/.cot-selfevolve/python/PYTHON.json` file. It then creates a virtual
    environment in the `.venv` directory and installs the necessary
    dependencies using pip and poetry.

    Raises:
        OSError: If the system is not Linux.
    """
    if platform.system() != "Linux":
        raise OSError("This script only supports Linux.")
    elif not os.path.exists(os.path.expanduser("~/.cot-selfevolve")):
        # Download Python binary
        url = "https://github.com/indygreg/python-build-standalone/releases/download/20210506/cpython-3.8.10-x86_64-unknown-linux-gnu-pgo-20210506T0943.tar.zst"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        destination = "python-build-standalone.tar.zst"
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Decompress .zst file
        decompressed_file = "python-build-standalone.tar"
        with open(destination, "rb") as f_in, open(decompressed_file, "wb") as f_out:
            dctx = pyzstd.ZstdDecompressor()
            f_out.write(dctx.decompress(f_in.read()))

        # Open decompressed .tar file
        if os.path.exists(os.path.expanduser("~/.cot-selfevolve")):
            shutil.rmtree(os.path.expanduser("~/.cot-selfevolve"))
        with tarfile.open(decompressed_file, "r") as tar:
            tar.extractall(os.path.expanduser("~/.cot-selfevolve"))

        # Clean up
        os.remove(destination)
        os.remove(decompressed_file)

    # Set the Python interpreter path
    python_metadata = json.load(
        open(os.path.expanduser("~/.cot-selfevolve/python/PYTHON.json"))
    )
    python_interpreter = os.path.join(
        os.path.expanduser("~/.cot-selfevolve/python"), python_metadata["python_exe"]
    )

    # Install venv
    _subprocess_run([python_interpreter, "-m", "venv", ".venv"])

    # Install dependencies
    _subprocess_run([".venv/bin/pip", "install", "-U", "pip"])
    _subprocess_run([".venv/bin/pip", "install", "-U", "poetry"])
    _subprocess_run([".venv/bin/python", "-m", "poetry", "install", "--no-root"])


def hashes(doc: str) -> str:
    return xxh64(doc).hexdigest()
