import glob
import json
import logging
import os
import sys
from typing import List

import chromadb
import pandas as pd
import tiktoken as tk
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm.auto import tqdm

load_dotenv()  # load env before importing other modules

from src.dataset import Dataset
from src.utils import hashes

# Tiktoken encoding
encoding = tk.encoding_for_model("gpt-3.5-turbo-0613")

# Chroma vector store
client = chromadb.Client(
    settings=Settings(
        chroma_api_impl="rest",
        chroma_server_host="0.0.0.0",
        chroma_server_http_port=8000,
    )
)
vector_store = Chroma(
    client=client,
    collection_name="stack_overflow",
    embedding_function=AzureOpenAIEmbeddings(
        openai_api_type="azure",
        openai_api_key=os.getenv("AZURE_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        deployment="text-embedding-ada-002",
        max_retries=30,
    ),
)

# Create a logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def render_posts(
    index: pd.DataFrame, post_id: int, max_token: int, min_comments: int, step: int
) -> List[str]:
    """
    Renders posts and their comments from Stack Overflow.

    This function retrieves a post and its comments from Stack Overflow, and
    renders them into a list of strings.
    Each string contains the post and a subset of its comments. The function
    ensures that the total number of tokens
    in each string does not exceed `max_token`. It also ensures that the post
    has at least `min_comments` comments.
    The function skips `step` comments at a time when rendering the post and
    comments.

    Args:
        index (pd.DataFrame): A DataFrame containing the post and comment IDs.
        post_id (int): The ID of the post to render.
        max_token (int): The maximum number of tokens allowed in each rendered
                         string.
        min_comments (int): The minimum number of comments that a post should
                            have.
        step (int): The number of comments to skip at a time when rendering the
                    post and comments.

    Returns:
        List[str]: A list of strings, each containing a post and a subset of
                its comments.
    """
    post_template = """
    Post: {post}

    Comments:
    """

    # Retrieve the post and comments
    post = json.load(
        open(os.path.join("stack_overflow", "posts", str(post_id) + ".json"), "r")
    ).get("Body", "")
    comments = []
    for comment_id in index[index["post_id"] == post_id]["comment_id"]:
        comment = json.load(
            open(
                os.path.join(
                    "stack_overflow",
                    "comments",
                    str(comment_id) + "_" + str(post_id) + ".json",
                ),
                "r",
            )
        ).get("Text", "")
        comments.append(comment)
    if len(comments) < min_comments:
        return [""]

    # Count the number of tokens from post and comments
    post_tokens = len(encoding.encode(post))
    if post_tokens > max_token:
        return [""]
    comments_tokens = [len(encoding.encode(comment)) for comment in comments]

    rendered_posts = []

    # Outer loop: iterate over the comments
    for i in range(0, len(comments_tokens), step):
        # Initialize the total number of tokens with the number of tokens in the post
        total_tokens = post_tokens

        # Initialize the list of comments for this post
        post_comments = []

        # Inner loop: try to add as many subsequent comments as possible
        for comment, comment_tokens in zip(comments[i:], comments_tokens[i:]):
            if total_tokens + comment_tokens > max_token:
                break
            total_tokens += comment_tokens
            post_comments.append("- comment: " + comment)

        # Render the post and comments, and add it to the list of rendered posts
        rendered_posts.append(
            post_template.format(post=post) + "\n".join(post_comments)
        )

    # Return the list of rendered posts
    return rendered_posts


def ingest_vector_store(docs: List[Document], vector_store: Chroma) -> None:
    """
    Ingests a list of documents into a vector store.

    This function takes a list of documents and a vector store as input. It
    first deduplicates the documents based on their hashed page content. Then,
    it checks if the unique documents are already in the vector store. If there
    are new unique documents that are not in the vector store, it ingests them
    into the vector store.

    Args:
        docs (List[Document]): A list of documents to be ingested into the
                               vector store.
        vector_store (Chroma): The vector store where the documents will be
                               ingested.
    """
    if len(docs) > 0:
        # Deduplicate docs
        unique_ids = set()
        unique_docs = []
        for doc in docs:
            doc_id = hashes(doc.page_content)
            if doc_id not in unique_ids:
                unique_ids.add(doc_id)
                unique_docs.append(doc)

        docs_in_store = set(vector_store.get()["ids"])
        unique_ids = unique_ids.difference(docs_in_store)
        unique_docs = [
            doc for doc in unique_docs if hashes(doc.page_content) in unique_ids
        ]

        if len(unique_docs) == 0:
            logger.info("No new docs to ingest")
        else:
            # Ingest into the vector store
            logger.info(f"Ingesting {len(unique_docs)} documents into the vector store")
            vector_store.add_documents(
                documents=unique_docs,
                ids=list(unique_ids),
            )
    else:
        logger.info("Empty docs")


if __name__ == "__main__":
    # Initialize the dataset
    stack_overflow_dataset = Dataset(
        dataset="stack_overflow", kwargs={"download": False}
    ).dataset

    # Generate a list of rendered posts
    if not os.path.exists("stack_overflow/rendered_posts"):
        os.makedirs("stack_overflow/rendered_posts")

    post_ids = stack_overflow_dataset.index["post_id"].unique()
    for post_id in tqdm(post_ids, total=len(post_ids)):
        rendered_posts = render_posts(
            index=stack_overflow_dataset.index,
            post_id=post_id,
            max_token=3000,
            min_comments=10,
            step=5,
        )
        if rendered_posts[0] != "":
            for i, rendered_post in enumerate(rendered_posts):
                with open(
                    os.path.join(
                        "stack_overflow", "rendered_posts", f"{post_id}_{i}.txt"
                    ),
                    "w",
                ) as f:
                    f.write(rendered_post)

    # Ingest into the vector store
    rendered_posts = glob.glob("stack_overflow/rendered_posts/*.txt")

    chunk_size = 20
    for i in tqdm(range(0, len(rendered_posts), chunk_size)):
        chunk = rendered_posts[i : i + chunk_size]
        docs = []
        for post in chunk:
            with open(post, "r") as f:
                docs.append(Document(page_content=f.read()))
        ingest_vector_store(docs=docs, vector_store=vector_store)
