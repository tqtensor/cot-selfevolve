import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import List

import chromadb
import pandas as pd
import psutil
from chromadb.config import Settings
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from minio import Minio

from src.dataset.minio_helper import Progress

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


def get_posts(file_name: str, tags: List[str] = [""]):
    """
    Extracts and saves posts from a Stack Overflow XML dump.

    This function iterates over an XML dump of Stack Overflow posts, extracts
    posts that contain any of the specified tags, and saves them to individual
    JSON files. It also saves metadata about the posts to a CSV file. If a post
    cannot be parsed due to a ParseError, it is skipped. If any other exception
    occurs, the function logs the error and stops.

    Args:
        file_name (str): The name of the XML file containing the Stack Overflow
                         dump.
        tags (List[str], optional): A list of tags to filter the posts by. If a
                                    post contains any of these tags, it is
                                    saved. If not provided, all posts are saved.
    """
    if not os.path.exists("stack_overflow/posts"):
        os.makedirs("stack_overflow/posts")

    posts_count = 1
    posts_metadata = []

    xmlit = iter(
        ET.iterparse(
            source=file_name,
            events=("start", "end"),
        )
    )
    event, _ = next(xmlit)
    while True:
        try:
            event, elem = next(xmlit)
            if event == "end" and elem.tag == "row":
                post = elem.attrib
                post_tags = post.get("Tags", "")
                if any(tag in post_tags for tag in tags):
                    posts_count += 1
                    posts_metadata.append(
                        {
                            "id": post.get("Id", 1),
                            "tags": post_tags,
                            "created_at": post.get("CreationDate", ""),
                        }
                    )
                    # Save post to file
                    if not os.path.exists(
                        os.path.join(
                            "stack_overflow", "posts", post.get("Id", 1) + ".json"
                        )
                    ):
                        with open(
                            os.path.join(
                                "stack_overflow",
                                "posts",
                                post.get("Id", 1) + ".json",
                            ),
                            "w",
                        ) as f:
                            json.dump(post, f, indent=4)

                    if posts_count % 1000 == 0:
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        logger.info(
                            f"Found {posts_count} posts. Current memory usage: {round(mem_info.rss / 1024 ** 2)} MB"
                        )

                # Clear the element to free up memory
                elem.clear()
        except ET.ParseError:
            logger.warning("ParseError occurred, skipping to next row")
            continue
        except StopIteration:
            logger.info(elem.attrib)
            logger.info("StopIteration occurred")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break

    # Save metadata to file
    df = pd.DataFrame(posts_metadata)
    df.to_csv("stack_overflow/posts_metadata.csv", index=False)

    logger.info("Completed!")


def get_comments(file_name: str, post_ids: set):
    """
    Extracts and saves comments from a Stack Overflow XML dump.

    This function iterates over an XML dump of Stack Overflow comments,
    extracts comments that are associated with any of the specified post IDs,
    and saves them to individual JSON files. It also saves metadata about the
    comments to a CSV file. If a comment cannot be parsed due to a ParseError,
    it is skipped. If any other exception occurs, the function logs the error
    and stops.

    Args:
        file_name (str): The name of the XML file containing the Stack Overflow
                         dump.
        post_ids (set): A set of post IDs to filter the comments by. If a
        comment is associated with any of these post IDs, it is saved.
    """
    if not os.path.exists("stack_overflow/comments"):
        os.makedirs("stack_overflow/comments")

    comments_count = 1
    comments_metadata = []

    xmlit = iter(
        ET.iterparse(
            source=file_name,
            events=("start", "end"),
        )
    )
    event, _ = next(xmlit)
    while True:
        try:
            event, elem = next(xmlit)
            if event == "end" and elem.tag == "row":
                comment = elem.attrib
                if int(comment.get("PostId", 1)) in post_ids:
                    comments_count += 1
                    comments_metadata.append(
                        {
                            "id": comment.get("Id", 1),
                            "post_id": comment.get("PostId", 1),
                            "created_at": comment.get("CreationDate", ""),
                        }
                    )

                    # Save comment to file
                    if not os.path.exists(
                        os.path.join(
                            "stack_overflow",
                            "comments",
                            comment.get("Id", 1)
                            + "_"
                            + comment.get("PostId", 1)
                            + ".json",
                        )
                    ):
                        with open(
                            os.path.join(
                                "stack_overflow",
                                "comments",
                                comment.get("Id", 1)
                                + "_"
                                + comment.get("PostId", 1)
                                + ".json",
                            ),
                            "w",
                        ) as f:
                            json.dump(comment, f, indent=4)

                    if comments_count % 10000 == 0:
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        logger.info(
                            f"Found {comments_count} comments. Current memory usage: {round(mem_info.rss / 1024 ** 2)} MB"
                        )

                # Clear the element to free up memory
                elem.clear()
        except ET.ParseError:
            logger.warning("ParseError occurred, skipping to next row")
            continue
        except StopIteration:
            logger.info(elem.attrib)
            logger.info("StopIteration occurred")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break

    # Save metadata to file
    df = pd.DataFrame(comments_metadata)
    df.to_csv("stack_overflow/comments_metadata.csv", index=False)

    logger.info("Completed!")


class StackOverflowDataset:
    def __init__(self, precomputed: bool = True) -> None:
        """
        Initializes the StackOverflow class.

        This method initializes the StackOverflow class. If `precomputed` is
        False, it downloads and processes the posts and comments from MinIO,
        and initializes the vector store. If `precomputed` is True, it raises a
        NotImplementedError, as precomputed vector stores are not supported.

        Args:
            precomputed (bool, optional): A flag indicating whether the vector
                                          store is precomputed. Defaults to
                                          False.

        Raises:
            NotImplementedError: If `precomputed` is True, as precomputed
                                 vector stores are not supported.
        """
        if not precomputed:
            # Download Posts and Comments from MinIO
            client = Minio(
                endpoint=os.getenv("MINIO_ENDPOINT"),
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=False,
            )
            if not os.path.exists("stackoverflow.com-Posts.7z") and not os.path.exists(
                "Posts.xml"
            ):
                client.fget_object(
                    bucket_name="bk-imp",
                    object_name="self-debug/stackoverflow.com-Posts.7z",
                    file_path="stackoverflow.com-Posts.7z",
                    progress=Progress(),
                )
            if not os.path.exists(
                "stackoverflow.com-Comments.7z"
            ) and not os.path.exists("Comments.xml"):
                client.fget_object(
                    bucket_name="bk-imp",
                    object_name="self-debug/stackoverflow.com-Comments.7z",
                    file_path="stackoverflow.com-Comments.7z",
                    progress=Progress(),
                )

            # Extract Posts and Comments
            if not os.path.exists("Posts.xml"):
                os.system("7z x stackoverflow.com-Posts.7z")
                os.remove("stackoverflow.com-Posts.7z")
            if not os.path.exists("Comments.xml"):
                os.system("7z x stackoverflow.com-Comments.7z")
                os.remove("stackoverflow.com-Comments.7z")

            # Get all related posts
            get_posts(
                file_name="Posts.xml",
                tags=[
                    "matplotlib",
                    "pandas",
                    "numpy",
                    "scipy",
                    "seaborn",
                    "sklearn",
                    "tensorflow",
                    "pytorch",
                ],
            )

            # Get list of PostId
            post_ids = set(
                pd.read_csv("stack_overflow/posts_metadata.csv")["id"].tolist()
            )

            # Get all comments
            get_comments(file_name="Comments.xml", post_ids=post_ids)

        # Build index of Posts and Comments
        if not os.path.exists("stack_overflow/index.csv"):
            posts_metadata = pd.read_csv("stack_overflow/posts_metadata.csv")[
                ["id", "tags"]
            ].rename(columns={"id": "post_id"})
            comments_metadata = pd.read_csv(
                "stack_overflow/comments_metadata.csv"
            ).rename(columns={"id": "comment_id"})
            self.index = pd.merge(
                posts_metadata, comments_metadata, on="post_id", how="inner"
            ).sort_values(by=["post_id", "comment_id", "created_at"])
            self.index.to_csv("stack_overflow/index.csv", index=False)
        else:
            self.index = pd.read_csv("stack_overflow/index.csv")

    @staticmethod
    def retrieve(query: str, k: int) -> List[Document]:
        return vector_store.similarity_search(query=query, k=k)


if __name__ == "__main__":
    # Initialize the StackOverflowDataset class from sratch
    dataset = StackOverflowDataset(precomputed=False)
    print(dataset.retrieve("How to plot a graph in Python?", k=5))
