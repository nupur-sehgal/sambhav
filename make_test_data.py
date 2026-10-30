"""
Build a bigger FAKE test collection for the chatbot.

Documents have the same shape as your real ones: title, content, subject and a
384-number embedding. They are written to a SEPARATE collection (default:
documents_test), so your real collection is never modified.

    python make_test_data.py                 # ~100 documents
    python make_test_data.py --scale 20      # ~2,000 documents (volume testing)
    python make_test_data.py --replace       # wipe and rebuild the test collection

Then set MONGODB_COLLECTION=documents_test in .env to point the chatbot at it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from pymongo.operations import SearchIndexModel

load_dotenv()

from db import COLLECTION_NAME, collection as main_collection, db  # noqa: E402

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
DEFAULT_COLLECTION = os.getenv("MONGODB_TEST_COLLECTION", "documents_test")

# --------------------------------------------------
# Seed content: (subject, title, content)
# --------------------------------------------------

SEED = [
    # ---- DBMS (the first two are your real documents) ----
    ("DBMS", "What is a DBMS?", "A DBMS is software used to create, manage, and access databases."),
    ("DBMS", "What is a primary key?", "A primary key uniquely identifies each record in a table."),
    ("DBMS", "What is a foreign key?", "A foreign key is a column that refers to the primary key of another table and links the two tables together."),
    ("DBMS", "What is normalization?", "Normalization is the process of organizing a database into tables to reduce redundancy and avoid update anomalies."),
    ("DBMS", "What is a transaction?", "A transaction is a group of database operations that run as a single unit: either all of them succeed or none of them do."),
    ("DBMS", "What are the ACID properties?", "ACID stands for atomicity, consistency, isolation and durability, the four guarantees that make database transactions reliable."),
    ("DBMS", "What is an index in a database?", "An index is a data structure that speeds up searches on a column at the cost of extra storage and slower writes."),
    ("DBMS", "What is a join?", "A join combines rows from two or more tables based on a related column between them."),
    ("DBMS", "What is SQL?", "SQL is the standard language used to define, query and modify data in relational databases."),
    ("DBMS", "What is a view?", "A view is a stored query that behaves like a virtual table and does not store data of its own."),
    ("DBMS", "What is a deadlock in a DBMS?", "A deadlock happens when two transactions each wait for a lock held by the other, so neither can continue."),
    ("DBMS", "What is denormalization?", "Denormalization deliberately adds redundant data to tables to make read queries faster."),
    ("DBMS", "What is a candidate key?", "A candidate key is a minimal set of columns that can uniquely identify every row in a table."),
    ("DBMS", "What is the difference between DELETE and TRUNCATE?", "DELETE removes rows one at a time and can use a WHERE clause, while TRUNCATE removes all rows quickly and cannot filter."),
    # ---- Operating Systems ----
    ("Operating Systems", "What is an operating system?", "An operating system is software that manages computer hardware and provides services that application programs need to run."),
    ("Operating Systems", "What is a process?", "A process is a program in execution, with its own memory space, registers and state."),
    ("Operating Systems", "What is a thread?", "A thread is the smallest unit of execution within a process, and threads in the same process share its memory."),
    ("Operating Systems", "What is virtual memory?", "Virtual memory lets a computer use disk space to extend RAM, giving each process the illusion of a large private address space."),
    ("Operating Systems", "What is paging?", "Paging divides memory into fixed-size pages so that a process's memory does not need to be contiguous in physical RAM."),
    ("Operating Systems", "What is a context switch?", "A context switch saves the state of the running process and loads the state of another so the CPU can be shared."),
    ("Operating Systems", "What is a semaphore?", "A semaphore is a synchronization variable used to control access to shared resources by multiple processes or threads."),
    ("Operating Systems", "What is a deadlock in an operating system?", "A deadlock occurs when a set of processes each hold a resource while waiting for another resource held by a process in the set."),
    ("Operating Systems", "What is CPU scheduling?", "CPU scheduling decides which ready process runs next on the CPU, using policies such as round robin or shortest job first."),
    ("Operating Systems", "What is a kernel?", "The kernel is the core of an operating system that controls hardware, memory and process execution."),
    ("Operating Systems", "What is thrashing?", "Thrashing happens when a system spends more time swapping pages in and out of memory than running processes."),
    ("Operating Systems", "What is a system call?", "A system call is the way a program asks the operating system's kernel to perform a privileged action such as reading a file."),
    # ---- Computer Networks ----
    ("Computer Networks", "What is an IP address?", "An IP address is a numerical label that identifies a device on a network and lets it send and receive data."),
    ("Computer Networks", "What is the OSI model?", "The OSI model describes network communication in seven layers, from the physical layer up to the application layer."),
    ("Computer Networks", "What is TCP?", "TCP is a connection-oriented protocol that delivers data reliably and in order between two hosts."),
    ("Computer Networks", "What is UDP?", "UDP is a connectionless protocol that sends data without guaranteeing delivery, which makes it fast and lightweight."),
    ("Computer Networks", "What is DNS?", "DNS translates human-readable domain names such as example.com into IP addresses."),
    ("Computer Networks", "What is a router?", "A router forwards data packets between different networks based on their destination IP addresses."),
    ("Computer Networks", "What is a subnet mask?", "A subnet mask splits an IP address into a network part and a host part."),
    ("Computer Networks", "What is HTTP?", "HTTP is the application-layer protocol that web browsers and servers use to exchange pages and data."),
    ("Computer Networks", "What is a firewall?", "A firewall filters network traffic according to security rules and blocks unwanted connections."),
    ("Computer Networks", "What is DHCP?", "DHCP automatically assigns IP addresses and other network settings to devices when they join a network."),
    ("Computer Networks", "What is a MAC address?", "A MAC address is a unique hardware identifier assigned to a network interface card."),
    ("Computer Networks", "What is the three-way handshake?", "The three-way handshake is the SYN, SYN-ACK and ACK exchange that TCP uses to open a connection."),
    # ---- Data Structures ----
    ("Data Structures", "What is an array?", "An array stores elements of the same type in contiguous memory and allows access by index in constant time."),
    ("Data Structures", "What is a linked list?", "A linked list is a sequence of nodes where each node stores a value and a reference to the next node."),
    ("Data Structures", "What is a stack?", "A stack is a last-in, first-out structure where elements are added and removed from the top."),
    ("Data Structures", "What is a queue?", "A queue is a first-in, first-out structure where elements are added at the back and removed from the front."),
    ("Data Structures", "What is a binary tree?", "A binary tree is a hierarchical structure in which each node has at most two children."),
    ("Data Structures", "What is a binary search tree?", "A binary search tree keeps smaller values in the left subtree and larger values in the right, which makes searching fast."),
    ("Data Structures", "What is a hash table?", "A hash table maps keys to values using a hash function, giving average constant-time lookups."),
    ("Data Structures", "What is a heap?", "A heap is a tree-based structure where each parent is ordered relative to its children, which makes it useful for priority queues."),
    ("Data Structures", "What is a graph?", "A graph is a set of vertices connected by edges and is used to model networks and relationships."),
    ("Data Structures", "What is a trie?", "A trie is a tree that stores strings by their characters, making prefix searches efficient."),
    ("Data Structures", "What is a priority queue?", "A priority queue removes the element with the highest priority first, regardless of insertion order."),
    ("Data Structures", "What is a circular queue?", "A circular queue connects the end of the queue back to the front so freed space can be reused."),
    # ---- Algorithms ----
    ("Algorithms", "What is binary search?", "Binary search finds an item in a sorted list by repeatedly halving the search range."),
    ("Algorithms", "What is Big O notation?", "Big O notation describes how an algorithm's running time or memory use grows as the input size grows."),
    ("Algorithms", "What is quicksort?", "Quicksort picks a pivot, partitions the data around it, and sorts each part recursively, with average time O(n log n)."),
    ("Algorithms", "What is merge sort?", "Merge sort splits a list in half, sorts each half and merges them, always running in O(n log n) time."),
    ("Algorithms", "What is dynamic programming?", "Dynamic programming solves a problem by breaking it into overlapping subproblems and storing their results to avoid repeated work."),
    ("Algorithms", "What is a greedy algorithm?", "A greedy algorithm makes the best local choice at each step in the hope of reaching a global optimum."),
    ("Algorithms", "What is recursion?", "Recursion is a technique where a function solves a problem by calling itself on a smaller version of the same problem."),
    ("Algorithms", "What is breadth-first search?", "Breadth-first search explores a graph level by level, visiting all neighbours before moving deeper."),
    ("Algorithms", "What is depth-first search?", "Depth-first search explores a graph by going as deep as possible along each branch before backtracking."),
    ("Algorithms", "What is Dijkstra's algorithm?", "Dijkstra's algorithm finds the shortest paths from one vertex to all others in a graph with non-negative edge weights."),
    ("Algorithms", "What is backtracking?", "Backtracking builds a solution step by step and abandons a path as soon as it cannot lead to a valid solution."),
    ("Algorithms", "What is a divide and conquer algorithm?", "Divide and conquer splits a problem into smaller independent parts, solves each, and combines the results."),
    # ---- Machine Learning ----
    ("Machine Learning", "What is machine learning?", "Machine learning is a field in which computers learn patterns from data instead of being explicitly programmed."),
    ("Machine Learning", "What is supervised learning?", "Supervised learning trains a model on labelled examples so it can predict labels for new data."),
    ("Machine Learning", "What is unsupervised learning?", "Unsupervised learning finds structure in unlabelled data, such as clusters or patterns."),
    ("Machine Learning", "What is overfitting?", "Overfitting happens when a model learns the training data too closely and performs poorly on new data."),
    ("Machine Learning", "What is a neural network?", "A neural network is a model made of layers of connected units that learn to map inputs to outputs."),
    ("Machine Learning", "What is gradient descent?", "Gradient descent is an optimization method that repeatedly adjusts parameters in the direction that reduces the loss."),
    ("Machine Learning", "What is cross-validation?", "Cross-validation estimates how well a model generalizes by training and testing it on several different splits of the data."),
    ("Machine Learning", "What is a decision tree?", "A decision tree makes predictions by asking a series of yes-or-no questions about the input features."),
    ("Machine Learning", "What is the bias-variance tradeoff?", "The bias-variance tradeoff is the balance between a model that is too simple and one that is too sensitive to the training data."),
    ("Machine Learning", "What is k-means clustering?", "K-means groups data into k clusters by assigning each point to the nearest cluster centre and updating the centres repeatedly."),
    ("Machine Learning", "What is a confusion matrix?", "A confusion matrix is a table that compares a classifier's predicted labels with the true labels."),
    ("Machine Learning", "What is regularization?", "Regularization adds a penalty on model complexity during training to reduce overfitting."),
    # ---- Python ----
    ("Python", "What is a list in Python?", "A list is an ordered, changeable collection of items written in square brackets."),
    ("Python", "What is a tuple in Python?", "A tuple is an ordered collection like a list, but it cannot be changed after it is created."),
    ("Python", "What is a dictionary in Python?", "A dictionary stores key-value pairs and lets you look up a value by its key."),
    ("Python", "What is a decorator in Python?", "A decorator is a function that wraps another function to add behaviour without changing its code."),
    ("Python", "What is a list comprehension?", "A list comprehension builds a new list from an existing iterable in a single compact expression."),
    ("Python", "What is a generator in Python?", "A generator produces values one at a time using yield, which saves memory for large sequences."),
    ("Python", "What is the GIL in Python?", "The Global Interpreter Lock lets only one thread run Python bytecode at a time in CPython."),
    ("Python", "What is a lambda function?", "A lambda is a small anonymous function written in one line with the lambda keyword."),
    ("Python", "What is a virtual environment in Python?", "A virtual environment is an isolated folder with its own Python packages so projects do not conflict."),
    ("Python", "What is exception handling in Python?", "Exception handling uses try and except blocks to catch errors and keep a program running."),
    ("Python", "What is a Python module?", "A module is a file of Python code that can be imported and reused in other programs."),
    ("Python", "What is pip?", "Pip is the standard tool for installing and managing Python packages."),
    # ---- Software Engineering ----
    ("Software Engineering", "What is the software development life cycle?", "The software development life cycle is the series of stages from planning and design to coding, testing and maintenance."),
    ("Software Engineering", "What is Agile?", "Agile is an approach to building software in short iterations with frequent feedback and changing requirements."),
    ("Software Engineering", "What is version control?", "Version control tracks changes to files over time so teams can collaborate and return to earlier versions."),
    ("Software Engineering", "What is unit testing?", "Unit testing checks that the smallest parts of a program, such as single functions, behave correctly."),
    ("Software Engineering", "What is continuous integration?", "Continuous integration automatically builds and tests code every time a change is merged."),
    ("Software Engineering", "What is refactoring?", "Refactoring restructures existing code to make it cleaner without changing what it does."),
    ("Software Engineering", "What is an API?", "An API is a defined way for one piece of software to request services or data from another."),
    ("Software Engineering", "What is a design pattern?", "A design pattern is a reusable solution to a common problem in software design."),
    ("Software Engineering", "What is technical debt?", "Technical debt is the future cost of rework caused by choosing a quick solution over a better one."),
    ("Software Engineering", "What is the waterfall model?", "The waterfall model builds software in fixed sequential phases, with each phase finished before the next begins."),
    ("Software Engineering", "What is a code review?", "A code review is when other developers examine a change before it is merged to catch bugs and improve quality."),
    ("Software Engineering", "What is DevOps?", "DevOps is a set of practices that joins software development and operations to deliver changes faster and more reliably."),
]

# --------------------------------------------------
# How the text is turned into an embedding
# --------------------------------------------------

RECIPES = {
    "content only": lambda d: d["content"],
    "title only": lambda d: d["title"],
    "title + newline + content": lambda d: f'{d["title"]}\n{d["content"]}',
    "title + space + content": lambda d: f'{d["title"]} {d["content"]}',
    "title: content": lambda d: f'{d["title"]}: {d["content"]}',
}
DEFAULT_RECIPE = "title + newline + content"
MATCH_THRESHOLD = 0.98


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect_recipe(model, real_docs: list[dict]) -> tuple[str, float | None]:
    """
    Work out which text was embedded in your real documents, by re-embedding
    each candidate and comparing with the stored vector. Returns (name, score);
    score is None when there is nothing to compare against.
    """
    if not real_docs:
        return DEFAULT_RECIPE, None

    scores: dict[str, float] = {}
    for name, make_text in RECIPES.items():
        try:
            scores[name] = min(
                cosine(model.encode(make_text(doc)), doc["embedding"])
                for doc in real_docs
            )
        except ValueError:  # vector lengths differ: a different embedding model
            scores[name] = -1.0

    best = max(scores, key=scores.get)
    if scores[best] < MATCH_THRESHOLD:
        return DEFAULT_RECIPE, scores[best]
    return best, scores[best]


# --------------------------------------------------
# Build, load, index
# --------------------------------------------------

def build_documents(scale: int) -> list[dict]:
    docs = []
    for round_number in range(scale):
        for subject, title, content in SEED:
            suffix = "" if round_number == 0 else f" (set {round_number + 1})"
            docs.append({"title": title + suffix, "content": content, "subject": subject})
    return docs


def ensure_vector_index(target, dimensions: int) -> bool:
    """Create the vector index if it is missing. Returns False if it needs doing by hand."""
    definition = {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"}
        ]
    }

    try:
        if list(target.list_search_indexes(VECTOR_INDEX)):
            print(f"Vector index '{VECTOR_INDEX}' already exists.")
            return True

        target.create_search_index(
            SearchIndexModel(definition=definition, name=VECTOR_INDEX, type="vectorSearch")
        )
        print(f"Creating vector index '{VECTOR_INDEX}' ...")
        return True
    except Exception as exc:
        print(f"\nCould not create the index automatically: {type(exc).__name__}: {exc}")
        print("Create it by hand in Atlas (Search & Vector Search > Create Search Index >")
        print(f"Vector Search > JSON editor), on collection '{target.name}', named '{VECTOR_INDEX}':")
        print(json.dumps(definition, indent=2))
        return False


def wait_for_index(target, timeout: int = 180) -> bool:
    """Poll until Atlas reports the index as queryable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        indexes = list(target.list_search_indexes(VECTOR_INDEX))
        if indexes and indexes[0].get("queryable"):
            return True
        time.sleep(5)
    return False


def run(model, name: str, scale: int, replace: bool, make_index: bool) -> dict:
    """Create the test collection. Raises SystemExit with a clear message on misuse."""
    if name == COLLECTION_NAME:
        sys.exit(
            f"Refusing to write to '{name}': that is your main collection. "
            "Choose another name with --collection."
        )

    target = db[name]
    existing = target.count_documents({})
    if existing and not replace:
        sys.exit(
            f"'{name}' already has {existing} documents. "
            "Use --replace to wipe and rebuild it, or pick another --collection."
        )

    real_docs = list(main_collection.find({"embedding": {"$exists": True}}).limit(3))
    recipe, score = detect_recipe(model, real_docs)
    if score is None:
        print(f"No embedded documents found in '{COLLECTION_NAME}'; using '{recipe}'.")
    elif score >= MATCH_THRESHOLD:
        print(f"Your real embeddings were made from: {recipe} (similarity {score:.3f}). Using the same.")
    else:
        print(
            f"Could not reproduce your real embeddings (best match {score:.2f}); "
            f"using '{recipe}'. Retrieval on test data may behave differently."
        )

    docs = build_documents(scale)
    vectors = model.encode(
        [RECIPES[recipe](doc) for doc in docs], batch_size=64, show_progress_bar=True
    )
    for doc, vector in zip(docs, vectors):
        doc["embedding"] = [float(x) for x in vector]

    dimensions = len(docs[0]["embedding"])
    if real_docs and len(real_docs[0]["embedding"]) != dimensions:
        print(
            f"Warning: test vectors have {dimensions} numbers but your real ones have "
            f"{len(real_docs[0]['embedding'])}. Check EMBEDDING_MODEL in .env."
        )

    if existing:
        target.delete_many({})
    for start in range(0, len(docs), 500):
        target.insert_many(docs[start:start + 500])
    print(f"Inserted {len(docs)} documents into '{name}'.")

    index_ready = None
    if make_index:
        if ensure_vector_index(target, dimensions):
            print("Waiting for Atlas to build the index (usually under a minute) ...")
            index_ready = wait_for_index(target)
            print("Index is ready." if index_ready else
                  "Index is still building. Check Atlas; document questions work once it is READY.")

    return {"collection": name, "documents": len(docs), "recipe": recipe, "dimensions": dimensions, "index_ready": index_ready}


def main():
    parser = argparse.ArgumentParser(description="Create a fake test collection.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"target collection (default {DEFAULT_COLLECTION})")
    parser.add_argument("--scale", type=int, default=1, help="repeat the seed set N times for volume tests (adds near-duplicates)")
    parser.add_argument("--replace", action="store_true", help="wipe and rebuild the target collection")
    parser.add_argument("--no-index", action="store_true", help="do not create the vector index")
    args = parser.parse_args()

    if args.scale < 1:
        sys.exit("--scale must be at least 1.")

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    summary = run(model, args.collection, args.scale, args.replace, not args.no_index)

    print("\nDone. Next:")
    print(f"  1. In .env set  MONGODB_COLLECTION={summary['collection']}")
    print("  2. Restart the app:  streamlit run app.py")
    print(f"  3. To go back to your real data, set MONGODB_COLLECTION={COLLECTION_NAME}")


if __name__ == "__main__":
    main()