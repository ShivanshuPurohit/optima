"""Per-epoch prompt sampling.

A kernel must not be able to special-case a fixed handful of prompts, so the
validator samples a fresh subset each epoch from a larger corpus, keyed by an
epoch seed. In production this corpus would be drawn from the real (agentic)
serving distribution and rotated/expanded each epoch; this is a stand-in that is
diverse enough to exercise varied shapes and stabilize the KL estimate.
"""

from __future__ import annotations

import random

CORPUS: tuple[str, ...] = (
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain, step by step, how a hash map handles collisions.",
    "Summarize the tradeoffs between TCP and UDP for a real-time game.",
    "Given a list of integers, describe an O(n) algorithm to find the majority element.",
    "Refactor a nested callback chain into async/await and explain why.",
    "What are the failure modes of two-phase commit, and how does Paxos help?",
    "Implement binary search and state its preconditions and invariants.",
    "Describe how a B-tree keeps itself balanced on insertion.",
    "Explain the CAP theorem with a concrete example for each pair.",
    "Walk through how TLS establishes a session key.",
    "Compare mutexes and channels for sharing state between threads.",
    "Explain how a generational garbage collector decides what to collect.",
    "Given a directed graph, outline Tarjan's algorithm for strongly connected components.",
    "Describe the memory hierarchy and why cache-oblivious algorithms matter.",
    "Explain backpropagation through a single linear layer with a bias.",
    "What is the difference between bagging and boosting, with an example each?",
    "Outline a rate limiter using a token bucket and discuss burst handling.",
    "Explain MVCC and how it avoids read locks in a database.",
    "Describe how consistent hashing reduces churn when a node leaves.",
    "Write a SQL query to find the second-highest salary per department.",
    "Explain how a bloom filter trades memory for false positives.",
    "Describe the actor model and where it fits versus shared memory.",
    "How does a CPU branch predictor work, and what is a misprediction penalty?",
    "Explain the difference between latency and throughput with an analogy.",
    "Outline how Raft elects a leader and commits a log entry.",
    "Describe how copy-on-write makes fork cheap.",
    "Explain what makes a hash function suitable for a hash table vs cryptography.",
    "Give an example where eventual consistency is acceptable and one where it is not.",
    "Describe how a JIT compiler decides what to optimize at runtime.",
    "Explain vectorization and when the compiler can and cannot do it for you.",
    "Walk through quicksort and explain the worst case and how to avoid it.",
    "Explain how attention computes a weighted sum and why it scales as O(n^2).",
    "Describe how paging and a TLB translate a virtual address.",
    "What is the difference between optimistic and pessimistic concurrency control?",
    "Explain how a reverse proxy and a load balancer differ in purpose.",
    "Describe the tradeoffs of column-oriented vs row-oriented storage.",
    "Explain how gradient checkpointing trades compute for memory.",
    "Outline how a merge sort can be parallelized across cores.",
    "Explain what a race condition is and give a minimal example.",
    "Describe how speculative decoding speeds up autoregressive generation.",
)


def sample_prompts(n: int, seed: int) -> list[str]:
    """Deterministically sample ``n`` prompts for an epoch.

    Without replacement when ``n <= len(CORPUS)``, otherwise with replacement so
    callers can request large workloads for throughput measurement.
    """
    rng = random.Random(seed)
    if n <= len(CORPUS):
        return rng.sample(list(CORPUS), n)
    return [rng.choice(CORPUS) for _ in range(n)]
