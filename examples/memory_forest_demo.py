"""
Minimal demonstration of the tree-structured cross-modal memory store.

Usage:
    python examples/memory_forest_demo.py --collection demo_forest
"""

from __future__ import annotations

import argparse
from typing import Iterable

from config import Config
from unimemrag.embedding.models.ClipEmbedding import ClipEmbedding
from unimemrag.memory_forest import (
    EventNode,
    LeafNode,
    MemoryForestStore,
    MemoryTree,
    RootNode,
    TreeRetrievalResult,
)


def build_sample_tree(doc_id: str) -> MemoryTree:
    root = RootNode(
        topic="Wildfire response and satellite monitoring overview.",
        root_id=doc_id,
        metadata={"fusion_alpha": 0.7},
    )
    events = [
        EventNode(
            event_id=f"{doc_id}::event-1",
            parent_id=doc_id,
            summary="Satellite identifies rapid spread across northern ridge line.",
            leaf_ids=(f"{doc_id}::event-1::leaf-1",),
        ),
        EventNode(
            event_id=f"{doc_id}::event-2",
            parent_id=doc_id,
            summary="Firefighters deploy aerial tankers with infrared guidance.",
            leaf_ids=(f"{doc_id}::event-2::leaf-1",),
        ),
    ]
    leaves = [
        LeafNode(
            leaf_id=f"{doc_id}::event-1::leaf-1",
            parent_id=f"{doc_id}::event-1",
            text=(
                "Thermal imaging from the Sentinel constellation reveals a new "
                "hotspot cluster moving east; response team escalates severity level."
            ),
        ),
        LeafNode(
            leaf_id=f"{doc_id}::event-2::leaf-1",
            parent_id=f"{doc_id}::event-2",
            text=(
                "Two water bombers execute alternating drops coordinated with drones "
                "tracking embers beyond containment lines."
            ),
        ),
    ]
    return MemoryTree(tree_id=doc_id, root=root, events=events, leaves=leaves)


def pretty_print(results: Iterable[TreeRetrievalResult]) -> None:
    for result in results:
        print(f"\nTree: {result.tree_id}")
        print(f"  Root score: {result.root.score:.4f} | topic: {result.root.payload.get('topic')}")
        for event in result.events:
            print(f"    Event score: {event.score:.4f} | summary: {event.payload.get('summary')}")
            for leaf in result.leaves.get(event.id, []):
                print(f"      Leaf score: {leaf.score:.4f} | text: {leaf.payload.get('content')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="memory_forest_demo")
    parser.add_argument("--text-query", default="infrared guided firefighting response")
    args = parser.parse_args()

    cfg = Config(collection=args.collection)
    embedder = ClipEmbedding()
    store = MemoryForestStore(cfg, vector_size=embedder.dim)

    tree = build_sample_tree("doc-wildfire")
    summary = store.ingest_trees([tree], embedder=embedder)
    print(f"Ingested | roots={summary['roots']} events={summary['events']} leaves={summary['leaves']}")

    results = store.retrieve(embedder, query_text=args.text_query, root_top_k=1, event_top_k=2, leaf_top_k=2)
    pretty_print(results)


if __name__ == "__main__":
    main()
