from pathlib import Path
import sys
sys.path.append("..")

from unimemrag.main import (
    RagPipelineConfig,
    create_retriever,
    load_viquae_dataset,
    index_text_corpus,
    index_image_corpus,
    build_rag_bundle,
    evaluate_bundle_with_model,
)


cfg = RagPipelineConfig(
    collection_name="mmlongbench_vrag",
    dataset_file="benchmark/MMLongBench/mmlb_data/vrag/viquae_K8_dep6.jsonl",
    image_root="benchmark/MMLongBench/mmlb_image",
    top_k=5,
)

retriever, embed_model, store = create_retriever(cfg)

dataset = load_viquae_dataset(cfg.dataset_file, max_samples=None)
index_text_corpus(dataset, embedder=embed_model, store=store, batch_size=cfg.text_batch_size)
index_image_corpus(dataset, embedder=embed_model, store=store, image_root=Path(cfg.image_root))

bundle = build_rag_bundle(dataset, retriever=retriever, image_root=Path(cfg.image_root), top_k=cfg.top_k)


from argparse import Namespace
from MMLongBench.vlm_model import load_LLM


args = Namespace(
    model_name_or_path="../../ckpts/Qwen2.5-VL-7B-Instruct",
    temperature=0.7, top_p=0.9,
    input_max_length=32768,
    generation_max_length=2048,
    generation_min_length=0,
    do_sample=False, stop_newline=False, use_chat_template=False,
    no_torch_compile=False, no_bf16=False, load_in_8bit=False,
    rope_theta=None, use_yarn=False, offload_state_dict=False,
    do_prefill=False, attn_implementation=None,
    image_resize=None, max_image_num=None, max_image_size=None,
    api_sleep=None, image_detail="auto",
)

model = load_LLM(args)

report = evaluate_bundle_with_model(model, bundle, num_workers=16, max_examples=None)

print(report["averaged_metrics"])