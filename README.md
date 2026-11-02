# FineVerify

Code and data for the paper [&#34;FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search&#34;](https://aclanthology.org/2026.emnlp-main.2599.pdf) (EMNLP 2026 Main)

## 📁 Contents

```text
.
|- data/                         # benchmarks and decomposed subquestions
|- prompts.py                    # prompt templates for decomposition, generation, verification, and judging
|- tools.py                      # shared helpers for JSONL I/O, search, scoring, and answer selection, etc.
|- gpt5_mini_decompose.py        # GPT-5-mini decomposition for live web-search benchmarks
|- gemini_decompose.py           # Gemini decomposition for live web-search benchmarks
|- gpt5_mini_bc_decompose.py     # GPT-5-mini decomposition for BrowseComp-Plus
|- gemini_bc_decompose.py        # Gemini decomposition for BrowseComp-Plus
|- gpt5_mini_fineverify.py       # GPT-5-mini FineVerify driver for live web-search benchmarks
|- gemini_fineverify.py          # Gemini FineVerify driver for live web-search benchmarks
|- BrowseComp-Plus/              # BrowseComp-Plus repository, including code and setup files
|- searcher/                     # searcher implementations from BrowseComp-Plus, for illustration.
|- search_agent/                 # FineVerify components for BrowseComp-Plus with custom tools
```

Note: files with `bc` or `browsecomp` in the name are for the BrowseComp-Plus benchmark.

## 🚀 Setup

Create the environment:

```bash
conda create -n fineverify python=3.10
conda activate fineverify
```

Install the dependencies:

```bash
pip install -r fineverify/requirements.txt
```

Set the API keys for the providers:

```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
```

You can also create a `.env` file in `fineverify/` with the keys:

```text
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

### BrowseComp-Plus Setup

BrowseComp-Plus requires FAISS retrieval. Follow the setup instructions in [`BrowseComp-Plus/README.md`](BrowseComp-Plus/README.md) to configure the FAISS searcher and download the local index files. Then activate the environment and install the remaining dependencies, for example:

```bash
source BrowseComp-Plus/.venv/bin/activate
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
conda install -c conda-forge openjdk=21
```

Compatibility note: current BrowseComp-Plus setup may create incompatibilities between newer versions of `google-genai` (for `gemini-3-flash-preview`) and `fastmcp`. We recommend using a separate environment for the BrowseComp-Plus Gemini driver. After basic setup, upgrade these packages to the specified versions:

```bash
pip install --upgrade google-genai==1.70.0
pip install --upgrade fastmcp==3.2.0
```

## Question Decomposition

Run question decomposition on web-search benchmarks, such as DeepSearchQA, with GPT-5-mini:

```bash
python fineverify/gpt5_mini_decompose.py \
  --input_file fineverify/data/DeepsearchQA.jsonl \
  --output_file fineverify/data/decomposed/gpt_5_mini_DSQA-decomposed.jsonl
```

Run question decomposition on BrowseComp-Plus with GPT-5-mini:

```bash
python fineverify/gpt5_mini_bc_decompose.py \
  --input_file fineverify/data/Browsecomp_plus.jsonl \
  --output_file fineverify/data/decomposed/gpt_5_mini_browsecomp_decomposed.jsonl
```

For Gemini, use the same commands with the Gemini script and output file names.

## FineVerify on Live Web-Search Benchmarks

These drivers use provider-native web search for answering and verification.

GPT-5-mini:

```bash
python fineverify/gpt5_mini_fineverify.py \
  --config fineverify/config_gpt5_mini_fineverify.yaml
```

Use `config_gpt5_mini_fineverify.yaml` for GPT-5-mini and `config_gemini_fineverify.yaml` for Gemini. The default configs run a small slice (`start: 0`, `end: 1`) for smoke testing. Set `end: -1` for full runs.

## FineVerify on BrowseComp-Plus

First, follow the [`BrowseComp-Plus/README.md`](BrowseComp-Plus/README.md) setup instructions to configure the FAISS searcher and download the local index files. Then run the FineVerify drivers with custom search tools that interface with the local searcher and indices.

Copy or move files under `fineverify/search_agent/` to the corresponding directory under `BrowseComp-Plus/search_agent` to keep dependencies aligned. You can also move `data/` folder under `BrowseComp-Plus/`.

Edit configs as needed in `search_agent/config_gpt5_mini_bc_fineverify.yaml` and `search_agent/config_gemini_bc_fineverify.yaml`.

### 1. GPT-5-mini on BrowseComp-Plus

Run the GPT-5-mini driver with custom tools for BrowseComp-Plus.

```bash
python BrowseComp-Plus/search_agent/gpt5_mini_bc_fineverify.py \
  --config BrowseComp-Plus/search_agent/config_gpt5_mini_bc_fineverify.yaml
```

### 2. Gemini with MCP Tools on BrowseComp-Plus

Following [`BrowseComp-Plus/docs/gemini.md`](BrowseComp-Plus/docs/gemini.md), set up the MCP servers with Qwen3-Embedding search tools. FineVerify uses two MCP servers: one for candidate-answer search and one for verification.

- `search_mcp_url`: candidate-answer search tool, default `http://127.0.0.1:8080/mcp`
- `verification_mcp_url`: verification search and get_doc tools, default `http://127.0.0.1:8081/mcp`

Run:

```bash
python BrowseComp-Plus/search_agent/gemini_bc_fineverify.py \
  --config BrowseComp-Plus/search_agent/config_gemini_bc_fineverify.yaml
```

## 📤 Outputs

Each FineVerify run writes to `output_dir`:

```text
final_results.jsonl        # selected answer per question
all_rounds.jsonl           # compact round-level summaries
failed_records.jsonl       # failures, when any
params_for_run.jsonl       # run configuration snapshots
run_summary.json           # aggregate usage and status
separated/                 # per-round and final debug JSON
```

`final_results.jsonl` is the main file for downstream analysis.

## 🛠️ Common Options

- `--T`, `--max-rounds`: maximum FineVerify rounds per question.
- `--start`, `--end`, `--limit`: select a slice of the input JSONL.
- `--num-threads`: process multiple records concurrently.
- `--force`: rerun records already present in `final_results.jsonl`.
- `--score-supported`, `--score-not-found`, `--score-contradicted`: verification scoring.
- `--early-stop-score`: stop once a round reaches this score, default `1.0`. Set to `0` to disable early stopping.

Use `--help` on any script for the full argument list.

## 📝 Notes

- We do not tune prompts for specific benchmarks or models. Feel free to adjust the prompt templates in `prompts.py` for potential improvements.
- Paths in the YAML configs are usually relative to `fineverify/` and are
  resolved by the drivers.
- The BrowseComp-Plus GPT driver does not require MCP, but it does require the
  configured searcher dependencies and local index files.
- The BrowseComp-Plus Gemini driver requires running MCP servers before starting the FineVerify job.

## Citation

If you find our code or data useful, please cite:

```bibtex
@inproceedings{zhao-etal-2026-fineverify,
    title = "{F}ine{V}erify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search",
    author = "Zhao, James Xu  and
      Chen, Hui  and
      Hooi, Bryan  and
      Ng, See-Kiong",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2599/",
    doi = "10.18653/2026.emnlp-main.2599",
    pages = "59033--59062",
    ISBN = "979-8-89176-459-0",
    abstract = "Agentic search requires language model agents to explore many sources and answer complex information-seeking questions. Scaling test-time compute is a promising way to improve these agents, but current approaches can fail, because correct answers are often sparse and score-based selection depends on model calibration. We propose FineVerify, a fine-grained self-verification framework that decomposes each question into checkable sub-questions, verifies sampled candidates against each sub-question, and selects the candidate with the highest aggregated score. This per-check structure turns selection into simpler local judgments and produces scores under the same explicit criteria. Across four agentic search benchmarks and two models, FineVerify consistently outperforms standard scaling baselines. With only four sampled trajectories, it improves GPT-5-mini by 8.2 accuracy points and Gemini-3-flash by 5.6{\%} on average. With 12 samples, FineVerify enables GPT-5-mini to surpass frontier GPT-5 on BrowseComp-Plus. Beyond accuracy, FineVerify produces interpretable verification traces that help audit benchmark errors, suggesting broader applications for inspecting agentic search systems."
}
```

## Contact

For questions or suggestions, feel free to contact: `xu.zhao@u.nus.edu`
