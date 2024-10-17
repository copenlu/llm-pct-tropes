# LLM Tropes: Revealing Fine-Grained Values and Opinions in Large Language Models

# Code

## Generating the data
If you wish to generate the data from scratch, perform the following (otherwise, the data is available on [huggingface](https://huggingface.co/datasets/copenlu/llm-pct-tropes))

The code to generate the bulk data is under `src/bulk_generate_pct_vllm.py`. After generating the data, you can get the predicted stance for the open-ended prompts using `src/open_to_closed_vllm.py`. The final consolidation is done using `src/consolidate_data.py`. This is orchestrated under `scripts/generate_data.sh` so you can simply run the following:

```
$ bash scripts/generate_data.sh
```

After running the script, the data can be found in the directories `data/bulk_consolidated/` and `data/bulk_basecase_consolidated/` for each model in a csv format. 

The tropes can then be exracted and generated using `src/tropes/trope_extraction.py`. Save the final tropes csv to `data/tropes.csv`

## Running the analysis
All of the analysis and figure generation can be found in the `src/analysis.ipynb` notebook.

# Dataset
The dataset for our work can be found on Huggingface Datasets here: https://huggingface.co/datasets/copenlu/llm-pct-tropes

# Citation
If you use our code or dataset, kindly cite using
```
@inproceedings{wright2024revealingfinegrainedvaluesopinions,
      title={LLM Tropes: Revealing Fine-Grained Values and Opinions in Large Language Models},
      author={Dustin Wright and Arnav Arora and Nadav Borenstein and Srishti Yadav and Serge Belongie and Isabelle Augenstein},
      year={2024},
      booktitle = {Findings of EMNLP},
      publisher = {Association for Computational Linguistics}
}
```
