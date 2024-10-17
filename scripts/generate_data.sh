#!/bin/bash

seed=1000
base_dir=`pwd`
data_dir=${base_dir}/data/

cd src/

# Generate all of the bulk data for each model
for model in 'meta-llama/Llama-2-13b-chat-hf' 'mistralai/Mixtral-8x7B-Instruct-v0.1' 'mistralai/Mistral-7B-Instruct-v0.2' 'HuggingFaceH4/zephyr-7b-beta' 'allenai/OLMo-7B-Instruct' 'meta-llama/Meta-Llama-3-8B-Instruct'; do
    # With demographics
    python bulk_generate_pct_vllm.py \
        --personas_file ${base_dir}/data/prompting/personas.json \
        --instructions_file ${base_dir}/data/prompting/instructions.json \
        --pct_questions_file ${base_dir}/data/political_compass/political_compass_questions.txt \
        --output_dir ${data_dir}/bulk \
        --model_id ${model} \
        --seed ${seed}
        
    python open_to_closed_vllm.py \
        --input_dir ${data_dir}/bulk \
        --output_dir ${data_dir}/bulk_converted \
        --model_id ${model}
        
    # Base case
    python bulk_generate_pct_vllm.py \
        --personas_file ${base_dir}/data/prompting/personas.json \
        --instructions_file ${base_dir}/data/prompting/instructions.json \
        --pct_questions_file ${base_dir}/data/political_compass/political_compass_questions.txt \
        --output_dir ${data_dir}/bulk_basecase \
        --model_id ${model} \
        --seed ${seed} \
        --base_case
        
    python open_to_closed_vllm.py \
        --input_dir ${data_dir}/bulk_basecase \
        --output_dir ${data_dir}/bulk_basecase_converted \
        --model_id ${model}
done

# Consolidate all of the data
python consolidate_data.py \
    --input_dir ${data_dir}/bulk_converted
    --output_dir ${data_dir}/bulk_consolidated

python consolidate_data.py \
    --input_dir ${data_dir}/bulk_basecase_converted
    --output_dir ${data_dir}/bulk_basecase_consolidated

cd ${base_dir}