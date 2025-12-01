pixiu_path=$(pwd)
export PYTHONPATH="$pixiu_path/src:$pixiu_path/src/financial-evaluation:$pixiu_path/src/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_LOG_LEVEL="ERROR"

TASKS="flare_fpb,\
flare_fiqasa,\
flare_headlines,\
flare_ner,\
flare_convfinqa,\
flare_finer_ord,\
flare_finred,\
flare_causal20_sc,\
flare_cd,\
flare_fnxl,\
flare_fsrl,\
flare_tsa,\
flare_fomc,\
flare_finarg_ecc_auc,\
flare_finarg_ecc_arc,\
flare_multifin_en,\
flare_ma,\
flare_mlesg,\
flare_finqa,\
flare_tatqa,\
flare_edtsum,\
flare_ectsum,\
flare_german,\
flare_australian,\
flare_cra_lendingclub,\
flare_cra_ccf,\
flare_cra_ccfraud,\
flare_cra_polish,\
flare_cra_taiwan,\
flare_cra_portoseguro,\
flare_cra_travelinsurace,\
flare_sm_bigdata,\
flare_sm_acl,\
flare_sm_cikm,\
flare_en_finterm"

python src/eval.py \
    --model hf-causal-vllm \
    --tasks "$TASKS" \
    --model_args use_accelerate=True,pretrained=google/gemma-3-1b-it,tokenizer=google/gemma-3-1b-it,use_fast=False,max_gen_toks=1024,dtype=float16,gpu_memory_utilization=0.8 \
    --no_cache \
    --batch_size 200000 \
    --model_prompt 'finma_prompt' \
    --num_fewshot 0 \
    --write_out \
    --output_path ./results/evaluation_results.json