#!/bin/bash
#SBATCH --job-name=qwen3-vl-xiaotongyang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4:00:00
#SBATCH --partition=a4xmaxpartition
#SBATCH --nodelist=infer3a4xm-a4xmaxnodeset-1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/home/xiaotongyang_google_com/%j-stdout.txt
#SBATCH --error=/home/xiaotongyang_google_com/%j-stderr.txt
#SBATCH --gres=gpu:4


CONTAINER_IMAGE="/home/xiaotongyang_google_com/nvidia+mlperf-inference-partner+nv-mlpinf-partner+v6.03-rc2-feb10-q3vl-aarch64+latest.sqsh"
HF_CACHE_HOST_DIR="/home/xiaotongyang_google_com/huggingface"

OUTPUT_HOST_DIR="/home/xiaotongyang_google_com/outputs/${SLURM_JOB_ID}/"
MODE=${MODE:-"performance_only"}
TARGET_QPS=${TARGET_QPS:-41.5}
HF_TOKEN=${HF_TOKEN:-""}

mkdir -p "${OUTPUT_HOST_DIR}"

# mounts="${HF_CACHE_HOST_DIR}:/root/.cache/huggingface,${OUTPUT_HOST_DIR}:/output/"
mounts="${HF_CACHE_HOST_DIR}:/root/.cache/huggingface,${OUTPUT_HOST_DIR}:/output/,/tmp/container_cache:/home/xiaotongyang_google_com/.cache"

export DYN_LOG=debug  # Change from 'info' to 'debug'
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024))
export TOKIO_WORKER_THREADS=32
export OMP_NUM_THREADS=64

export VLLM_USE_TRITON_POS_EMBED=1
export VLLM_MM_ENCODER_FP8_ATTN=1

echo "Starting job at: $(date)"
echo $SLURM_GPUS_ON_NODE

mkdir -p /tmp/container_cache

# NUMA binding: GPU 0-1 use NUMA node 0, GPU 2-3 use NUMA node 1
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${mounts}" \
    --no-container-mount-home \
    --mpi=pmix \
    bash -c " \
        export HF_HOME=/root/.cache/huggingface && \
        export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer && \
        export XDG_CACHE_HOME=/tmp/.cache && \
        export TRITON_CACHE_DIR=/tmp/.triton && \
        mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm \
        --use-http-client \
        --max-concurrency=2048 \
        --dynamo.vllm.enable_numa_binding=true \
        --dynamo.frontend.enable_numa_binding=true \
        --dynamo.model.repo_id=nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0 \
        --dynamo.model.revision=main \
        --settings.test.scenario server \
        --settings.test.server_target_qps ${TARGET_QPS} \
        --dynamo.num_warmup_requests_per_vllm_instance 100 \
        --settings.test.mode ${MODE} \
        --settings.test.qsl_rng_seed 2465351861681999779 \
        --settings.test.sample_index_rng_seed 14276810075590677512 \
        --settings.test.schedule_rng_seed 3936089224930324775 \
        --settings.logging.log_output.outdir /output/ \
        --dynamo.vllm.cli=--tensor-parallel-size=1 \
        --dynamo.vllm.cli=--pipeline-parallel-size=1 \
        --dynamo.vllm.cli=--data-parallel-size=1 \
        --dynamo.vllm.cli=--enable-expert-parallel \
        --dynamo.vllm.cli=--all2all-backend=flashinfer_all2allv \
        --dynamo.vllm.cli=--async-scheduling \
        --dynamo.vllm.cli=--max-model-len=32768 \
        --dynamo.vllm.cli=--max-num-seqs=1024 \
        --dynamo.vllm.cli=--mm-encoder-attn-backend=FLASHINFER \
        --dynamo.vllm.cli=--max-num-batched-tokens=8192 \
        --dynamo.vllm.cli=--scheduling-policy=sjf \
        --dynamo.vllm.cli=--compilation-config='{
            \"max_cudagraph_capture_size\": 8192,
            \"cudagraph_capture_sizes\": [
                1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
                256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
                496, 
                512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544, 4608, 4672, 4736, 4800, 4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312, 5376, 5440, 5504, 5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080, 6144, 6208, 6272, 6336, 6400, 6464, 6528, 6592, 6656, 6720, 6784, 6848, 6912, 6976, 7040, 7104, 7168, 7232, 7296, 7360, 7424, 7488, 7552, 7616, 7680, 7744, 7808, 7872, 7936, 8000, 8064, 8128,
                8192
            ]
        }' \
        --dynamo.vllm.cli=--override-generation-config='{\"max_new_tokens\": 150}' \
        --dynamo.vllm.cli=--limit-mm-per-prompt.video=0 \
        --dynamo.vllm.cli=--mm-processor-cache-gb=0 \
        --dynamo.vllm.cli=--no-enable-prefix-caching \
        --dynamo.vllm.cli=--enable-multimodal \
        --dynamo.vllm.cli=--connector=none \
        --dynamo.vllm.cli=--kv-events-config='{\"publisher\":\"null\"}' \
        --dynamo.vllm.cli=--distributed-executor-backend=mp; \
        EXIT_CODE=\$?; \
        if [ \$SLURM_LOCALID -eq 0 ]; then \
            if [ \$EXIT_CODE -eq 0 ]; then \
                if [ \"${MODE}\" == \"accuracy_only\" ]; then \
                    mlperf-inf-mm-q3vl evaluate --filename=/output/mlperf_log_accuracy.json; \
                    mv accuracy.txt /output/accuracy.txt; \
                fi; \
            else \
                echo \"Previous numactl command failed with exit code \$EXIT_CODE\"; \
                exit \$EXIT_CODE; \
            fi; \
        fi; \
    "