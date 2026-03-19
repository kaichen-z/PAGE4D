MAX_RETRIES=40
RETRY_COUNT=0

TRAINING_CMD="CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29509 /workspace/code/12_4d/PAGE4D_git/training/launch_gra.py --config training_final"

LOG_DIR="/workspace/code/12_4d/PAGE4D_git/logs"
mkdir -p "$LOG_DIR"
FULL_LOG="$LOG_DIR/training_final.log"

echo "Starting training (gradient checkpointing, max $MAX_RETRIES retries)" | tee -a "$FULL_LOG"
echo "Command: $TRAINING_CMD" | tee -a "$FULL_LOG"
echo "========================================" | tee -a "$FULL_LOG"

while [ $RETRY_COUNT -le $MAX_RETRIES ]; do
    echo "=== ATTEMPT $((RETRY_COUNT + 1)) AT $(date) ===" | tee -a "$FULL_LOG"
    eval $TRAINING_CMD 2>&1 | tee -a "$FULL_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    echo "=== ATTEMPT $((RETRY_COUNT + 1)) ENDED WITH EXIT CODE $EXIT_CODE AT $(date) ===" | tee -a "$FULL_LOG"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!" | tee -a "$FULL_LOG"
        break
    else
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "Training failed (exit code $EXIT_CODE). Restarting in 10 seconds..." | tee -a "$FULL_LOG"
            sleep 10
            RETRY_COUNT=$((RETRY_COUNT + 1))
        else
            echo "Training failed after $((MAX_RETRIES + 1)) attempts. Stopping." | tee -a "$FULL_LOG"
            break
        fi
    fi
done

echo "Script finished at $(date). Check full logs: $FULL_LOG" | tee -a "$FULL_LOG"
