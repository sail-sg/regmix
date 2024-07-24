export OUT_FOLDER="eval_out"
# custom name for saving the output
export model_name="data-mixture-regmix-1b-seed-1"
export model_args="pretrained=sail/data-mixture-regmix-1b,revision=seed-1"
# task list
tasks=(
    'social_iqa'
    'hellaswag'
    'piqa'
    'openbookqa'
    'lambada_standard'
    'sciq'
    'arc_easy'
    'copa'
    'race'
    'logiqa'
    'qqp'
    'winogrande'
    'multirc'
)

for few_shot in 0 1 2 3 4 5; do
    for task in "${tasks[@]}"; do
        # print the task name
        echo "Evaluating task: $task"

        lm_eval --model hf \
            --model_args $model_args \
            --tasks $task \
            --batch_size auto:4 \
            --num_fewshot $few_shot \
            --output_path $OUT_FOLDER/$few_shot/$model_name/$task
    done
done
