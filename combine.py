import pandas as pd


predicted_files = [
    "outputs/runtime_estimation_operator-level-benchmark_H100.csv",
    "outputs/runtime_estimation_operator-level-cost-model_H100_CFG1.csv",
    "outputs/runtime_estimation_operator-level-cost-model_H100_CFG2.csv",
    "outputs/runtime_estimation_operator-level-learned-model_H100_CFG2.csv",
]

models = ["timm_vit", "gemma_2b", "hf_clip", "llama_v3_1b", "hf_T5"]
res = []
for file in predicted_files:
    print(file)
    predicted_df = pd.read_csv(file, header=None)
    actual_df = pd.read_csv("outputs/real_execution_H100.csv", header=None)

    predicted_df.columns = [
        "model_name", "batch_size", "seq_length", "image_size", "precision", 
        "activation_checkpointing", "estimation_type", "predicted_time", "time_for_estimation"
    ]

    actual_df.columns = [
        "model_name", "batch_size", "seq_length", "image_size", "precision", 
        "activation_checkpointing", "actual_time", "memory1", "memory2"
    ]

    actual_df = actual_df.drop(columns=["memory1", "memory2"])

    combined_df = pd.merge(
        predicted_df, actual_df, 
        on=["model_name", "batch_size", "seq_length", "image_size", "precision", "activation_checkpointing"],
        how="left"
    )


    combined_df["accuracy"] = combined_df["actual_time"] / combined_df["predicted_time"]
    
    filtered_df = combined_df[combined_df["model_name"].isin(models)]
    result = filtered_df.groupby(["estimation_type"])["accuracy"].describe()
    print(result)
    
    # for model in models:
    #     print(model)
    #     print(combined_df.query(f"model_name == '{model}'").groupby("estimation_type")["accuracy"].describe())
    
    
    res.append(combined_df)



# model = ["timm_vit", "gemma_2b", "hf_clip", "llama_v3_1b", "hf_T5"][2]
# combined_df = pd.concat(res, axis=0)

# combined_df['estimation_type'] = combined_df['estimation_type'].replace({
#     'operator-level-benchmark': 'Benchmark',
#     'operator-level-cost-model': 'Cost Model',
#     'operator-level-learned-model': 'Learned'
# })
# print(combined_df.query(f"model_name == '{model}'"))
# # combined_df = combined_df.query(f"model_name == '{model}' and batch_size == 32 and precision == 'FP' and activation_checkpointing").drop(columns=["model_name"])
# combined_df = combined_df.query(f"model_name == '{model}' and batch_size == 32 and precision == 'FP'").drop(columns=["model_name"])
# print(combined_df)
# combined_df.to_csv(f'outputs/{model}.csv', index=False, float_format='%.2f')

