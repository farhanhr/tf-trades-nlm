config = {
    "epochs": 100,
    "batch_size": 32,
    "momentum": 0.9,
    "step_size": 1, #pixels
    "epsilon": 4,  #pixels
    "perturb_steps": 10,
    "beta": 5.0,
    "val_subset_size": 512,
    "dataset_path": "data/brain_tumor_dataset",
    "checkpoint_dir": "models/saved_models/checkpoints/",
    "final_model_dir": "models/saved_models/final/",
}