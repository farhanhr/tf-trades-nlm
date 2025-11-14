import tensorflow as tf
from data.data_loader import get_testing_data
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from inference.evaluate import model_classification_report
from data.preprocessors import compose_preprocessors, nlm_denoise_bpda, bilateral_denoise_tf
from inference.evaluate_ART import evaluate, classification_report_art, batch_evaluation
from inference.evaluate_cleverhans import batch_evaluation_cleverhans_attacks
from configs.train_config import config

def main():
    test_gen = get_testing_data(config["dataset_path"], batch_size=config["batch_size"])
    # preprocess_fn = compose_preprocessors([
    # # bilateral_denoise_tf(spatial_sigma=1.0, intensity_sigma=25.0, kernel_size=3),
    # # nlm_denoise_bpda(h=0.3, patch_size=3, window_size=5),
    # vgg_preprocess
    # ])
    # model_name = "BrainTumorMRI_VGG16_03112025_2358_model.h5" 
    # # # results = model_classification_report(model_name, test_gen, preprocess_fn=vgg_wrapper)
    # # evaluate(model_name, test_gen, epsilon=(2.55), step_size=0.51, preprocess_fn=preprocess_fn)
    # classification_report_art(model_name, test_gen, epsilon=(2.55), step_size=0.51, preprocess_fn=preprocess_fn)
    
    
    base_vgg_fn = compose_preprocessors([
    vgg_preprocess
    ])
    trades_only_vgg_fn = compose_preprocessors([
    vgg_preprocess
    ])
    nlm_fn = compose_preprocessors([
    nlm_denoise_bpda(h=0.3, patch_size=3, window_size=5),
    vgg_preprocess
    ])
    bf_fn = compose_preprocessors([
    bilateral_denoise_tf(spatial_sigma=1.0, intensity_sigma=25.0, kernel_size=3),
    vgg_preprocess
    ])
    bf_nlm_fn = compose_preprocessors([
    bilateral_denoise_tf(spatial_sigma=1.0, intensity_sigma=25.0, kernel_size=3),
    nlm_denoise_bpda(h=0.3, patch_size=3, window_size=5),
    vgg_preprocess
    ])
    
    base_model = "BrainTumorMRI_VGG16_03112025_2358_model.h5"
    trades_only = "checkpoints/11112025_063443/[Best_checkpoint]TRADES_Only_B(5)EPS(2.55)LR(0.0001)_model_11112025_063443.h5"
    trades_nlm = "checkpoints/13112025_065531/[Best_checkpoint]TRADES_NLM_B(5)EPS(2.55)LR(0.0001)_model_13112025_065531.h5"
    trades_bf = "checkpoints/12112025_164125/[Best_checkpoint]TRADES_BF_B(5)EPS(2.55)LR(0.0001)_model_12112025_164125.h5"
    trades_bf_nlm = "checkpoints/13112025_135537/[Best_checkpoint]TRADES_NLM_BF_B(5)EPS(2.55)LR(0.0001)_model_13112025_135537.h5"
    
    batch_model = [base_model, trades_only, trades_nlm, trades_bf, trades_bf_nlm]
    batch_preps = [base_vgg_fn, trades_only_vgg_fn, nlm_fn, bf_fn, bf_nlm_fn]
    
    batch_evaluation(
        model_name=batch_model,
        test_gen=test_gen,
        epsilon=[2.55, 4.0, 8.0],
        step_size=[0.64, 1.0, 2.0],
        preprocess_fn=batch_preps,
        out_dir="inference/result/"
    )
    batch_evaluation_cleverhans_attacks(
        model_name=batch_model,
        test_gen=test_gen,
        epsilon=[2.55, 4.0, 8.0],
        step_size=[0.64, 1.0, 2.0],
        preprocess_fn=batch_preps,
        out_dir="inference/result/cleverhans"
    )

    print("Done evaluate_art")

if __name__ == "__main__":
    main()
