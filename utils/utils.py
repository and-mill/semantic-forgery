import random

import torch
import numpy as np

import PIL.Image


# Gaussian Shading FPR=10-6
# caluclated using get_GS_thresholds, but you need mp math and it is not easy to install via pip
# This is the threshold for the multi-bit setting with 100k users
GS_THRESHOLDS = 0.70703125

# Tree-Ring FPR=1%
# empricially measured with 5000 watermarked vs 5000 non-watermarked images on each model
TR_THRESHOLD_SDXL = 0.0261008646426459
TR_THRESHOLD_PIXART = 0.015917123872865434
TR_THRESHOLD_FLUX = 0.02185009852518841
TR_THRESHOLD_FOR_MODEL = {
    "stabilityai/stable-diffusion-xl-base-1.0": TR_THRESHOLD_SDXL,
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": TR_THRESHOLD_PIXART,
    "black-forest-labs/FLUX.1-dev": TR_THRESHOLD_FLUX
}


def get_detection_threshold(wm_type: str,
                            model_name: str = None) -> float:
    """
    Get the detection threshold for the given model

    @param wm_type: str
    @param model_name: str

    @return: float
    """
    if wm_type == "GS":
        return GS_THRESHOLDS
    elif wm_type:
        assert model_name, "Model name must be provided for TR"
        return TR_THRESHOLD_FOR_MODEL.get(model_name, TR_THRESHOLD_SDXL)
    else:
        raise ValueError("Unknown watermark type")
    

def check_if_detection_successful(wm_type: str, threshold: float, value: float) -> bool:
    """
    Check if the detection is successful

    @param wm_type: str
    @param threshold: float
    @param value: float

    @return: bool
    """
    if wm_type == "GS":
        return value > threshold
    elif wm_type == "TR":
        return value < threshold
    else:
        raise ValueError("Unknown watermark type")


def set_random_seed(seed=0):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


##############################################################################################################################################################################################

def get_GS_thresholds(num_bits=256, NUM_USERS=100*1000):
    """
    This implements the calcualtion of theoretical thresholds for different FPRS in Gaussian Shading as per the instructions in their appendix.
    You do not need it though, as the threshold is provided above for our used settings.

    Need to install mpmath. Did not manage to do it via pip.
    """
    INTERESTING_FPRs_GS = [10**-i for i in [6, 16, 32, 64]]
    from scipy.special import betainc
    import mpmath as mp
    mp.mp.dps = 100


    # Number of bits in the WM message
    #K = 256
    K = num_bits
    N = NUM_USERS
    
    def beta_func(τ, k=K):
        a = τ + 1
        b = k - τ
        return betainc(a, b, 0.5)

    # Generating thresholds and their corresponding single-user FPR
    thresholds = range(K//2, K + 1)
    single_user_FPRs = [beta_func(τ) for τ in thresholds]

    # Calculating multi-user FPR
    single_user_FPRs = np.array(single_user_FPRs)  # Ensure it's a numpy array if not already

    # Convert single_user_FPRs to mpmath floats for higher precision
    single_user_FPRs_mp = [mp.mpf(fpr) for fpr in single_user_FPRs]

    # Compute the result with high precision
    multi_user_FPRs = [1 - mp.exp(-N * fpr) for fpr in single_user_FPRs_mp]
    
    
    def find_first_index_below_threshold(values, a):
        return next((i for i, x in enumerate(values) if x < a), None)

    # zero-bit / detection-only scenario
    FPRs_GS = [beta_func(τ) for τ in thresholds]
    THRESHOLD_INDEXES_FOR_FPRs_GS = {fpri: find_first_index_below_threshold(FPRs_GS, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS.items()}
    
    print("single:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")
        
    # multi-bit / detection & attribution scenario
    FPRs_GS_MULTI = multi_user_FPRs
    THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI = {fpri: find_first_index_below_threshold(FPRs_GS_MULTI, fpri) for fpri in INTERESTING_FPRs_GS}
    THRESHOLD_FOR_FPRs_GS_MULTI = {fpri: thresholds[index] for fpri, index in THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI.items()}
    THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI = {fpri: thres / K for fpri, thres in THRESHOLD_FOR_FPRs_GS_MULTI.items()}
    print("multi:")
    for fpri, thres in THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI.items():
        print(f"\tThreshold for fpri {fpri:<10}:{thres:<30}")

    return {"INTERESTING_FPRs_GS": INTERESTING_FPRs_GS,
            "THRESHOLD_INDEXES_FOR_FPRs_GS": THRESHOLD_INDEXES_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FOR_FPRs_GS": THRESHOLD_FOR_FPRs_GS_MULTI,
            "THRESHOLD_FLOAT_FOR_FPRs_GS": THRESHOLD_FLOAT_FOR_FPRs_GS_MULTI}


if __name__ == "__main__":
    print(get_GS_thresholds())
