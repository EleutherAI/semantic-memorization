# API - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
from scipy.stats import pearsonr as pearson_correlation

# API - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
from scipy.stats import spearmanr as spearman_correlation

# Paper - https://arxiv.org/abs/1909.10140
# API - https://swarnakumar.github.io/xicorpy/xi/
from xicorpy import compute_xi_correlation as xi_correlation

# Paper - https://arxiv.org/abs/1910.12327
# API - https://swarnakumar.github.io/xicorpy/foci/
from xicorpy import select_features_using_foci as feature_ordering_conditional_independence
