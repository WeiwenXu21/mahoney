from sklearn.utils.estimator_checks import check_estimator
from mahoney.nmf_skl import NMF_dcomp, NMF_extract

assert check_estimator(NMF_dcomp)
assert check_estimator(NMF_extract)
