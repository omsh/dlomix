import logging

from dlomix.models import ChargeStatePredictorTorch

logger = logging.getLogger(__name__)


# ------------------ CS | check for existence of model & its parameters ------------------


def basic_model_existence_test_torch(model):
    logger.info(model)
    assert model is not None

    assert len(list(model.parameters())) > 0


def test_dominant_chargestate_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="dominant")
    basic_model_existence_test_torch(model)

def test_observed_chargestate_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="observed")
    basic_model_existence_test_torch(model)

def test_chargestate_distribution_model_torch():
    model = ChargeStatePredictorTorch(model_flavour="relative")
    basic_model_existence_test_torch(model)


# ------------------ CS | comparison of tf & torch ------------------

def test_tf_torch_equivalence_chargestate_model_shapes():
    
# to compare tf & torch: shapes at beginnin & end of forward & in between the layers - compare

# test in run script: simple data --> fit & compare outputs
