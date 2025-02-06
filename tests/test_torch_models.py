import logging

from dlomix.models import DominantChargeStatePredictorTorch

logger = logging.getLogger(__name__)


def test_dominant_chargestate_model_torch():
    model = DominantChargeStatePredictorTorch()
    logger.info(model)
    assert model is not None

    # logger.info()

    # check model.parameters not empty --> nn.Module.parameters()


# to compare tf & torch: shapes at beginnin & end of forward & in between the layers - compare

# test in run script: simple data --> fit & compare outputs
