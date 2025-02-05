import torch
import torch.nn as nn

class MaskedIonmobLoss(nn.Module):
    def __init__(self, use_mse: bool = True):
        """
        This loss can be used for the training of an ion-mobility predictor that
        outputs both expected ion mean and ion standard deviation
        Args:
            use_mse: if false, MAE will be used instead of MSE (ionmob-style)
        """

        super(MaskedIonmobLoss, self).__init__()

        if use_mse:
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.L1Loss()

    def forward(self, outputs, targets):
        """
        Ionmob mobel has three outputs, CCS, deep residues, CCS-std
        Args:
            outputs: Tuple[CCS, deep_residues, CCS_STD]
            targets: Tuple[CCS, _, CCS_STD]

        Returns:
            Combined loss of predicted CCS and masked CCS STD since STD is
            not available in the current training data for all the datapoints
        """
        total_output, _, ccs_std_output = outputs
        target_total, _, target_ccs_std = targets

        # compute the loss on the CCS
        loss_total = self.loss_function(total_output, target_total)

        # compute the loss on the CCS-std
        mask = target_ccs_std != -1
        # at least one value in the batch for CCS-STD was non-zero
        if mask.sum() > 0:
            loss_ccs_std = self.loss_function(ccs_std_output[mask], target_ccs_std[mask])
        else:
            loss_ccs_std = torch.tensor(0.0, device=total_output.device)

        # Final loss (sum or weighted combination)
        loss = loss_total + loss_ccs_std
        return loss
