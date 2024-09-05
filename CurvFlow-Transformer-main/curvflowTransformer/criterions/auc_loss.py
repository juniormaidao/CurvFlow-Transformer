from sklearn.metrics import roc_auc_score
from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion

# ... （之前的代码）
from graphormer.criterions.binary_logloss import GraphPredictionBinaryLogLoss


@register_criterion("auc_loss", dataclass=FairseqDataclass)
class GraphPredictionAUCLoss(GraphPredictionBinaryLogLoss):
    """
    Implementation for computing AUC as the evaluation metric.
    """

    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])

        # Compute probability scores
        prob_scores = torch.sigmoid(logits).cpu().detach().numpy()
        targets = targets[: logits.size(0)].cpu().detach().numpy()

        # Compute AUC
        auc = roc_auc_score(targets, prob_scores)

        logging_output = {
            "auc": auc,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
            # You can add other logging information if needed
        }
        return auc, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        auc_sum = sum(log.get("auc", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("auc", auc_sum / sample_size, sample_size, round=3)
        # Add other metrics if needed

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True