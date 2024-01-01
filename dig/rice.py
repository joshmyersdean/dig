import torchmetrics

def rice(pred, init, gt):
    total = 0
    for i in range(pred.size(0)):
        a = torchmetrics.functional.jaccard_index(pred[i].unsqueeze(0).int(), gt[i].unsqueeze(0).int(),
                                                  task='binary', num_classes=3, ignore_index=255, average=None)
        b = torchmetrics.functional.jaccard_index(init[i].unsqueeze(0).int(), gt[i].unsqueeze(0).int(),
                                                  task='binary', num_classes=3, ignore_index=255, average=None)
        total += (a - b) / (1 - b) if a >= b else (a / b) - 1

    return (total / pred.size(0)).item()
