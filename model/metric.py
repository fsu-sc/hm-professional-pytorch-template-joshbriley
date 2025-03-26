import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_squared_error(output, target):
    """
    Computes the Mean Squared Error (MSE) between the model output and the target.
    """
    with torch.no_grad():
        mse = torch.mean((output - target) ** 2).item()
    return mse


def mean_absolute_error(output, target):
    """
    Computes the Mean Absolute Error (MAE) between the model output and the target.
    """
    with torch.no_grad():
        mae = torch.mean(torch.abs(output - target)).item()
    return mae


def training_loss(loss_fn, output, target):
    """
    Computes the training loss using the provided loss function.
    """
    with torch.no_grad():
        loss = loss_fn(output, target).item()
    return loss


def validation_loss(loss_fn, output, target):
    """
    Computes the validation loss using the provided loss function.
    """
    with torch.no_grad():
        loss = loss_fn(output, target).item()
    return loss
