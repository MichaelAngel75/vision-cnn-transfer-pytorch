import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss, scheduler=None):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        # DONE: YOUR CODE HERE: transfer the model to the GPU
        # HINT: use .cuda()
        model = model.cuda()

    # DONE: YOUR CODE HERE: set the model to training mode
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        # DONE:  YOUR CODE HERE:
        optimizer.zero_grad()
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data)  # DONE: YOUR CODE HERE
        # 3. calculate the loss
        loss_value  =  loss(output, target)  # DONE:  YOUR CODE HERE
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        # DONE:  YOUR CODE HERE:
        loss_value.backward()
        # 5. perform a single optimization step (parameter update)
        # DONE:  YOUR CODE HERE:
        optimizer.step()

        # 🔁 STEP THE LR SCHEDULER — if batch-based
        if scheduler and not isinstance(isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
            scheduler.step()

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # set the model to evaluation mode
        # DONE:  YOUR CODE HERE
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data) # DONE:  YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = loss(output, target)  # DONE:  YOUR CODE HERE

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here: 
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    # scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2) # DONE:  YOUR CODE HERE
    # Determine scheduler based on optimizer type
    scheduler = None
    if isinstance(optimizer, torch.optim.SGD):
    # ---- scheduler learning for sgd -------
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=4,
            verbose=True,
            min_lr=1e-5
        )
    elif isinstance(optimizer, torch.optim.Adam):
        # --- the below to be used for Adam optimizer ---
        scheduler  = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,  #  from 0.001 - 0.01
            steps_per_epochs=len(data_loaders["train"]),
            epochs=n_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
    else:
        raise ValueError("Unsupported optimizer type for scheduler setup.")    

    for epoch in range(1, n_epochs + 1):
        if isinstance(optimizer, torch.optim.SGD):
            train_loss = train_one_epoch(
                data_loaders["train"], model, optimizer, loss, scheduler=None
            )
        elif isinstance(optimizer, torch.optim.Adam):
            train_loss = train_one_epoch(
                # --- only for adam optimizer since that should be at the level of the batch 
                data_loaders["train"], model, optimizer, loss, scheduler=scheduler
            )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            # DONE:  YOUR CODE HERE
            # --------------------------------------------------------------------------------------------
            # torch.save(model.state_dict(), save_path)

            # --------------------------------------------------------------------------------------------
            # 🔧 Switch to TorchScript saving (create an example input for tracing)
            example_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)  # adjust shape as needed
            # 🔧 Convert to TorchScript using tracing
            traced_model = torch.jit.trace(model, example_input)
            # 🔧 Save the scripted model
            traced_model.save(save_path)

            # --------------------------------------------------------------------------------------------
            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        # DONE:  YOUR CODE HERE
        if isinstance(optimizer, torch.optim.SGD):
            scheduler.step(valid_loss)
        # -- Angel:  Print the learning rate how much is decreasing to avoid reaching a plateau
        print(optimizer.param_groups[0]["lr"])

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        # DONE: YOUR CODE HERE
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data)   # DONE:  YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = loss(logits, target)  # DONE:  YOUR CODE HERE

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred  = torch.argmax(logits, dim=1)  # DONE: YOUR CODE HERE

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
