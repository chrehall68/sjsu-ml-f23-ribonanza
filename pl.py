from datasets import load_from_disk, concatenate_datasets, Dataset, Array2D
import torch
import torch.utils.data as data
from constants import NUM_REACTIVITIES, NUM_BPP
from models import AttentionModel, DEVICE, train_batch
from tqdm import tqdm


def pl(
    model: torch.nn.Module, batch_size: int, error_interval: float, device: torch.device
):
    """
    Pseudo-label test data (generate model predictions for each item in the test set)

    Note: the model pseudo-labeling the data should be pretty good,
    otherwise pseudo-labeling will only add noise to the training process
    """
    # columns to keep
    columns = ["simple_tokens", "bpp", "outputs", "output_masks"]

    def pred(rows):
        """
        Run inference on a batch, and store the outputs
        """
        tokens = rows["simple_tokens"].to(device)
        bpp = rows["bpp"].to(device)

        weights = torch.zeros((tokens.shape[0], NUM_REACTIVITIES, 2))
        weights[tokens != 0] = 1

        with torch.no_grad():
            preds = model(tokens, bpp).cpu()

        rows["outputs"] = preds
        rows["output_masks"] = weights

        return rows

    model = model.eval().to(device)
    ds = load_from_disk("test_data_preprocessed").with_format("torch")
    ds = (
        ds.map(pred, batch_size=batch_size, batched=True, load_from_cache_file=False)
        .cast_column("outputs", Array2D((NUM_REACTIVITIES, 2), "float32"))
        .cast_column("output_masks", Array2D((NUM_REACTIVITIES, 2), "float32"))
        .cast_column("bpp", Array2D((NUM_REACTIVITIES, NUM_BPP), "float32"))
        .select_columns(columns)
    )

    other_ds = (
        load_from_disk("train_data_full_preprocessed")
        .with_format("torch")
        .select_columns(columns)
    )

    ds: Dataset = concatenate_datasets([ds, other_ds]).with_format("torch")
    return ds


def train_ds(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    ds: data.Dataset,
    batch_size: int,
    device: torch.device,
):
    """
    Train the model on an entire dataset

    Arguments:
        - model: torch.nn.Module - the model to train
        - optim: torch.optim.Optimizer - the optimizer to use for the model
        - ds: data.Dataset - The dataset to use
        - batch_size: int - the batch size to use
        - device: torch.device - the device to train on
    """
    model = model.train().to(device)

    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    epoch_mae = 0
    epoch_weighted_mae = 0

    # iterate through dl
    for batch in (prog := tqdm(dl)):
        tokens: torch.Tensor = batch["simple_tokens"]
        bpp: torch.Tensor = batch["bpp"]
        outs: torch.Tensor = batch["outputs"]
        masks: torch.Tensor = batch["output_masks"]

        tokens = tokens.to(device)
        bpp = bpp.to(device)
        outs = outs.to(device)
        masks = masks.to(device)

        weighted_mae, mae = train_batch(model, tokens, bpp, outs, masks, optim)

        epoch_weighted_mae += weighted_mae
        epoch_mae += mae
        prog.set_postfix_str(f"mae: {mae:.5f}, weighted_mae: {weighted_mae:.5f}")

    epoch_weighted_mae /= len(dl)
    epoch_mae /= len(dl)
    print("mae", epoch_mae, "weighted mae:", epoch_weighted_mae)


def ssl(
    name: str,
    lr: float = 1e-4,
    error_interval: float = 0.15,
    train_batch_size: int = 32,
    label_batch_size: int = 64,
    epochs: int = 5,
    sub_epochs: int = 2,
    model_dict: dict = dict(
        latent_dim=32,
        n_heads=1,
        enc_layers=4,
        dec_layers=4,
        ff_dim=2048,
    ),
):
    """
    Semi-Supervised Learning. Generates predictions on the test set,
    then uses those predictions as train data (along with actual train data)

    Arguments:
        - name: str - the name of the run and of the model to load
        - lr: float - the learning rate to use. Defaults to `1e-4`
        - error_interval: float - the maximum predicted error to allow for pseudolabels
            to be treated as real labels. Defaults to 0.15
        - train_batch_size: int - batch size for training (train mode). Defaults to `32`
        - label_batch_size: int - batch size for labeling (inference mode). Defaults to `64`
        - epochs: int - number of epochs to train for. One epoch is one complete pass on the
            test set concatenated with the train set and one complete pass on just the train set.
            Defaults to `5`.
        - sub_epochs: int - number of repetitions to do on each dataset in one epoch. Defaults to `2`
        - model_dict: dict - a dictionary containing all the arguments to be passed when instantiating
            the `AttentionModel`
    """
    # set seed for reproducibility
    torch.manual_seed(2023)

    # make model
    model = AttentionModel(**model_dict)
    model.load_state_dict(torch.load(f"{name}_model.pt"))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # pl process
    for i in range(epochs):
        # make preds
        ds = pl(model, label_batch_size, error_interval, DEVICE)

        # train on pseudo-labeled data
        print("pl epoch", i + 1)
        for sub in range(sub_epochs):
            print("sub epoch", sub + 1)
            train_ds(model, optim, ds, train_batch_size, DEVICE)
        print("cleaning up", ds.cleanup_cache_files(), "cache files")
        torch.save(model.state_dict(), f"{name}_model.pt")

        # train on regular data
        print("regular epoch", i + 1)
        ds = load_from_disk("train_data_full_preprocessed").with_format("torch")
        for sub in range(sub_epochs):
            print("sub epoch", sub + 1)
            train_ds(model, optim, ds, train_batch_size, DEVICE)
        torch.save(model.state_dict(), f"{name}_model.pt")
