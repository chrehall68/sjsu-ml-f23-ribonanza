# imports
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import os
from constants import NUM_BPP, NUM_REACTIVITIES
from typing import Tuple

# used for better attention mechanisms
import xformers.components.attention as attentions
import xformers.components.attention.utils as att_utils
import xformers.components as components


# if no gpu available, use cpu. if on macos>=13.0, use mps
DEVICE = "cpu"

if torch.backends.mps.is_built():
    DEVICE = "mps"
elif torch.backends.cuda.is_built():
    DEVICE = "cuda"

DEVICE = torch.device(DEVICE)
print(DEVICE)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attention: components.Attention,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        *args,
        **kwargs,
    ) -> None:
        super(CustomTransformerEncoderLayer, self).__init__()
        self.attention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=attention,
            use_rotary_embeddings=True,
            **kwargs,
        )
        self.l1 = nn.LayerNorm(latent_dim)

        self.gelu = nn.GELU()
        self.ff1 = nn.Linear(latent_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, latent_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        # MHA self attention, add, norm
        x = self.l1(self.attention(x, att_mask=attention_mask) + x)

        # ff, add, norm
        x = self.l1(self.gelu(self.ff2(self.gelu(self.ff1(x)))) + x)

        return x


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attention: components.Attention,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        *args,
        **kwargs,
    ) -> None:
        super(CustomTransformerDecoderLayer, self).__init__()
        self.crossattention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=attention,
            use_rotary_embeddings=True,
            **kwargs,
        )

        self.selfattention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=attention,
            use_rotary_embeddings=True,
            **kwargs,
        )
        self.l1 = nn.LayerNorm(latent_dim)

        self.gelu = nn.GELU()
        self.ff1 = nn.Linear(latent_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, latent_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, attention_mask: torch.Tensor):
        # MHA self attention, add norm
        x = self.l1(self.selfattention(x, att_mask=attention_mask) + x)

        # MHA cross attention, add, norm
        x = self.l1(
            self.crossattention(key=ctx, query=ctx, value=x, att_mask=attention_mask)
            + x
        )

        # ff, add, norm
        x = self.l1(self.gelu(self.ff2(self.gelu(self.ff1(x)))) + x)

        return x


class CustomTransformerEncoder(nn.Module):
    def __init__(
        self,
        attention_type: components.Attention,
        n_layers: int,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        **kwargs,
    ) -> None:
        super(CustomTransformerEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(
                str(i),
                CustomTransformerEncoderLayer(
                    attention=attention_type,
                    latent_dim=latent_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    **kwargs,
                ),
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        for module in self._modules.values():
            x = module(x, attention_mask=attention_mask)
        return x


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self,
        attention_type: components.Attention,
        n_layers: int,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        **kwargs,
    ) -> None:
        super(CustomTransformerDecoder, self).__init__()
        for i in range(n_layers):
            self.add_module(
                str(i),
                CustomTransformerDecoderLayer(
                    attention=attention_type,
                    latent_dim=latent_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    **kwargs,
                ),
            )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, attention_mask: torch.Tensor):
        for module in self._modules.values():
            x = module(x, ctx, attention_mask=attention_mask)
        return x


class AttentionModel(nn.Module):
    def __init__(
        self,
        attention_type: attentions.Attention = attentions.ScaledDotProduct(dropout=0.1),
        latent_dim: int = 128,
        ff_dim: int = 1024,
        n_heads: int = 2,
        enc_layers: int = 1,
        dec_layers: int = 1,
    ) -> None:
        super(AttentionModel, self).__init__()

        # data
        self.n_heads = n_heads
        self.latent_dim = latent_dim

        # input is bpp and one-hot encoded bases
        self.proj = nn.Linear(NUM_BPP + 4, latent_dim)

        # positional embedding encoder/decoder layers
        self.has_encoder = enc_layers >= 1
        self.has_decoder = dec_layers >= 1
        if self.has_encoder:
            self.encoder_layers = CustomTransformerEncoder(
                latent_dim=latent_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                attention_type=attention_type,
                n_layers=enc_layers,
            )
        if self.has_decoder:
            self.decoder_layers = CustomTransformerDecoder(
                latent_dim=latent_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                attention_type=attention_type,
                n_layers=dec_layers,
            )

        # output head
        # dms, 2a3
        self.head = nn.Linear(latent_dim, 2)

        # activations
        self.gelu = nn.GELU()

        self.register_buffer(
            "oh",
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        )

    def forward(self, tokens: torch.Tensor, bpp: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            - tokens: torch.Tensor - should have shape B,457
            - bpp: torch.Tensor - should have shape B,457,NUM_BPP
        """
        mask = att_utils.maybe_merge_masks(
            att_mask=None,
            key_padding_mask=tokens != 0,
            batch_size=tokens.shape[0],
            num_heads=self.n_heads,
            src_len=NUM_REACTIVITIES,
        )

        # project inputs and bpp to latent_dim
        x = self.proj(
            torch.concat([self.oh[tokens.to(torch.int)].to(bpp.dtype), bpp], dim=-1)
        )

        # add sinusoidal embedding and then perform attention
        if self.has_decoder and self.has_encoder:
            x = self.decoder_layers(
                x, ctx=self.encoder_layers(x, attention_mask=mask), attention_mask=mask
            )
        elif self.has_encoder:
            x = self.encoder_layers(x, attention_mask=mask)
        elif self.has_decoder:
            x = self.decoder_layers(x, ctx=x, attention_mask=mask)

        # final result
        x = self.gelu(self.head(x))
        return x


def unweightedL1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    l1=nn.L1Loss(reduction="none"),
) -> torch.Tensor:
    """
    MAE Loss function where sample weights are only used to determine masks.

    Arguments:
        - y_pred: torch.Tensor[B, NUM_REACTIVITIES, 2] - the predicted reactivities
        - y_true: torch.Tensor[B, NUM_REACTIVITIES, 2] - the true reactivities

    Returns:
        - torch.Tensor - loss for reactivity predictions
    """
    return (l1(y_pred, y_true))[weights != 0].mean()


def weightedL1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    l1=nn.L1Loss(reduction="none"),
) -> torch.Tensor:
    """
    MAE loss function that takes into account sample weights

    Arguments:
        - y_pred: torch.Tensor[B, NUM_REACTIVITIES, 2] - the predicted reactivities
        - y_true: torch.Tensor[B, NUM_REACTIVITIES, 2] - the true reactivities

    Returns:
        - torch.Tensor - loss for reactivity predictions
    """
    return (l1(y_pred, y_true) * weights)[weights != 0].mean()


def train_batch(
    m: nn.Module,
    tokens: torch.Tensor,
    bpp: torch.Tensor,
    outs: torch.Tensor,
    masks: torch.Tensor,
    m_optim: torch.optim.Optimizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the loss on a batch and perform the corresponding weight updates.
    Used for training purposes

    Returns:
        - Tuple[torch.Tensor, torch.Tensor] - weighted_loss, unweighted_loss
    """
    m_optim.zero_grad()
    preds = m(tokens, bpp)

    # get the weighted mae
    weighted_loss = weightedL1(preds, outs, masks)
    weighted_loss.backward()

    # calculate gradients
    m_optim.step()

    with torch.no_grad():
        unweighted_loss = unweightedL1(preds, outs, masks)

    # return weighted and unweighted mae loss
    return (
        weighted_loss.detach().cpu(),
        unweighted_loss.detach().cpu(),
    )


def noupdate_batch(
    m: nn.Module,
    tokens: torch.Tensor,
    bpp: torch.Tensor,
    outs: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the loss on a batch without performing any updates.
    Used for validation purposes

    Returns:
        - Tuple[torch.Tensor, torch.Tensor] - weighted_loss, unweighted_loss
    """
    with torch.no_grad():
        preds = m(tokens, bpp)
        weighted_loss = weightedL1(preds, outs, masks)
        unweighted_loss = unweightedL1(preds, outs, masks)

    # return weighted and unweighted mae loss
    return (weighted_loss.cpu(), unweighted_loss.cpu())


def masked_train(
    m: nn.Module,
    m_optim: torch.optim.Optimizer,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    writer: SummaryWriter,
    model_name: str,
    epochs: int = 1,
    device: str = DEVICE,
):
    """
    Train the given model.

    Arguments:
        - m: nn.Module - the model to train.
        - m_optim: torch.optim.Optimizer - the optimizer to use for the model
        - train_dataloader: data.Dataloader - the dataloader that provides the batched training data
        - val_dataloader: data.Dataloader - the dataloader that provides the batched validation data
        - writer: SummaryWriter - the logger to use
        - model_name: str - the name to save the model as
        - epochs: int - how many epochs to train for. Defaults to `1`.
        - device: str - the device to train on, defaults to `DEVICE`
    """
    m = m.to(device)

    best_val_mae = 100.0  # arbitrary large number
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        epoch_mae = 0.0
        epoch_weighted_mae = 0.0

        m = m.train()
        for tdata in (prog := tqdm(train_dataloader, desc="batch")):
            tokens = tdata["simple_tokens"]
            bpp = tdata["bpp"]
            outs = tdata["outputs"]
            masks = tdata["output_masks"]

            tokens = tokens.to(device)
            bpp = bpp.to(device)
            outs = outs.to(device)
            masks = masks.to(device)

            weighted_mae, mae = train_batch(m, tokens, bpp, outs, masks, m_optim)

            epoch_weighted_mae += weighted_mae
            epoch_mae += mae

            # log
            prog.set_postfix_str(f"MAE: {mae:.5f}, WMAE: {weighted_mae:.5f}")

            # break  # used for sanity check
        epoch_weighted_mae /= len(train_dataloader)
        epoch_mae /= len(train_dataloader)

        # do validation
        val_mae = 0.0
        val_weighted_mae = 0.0
        m = m.eval()
        for vdata in val_dataloader:
            tokens = vdata["simple_tokens"]
            bpp = vdata["bpp"]
            outs = vdata["outputs"]
            masks = vdata["output_masks"]

            tokens = tokens.to(device)
            bpp = bpp.to(device)
            outs = outs.to(device)
            masks = masks.to(device)
            weighted_mae, mae = noupdate_batch(m, tokens, bpp, outs, masks)

            val_weighted_mae += weighted_mae
            val_mae += mae
        val_weighted_mae /= len(val_dataloader)
        val_mae /= len(val_dataloader)

        print(
            f"Epoch MAE: {epoch_mae:.5f}\tEpoch WMAE: {epoch_weighted_mae:.5f}\t"
            + f"Val MAE: {val_mae:.5f}\tVal WMAE: {val_weighted_mae:.5f}\t"
        )

        writer.add_scalar("epoch_mae", epoch_mae, global_step=epoch)
        writer.add_scalar("val_mae", val_mae, global_step=epoch)

        # save every epoch
        torch.save(m.state_dict(), f"{model_name}_model.pt")

        # save the best model as well
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(m.state_dict(), "best.pt")


def train(
    run_name: str,
    dataset_name: str,
    lr: float = 1e-4,
    batch_size: int = 32,
    val_split: float = 0.1,
    epochs: int = 10,
    model_dict: dict = dict(
        latent_dim=32,
        n_heads=1,
        enc_layers=4,
        dec_layers=4,
        ff_dim=2048,
    ),
):
    """
    Train a model from start to finish, taking care of data loading,
    optimizers, etc

    Arguments:
        - run_name: str - the name of the run to log as
        - dataset_name: str - the name of the dataset, either "2a3" or "dms"
        - lr: float - the learning rate to use. Defaults to 1e-4
        - batch_size: int - the batch size to use when training and running validation. Defaults to 64
        - val_split: float - the size of the validation, from 0 to 1. Defaults to 0.1
        - epochs: int - the number of epochs to train for. Defaults to 10
        - model_dict: dict - a dictionary containing all the arguments to be passed when instantiating
            the `AttentionModel`
    """
    # set seed for reproducibility
    torch.manual_seed(2023)

    # load and process dataset
    columns = ["simple_tokens", "outputs", "output_masks", "bpp"]

    dataset = Dataset.load_from_disk(
        f"train_data_{dataset_name}_preprocessed"
    ).with_format("torch")
    split = dataset.train_test_split(test_size=val_split).select_columns(columns)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(dataset)
    print(
        f"{dataset_name} - train set is len",
        len(train_dataset),
        "and val dataset is len",
        len(val_dataset),
    )

    # load into batches
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # create logger
    writer = SummaryWriter(f"runs/{run_name}")

    # create model + optimizer
    model = AttentionModel(**model_dict).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # load old weights if possible
    if os.path.exists(f"{run_name}_model.pt"):
        try:
            model.load_state_dict(torch.load(f"{run_name}_model.pt"))
            print(f"loaded previous {run_name} weights")
        except Exception as e:
            print(f"not loading previous {run_name} weights because", e)
            pass

    # log # of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total {run_name} model params:", params)

    # train
    masked_train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        writer=writer,
        model_name=run_name,
        epochs=epochs,
    )
