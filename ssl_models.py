# imports
import random
from typing import Callable
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import os
from constants import NUM_BPP, NUM_REACTIVITIES

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


class CustomTransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        att_factory: Callable[[], components.Attention],
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        device: str = DEVICE,
        *args,
        **kwargs,
    ) -> None:
        super(CustomTransformerEncoderLayer, self).__init__()
        self.attention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=att_factory(),
            use_rotary_embeddings=True,
            **kwargs,
        ).to(device)
        self.layer_norm = torch.nn.LayerNorm(latent_dim).to(device)

        self.ff1 = torch.nn.Linear(latent_dim, ff_dim).to(device)
        self.ff2 = torch.nn.Linear(ff_dim, latent_dim).to(device)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        # MHA self attention, add, norm
        if self.attention.attention.supports_attention_mask:
            x = self.layer_norm(self.attention(x, att_mask=attention_mask) + x)
        else:
            x = self.layer_norm(self.attention(x) + x)

        # ff, add, norm
        x = self.layer_norm(self.gelu(self.ff2(self.gelu(self.ff1(x)))) + x)

        return x


class CustomTransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        att_factory: Callable[[], components.Attention],
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        device: str = DEVICE,
        *args,
        **kwargs,
    ) -> None:
        super(CustomTransformerDecoderLayer, self).__init__()
        self.crossattention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=att_factory(),
            use_rotary_embeddings=True,
            **kwargs,
        ).to(device)

        self.selfattention = components.MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=att_factory(),
            use_rotary_embeddings=True,
            **kwargs,
        ).to(device)
        self.layer_norm = torch.nn.LayerNorm(latent_dim).to(device)

        self.ff1 = torch.nn.Linear(latent_dim, ff_dim).to(device)
        self.ff2 = torch.nn.Linear(ff_dim, latent_dim).to(device)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, attention_mask: torch.Tensor):
        # MHA self attention, add norm
        if self.selfattention.attention.supports_attention_mask:
            x = self.layer_norm(self.selfattention(x, att_mask=attention_mask) + x)
        else:
            x = self.layer_norm(self.selfattention(x) + x)

        # MHA cross attention, add, norm
        if self.crossattention.attention.supports_attention_mask:
            x = self.layer_norm(
                self.crossattention(
                    key=ctx, query=ctx, value=x, att_mask=attention_mask
                )
                + x
            )
        else:
            x = self.layer_norm(self.crossattention(key=ctx, query=ctx, value=x) + x)

        # ff, add, norm
        x = self.layer_norm(self.gelu(self.ff2(self.gelu(self.ff1(x)))) + x)

        return x


class CustomTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        att_factory: Callable[[], components.Attention],
        n_layers: int,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        device: str = DEVICE,
        **kwargs,
    ) -> None:
        super(CustomTransformerEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(
                str(i),
                CustomTransformerEncoderLayer(
                    att_factory=att_factory,
                    latent_dim=latent_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    device=device,
                    **kwargs,
                ),
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        for module in self._modules.values():
            x = module(x, attention_mask=attention_mask)
        return x


class CustomTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        att_factory: Callable[[], components.Attention],
        n_layers: int,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        device: str = DEVICE,
        **kwargs,
    ) -> None:
        super(CustomTransformerDecoder, self).__init__()
        for i in range(n_layers):
            self.add_module(
                str(i),
                CustomTransformerDecoderLayer(
                    att_factory=att_factory,
                    latent_dim=latent_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    device=device,
                    **kwargs,
                ),
            )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, attention_mask: torch.Tensor):
        for module in self._modules.values():
            x = module(x, ctx, attention_mask=attention_mask)
        return x


class AttentionModel(torch.nn.Module):
    def __init__(
        self,
        att_factory: attentions.Attention,
        latent_dim: int,
        ff_dim: int,
        n_heads: int,
        enc_layers: int,
        dec_layers: int,
        device: str = DEVICE,
    ) -> None:
        super(AttentionModel, self).__init__()

        # data
        self.n_heads = n_heads
        self.latent_dim = latent_dim

        self.proj = torch.nn.Linear(NUM_BPP * 4 + 6 + 4, latent_dim).to(device)

        # positional embedding encoder/decoder layers
        self.has_encoder = enc_layers >= 1
        self.has_decoder = dec_layers >= 1
        if self.has_encoder:
            self.encoder_layers = CustomTransformerEncoder(
                latent_dim=latent_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                device=device,
                att_factory=att_factory,
                n_layers=enc_layers,
            )
        if self.has_decoder:
            self.decoder_layers = CustomTransformerDecoder(
                latent_dim=latent_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                att_factory=att_factory,
                n_layers=dec_layers,
            )

        # output head
        self.head = torch.nn.Linear(latent_dim, 1).to(device)
        self.final_result = torch.nn.Linear(NUM_REACTIVITIES, NUM_REACTIVITIES).to(
            device
        )

        # activations
        self.gelu = torch.nn.GELU()

    def forward(
        self,
        bases: torch.Tensor,
        bpp: torch.Tensor,
        mfe: torch.Tensor,
        capr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            - bases: torch.Tensor - should have shape B,457,4
            - bpp: torch.Tensor - should have shape B,457,NUM_BPP
            - mfe: torch.Tensor - B,457,3*NUM_BPP
            - capr: torch.Tensor - B,457,6
        """
        mask = att_utils.maybe_merge_masks(
            att_mask=None,
            key_padding_mask=bases.any(dim=-1) != 0,
            batch_size=bases.shape[0],
            num_heads=self.n_heads,
            src_len=NUM_REACTIVITIES,
        )

        # project inputs and bpp to latent_dim
        x = self.gelu(self.proj(torch.concat([bases, bpp, mfe, capr], dim=-1)))

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
        x = self.gelu(self.head(x).flatten(start_dim=1))
        return x


def contrastive_loss(
    i_pred: torch.Tensor,
    j_pred: torch.Tensor,
    l1=torch.nn.L1Loss(reduction="none"),
):
    """
    Contrastive loss
    """
    return (l1(i_pred, j_pred)).mean()

def train_unlabelled_batch(
    m: torch.nn.Module,
    bases: torch.Tensor,
    bpp: torch.Tensor,
    mfe: torch.Tensor,
    capr: torch.Tensor,
    m_optim: torch.optim.Optimizer,
):
    """
    Get the loss on a batch and perform the corresponding weight updates.
    Used for training purposes
    """
    m_optim.zero_grad()

    i_bpp = bpp
    j_bpp = bpp
    i_bpp[:, :, 1] = i_bpp[:, :, 0]
    j_bpp[:, :, 0] = j_bpp[:, :, 1]

    i_mfe = mfe
    j_mfe = mfe
    i_mfe[:, :, 3:] = i_mfe[:, :, :3]
    j_mfe[:, :, :3] = j_mfe[:, :, 3:]

    i_preds = m(bases, i_bpp, i_mfe, capr)
    j_preds = m(bases, j_bpp, j_mfe, capr)

    # get contrastive loss
    loss = contrastive_loss(i_preds, j_preds)
    loss.backward()

    # calculate gradients
    m_optim.step()

    # return weighted and unweighted mae loss
    return loss.to(torch.float32).detach().cpu()


def ssl_train_loop(
    m: torch.nn.Module,
    m_optim: torch.optim.Optimizer,
    m_sched: torch.optim.lr_scheduler.LRScheduler,
    train_dataloader: data.DataLoader,
    writer: SummaryWriter,
    model_name: str,
    epochs: int = 1,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float32,
):
    """
    Train the given model.

    Arguments:
        - m: torch.nn.Module - the model to train.
        - m_optim: torch.optim.Optimizer - the optimizer to use for the model
        - m_sched: torch.optim.lr_scheduler.LRScheduler - the scheduler to use to adjust the lr
        - train_dataloader: data.Dataloader - the dataloader that provides the batched training data
        - writer: SummaryWriter - the summary writer to use for tensorboard logging
        - model_name: str - the name of the model (what to save it as)
        - epochs: int - how many epochs to train for. Defaults to `1`.
        - device: torch.device - the device to train on, defaults to `DEVICE`
        - dtype: torch.dtype - the dtype to use for training
    """
    m = m.to(device, dtype)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        epoch_mae = 0.0

        m = m.train()
        for tdata in (prog := tqdm(train_dataloader, desc="batch")):
            bases = tdata["bases"]
            bpp = tdata["bpp"]
            mfe = tdata["mfe"]
            capr = tdata["capr"]

            bases = bases.to(device, dtype)
            bpp = bpp.to(device, dtype)
            mfe = mfe.to(device, dtype)
            capr = capr.to(device, dtype)

            mae = train_unlabelled_batch(
                m, bases, bpp, mfe, capr, m_optim
            )

            epoch_mae += mae

            # log
            prog.set_postfix_str(
                f"mae_loss: {mae:.5f}"
            )

            # break  # used for sanity check
        epoch_mae /= len(train_dataloader)


        print(
            f"Epoch MAE: {epoch_mae:.5f}"
        )

        # log to tensorboard
        writer.add_scalar("epoch_mae", epoch_mae, global_step=epoch)
        writer.add_scalar("lr", m_sched.get_last_lr()[0], global_step=epoch)

        # save every epoch
        torch.save(m.state_dict(), f"{model_name}_ssl_model.pt")

        # update lr
        m_sched.step()


def create_scaled_dot_product_attention(dropout: float = 0.1) -> attentions.Attention:
    """
    Factory for creating ScaledDotProduct attention

    Arguments:
        - dropout: float - dropout to use for the attention. Defaults to 0.1
    """
    return attentions.ScaledDotProduct(dropout=dropout)


def create_global_attention(dropout: float = 0.1) -> attentions.Attention:
    """
    Factory for creating GlobalAttention

    Arguments:
        - dropout: float - dropout to use for the attention. Defaults to 0.1
    """
    ret = attentions.GlobalAttention(
        dropout=dropout,
        attention_query_mask=torch.ones((NUM_REACTIVITIES, 1), dtype=torch.bool),
    )
    ret.supports_attention_mask = True
    return ret


def create_lin_attention(dropout: float = 0.1) -> attentions.Attention:
    """
    Factory for creating LinFormerAttention

    Arguments:
        - dropout: float - dropout to use for the attention. Defaults to 0.1
    """
    return attentions.LinformerAttention(dropout=dropout, seq_len=NUM_REACTIVITIES)


def ssl_train(
    run_name: str,
    dataset_name: str,
    lr: float = 3e-4,
    scheduler: str = "cosine",
    batch_size: int = 32,
    samples: int = 0,
    epochs: int = 10,
    model_dict: dict = dict(
        latent_dim=32,
        n_heads=1,
        enc_layers=4,
        dec_layers=4,
        ff_dim=2048,
    ),
    att_factory: Callable[
        [], attentions.Attention
    ] = create_scaled_dot_product_attention,
    save_prefix: str = "",
    load_prefix: str = "",
    dtype: torch.dtype = torch.float32,
):
    """
    Train a model from start to finish, taking care of data loading,
    optimizers, etc

    Arguments:
        - run_name: str - the name of the run to log as
        - dataset_name: str - the name of the dataset, either "2a3" or "dms"
        - lr: float - the learning rate to use. Defaults to 3e-4
        - scheduler: str - the scheduler to use. Only 'cosine' causes a difference
        - batch_size: int - the batch size to use when training and running validation. Defaults to 64
        - val_split: float - the size of the validation, from 0 to 1. Defaults to 0.1
        - epochs: int - the number of epochs to train for. Defaults to 10
        - model_dict: dict - a dictionary containing all the arguments to be passed when instantiating
            the `AttentionModel`
        - att_factory: Callable[[], attentions.Attention] - factory that produces the attention type
    """
    # load and process dataset
    columns = ["bases", "bpp", "mfe", "capr"]

    dataset = Dataset.load_from_disk(
        f"test_data_preprocessed"
    ).with_format("torch")

    if samples > 0:
        dataset = dataset.select(random.sample(range(0, len(dataset)), samples))
    print(dataset)

    # load into batches
    train_dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # create logger
    writer = SummaryWriter(f"runs/{run_name}")

    # create model + optimizer + scheduler
    model = AttentionModel(
        **model_dict,
        att_factory=att_factory,
    ).to(DEVICE, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # load old weights if possible
    if os.path.exists(f"{load_prefix}{dataset_name}_model.pt"):
        try:
            model.load_state_dict(torch.load(f"{load_prefix}{dataset_name}_model.pt"))
            print(f"loaded previous {load_prefix}{dataset_name} weights")
        except Exception as e:
            print(f"not loading previous {load_prefix}{dataset_name} weights because", e)
            pass

    # log # of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total {dataset_name} model params:", params)

    # train
    ssl_train_loop(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        writer=writer,
        model_name=save_prefix+dataset_name,
        epochs=epochs,
        dtype=dtype,
    )
