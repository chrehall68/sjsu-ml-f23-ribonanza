from models import DEVICE, AttentionModel
import torch
import torch.utils.data as data
from tqdm import tqdm
from datasets import Dataset
import os


def pipeline(
    model: torch.nn.Module,
    input_ds: str,
    out: str,
    batch_size: int,
):
    """
    Make predictions on the test dataset and write them to a csv file

    Parameters:
        - model: torch.nn.Module - the model trained on both the 2a3 and dms distributions
        - input_ds: str - name of the dataset to load
        - out: str - name of the file to write to
        - batch_size: int - size of the batches to use to process the data.
            In general, larger batch sizes mean faster runtime
    """
    ds = Dataset.load_from_disk(input_ds).with_format("torch")
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    iterable = iter(loader)

    with open(out, "w") as outfile:
        # write the header
        outfile.write("id,reactivity_DMS_MaP,reactivity_2A3_MaP\n")

        for _ in tqdm(range(len(loader))):
            # get the next group of data
            tdata = next(iterable)
            tokens = tdata["simple_tokens"].to(DEVICE)
            bpp = tdata["bpp"].to(DEVICE)
            min_ids = tdata["id_min"].numpy()
            max_ids = tdata["id_max"].numpy()

            # make predictions w/o gradients
            with torch.no_grad():
                preds = model(tokens, bpp).cpu().numpy()

            # write preds
            for i in range(tokens.shape[0]):
                outfile.writelines(
                    map(
                        # dms is index 0, 2a3 is index 1
                        lambda seq_idx: f"{seq_idx},{preds[i, seq_idx-min_ids[i], 0]:.3f},{preds[i, seq_idx-min_ids[i], 1]:.3f}\n",
                        # +1 since the id_max is inclusive
                        range(min_ids[i], max_ids[i] + 1),
                    )
                )


def submit(
    batch_size: int = 64,
    model_dict: dict = dict(
        latent_dim=32,
        n_heads=1,
        enc_layers=4,
        dec_layers=4,
        ff_dim=2048,
    ),
    model_name: str = "full",
):
    """
    Generate a submission.csv.zip file for submitting

    Arguments:
        - batch_size: int - the batch size to use during inference. Defaults to 64
        - model_2a3_dict: dict - a dictionary containing all the arguments to be passed when instantiating
            the 2a3 `AttentionModel`
        - model_dms_dict: a dictionary containing all the arguments to be passed when instantiating
            the dms `AttentionModel`
    """
    # initialize models
    model = AttentionModel(**model_dict)

    # load weights
    model.load_state_dict(torch.load(f"{model_name}_model.pt"))

    # set in evaluation mode and move to device
    model.eval().to(DEVICE)

    pipeline(
        model,
        "test_data_preprocessed",
        "submission.csv",
        batch_size=batch_size,
    )

    # zip our submission into an easily-uploadable zip file
    print("zipping submissions. This may take a while...")
    os.system("zip submission.csv.zip submission.csv")
    print("Done zipping submissions!")
