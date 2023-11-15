from preprocessing import (
    filter_train_data,
    filter_DMS,
    filter_2A3,
    preprocess_csv,
    process_data_test,
)
from models import train, create_lin_attention
from ssl_models import ssl_train
from submit import submit
from torch import float32

if __name__ == "__main__":
    # filter out data
    filter_train_data()
    filter_DMS()
    filter_2A3()

    # preprocess csvs
    preprocess_csv("train_data_2a3_preprocessed", "train_data_2a3.csv")
    preprocess_csv("train_data_dms_preprocessed", "train_data_dms.csv")
    preprocess_csv(
        "test_data_preprocessed",
        "test_sequences.csv",
        map_fn=process_data_test,
        extra_cols_to_keep=["id_min", "id_max"],
    )

    # train models
    train(
        "2a3_linearfold",
        dataset_name="2a3",
        epochs=50,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="pressl_",
    )
    train(
        "dms_linearfold",
        dataset_name="dms",
        epochs=50,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="pressl_",
    )

    # ss train models
    ssl_train(
        "2a3_linearfold",
        dataset_name="2a3",
        epochs=10,
        samples=1000,
        batch_size=100,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="ssl_",
        load_prefix="pressl_",
    )
    ssl_train(
        "dms_linearfold",
        dataset_name="dms",
        epochs=10,
        samples=1000,
        batch_size=100,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="ssl_",
        load_prefix="pressl_",
    )

    # train models again
    train(
        "2a3_linearfold",
        dataset_name="2a3",
        epochs=40,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="postssl_",
        load_prefix="ssl_",
    )
    train(
        "dms_linearfold",
        dataset_name="dms",
        epochs=40,
        att_factory=create_lin_attention,
        dtype=float32,
        save_prefix="postssl_",
        load_prefix="ssl_",
    )

    # submit predictions (load regular model)
    '''submit(
        batch_size=64,
        dtype=float32,
        model_2a3_att_factory=create_lin_attention,
        model_dms_att_factory=create_lin_attention,
    )'''

    # submit predictions (load ssl model)
    submit(
        batch_size=64,
        dtype=float32,
        model_2a3_att_factory=create_lin_attention,
        model_dms_att_factory=create_lin_attention,
        model_2a3_params="postssl_2a3_model.pt",
        model_dms_params="postssl_dms_model.pt",
    )
