"""This module saves a Keras model to BentoML."""

from pathlib import Path

import tensorflow as tf
import bentoml


def load_model_and_save_it_to_bento(model_path: Path, model_name : str) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    # model = keras.models.load_model(model_file)

    model = tf.keras.saving.load_model(model_path)
    bento_model = bentoml.keras.save_model(model_name, model)

    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("service/models/model_p_2_1_best.h5"), "p_model")
    load_model_and_save_it_to_bento(Path("service/models/model_s_2_1_best.h5"), "s_model")
    load_model_and_save_it_to_bento(Path("service/models/model_mag_1.h5"), "mag_model")
    load_model_and_save_it_to_bento(Path("service/models/model_dist_1.h5"), "dist_model")