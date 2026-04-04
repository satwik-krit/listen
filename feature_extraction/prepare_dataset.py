import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from tqdm import tqdm
import joblib

import config
from features import get_normal_baseline, process_file, create_master_mask


def process_file_and_save(
    file_path,
    scaler_mel,
    scaler_delta,
    scaler_delta2,
    master_noise,
    input_dir,
    output_dir,
):
    try:
        mel_data, audio_features = process_file(
            file_path, scaler_mel, scaler_delta, scaler_delta2, master_noise
        )
        relative_path = file_path.relative_to(input_dir)
        print(relative_path)
        target_path_mel = output_dir / relative_path.with_suffix(".npy")

        # Extract the base name without the .wav extension
        base_name = relative_path.stem

        target_path_mel = output_dir / relative_path.with_name(
            f"{base_name}_mel_data.npy"
        )
        target_path_audio_features = output_dir / relative_path.with_name(
            f"{base_name}_audio_features.npy"
        )
        target_path_mel.parent.mkdir(parents=True, exist_ok=True)
        target_path_audio_features.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path_mel, mel_data.astype(np.float32))
        np.save(target_path_audio_features, audio_features.astype(np.float32))

        label = str(relative_path).split("\\")[0]
        return 1 if label == "abnormal" else 0
    except Exception as e:
        print(f"error on {file_path.name}: {e}")
        return -1


def batch_process(input_dir, output_dir, scaler_dir):
    print(f"STARTING BATCH PROCESSING for {input_dir}...")
    start = time.perf_counter()
    all_files = list(input_dir.rglob("*.wav"))

    # Create the noise for our master mask
    print("CREATING MASTER MASK...")
    # Last 20 files are most likely normal, so pass them without checking
    master_noise = create_master_mask(all_files[-1:-20:-1])

    s_mel, s_delta, s_delta2 = get_normal_baseline(all_files[-1:-20:-1])

    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(s_mel, scaler_dir / "scaler_mel.pkl")
    joblib.dump(s_delta, scaler_dir / "scaler_delta.pkl")
    joblib.dump(s_delta2, scaler_dir / "scaler_delta2.pkl")
    joblib.dump(master_noise, scaler_dir / "master_noise.pkl")

    worker_func = partial(
        process_file_and_save,
        input_dir=input_dir,
        output_dir=output_dir,
        scaler_mel=s_mel,
        scaler_delta=s_delta,
        scaler_delta2=s_delta2,
        master_noise=master_noise,
    )

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(worker_func, all_files, chunksize=20), total=len(all_files)
            )
        )

    end = time.perf_counter()
    print(f"FINISHED in {round(end - start, 2)} seconds.")
    print(
        f"\t{results.count(0)} NORMAL files\n\t{results.count(1)} ABNORMAL files\n\t{results.count(-1)} ERRORS"
    )


if __name__ == "__main__":
    for i in range(len(config.INPUT_DIRS)):
        batch_process(
            input_dir=config.INPUT_DIRS[i],
            output_dir=config.OUTPUT_DIRS[i],
            scaler_dir=config.SCALER_DIRS[i],
        )
