## Preprocess dataset for training
Simply run `py feature_extraction\preprocess.py` from the directory of the project.
This will create everything required - scalers, masks and of course the extracted features.

## To process incoming audio
Call `process_incoming_audio` from `inference.py`.
It takes two arguments
  - `file_path`: A valid path to the audio clip to be processed.
  - `no_mel`: Whether or not to extract mel features from the audio as well. Extracts the raw audio features in either case. Good when processing for edge deployments and server deployments.
