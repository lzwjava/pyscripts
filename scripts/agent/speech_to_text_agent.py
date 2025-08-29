import os
import argparse
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from scripts.llm.openrouter_client import call_openrouter_api


MAX_AUDIO_LENGTH_SECS = 20 * 60 * 60
OUTPUT_DIRECTORY = "assets/transcriptions"


def run_batch_recognize(audio_gcs_uri, output_gcs_folder, language_code="en-US"):
    """
    Transcribes an audio file using Google Cloud Speech-to-Text Batch API.

    Args:
        audio_gcs_uri: GCS URI of the audio file.
        output_gcs_folder: GCS URI of the folder to store the transcription.
        language_code: Language code for transcription (e.g., "en-US", "cmn-CN").
    """
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config={},
        features=cloud_speech.RecognitionFeatures(
            enable_word_confidence=True,
            enable_word_time_offsets=True,
        ),
        model="long",
        language_codes=[language_code],
    )

    output_config = cloud_speech.RecognitionOutputConfig(
        gcs_output_config=cloud_speech.GcsOutputConfig(uri=output_gcs_folder),
    )

    files = [cloud_speech.BatchRecognizeFileMetadata(uri=audio_gcs_uri)]

    request = cloud_speech.BatchRecognizeRequest(
        recognizer="projects/graphite-ally-445108-k3/locations/global/recognizers/_",
        config=config,
        files=files,
        recognition_output_config=output_config,
    )
    operation = client.batch_recognize(request=request)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=3 * MAX_AUDIO_LENGTH_SECS)
    print(response)


def process_audio_file(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_file)
    if not filename.endswith(".m4a"):
        print(f"Error: {filename} is not a .m4a file.")
        return

    output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    if os.path.exists(output_filename):
        print(f"Skipping {filename}: {output_filename} already exists.")
        return

    print(f"Processing: {filename}")
    try:
        # Determine language based on filename suffix
        if filename.endswith("-zh.m4a"):
            language_code = "cmn-CN"
        else:
            language_code = "en-US"

        # Construct GCS URIs
        gcs_audio_uri = f"gs://test2x/audio-files/{filename}"
        gcs_output_uri = f"gs://test2x/transcripts/{os.path.splitext(filename)[0]}"

        # Upload the file to GCS if it doesn't exist
        storage_client = storage.Client()
        bucket = storage_client.bucket("test2x")
        blob = bucket.blob(f"audio-files/{filename}")
        if not blob.exists():
            blob.upload_from_filename(input_file)
            print(f"Uploaded {filename} to GCS.")
        else:
            print(f"{filename} already exists in GCS.")

        run_batch_recognize(
            audio_gcs_uri=gcs_audio_uri,
            output_gcs_folder=gcs_output_uri,
            language_code=language_code,
        )
        print(f"File {filename} processed.\n")

        # Download the transcription
        blobs = storage_client.list_blobs(
            "test2x", prefix=f"transcripts/{os.path.splitext(filename)[0]}"
        )

        for blob in blobs:
            if blob.name.endswith(".json"):
                local_output_path = os.path.join(
                    output_dir, os.path.basename(blob.name)
                )
                blob.download_to_filename(local_output_path)
                print(f"Downloaded {blob.name} to {local_output_path}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

    print(f"Processing complete for {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a single Voice Memo (.m4a) file to generate transcription."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input path for the Voice Memo file.",
    )

    args = parser.parse_args()

    process_audio_file(
        input_file=args.input_file,
        output_dir=OUTPUT_DIRECTORY,
    )
