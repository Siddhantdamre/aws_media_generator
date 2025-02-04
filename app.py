import boto3
import time
import logging

logging.basicConfig(level=logging.DEBUG)
boto3.set_stream_logger(name="botocore")
# AWS Clients
transcribe_client = boto3.client('transcribe', region_name='us-east-1')
rekognition_client = boto3.client('rekognition', region_name='us-east-1')
bedrock_client = boto3.client('bedrock', region_name='us-east-1')  # Bedrock support

print("AWS clients initialized!")
def transcribe_audio(audio_file_uri):
    # Create a unique job name
    job_name = f"transcription-job-{int(time.time())}"
    
    # Start transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audio_file_uri},
        MediaFormat='wav',  # Change if your file is MP3, MP4, etc.
        LanguageCode='en-US'
    )
    print(f"Started transcription job: {job_name}")

    # Wait for the job to complete
    while True:
        job_status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if job_status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Waiting for transcription to complete...")
        time.sleep(10)

    if job_status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = job_status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        print(f"Transcription complete! Transcript file: {transcript_uri}")
        return transcript_uri
    else:
        print("Transcription failed.")
        return None
    
def analyze_with_rekognition(image_path):
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    response = rekognition_client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=75
    )
    for label in response['Labels']:
        print(f"Detected label: {label['Name']} with confidence: {label['Confidence']}%")

from PIL import Image
import requests
from io import BytesIO

def generate_image(description_text):
    try:
        response = bedrock_client.invoke_model(
            ModelId="stability-ai-stable-diffusion-v1-5",
            Prompt=description_text,
            Parameters={"max_steps": 50, "width": 512, "height": 512}
)
        image_data = response.get('Body').read()  # Ensure this matches Bedrock's API response structure
        image = Image.open(BytesIO(image_data))
        image.save("generated_image.png")
        print("Image saved as generated_image.png!")
    except Exception as e:
        print(f"Error generating image: {e}")

def main():
    # Provide your audio file URL in S3
    audio_file_uri = "s3://sketchaibucket/audio-file.wav"
    
    print("Starting transcription...")
    transcript_uri = transcribe_audio(audio_file_uri)
    if transcript_uri:
        # Fetch the transcript file
        transcript_text = requests.get(transcript_uri).json()['results']['transcripts'][0]['transcript']
        print(f"Transcribed Text: {transcript_text}")
        
        print("Generating image from text...")
        generate_image(transcript_text)

        print("Analyzing with Rekognition...")
        analyze_with_rekognition("generated_image.png")
        
        

if __name__ == "__main__":
    main()
