import os
from moviepy.editor import VideoFileClip
from googletrans import Translator
import speech_recognition as sr
import pyttsx3

def convert_anime_to_english(input_file, output_file):
    try:
        # Load the video file
        video = VideoFileClip(input_file)
        audio = video.audio
        audio_file = "temp_audio.wav"
        audio.write_audiofile(audio_file)

        # Initialize recognizer and translator
        recognizer = sr.Recognizer()
        translator = Translator()
        engine = pyttsx3.init()

        # Load the audio file for speech recognition
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        # Recognize speech in the audio file
        try:
            text = recognizer.recognize_google(audio_data, language='ja-JP')
            print("Recognized Japanese text:", text)
        except sr.UnknownValueError:
            print("Could not understand audio")
            return
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return

        # Translate the recognized text to English
        translated_text = translator.translate(text, src='ja', dest='en').text
        print("Translated text:", translated_text)

        # Convert the translated text to speech
        engine.save_to_file(translated_text, "translated_audio.mp3")
        engine.runAndWait()

        # Combine the original video with the translated audio
        final_video = video.set_audio("translated_audio.mp3")
        final_video.write_videofile(output_file)

        # Clean up temporary files
        os.remove(audio_file)
        os.remove("translated_audio.mp3")
        print("Conversion completed successfully. Output file:", output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_video = "input_anime.mp4"  # Replace with your input file
    output_video = "output_anime.mp4"  # Desired output file name
    convert_anime_to_english(input_video, output_video)
