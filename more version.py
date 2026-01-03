import speech_recognition as sr
import requests

def translate_text(text, target_language='en'):
    """Translate the given text to the target language using an external API."""
    try:
        # Replace 'YOUR_API_KEY' with your actual API key for the translation service
        api_key = 'YOUR_API_KEY'
        url = f"https://api.translation-service.com/translate?text={text}&target={target_language}&key={api_key}"
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        
        translation = response.json().get('translatedText')
        return translation
    except requests.exceptions.RequestException as e:
        print(f"Error during translation: {e}")
        return None

def recognize_audio():
    """Capture audio from the microphone and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak the dialogue from the anime...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio, language='auto')
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def main():
    """Main function to run the anime translation software."""
    print("Anime Translation Software")
    
    # Step 1: Recognize audio
    audio_text = recognize_audio()
    if audio_text:
        print(f"Recognized text: {audio_text}")
        
        # Step 2: Translate the recognized text
        translated_text = translate_text(audio_text)
        if translated_text:
            print(f"Translated text: {translated_text}")
        else:
            print("Translation failed.")
    else:
        print("No audio recognized.")

if __name__ == "__main__":
    main()
