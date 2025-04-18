import json
import os
import time
import wikipedia
import requests
import random
import string
import yt_dlp
import urllib.parse
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import pandas as pd
import yaml
from difflib import get_close_matches
from fuzzywuzzy import fuzz
import warnings
from bs4.builder import HTMLParserTreeBuilder
from utils import schedule_deletion

# Suppress the GuessedAtParserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")


# ---------- Predefined Q&A and Books Loading ----------
def load_qna():
    qna_file = "qna.json"
    if os.path.exists(qna_file):
        with open(qna_file, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def load_books():
    books = {}
    book_paths = ["books/ArticlesNCSS.txt"]
    for path in book_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                books[os.path.basename(path)] = file.read()
            print(f"Loaded: {path}")
        else:
            print(f"Warning: {path} not found.")
    return books


# ---------- Spell Checker ----------
spell = SpellChecker()


def correct_spelling(query):
    words = query.split()
    corrected_words = [
        spell.correction(word) if spell.correction(word) else word for word in words
    ]
    return " ".join(corrected_words)


# ---------- Wikipedia Search ----------
def search_wikipedia(query):
    try:
        # Force the usage of the 'html.parser'
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return None
    except Exception as e:
        return None


# ---------- Music Search and Audio Download ----------
def random_filename():
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


def search_music_yt(query):
    ydl_opts = {"default_search": "ytsearch1", "noplaylist": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(query, download=False)
            if "entries" in info and info["entries"]:
                entry = info["entries"][0]
                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                return video_url
        except Exception as e:
            print(f"Music search error: {str(e)}")
    return None


def download_audio(youtube_url):
    filename = random_filename()
    output_path = f"{filename}.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{filename}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


# ---------- Session Memory, Training Data, and Logging ----------
def load_memory():
    memory_file = "user_memory.json"
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_memory(memory):
    with open("user_memory.json", "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


def load_training_data():
    data_file = "user_data.json"
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_training_data(data):
    with open("user_data.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def log_chat(user, query, response, rating=None):
    log_file = "chat_logs.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            logs = json.load(file)

    log_entry = {
        "user": user,
        "query": query,
        "response": response,
        "timestamp": time.time(),
    }

    if rating is not None:
        log_entry["rating"] = rating

    logs.append(log_entry)

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4)


def log_feedback(user, query, rating):
    feedback_file = "feedback.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as file:
            feedback_data = json.load(file)
    else:
        feedback_data = []
    feedback_data.append(
        {"user": user, "query": query, "rating": rating, "timestamp": time.time()}
    )
    with open(feedback_file, "w", encoding="utf-8") as file:
        json.dump(feedback_data, file, indent=4)


# ---------- Load Datasets ----------
dataset_folder = "dataset"


def load_replacements(file_path="replacements.json"):
    try:
        with open(file_path, "r") as file:
            replacements = json.load(file)
        return replacements
    except Exception as e:
        print(f"Error loading replacements: {e}")
        return {}


# Load the replacement phrases
replacements = load_replacements()

# Function to replace phrases in bot responses
def apply_replacements(response):
    for old_phrase, new_phrase in replacements.items():
        response = response.replace(old_phrase, new_phrase)
    return response


def load_csv(file_path):
    csv_data = pd.read_csv(file_path)
    print(f"Columns in {file_path}: {csv_data.columns}")
    return csv_data


def load_json(file_path):
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            print(f"Loaded JSON data from {file_path}")
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {str(e)}")
            return []


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def load_yml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_parquet(file_path):
    return pd.read_parquet(file_path)


def load_txt(file_path):
    with open(file_path, "r") as file:
        return file.readlines()


def load_all_datasets():
    combined_data = []
    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        print(f"Loading {filename}...")
        if filename.endswith(".csv"):
            csv_data = load_csv(file_path)
            combined_data.extend(process_csv(csv_data))
        elif filename.endswith(".json"):
            combined_data.extend(process_json(load_json(file_path)))
        elif filename.endswith(".jsonl"):
            combined_data.extend(process_jsonl(load_jsonl(file_path)))
        elif filename.endswith(".yml"):
            combined_data.extend(process_yml(load_yml(file_path)))
        elif filename.endswith(".parquet"):
            combined_data.extend(process_parquet(load_parquet(file_path)))
        elif filename.endswith(".txt"):
            combined_data.extend(process_txt(load_txt(file_path)))
    return combined_data


# ---------- Process Datasets ----------
def process_csv(csv_data):
    if "question" in csv_data.columns and "response" in csv_data.columns:
        return list(zip(csv_data["question"], csv_data["response"]))
    return []


def process_json(json_data):
    combined = []
    if isinstance(json_data, list):
        for item in json_data:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            if prompt and response:
                combined.append((prompt, response))
    elif isinstance(json_data, dict):
        prompt = json_data.get("prompt", "")
        response = json_data.get("response", "")
        if prompt and response:
            combined.append((prompt, response))
    return combined


def process_jsonl(jsonl_data):
    combined = []
    for item in jsonl_data:
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        if prompt and response:
            combined.append((prompt, response))
    return combined


def process_yml(yml_data):
    if isinstance(yml_data, list):
        return [
            (item.get("patterns", ""), item.get("responses", "")) for item in yml_data
        ]
    return []


def process_parquet(parquet_data):
    if "prompt" in parquet_data.columns and "response" in parquet_data.columns:
        return list(zip(parquet_data["prompt"], parquet_data["response"]))
    elif "question" in parquet_data.columns and "answer" in parquet_data.columns:
        return list(zip(parquet_data["question"], parquet_data["answer"]))
    return []


def process_txt(txt_data):
    combined = []
    for i in range(0, len(txt_data) - 1, 2):
        combined.append((txt_data[i], txt_data[i + 1]))
    if len(txt_data) % 2 != 0:
        combined.append((txt_data[-1], "Sorry, I don't understand."))
    return combined


# Load all datasets
dataset_data = load_all_datasets()
replacements = load_replacements()

# ---------- Get Best Match from Dataset ----------
def get_best_match(user_input, data):
    best_match = None
    best_score = 0
    for question, response in data:
        score = fuzz.partial_ratio(user_input.lower(), question.lower())
        if score > best_score:
            best_score = score
            best_match = response
    if best_score >= 70:
        return best_match
    else:
        return "Sorry, I didn't quite get that."


# ---------- Initialization ----------
qa_responses = load_qna()
books = load_books()
memory = load_memory()
training_data = load_training_data()

# ---------- Main Command-Line Loop ----------


def run_cli_chatbot():
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            break

    # Check predefined Q&A (case-insensitive match)
    if prompt.lower() in qa_responses:
        response = qa_responses[prompt.lower()]
        response = apply_replacements(response)  # Apply replacements here
        print(f"Bot: {response}")

        # Additionally provide Wikipedia info if available
        wiki_info = search_wikipedia(prompt)
        if wiki_info:
            print(f"🌐 Wikipedia says: {wiki_info}")

    # Music request handling
    elif prompt.lower().startswith("play "):
        song_name = prompt[5:].strip()
        print(f"🎵 Searching music: {song_name} (Please wait)...")
        youtube_link = search_music_yt(song_name)
        if youtube_link:
            print(f"🎵 Found YouTube link: {youtube_link}")
            print("🎧 Downloading audio (light-speed mode)...")
            audio_file = download_audio(youtube_link)
            schedule_deletion(audio_file)
            print(f"🎶 Music ready! Download the audio file: {audio_file}")
        else:
            print("❌ No music found.")

    # Normal query: check dataset for best match, then Wikipedia, then books.
    else:
        response = get_best_match(prompt, dataset_data)
        response = apply_replacements(response)  # Apply replacements here
        print(f"Bot: {response}")

        # Wikipedia lookup
        wiki_info = search_wikipedia(prompt)
        if wiki_info:
            response = f"🌐 Wikipedia says: {wiki_info}"
            print(f"Bot: {response}")

    # Update session memory for default_user
    if "default_user" not in memory:
        memory["default_user"] = []

    # Save the conversation line
    memory["default_user"].append(f":User  {prompt}\nBot: {response}")

    # Keep only the last 10 messages
    memory["default_user"] = memory["default_user"][-10:]
    save_memory(memory)

    # Store training data
    if prompt not in training_data:
        training_data[prompt] = response
        save_training_data(training_data)

    # Ask for feedback
    rating = None
    fb = input("Rate the response (1-5) or press Enter to skip: ").strip()
    if fb:
        try:
            fb_int = int(fb)
            if 1 <= fb_int <= 5:
                rating = fb_int
                print("Feedback recorded.")
            else:
                print("Invalid rating. Skipping feedback.")
        except ValueError:
            print("Invalid input. Skipping feedback.")

    # Log chat with rating
    log_chat("default_user", prompt, response, rating=rating)


if __name__ == "__main__":
    run_cli_chatbot()


def get_bot_response(prompt, mode=None):
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a message."

    # 🎵 Music mode
    if mode == "music":
        song_name = prompt
        youtube_link = search_music_yt(song_name)
        if youtube_link:
            try:
                audio_file = download_audio(youtube_link)
                file_url = f"/music/{audio_file}"
                return f"""
<div class="music-reply">
  <div class="music-header">
    <img src='/static/images/bot-button-logo.png' class='thumbnail'/>
    <div class="track-title">🎵 <strong>{song_name}</strong></div>
  </div>
  <audio controls autoplay class="music-player audio-player">
    <source src="{file_url}" type="audio/mpeg">
    Your browser does not support audio.
  </audio>
</div>
"""
            except Exception as e:
                return f"⚠️ Failed to get audio: {str(e)}"
        return "🎵 Song not found."


    # 🌐 Wikipedia-only mode
    elif mode == "pedia":
        wiki = search_wikipedia(prompt)
        return f"🌐 {wiki}" if wiki else "No Wikipedia info found."

    # 📚 Books-only mode
    elif mode == "books":
        for book, content in books.items():
            if prompt.lower() in content.lower():
                return f"📘 From {book}:\n\n{content[:400]}..."
        return "📕 Not found in books."

    # 🧠 GPT-style (custom dataset + Wikipedia)
    elif mode == "gpt":
        response = get_best_match(prompt, dataset_data)
        response = apply_replacements(response)
        wiki = search_wikipedia(prompt)
        if wiki:
            response += f"\n🌐 {wiki}"
        return response

    # ✅ Default QA responses
    if prompt.lower() in qa_responses:
        return apply_replacements(qa_responses[prompt.lower()])

    # 🎧 Quick "play" shortcut outside of music mode
    if prompt.lower().startswith("play "):
        song = prompt[5:].strip()
        yt = search_music_yt(song)
        return f"🎵 {yt}" if yt else "🎵 Not found."

    # 🤖 Default GPT-style fallback
    response = get_best_match(prompt, dataset_data)
    response = apply_replacements(response)

    wiki = search_wikipedia(prompt)
    if wiki:
        response += f"\n🌐 {wiki}"

    return response
