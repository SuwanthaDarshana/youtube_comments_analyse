import re
import string
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from youtube_comment_downloader import YoutubeCommentDownloader
from googletrans import Translator

# ---- Helper: Extract YouTube Video ID ----
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

# ---- Comment Extraction ----
def get_youtube_comments(video_url, max_comments=100):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        downloader = YoutubeCommentDownloader()
        comments = []
        for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
            comments.append(comment['text'])
            if len(comments) >= max_comments:
                break
        return comments
    except Exception as e:
        print("Error while fetching comments:", e)
        return []

# ---- Translation ----
translator = Translator()
def is_sinhala(text):
    return bool(re.search(r'[\u0D80-\u0DFF]', text))

def translate_to_sinhala_if_needed(comments):
    translated_comments = []
    for comment in comments:
        comment = comment.strip()
        if not comment:
            continue
        if is_sinhala(comment):
            translated_comments.append(comment)
        else:
            try:
                translation = translator.translate(comment, src='auto', dest='si')
                translated_comments.append(translation.text.strip())
            except Exception as e:
                print(f"Translation failed: {e}")
                translated_comments.append(comment)
    return translated_comments

# ---- Preprocessing ----
sinhala_stopwords = set("""සහ සමග සමඟ අහා ආහ් ආ ඕහෝ අනේ අඳෝ අපොයි අපෝ අයියෝ ආයි ඌයි චී චිහ් චික් හෝ‍ දෝ දෝහෝ මෙන්
සේ වැනි බඳු වන් අයුරු අයුරින් ලෙස වැඩි ශ්‍රී හා ය නිසා නිසාවෙන් බවට බව බවෙන් නම් වැඩි සිටදී මහා මහ
පමණ පමණින් පමන වන විට විටින් මේ මෙලෙස මෙයින් ඇති ලෙස සිදු වශයෙන් යන සඳහා මගින් හෝ‍ ඉතා ඒ එම ද
අතර විසින් සමග පිළිබඳව පිළිබඳ තුළ බව වැනි මහ මෙම මෙහි මේ වෙත වෙතින් වෙතට වෙනුවෙන් වෙනුවට
වෙන ගැන නෑ අනුව නව පිළිබඳ විශේෂ දැනට එහෙන් මෙහෙන් එහේ මෙහේ ම තවත් තව දක්වා ට ගේ එ ක ක්
බවත් බවද මත ඇතුලු ඇතුළු මෙසේ වැඩි වඩා වඩාත්ම නිති නිතිත් නිතොර නිතර ඉක්බිති දැන් යලි පුන ඉතින් සිට
සිටන් පටන් තෙක් දක්වා සා තාක් තුවක් පවා ද හෝ‍ වත් විනා හැර මිස මුත් කිම කිම් ඇයි මන්ද හෙවත් නොහොත්
පතා පාසා ගානෙ තව ඉතා බොහෝ වහා සෙද සැනින් හනික එම්බා එම්බල බොල නම් වනාහි කලී ඉඳුරා අන්න ඔන්න
මෙන්න උදෙසා පිණිස සඳහා අරබයා නිසා එනිසා එබැවින් බැවින් හෙයින් සේක් සේක ගැන අනුව පරිදි විට තෙක්
මෙතෙක් මේතාක් තුරු තුරා තුරාවට තුලින් නමුත් එනමුත් වස් මෙන් ලෙස පරිදි එහෙත්""".split())  

def remove_links(text): return re.sub(r'https?://\S+|www\.\S+', '', text)
def remove_english(text): return re.sub(r'[a-zA-Z]', '', text)
def remove_punctuations(text): return text.translate(str.maketrans('', '', string.punctuation))
def remove_numbers(text): return re.sub(r'\d+', '', text)
def remove_non_sinhala(text): return re.sub(r'[^\u0D80-\u0DFF\s]', '', str(text))
def remove_emojis(text):
    emoji_pattern = re.compile("[" +
        u"\U0001F600-\U0001F64F" +
        u"\U0001F300-\U0001F5FF" +
        u"\U0001F680-\U0001F6FF" +
        u"\U0001F1E0-\U0001F1FF" +
        u"\U00002700-\U000027BF" +
        u"\U000024C2-\U0001F251" +
        "]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_sinhala_stopwords(text):
    return " ".join([word for word in text.split() if word not in sinhala_stopwords])

def full_clean(text):
    text = text.strip()
    text = remove_links(text)
    text = remove_emojis(text)
    text = remove_english(text)
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_non_sinhala(text)
    text = remove_sinhala_stopwords(text)
    return text.strip()

def preprocess_comments(comments):
    return [full_clean(c) for c in comments if c.strip()]

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained("./notebooks/saved_model", local_files_only=True)
def tokenize_combined_text(text):
    return tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

# ---- Load Model and LabelEncoder ----
model = AutoModelForSequenceClassification.from_pretrained("./notebooks/saved_model", local_files_only=True)
label_encoder = joblib.load("./notebooks/saved_model/label_encoder.joblib")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---- Main Function ----
def predict_youtube_category(video_url):
    print("\n🔍 Fetching comments...")
    comments = get_youtube_comments(video_url)
    if not comments:
        print("❌ No comments fetched.")
        return None

    print(f"🗣 Translating and cleaning {len(comments)} comments...")
    comments = translate_to_sinhala_if_needed(comments)
    cleaned_comments = preprocess_comments(comments)

    print("\n🧹 Preprocessed Comments Sample:")
    for i, comment in enumerate(cleaned_comments[:100], start=1):
        print(f"{i}. {comment}")

    if not cleaned_comments:
        print("❌ All comments were empty after cleaning.")
        return None

    combined_text = " ".join(cleaned_comments)
    inputs = tokenize_combined_text(combined_text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("🤖 Predicting category...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    print(f"✅ Predicted Category: **{predicted_label}**")
    return predicted_label
