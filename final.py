import torch
from sentence_transformers import SentenceTransformer, util
import easyocr
import cv2
import numpy as np
import os
import re
from difflib import SequenceMatcher

# --- Step 1: Load the Model and Pre-computed Embeddings ---

print("Loading sentence transformer model and embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data we saved from the training script
try:
    saved_data = torch.load('embeddings.pt', map_location=torch.device('cpu'))
    image_filenames = saved_data['filenames']
    image_embeddings = saved_data['embeddings']
    print("✅ Model and pre-computed embeddings loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'embeddings.pt' not found.")
    print("Please run 'create_embeddings.py' first to generate the embeddings file.")
    exit()

# Initialize EasyOCR reader
print("Loading EasyOCR... (This may take a moment on first run)")
reader = easyocr.Reader(['en'])
print("✅ EasyOCR loaded successfully.")

# --- Step 2: The Search Function ---
def find_most_relevant_image(query):
    """
    Takes a user query, embeds it, and finds the most similar image using pre-loaded embeddings.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, image_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    confidence_score = cosine_scores[0][best_match_index].item()
    return image_filenames[best_match_index], confidence_score

# --- Step 3: Extract Words from Query ---
def extract_words_from_query(query):
    """
    Extract meaningful words from the query (excluding common stop words)
    """
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'what', 'which',
                  'who', 'where', 'when', 'why', 'how', 'this', 'that', 'these', 'those'}
    
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    return meaningful_words

# --- Step 3.5: Fuzzy Matching Function ---
def calculate_similarity(word1, word2):
    """
    Calculate similarity between two words using SequenceMatcher.
    Returns a score between 0 and 1.
    """
    # Remove special characters and normalize
    word1_clean = re.sub(r'[^a-zA-Z]', '', word1.lower())
    word2_clean = re.sub(r'[^a-zA-Z]', '', word2.lower())
    
    return SequenceMatcher(None, word1_clean, word2_clean).ratio()

def find_matching_words(query_words, detected_words, threshold=0.6):
    """
    Find which detected words match query words using fuzzy matching.
    Returns a list of (detected_word, matching_query_word, similarity_score) tuples.
    """
    matches = []
    
    for detected in detected_words:
        best_match = None
        best_score = 0
        
        # Clean detected word (remove punctuation, spaces)
        detected_clean = re.sub(r'[^a-zA-Z]', '', detected.lower())
        
        for query_word in query_words:
            # Try direct similarity
            score = calculate_similarity(detected, query_word)
            
            # Also check if query word is contained in detected word or vice versa
            if query_word.lower() in detected_clean or detected_clean in query_word.lower():
                score = max(score, 0.8)
            
            # Check word-by-word for multi-word labels
            detected_parts = detected.lower().split()
            for part in detected_parts:
                part_clean = re.sub(r'[^a-zA-Z]', '', part)
                part_score = calculate_similarity(part_clean, query_word)
                if part_score > score:
                    score = part_score
            
            if score > best_score:
                best_score = score
                best_match = query_word
        
        if best_score >= threshold:
            matches.append((detected, best_match, best_score))
    
    return matches

# --- Step 4: Process Image with OCR and Blank Words ---
def process_image_with_blanking(image_path, query_words, output_path, threshold=0.6):
    """
    Process the image with OCR and blank out words that are NOT in query_words.
    Uses fuzzy matching to find similar words.
    If no matches found, keep all words visible.
    Returns the list of all detected words and matched words.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return [], []
    
    print(f"\nProcessing image with OCR: {image_path}")
    
    # Run OCR on the image
    result = reader.readtext(image_path)
    
    # Load the original image
    image = cv2.imread(image_path)
    
    all_detected_words = []
    
    if result:
        print(f"Text detected! Found {len(result)} text regions.")
        
        # Extract all detected words
        all_detected_words = [text for (bbox, text, prob) in result]
        
        # Find matches using fuzzy matching
        matches = find_matching_words(query_words, all_detected_words, threshold)
        
        # Create a set of words to keep visible
        words_to_keep_set = set([match[0] for match in matches])
        
        # If no matches found, keep all words visible
        if len(matches) == 0:
            print("⚠️  No matching words found. Showing all labels.")
            words_to_keep_set = set(all_detected_words)
        else:
            print(f"\n🎯 Found {len(matches)} matching labels:")
            for detected, query, score in sorted(matches, key=lambda x: x[2], reverse=True):
                print(f"   '{detected}' ← matches '{query}' (similarity: {score:.2f})")
        
        for (bbox, text, prob) in result:
            # Check if this word should be kept visible
            should_blank = text not in words_to_keep_set
            
            box_points = np.array(bbox, dtype=np.int32)
            
            if should_blank:
                # Fill the box with solid color to hide the word
                cv2.fillPoly(
                    image,
                    [box_points],
                    color=(255, 255, 255)  # White fill to blank out the text
                )
                
                # Draw a border around the blanked box
                cv2.polylines(
                    image, 
                    [box_points], 
                    isClosed=True, 
                    color=(200, 200, 200),  # Light gray border
                    thickness=2
                )
                print(f"  Blanked: {text}")
            else:
                # Draw a green outline for words to keep
                cv2.polylines(
                    image, 
                    [box_points], 
                    isClosed=True, 
                    color=(0, 255, 0),  # Green box
                    thickness=2
                )
                print(f"  Kept visible: {text}")
        
        # Save the processed image
        cv2.imwrite(output_path, image)
        print(f"\n✅ Saved processed image to: {output_path}")
        
        return all_detected_words, matches
    else:
        print("No text was detected in the image.")
        # Just save the original image
        cv2.imwrite(output_path, image)
        return [], []

# --- Step 5: Main Interactive Application ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   INTERACTIVE IMAGE MAPPER WITH SMART BLANKING")
    print("="*60)
    print("\nHow it works:")
    print("1. Enter a biology question or sentence")
    print("2. System finds the most relevant diagram")
    print("3. Extracts words from your question")
    print("4. Shows only matching labels in the diagram (blanks others)")
    print("5. If no words match, shows all labels")
    print("\nType 'exit' or 'quit' to close the program.\n")

    while True:
        print("-" * 60)
        user_query = input("\nYour Question: ").strip()
        
        if user_query.lower() in ['exit', 'quit']:
            print("\nThank you for using the Image Mapper. Goodbye!")
            break
        
        if not user_query:
            print("Please enter a question.")
            continue
        
        # Step 1: Find the most relevant image
        print("\n🔍 Finding relevant image...")
        mapped_image, score = find_most_relevant_image(user_query)
        print(f"✅ Best Match: {mapped_image}")
        print(f"   Confidence Score: {score:.2f}")
        
        # Step 2: Extract words from the query
        query_words = extract_words_from_query(user_query)
        print(f"\n📝 Words extracted from query: {query_words}")
        
        # Step 3: Build the image path
        image_path = os.path.join("chapter5", mapped_image)
        output_path = f"output_{mapped_image}"
        
        # Step 4: Process the image with OCR (using fuzzy matching threshold of 0.6)
        detected_words, matches = process_image_with_blanking(image_path, query_words, output_path, threshold=0.6)
        
        # Step 5: Prepare summary
        print(f"\n📊 Summary:")
        print(f"   Total words detected in image: {len(detected_words)}")
        print(f"   Words from query that matched: {len(matches)}")
        
        if matches:
            matched_labels = [match[0] for match in matches]
            print(f"   Matched labels: {matched_labels}")
            print(f"   → Kept these visible, blanked the rest")
        else:
            print(f"   No matches found")
            print(f"   → Showing all labels: {detected_words}")
        
        print(f"\n✨ Output saved to: {output_path}")
