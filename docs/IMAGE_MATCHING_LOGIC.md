# Image Matching Logic - Updated

## Overview
The image processing system has been updated with a **simpler and more predictable** word matching algorithm that uses exact word matching instead of fuzzy similarity scores.

## How It Works

### 1. **Word Extraction from Query**
- Extracts meaningful words from the user's question
- Removes common stop words (the, a, is, etc.)
- Only keeps words with 3+ characters

### 2. **OCR Detection**
- Uses EasyOCR to detect all text in the selected diagram
- Extracts bounding boxes for each detected word/phrase

### 3. **Intelligent Matching Logic**

#### **Exact Word Matching**
The system now uses simple, predictable exact matching:

```
Query Word: "oxygen"
OCR Text: "oxygen" -> MATCH (exact)
OCR Text: "Oxygen" -> MATCH (case-insensitive)
OCR Text: "oxygenated" -> MATCH (substring, 4+ chars)
OCR Text: "O2" -> NO MATCH (different)
```

#### **Matching Rules**
1. **Exact match**: Normalized query word exactly matches normalized OCR text
2. **Substring match**: Query word (4+ chars) is contained in OCR text or vice versa
3. **Multi-word OCR**: Each word in multi-word OCR text is checked separately

### 4. **Display Behavior**

#### **Scenario A: Exact Matches Found**
```
Query: "What is photosynthesis?"
Query Words: ["photosynthesis"]
OCR Detected: ["leaf", "chloroplast", "photosynthesis", "sunlight", "water"]
Matches: ["photosynthesis"]

Result:
Show ONLY "photosynthesis" (GREEN outline)
Blank out: "leaf", "chloroplast", "sunlight", "water"
```

#### **Scenario B: No Matches Found**
```
Query: "Explain cellular respiration"
Query Words: ["explain", "cellular", "respiration"]
OCR Detected: ["mitochondria", "glucose", "ATP", "energy"]
Matches: [] (none)

Result:
Show ALL words (ORANGE outline)
Display all detected words: "mitochondria", "glucose", "ATP", "energy"
```

## Color Coding

| Color | Meaning |
|-------|---------|
| **Green** | Exact word matches from query (highlighted, rest blanked) |
| **Orange** | All words visible (no matches found) |
| **White/Blanked** | Words that don't match (when green matches exist) |

## Advantages of New Logic

### **Predictable**
- No fuzzy thresholds or similarity scores
- Easy to understand: exact word = match, no word = show all

### **Intelligent**
- Automatically shows all labels when query doesn't match
- Students won't miss important diagram information

### **Simple**
- No complex algorithms
- Fast processing
- Consistent results

### **Educational**
- Students see relevant labels highlighted
- Or see complete diagram with all labels when exploring new topics

## Examples

### Example 1: Biology Question
```
Question: "What are the parts of a plant cell?"
Keywords: ["parts", "plant", "cell"]

Diagram OCR: ["nucleus", "cell wall", "chloroplast", "vacuole", "mitochondria"]
Match Found: "cell" in "cell wall"

Result: Shows "cell wall" (green), blanks others
```

### Example 2: Chemistry Question
```
Question: "Explain the water cycle"
Keywords: ["explain", "water", "cycle"]

Diagram OCR: ["evaporation", "condensation", "precipitation", "collection"]
Match Found: None

Result: Shows ALL words (orange) - "evaporation", "condensation", "precipitation", "collection"
```

### Example 3: Physics Question
```
Question: "How does a circuit work?"
Keywords: ["circuit", "work"]

Diagram OCR: ["battery", "circuit", "bulb", "switch", "wire"]
Match Found: "circuit"

Result: Shows "circuit" (green), blanks others
```

## Technical Implementation

### Word Normalization
```python
def _normalize_word(word):
    # Remove special characters, convert to lowercase
    return re.sub(r'[^a-z0-9]', '', word.lower())
    
    # "Oxygen!" -> "oxygen"
    # "CO2" -> "co2"
    # "Cell-Wall" -> "cellwall"
```

### Matching Algorithm
```python
def _find_matching_words(query_words, detected_words):
    matches = []
    
    for detected in detected_words:
        # 1. Check exact match
        # 2. Check substring match (4+ chars)
        # 3. Check multi-word separately
        
        if match_found:
            matches.append((detected, query_word, 1.0))
    
    return matches
```

### Display Logic
```python
if len(matches) > 0:
    # Show only matched words (green)
    show_words = matched_words
    color = GREEN
else:
    # Show all words (orange)
    show_words = all_detected_words
    color = ORANGE
```

## Benefits for Students

1. **Clear Feedback**: Students know exactly which words matched
2. **Complete Information**: Never lose important labels due to no matches
3. **Learning Support**: Highlighted terms help focus on relevant concepts
4. **Visual Clarity**: Color coding makes it obvious what's happening

## Usage in Web Interface

The web interface displays:
- Green box: "Exact Word Matches Found" (when matches exist)
- Orange box: "No Exact Matches - Showing All Words" (when no matches)
- List of matched words or all detected words
- Processed diagram with appropriate color coding

## Conclusion

The new matching logic is **simple, predictable, and educational**. It ensures students always get useful visual information, whether through targeted highlights or complete labeled diagrams.
