import json
import os
import time
import re
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# --- Configuration ---
try:
    genai.configure(api_key="AIzaSyAtNxpha-7A0LF0FbR0jvVH8IO-IQ4QqUw")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

INPUT_FILENAME = "textual_content.json"
OUTPUT_FILENAME = "syllabus_map_1.json"
CHAPTERS_TO_PROCESS = [4,7,8]

MASTER_PROMPT = """
You are an expert scientific data processor specializing in CBSE Class 10 Science textbooks. 
Your task is to analyze the provided textbook excerpt on a specific subtopic and extract all key concepts relevant for students studying for board exams.

These key concepts will be later used to generate question-answer pairs for CBSE Class 10 Science students. 
Include all definitions, core principles / laws / formulas, chemical equations, and examples / observations. 
Focus on content that is relevant for exam questions.

**Output Format Constraints:**
* Return a single JSON array of strings, with each string being a complete key concept exactly as it appears in the text.
* Do not add information not present in the text.
* Skip meta-text: introductions, chapter overviews, instructions, or other non-informative sentences.
* Do not number items, do not include extra commentary.
* Output only the raw JSON array.

**Note:** Only analyze the text provided in this input. Do not attempt to recall content from other chapters or subtopics.

**Example Output Format:**
`["Concept one as a full sentence.", "Another concept as a full sentence.", "$CaO(s) + H_2O(l) \\\\rightarrow Ca(OH)_2(aq) + Heat$"]`

**Text to process:**  
"""

def get_key_concepts_from_gemini(text_content):
    """Send text to Gemini 2.5-Pro API and return key concepts with robust retries."""
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = MASTER_PROMPT + text_content
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            key_concepts = json.loads(cleaned_text)
            # Respect free-tier per-minute quota: max 2 requests/min
            time.sleep(35)
            return key_concepts
        except ResourceExhausted as e:
            # Quota exceeded (per-minute or daily)
            retry_after = getattr(e, "retry_delay", 30)
            print(f"   - Quota exceeded. Waiting {retry_after:.1f} seconds before retrying...")
            time.sleep(retry_after)
        except json.JSONDecodeError:
            print(f"   - JSON parsing failed. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"   - API call failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    print("   - Max retries reached. Logging as error placeholder.")
    return ["ERROR: Could not process this text."]

def clean_heading_name(heading):
    return re.sub(r'^\d+(\.\d+)*\s*', '', heading).strip()

def process_headings_into_structure(content_list):
    topics = []
    topic_map = {}

    for item in content_list:
        heading = item.get("heading_name", "")
        if heading.count('.') < 2:
            topic_obj = {
                "topic_name": clean_heading_name(heading),
                "key_concepts": item.get("key_concepts", []),
                "questions": [],
                "activity_contexts": [],
                "subtopics": []
            }
            topics.append(topic_obj)
            if heading.split(' ')[0]:
                topic_map[heading.split(' ')[0]] = topic_obj

    for item in content_list:
        heading = item.get("heading_name", "")
        if heading.count('.') >= 2:
            parent_key = ".".join(heading.split(' ')[0].split('.')[:2])
            if parent_key in topic_map:
                subtopic_obj = {
                    "subtopic_name": clean_heading_name(heading),
                    "key_concepts": item.get("key_concepts", []),
                    "questions": [],
                    "activity_contexts": []
                }
                topic_map[parent_key]["subtopics"].append(subtopic_obj)
            else:
                print(f"  - Warning: Parent for '{heading}' not found. Treating as main topic.")
                topic_obj = {
                    "topic_name": clean_heading_name(heading),
                    "key_concepts": item.get("key_concepts", []),
                    "questions": [],
                    "activity_contexts": [],
                    "subtopics": []
                }
                topics.append(topic_obj)
    return topics

def main():
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            all_chapters_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILENAME}' not found.")
        return

    final_syllabus_map = []
    failed_headings = []

    print("Starting automated key concept extraction using Gemini 2.5-Pro...")

    for chapter_data in all_chapters_data:
        chapter_num = chapter_data.get("chapter_number")
        chapter_name = chapter_data.get("chapter_name")
        print(f"\n--- Processing Chapter {chapter_num}: {chapter_name} ---")

        processed_content_list = []

        if chapter_num in CHAPTERS_TO_PROCESS:
            for content_item in chapter_data.get("content", []):
                heading = content_item.get("heading_name")
                raw_text = content_item.get("raw_text", "").strip()

                print(f"\nProcessing heading: {heading}")
                if raw_text:
                    concepts = get_key_concepts_from_gemini(raw_text)
                    if "ERROR" in concepts[0]:
                        failed_headings.append(f"{chapter_num} - {heading}")
                    print(f"   - Extracted {len(concepts)} key concepts.")
                else:
                    print("   - No text found. Skipping.")
                    concepts = []

                processed_content_list.append({
                    "heading_name": heading,
                    "key_concepts": concepts
                })

        nested_topics = process_headings_into_structure(processed_content_list)

        new_chapter_obj = {
            "chapter_number": chapter_num,
            "chapter_name": chapter_name,
            "topics": nested_topics
        }
        final_syllabus_map.append(new_chapter_obj)

    if failed_headings:
        print("\nThe following headings failed and may need manual review:")
        for fh in failed_headings:
            print(f"   - {fh}")

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(final_syllabus_map, f, indent=2, ensure_ascii=False)

    print(f"\nSuccess! Output saved to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()
