"""
JSON Structure Validator
Checks if your JSON file matches the expected format
"""

import json
import sys

def validate_structure(json_file_path):
    """Validate and display JSON structure"""
    
    print("="*60)
    print("JSON Structure Validator")
    print("="*60)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✅ Valid JSON file loaded")
        print(f"Root type: {type(data).__name__}")
        
        # Determine structure
        if isinstance(data, list):
            print(f"Root is a LIST with {len(data)} item(s)")
            
            if len(data) == 0:
                print("❌ ERROR: List is empty!")
                return
            
            # Check first item
            first_item = data[0]
            print(f"\nFirst item type: {type(first_item).__name__}")
            
            if isinstance(first_item, dict):
                print("\n📋 First item keys:")
                for key in first_item.keys():
                    print(f"  - {key}: {type(first_item[key]).__name__}")
                
                # Analyze as chapters
                analyze_chapters(data)
            else:
                print(f"❌ ERROR: Expected dict, got {type(first_item).__name__}")
        
        elif isinstance(data, dict):
            print("Root is a DICT (single chapter)")
            print("\n📋 Root keys:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key]).__name__}")
            
            analyze_chapters([data])
        
        else:
            print(f"❌ ERROR: Root must be list or dict, got {type(data).__name__}")
    
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{json_file_path}' not found!")
    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Invalid JSON format!")
        print(f"   {str(e)}")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

def analyze_chapters(chapters):
    """Analyze chapter structure"""
    
    print("\n" + "="*60)
    print("CHAPTER ANALYSIS")
    print("="*60)
    
    for i, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            print(f"\n❌ Chapter {i}: Not a dict! Type: {type(chapter).__name__}")
            continue
        
        chapter_num = chapter.get('chapter_number', '?')
        chapter_name = chapter.get('chapter_name', 'Unknown')
        
        print(f"\n📖 Chapter {chapter_num}: {chapter_name}")
        
        # Check topics
        topics = chapter.get('topics', [])
        print(f"   Topics: {len(topics)}")
        
        if not topics:
            print("   ⚠️  No topics found!")
            continue
        
        for j, topic in enumerate(topics):
            if not isinstance(topic, dict):
                print(f"   ❌ Topic {j}: Not a dict!")
                continue
            
            topic_name = topic.get('topic_name', 'Unknown')
            key_concepts = topic.get('key_concepts', [])
            questions = topic.get('questions', [])
            subtopics = topic.get('subtopics', [])
            activity_contexts = topic.get('activity_contexts', [])
            
            print(f"\n   📌 Topic {j+1}: {topic_name}")
            print(f"      - Key concepts: {len(key_concepts)}")
            print(f"      - Question sets: {len(questions)}")
            print(f"      - Subtopics: {len(subtopics)}")
            print(f"      - Activity contexts: {len(activity_contexts)}")
            
            # Count questions by mark
            if questions:
                for q_set in questions:
                    if isinstance(q_set, dict):
                        for mark_type in ['one_mark_questions', 'two_mark_questions', 
                                        'three_mark_questions', 'four_mark_questions', 
                                        'five_mark_questions']:
                            if mark_type in q_set:
                                count = len(q_set[mark_type])
                                if count > 0:
                                    print(f"         • {mark_type}: {count}")
            
            # Analyze subtopics
            if subtopics:
                for k, subtopic in enumerate(subtopics):
                    if isinstance(subtopic, dict):
                        sub_name = subtopic.get('subtopic_name', 'Unknown')
                        sub_concepts = len(subtopic.get('key_concepts', []))
                        sub_questions = len(subtopic.get('questions', []))
                        
                        print(f"      └─ Subtopic {k+1}: {sub_name}")
                        print(f"         - Key concepts: {sub_concepts}")
                        print(f"         - Question sets: {sub_questions}")
    
    print("\n" + "="*60)
    print("✅ STRUCTURE VALIDATION COMPLETE")
    print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_validator.py <path_to_json_file>")
        print("\nExample: python json_validator.py syllabus_map_1.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    validate_structure(json_file)

if __name__ == "__main__":
    main()