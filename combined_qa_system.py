"""
Combined Science Q&A System with Image Mapping
----------------------------------------------
✅ Runs RAG text generation and image mapping in parallel
✅ Takes questions in RAG format (with optional marks)
✅ Returns both text answer and relevant diagram
✅ Saves processed image with highlighted labels

Requires:
    pip install ollama chromadb sentence-transformers easyocr opencv-python torch
"""

import json
import ollama
import chromadb
from chromadb.utils import embedding_functions
import torch
from sentence_transformers import SentenceTransformer, util
import easyocr
import cv2
import numpy as np
import os
import re
import glob
import time
from difflib import SequenceMatcher
import concurrent.futures
import threading


class CombinedScienceQA:
    def __init__(self, json_file_path=None, model_name="llama3.2", db_path="./science_db", 
                 embeddings_path="embeddings.pt", images_folder="chapter5"):
        """
        Initialize the combined system with both RAG and image mapping capabilities.
        """
        self.model_name = model_name
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        self.images_folder = images_folder
        
        # Initialize RAG system
        print("🔗 Initializing RAG system...")
        self._init_rag_system(json_file_path)
        
        # Initialize Image Mapping system
        print("🖼️  Initializing Image Mapping system...")
        self._init_image_mapping()
        
        print("✅ Combined system ready!\n")
    
    # ==================== RAG SYSTEM INITIALIZATION ====================
    
    def _init_rag_system(self, json_file_path):
        """Initialize the RAG system with persistent ChromaDB storage."""
        print(f"   Using persistent ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # SentenceTransformer for embeddings
        self.rag_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get existing collection
        self.collection = self.client.get_or_create_collection(
            name="science_qa",
            embedding_function=self.rag_embedding_function
        )
        
        existing_count = self.collection.count()
        
        if existing_count == 0 and json_file_path:
            print("   🧩 No data found — indexing from JSON...")
            self._load_rag_data(json_file_path)
            print(f"   ✅ Indexed {self.collection.count()} documents in total.")
        elif existing_count > 0:
            print(f"   📦 Loaded existing database with {existing_count} documents.")
        else:
            raise ValueError("   ❌ No database found and no JSON file provided!")
    
    def _init_image_mapping(self):
        """Initialize the image mapping system with pre-computed embeddings."""
        print(f"   Loading image embeddings from: {self.embeddings_path}")
        
        # Load sentence transformer model for images
        self.image_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load pre-computed embeddings
        try:
            saved_data = torch.load(self.embeddings_path, map_location=torch.device('cpu'))
            self.image_filenames = saved_data['filenames']
            self.image_embeddings = saved_data['embeddings']
            print(f"   ✅ Loaded {len(self.image_filenames)} image embeddings.")
        except FileNotFoundError:
            print(f"   ❌ Error: '{self.embeddings_path}' not found.")
            print("   Please run 'create_embeddings.py' first to generate the embeddings file.")
            raise
        
        # Initialize EasyOCR reader
        print("   Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        print("   ✅ EasyOCR loaded successfully.")
    
    # ==================== RAG SYSTEM METHODS ====================
    
    def _process_questions(self, questions_list, metadata_base, doc_id_start):
        """Process questions from JSON structure."""
        documents, metadatas, ids = [], [], []
        doc_id = doc_id_start

        for q_set in questions_list:
            for mark_type in ['one_mark_questions', 'two_mark_questions',
                              'three_mark_questions', 'four_mark_questions', 'five_mark_questions']:
                if mark_type in q_set:
                    for qa in q_set[mark_type]:
                        question = qa.get('question', '')
                        answer = qa.get('answer', '')
                        if not question or not answer:
                            continue

                        doc_text = f"Question: {question}\nAnswer: {answer}"
                        documents.append(doc_text)

                        marks = mark_type.split('_')[0]
                        meta = metadata_base.copy()
                        meta.update({
                            'type': 'qa',
                            'marks': marks,
                            'question': question,
                            'answer': answer
                        })
                        metadatas.append(meta)
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1
        return documents, metadatas, ids, doc_id

    def _process_subtopics(self, subtopics_list, metadata_base, doc_id_start, level=1):
        """Process subtopics from JSON structure."""
        documents, metadatas, ids = [], [], []
        doc_id = doc_id_start

        for subtopic in subtopics_list:
            subtopic_name = subtopic.get('subtopic_name', 'Unknown')
            meta_base = metadata_base.copy()
            meta_base[f'subtopic_level_{level}'] = subtopic_name

            key_concepts = subtopic.get('key_concepts', [])
            if key_concepts:
                concept_text = "\n".join([str(c) for c in key_concepts if c])
                if concept_text.strip():
                    documents.append(concept_text)
                    meta = meta_base.copy()
                    meta['type'] = 'key_concepts'
                    metadatas.append(meta)
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1

            questions = subtopic.get('questions', [])
            if questions:
                q_docs, q_metas, q_ids, doc_id = self._process_questions(
                    questions, meta_base, doc_id)
                documents.extend(q_docs)
                metadatas.extend(q_metas)
                ids.extend(q_ids)

            activity_contexts = subtopic.get('activity_contexts', [])
            if activity_contexts:
                activity_text = "\n".join([str(a) for a in activity_contexts if a])
                if activity_text.strip():
                    documents.append(activity_text)
                    meta = meta_base.copy()
                    meta['type'] = 'activity'
                    metadatas.append(meta)
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1

            nested_subtopics = subtopic.get('subtopics', [])
            if nested_subtopics:
                nested_docs, nested_metas, nested_ids, doc_id = self._process_subtopics(
                    nested_subtopics, meta_base, doc_id, level + 1)
                documents.extend(nested_docs)
                metadatas.extend(nested_metas)
                ids.extend(nested_ids)

        return documents, metadatas, ids, doc_id

    def _load_rag_data(self, json_file_path):
        """Load JSON data and create vector embeddings for RAG."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chapters = data if isinstance(data, list) else [data]
        documents, metadatas, ids, doc_id = [], [], [], 0

        for chapter in chapters:
            chapter_num = chapter.get('chapter_number', 'Unknown')
            chapter_name = chapter.get('chapter_name', 'Unknown')

            for topic in chapter.get('topics', []):
                topic_name = topic.get('topic_name', 'Unknown')
                metadata_base = {
                    'chapter': str(chapter_num),
                    'chapter_name': chapter_name,
                    'topic': topic_name
                }

                key_concepts = topic.get('key_concepts', [])
                if key_concepts:
                    concept_text = "\n".join([str(c) for c in key_concepts if c])
                    if concept_text.strip():
                        documents.append(concept_text)
                        meta = metadata_base.copy()
                        meta['type'] = 'key_concepts'
                        metadatas.append(meta)
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1

                questions = topic.get('questions', [])
                if questions:
                    q_docs, q_metas, q_ids, doc_id = self._process_questions(
                        questions, metadata_base, doc_id)
                    documents.extend(q_docs)
                    metadatas.extend(q_metas)
                    ids.extend(q_ids)

                activity_contexts = topic.get('activity_contexts', [])
                if activity_contexts:
                    activity_text = "\n".join([str(a) for a in activity_contexts if a])
                    if activity_text.strip():
                        documents.append(activity_text)
                        meta = metadata_base.copy()
                        meta['type'] = 'activity'
                        metadatas.append(meta)
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1

                subtopics = topic.get('subtopics', [])
                if subtopics:
                    sub_docs, sub_metas, sub_ids, doc_id = self._process_subtopics(
                        subtopics, metadata_base, doc_id)
                    documents.extend(sub_docs)
                    metadatas.extend(sub_metas)
                    ids.extend(sub_ids)

        # Add to ChromaDB in batches
        if documents:
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
    
    def _retrieve_context(self, query, marks=None, n_results=5):
        """Retrieve context from ChromaDB."""
        where_filter = {"marks": marks} if marks else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def _generate_text_answer(self, query, marks=None):
        """Generate text answer using RAG system."""
        # Set default marks to 4 if not provided
        if marks is None:
            marks = "four"
        
        # For multi-part questions, retrieve more results to cover all topics
        # Check if query contains "and" suggesting multiple topics
        n_results = 10 if ' and ' in query.lower() else 5
        
        results = self._retrieve_context(query, marks=marks, n_results=n_results)

        # Check if no relevant information found (out of syllabus)
        if not results or not results.get('documents') or not results['documents'][0]:
            return "This topic appears to be out of syllabus. No relevant information found in the curriculum database."

        # Additional relevance check using semantic similarity
        # Use the sentence-transformer model to compute embeddings
        # and check cosine similarity between the query and retrieved documents.
        # If best cosine similarity is below threshold, treat as not relevant.
        try:
            docs_list = results.get('documents', [[]])[0]
            if docs_list:
                # Encode query and documents
                query_emb = self.image_model.encode(query, convert_to_tensor=True)
                docs_emb = self.image_model.encode(docs_list, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, docs_emb)
                # cos_scores is tensor shape [1, N]
                best_sim = float(torch.max(cos_scores).item())
                min_similarity_threshold = 0.30  # Stricter threshold
                if best_sim < min_similarity_threshold:
                    return "This question does not appear to be relevant to the curriculum. Please ask questions related to the science syllabus."
        except Exception:
            # If embedding/similarity check fails, continue with existing behavior
            pass

        # Build the context
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            chapter_info = f"Chapter {metadata.get('chapter', '')}: {metadata.get('chapter_name', '')}"
            topic_info = f"Topic: {metadata.get('topic', '')}"
            subtopic_levels = [k for k in metadata if k.startswith('subtopic_level_')]
            if subtopic_levels:
                subtopic_info = " > ".join([metadata[k] for k in sorted(subtopic_levels)])
                topic_info += f" > {subtopic_info}"
            context_parts.append(f"[{chapter_info} | {topic_info}]\n{doc}")

        context = "\n\n".join(context_parts)

        # Calculate number of points: 2 * marks
        marks_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        marks_value = marks_map.get(marks, 4)
        num_points = 2 * marks_value

        # Generate prompt with point-based structure
        prompt = f"""You are a science teacher helping students prepare for exams.

Context from textbook:
{context}

Student's Question ({marks} marks): {query}

CRITICAL Instructions:
- Provide a direct answer with NO introductory phrases like "Here's your answer" or "Let me explain"
- Structure the answer as EXACTLY {num_points} clear, concise points
- Number each point (1., 2., 3., etc.)
- Each point should be factual and relevant
- Keep the depth appropriate for a {marks}-mark question
- Do NOT include any conversational elements
- Start directly with point 1

IMPORTANT: If the question asks about MULTIPLE topics (e.g., "stomach and liver", "photosynthesis and respiration"), you MUST cover ALL topics mentioned in the question. Divide the {num_points} points proportionally between all topics.

Answer:"""

        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response']
    
    # ==================== IMAGE MAPPING METHODS ====================
    
    def _find_most_relevant_image(self, query):
        """Find the most relevant image using pre-loaded embeddings."""
        query_embedding = self.image_model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.image_embeddings)
        best_match_index = torch.argmax(cosine_scores).item()
        confidence_score = cosine_scores[0][best_match_index].item()
        return self.image_filenames[best_match_index], confidence_score
    
    def _extract_words_from_query(self, query):
        """Extract meaningful words from the query (excluding common stop words)."""
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'what', 'which',
                      'who', 'where', 'when', 'why', 'how', 'this', 'that', 'these', 'those'}
        
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        return meaningful_words
    
    def _normalize_word(self, word):
        """Normalize a word by removing special characters and converting to lowercase."""
        return re.sub(r'[^a-z0-9]', '', word.lower())
    
    def _find_matching_words(self, query_words, detected_words):
        """
        Find which detected words match query words using simple exact word matching.
        
        Logic:
        1. If an exact word from the query matches a word in OCR text, it's highlighted
        2. Handles multi-word OCR text by splitting and checking each word
        3. If NO matches found, returns empty list (which will show all words)
        
        Returns:
            List of tuples: (detected_text, matched_query_word, confidence=1.0)
        """
        matches = []
        
        # Normalize all query words for comparison
        normalized_query_words = {self._normalize_word(qw): qw for qw in query_words}
        
        for detected in detected_words:
            # Check if the entire detected text matches any query word
            detected_normalized = self._normalize_word(detected)
            
            # Direct exact match
            if detected_normalized in normalized_query_words:
                original_query_word = normalized_query_words[detected_normalized]
                matches.append((detected, original_query_word, 1.0))
                continue
            
            # Check if any query word is contained in the detected text
            # or vice versa (for partial matches like "oxygen" in "oxygenated")
            match_found = False
            for norm_qw, original_qw in normalized_query_words.items():
                # Check if query word is substring of detected word (at least 4 chars)
                if len(norm_qw) >= 4 and norm_qw in detected_normalized:
                    matches.append((detected, original_qw, 1.0))
                    match_found = True
                    break
                # Check if detected word is substring of query word (at least 4 chars)
                elif len(detected_normalized) >= 4 and detected_normalized in norm_qw:
                    matches.append((detected, original_qw, 1.0))
                    match_found = True
                    break
            
            if match_found:
                continue
            
            # For multi-word detected text, check each word separately
            detected_words_list = detected.lower().split()
            for detected_word in detected_words_list:
                detected_word_normalized = self._normalize_word(detected_word)
                
                # Exact match with any query word
                if detected_word_normalized in normalized_query_words:
                    original_query_word = normalized_query_words[detected_word_normalized]
                    matches.append((detected, original_query_word, 1.0))
                    break
                
                # Substring match (at least 4 chars)
                for norm_qw, original_qw in normalized_query_words.items():
                    if len(norm_qw) >= 4 and norm_qw in detected_word_normalized:
                        matches.append((detected, original_qw, 1.0))
                        break
                    elif len(detected_word_normalized) >= 4 and detected_word_normalized in norm_qw:
                        matches.append((detected, original_qw, 1.0))
                        break
        
        return matches
    
    def _process_image_with_blanking(self, image_path, query_words, output_path):
        """
        Process the image with OCR and intelligent word highlighting.
        
        Logic:
        - If exact words from the query match OCR text → show ONLY those, blank the rest
        - If NO words match → show ALL words detected (no blanking)
        
        Args:
            image_path: Path to the input image
            query_words: List of words extracted from the query
            output_path: Path to save the processed image
            
        Returns:
            Dictionary with detection results and match information
        """
        if not os.path.exists(image_path):
            return {
                'error': f"Image not found at {image_path}",
                'all_detected_words': [],
                'matches': []
            }
        
        # Run OCR on the image
        result = self.reader.readtext(image_path)
        
        # Load the original image
        image = cv2.imread(image_path)
        
        all_detected_words = []
        matches = []
        
        if result:
            # Extract all detected words
            all_detected_words = [text for (bbox, text, prob) in result]
            
            # Find matches using simple exact word matching
            matches = self._find_matching_words(query_words, all_detected_words)
            
            # INTELLIGENT DECISION:
            # If matches found → show only matched words (blank the rest)
            # If NO matches → show all words (no blanking)
            if len(matches) > 0:
                # Show only matched words
                words_to_keep_set = set([match[0] for match in matches])
                show_all = False
            else:
                # No matches, show all words
                words_to_keep_set = set(all_detected_words)
                show_all = True
            
            for (bbox, text, prob) in result:
                box_points = np.array(bbox, dtype=np.int32)
                
                if text in words_to_keep_set:
                    # Keep this word visible with green outline
                    if show_all:
                        # Show all mode - use blue outline
                        cv2.polylines(image, [box_points], isClosed=True, 
                                    color=(255, 165, 0), thickness=2)  # Orange for "show all"
                    else:
                        # Matched word - use green outline
                        cv2.polylines(image, [box_points], isClosed=True, 
                                    color=(0, 255, 0), thickness=3)  # Green for matched
                else:
                    # Blank out this word (white fill)
                    cv2.fillPoly(image, [box_points], color=(255, 255, 255))
                    cv2.polylines(image, [box_points], isClosed=True, 
                                color=(220, 220, 220), thickness=1)
            
            # Save the processed image
            cv2.imwrite(output_path, image)
        else:
            # No OCR results, just save the original image
            cv2.imwrite(output_path, image)
        
        return {
            'all_detected_words': all_detected_words,
            'matches': matches,
            'output_path': output_path,
            'show_all_mode': len(matches) == 0 and len(all_detected_words) > 0
        }
    
    # ==================== PARALLEL PROCESSING ====================
    
    def _cleanup_old_images(self, keep_latest=5):
        """Clean up old processed images to prevent accumulation."""
        try:
            # Find all output images
            output_images = glob.glob('output_*')
            if len(output_images) > keep_latest:
                # Sort by modification time
                output_images.sort(key=os.path.getmtime)
                # Delete older images, keep the latest ones
                for img in output_images[:-keep_latest]:
                    try:
                        os.remove(img)
                    except Exception:
                        pass  # Ignore errors during cleanup
        except Exception:
            pass  # Ignore cleanup errors
    
    def process_question(self, query, marks=None):
        """
        Process a question using both RAG and image mapping systems in parallel.
        Returns both text answer and processed image information.
        """
        # Clean up old images first
        self._cleanup_old_images(keep_latest=5)
        
        # Parse marks from query if present
        marks_match = re.search(r'\[(\w+)\s*mark', query, re.IGNORECASE)
        if marks_match:
            marks = marks_match.group(1).lower()
            query = re.sub(r'\[.*?mark.*?\]', '', query, flags=re.IGNORECASE).strip()
            marks_map = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
            if marks in marks_map:
                marks = marks_map[marks]
        
        # If marks still None, default to "four" (will be handled in _generate_text_answer)
        if marks is None:
            marks = "four"
        
        # Store results from parallel tasks
        text_answer = None
        image_result = None
        text_error = None
        image_error = None
        
        # Define task functions
        def generate_text():
            nonlocal text_answer, text_error
            try:
                text_answer = self._generate_text_answer(query, marks)
            except Exception as e:
                text_error = str(e)
        
        def process_image():
            nonlocal image_result, image_error
            try:
                # Find most relevant image
                mapped_image, score = self._find_most_relevant_image(query)
                
                # Check confidence threshold (20%)
                if score < 0.20:
                    image_result = {
                        'low_confidence': True,
                        'confidence_score': score,
                        'message': 'No relevant diagram found. This topic might be out of syllabus or no matching images are available.'
                    }
                    return
                
                # Extract words from query
                query_words = self._extract_words_from_query(query)
                
                # Build image paths with timestamp to prevent caching
                import time
                timestamp = int(time.time() * 1000)  # milliseconds
                image_path = os.path.join(self.images_folder, mapped_image)
                base_name, ext = os.path.splitext(mapped_image)
                output_path = f"output_{base_name}_{timestamp}{ext}"
                
                # Process image with OCR (using simple exact matching)
                processing_result = self._process_image_with_blanking(
                    image_path, query_words, output_path
                )
                
                image_result = {
                    'mapped_image': mapped_image,
                    'confidence_score': score,
                    'query_words': query_words,
                    'image_path': image_path,
                    'output_path': processing_result.get('output_path', output_path),
                    'all_detected_words': processing_result.get('all_detected_words', []),
                    'matches': processing_result.get('matches', []),
                    'show_all_mode': processing_result.get('show_all_mode', False),
                    'error': processing_result.get('error')
                }
            except Exception as e:
                image_error = str(e)
        
        # Execute both tasks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(generate_text)
            image_future = executor.submit(process_image)
            
            # Wait for both to complete
            concurrent.futures.wait([text_future, image_future])
        
        return {
            'query': query,
            'marks': marks,
            'text_answer': text_answer,
            'text_error': text_error,
            'image_result': image_result,
            'image_error': image_error
        }
    
    # ==================== INTERACTIVE MODE ====================
    
    def interactive_mode(self):
        """Run the interactive Q&A mode."""
        print("\n" + "="*70)
        print("   COMBINED SCIENCE Q&A SYSTEM WITH IMAGE MAPPING")
        print("="*70)
        print("\nHow it works:")
        print("  1. Enter your science question")
        print("  2. System generates structured answer in points (2×marks+2)")
        print("  3. System finds and displays relevant diagram (in parallel)")
        print("  4. Both results are shown together")
        print("\nAnswer Format:")
        print("  - Answers are structured as numbered points")
        print("  - Number of points = 2 × marks")
        print("  - Default: 4 marks (8 points) if not specified")
        print("\nCommands:")
        print("  - Type your question directly")
        print("  - Add '[X marks]' for specific mark value (1-5)")
        print("  - Example: 'What is ecosystem? [2 marks]' → 4 points")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("\n🎓 Your Question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Happy studying!")
                    break
                
                print("\n⏳ Processing (running text generation and image mapping in parallel)...")
                print("-" * 70)
                
                # Process the question with both systems in parallel
                result = self.process_question(user_input)
                
                # Display text answer
                print("\n📝 TEXT ANSWER:")
                print("-" * 70)
                if result['text_error']:
                    print(f"❌ Error generating text answer: {result['text_error']}")
                else:
                    print(result['text_answer'])
                print("-" * 70)
                
                # Display image information
                print("\n🖼️  IMAGE MAPPING:")
                print("-" * 70)
                if result['image_error']:
                    print(f"❌ Error processing image: {result['image_error']}")
                elif result['image_result']:
                    img_res = result['image_result']
                    
                    if img_res.get('low_confidence'):
                        print(f"⚠️  Low Confidence Match (Score: {img_res['confidence_score']:.2f})")
                        print(f"   {img_res['message']}")
                    elif img_res.get('error'):
                        print(f"❌ {img_res['error']}")
                    else:
                        print(f"✅ Best Match: {img_res['mapped_image']}")
                        print(f"   Confidence Score: {img_res['confidence_score']:.2f}")
                        print(f"\n📝 Words extracted from query: {', '.join(img_res['query_words'])}")
                        print(f"\n📊 Detection Summary:")
                        print(f"   • Total words detected in image: {len(img_res['all_detected_words'])}")
                        print(f"   • Exact matches found: {len(img_res['matches'])}")
                        
                        if img_res['matches']:
                            print(f"\n🎯 Exact Word Matches Found:")
                            matched_labels = []
                            for detected, query, score in img_res['matches']:
                                print(f"   ✓ '{detected}' ← matched with '{query}'")
                                matched_labels.append(detected)
                            print(f"\n   📌 Result: Showing ONLY matched words (green outline), rest blanked out")
                        else:
                            print(f"\n   ℹ️  No exact matches found with query words")
                            print(f"   📌 Result: Showing ALL detected words (orange outline)")
                            print(f"   📝 All words: {', '.join(img_res['all_detected_words'])}")
                        
                        print(f"\n✨ Processed image saved to: {img_res['output_path']}")
                else:
                    print("❌ No image result generated")
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy studying!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")


def main():
    """Main entry point for the combined system."""
    import sys
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python combined_qa_system.py <path_to_json_file> [model_name]")
        print("  python combined_qa_system.py --chat [model_name]")
        print()
        print("Examples:")
        print("  python combined_qa_system.py text/syllabus_map_1.json")
        print("  python combined_qa_system.py --chat llama3.2")
        sys.exit(1)
    
    if sys.argv[1] == "--chat":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
        system = CombinedScienceQA(model_name=model_name)
        system.interactive_mode()
    else:
        json_file = sys.argv[1]
        # If relative path without folder, check in text folder
        if not os.path.exists(json_file) and not json_file.startswith('text'):
            potential_path = os.path.join('text', json_file)
            if os.path.exists(potential_path):
                json_file = potential_path
        
        model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
        system = CombinedScienceQA(json_file_path=json_file, model_name=model_name)
        system.interactive_mode()


if __name__ == "__main__":
    main()
