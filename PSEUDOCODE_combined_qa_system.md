# DETAILED PSEUDOCODE - combined_qa_system.py

## Overview
This file implements a Combined Science Q&A System that integrates two major components:
1. **RAG (Retrieval-Augmented Generation) Text System** - Answers questions using a vector database
2. **Image Mapping System** - Finds and processes relevant diagrams with intelligent word highlighting

Both systems run in parallel for efficient response generation.

---

## Main Components

### 1. CLASS: CombinedScienceQA

#### A. INITIALIZATION (__init__)
```
FUNCTION __init__(json_file_path, model_name, db_path, embeddings_path, images_folder):
    SET self.model_name = model_name (default: "llama3.2")
    SET self.db_path = db_path (default: "./science_db")
    SET self.embeddings_path = embeddings_path (default: "embeddings.pt")
    SET self.images_folder = images_folder (default: "chapter5")
    
    PRINT "Initializing RAG system..."
    CALL _init_rag_system(json_file_path)
    
    PRINT "Initializing Image Mapping system..."
    CALL _init_image_mapping()
    
    PRINT "Combined system ready!"
END FUNCTION
```

---

### 2. RAG SYSTEM INITIALIZATION

#### A. _init_rag_system(json_file_path)
```
FUNCTION _init_rag_system(json_file_path):
    PRINT "Using persistent ChromaDB at: {db_path}"
    
    // Initialize ChromaDB persistent client
    SET self.client = ChromaDB.PersistentClient(path=self.db_path)
    
    // Initialize sentence transformer for embeddings
    SET self.rag_embedding_function = SentenceTransformer("all-MiniLM-L6-v2")
    
    // Get or create collection in ChromaDB
    SET self.collection = self.client.get_or_create_collection(
        name="science_qa",
        embedding_function=self.rag_embedding_function
    )
    
    // Check if database already has data
    SET existing_count = self.collection.count()
    
    IF existing_count == 0 AND json_file_path is provided:
        PRINT "No data found — indexing from JSON..."
        CALL _load_rag_data(json_file_path)
        PRINT "Indexed {count} documents in total"
    ELSE IF existing_count > 0:
        PRINT "Loaded existing database with {existing_count} documents"
    ELSE:
        THROW ERROR "No database found and no JSON file provided!"
    END IF
END FUNCTION
```

#### B. _load_rag_data(json_file_path)
```
FUNCTION _load_rag_data(json_file_path):
    // Load JSON data from file
    OPEN json_file_path FOR READING
    PARSE JSON data into variable 'data'
    
    // Normalize data structure (ensure it's a list)
    IF data is a list:
        SET chapters = data
    ELSE:
        SET chapters = [data]
    END IF
    
    // Initialize storage for documents
    CREATE empty lists: documents, metadatas, ids
    SET doc_id = 0
    
    // Process each chapter
    FOR EACH chapter IN chapters:
        EXTRACT chapter_number from chapter
        EXTRACT chapter_name from chapter
        
        // Process each topic in the chapter
        FOR EACH topic IN chapter.topics:
            EXTRACT topic_name from topic
            
            CREATE metadata_base = {
                'chapter': chapter_number,
                'chapter_name': chapter_name,
                'topic': topic_name
            }
            
            // Process key concepts
            IF topic has key_concepts:
                CONCATENATE all key_concepts into concept_text
                IF concept_text is not empty:
                    ADD concept_text to documents
                    CREATE metadata with type='key_concepts'
                    ADD metadata to metadatas
                    ADD "doc_{doc_id}" to ids
                    INCREMENT doc_id
                END IF
            END IF
            
            // Process questions
            IF topic has questions:
                CALL _process_questions(questions, metadata_base, doc_id)
                APPEND returned documents, metadatas, ids
                UPDATE doc_id
            END IF
            
            // Process activity contexts
            IF topic has activity_contexts:
                CONCATENATE all activity_contexts into activity_text
                IF activity_text is not empty:
                    ADD activity_text to documents
                    CREATE metadata with type='activity'
                    ADD metadata to metadatas
                    ADD "doc_{doc_id}" to ids
                    INCREMENT doc_id
                END IF
            END IF
            
            // Process subtopics (recursive)
            IF topic has subtopics:
                CALL _process_subtopics(subtopics, metadata_base, doc_id)
                APPEND returned documents, metadatas, ids
                UPDATE doc_id
            END IF
        END FOR
    END FOR
    
    // Add all documents to ChromaDB in batches
    IF documents list is not empty:
        SET batch_size = 1000
        FOR i FROM 0 TO length(documents) STEP batch_size:
            SET end_idx = min(i + batch_size, length(documents))
            CALL self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        END FOR
    END IF
END FUNCTION
```

#### C. _process_questions(questions_list, metadata_base, doc_id_start)
```
FUNCTION _process_questions(questions_list, metadata_base, doc_id_start):
    CREATE empty lists: documents, metadatas, ids
    SET doc_id = doc_id_start
    
    FOR EACH question_set IN questions_list:
        FOR EACH mark_type IN ['one_mark', 'two_mark', 'three_mark', 'four_mark', 'five_mark']:
            IF question_set has mark_type questions:
                FOR EACH qa IN question_set[mark_type]:
                    EXTRACT question from qa
                    EXTRACT answer from qa
                    
                    IF question and answer are not empty:
                        // Create document text
                        SET doc_text = "Question: {question}\nAnswer: {answer}"
                        ADD doc_text to documents
                        
                        // Create metadata
                        EXTRACT marks value from mark_type (e.g., 'two' from 'two_mark_questions')
                        COPY metadata_base to meta
                        ADD to meta: type='qa', marks=marks, question=question, answer=answer
                        ADD meta to metadatas
                        
                        ADD "doc_{doc_id}" to ids
                        INCREMENT doc_id
                    END IF
                END FOR
            END IF
        END FOR
    END FOR
    
    RETURN documents, metadatas, ids, doc_id
END FUNCTION
```

#### D. _process_subtopics(subtopics_list, metadata_base, doc_id_start, level)
```
FUNCTION _process_subtopics(subtopics_list, metadata_base, doc_id_start, level=1):
    CREATE empty lists: documents, metadatas, ids
    SET doc_id = doc_id_start
    
    FOR EACH subtopic IN subtopics_list:
        EXTRACT subtopic_name from subtopic
        
        COPY metadata_base to meta_base
        ADD to meta_base: "subtopic_level_{level}" = subtopic_name
        
        // Process key concepts for this subtopic
        IF subtopic has key_concepts:
            CONCATENATE all key_concepts into concept_text
            IF concept_text is not empty:
                ADD concept_text to documents
                COPY meta_base to meta
                ADD type='key_concepts' to meta
                ADD meta to metadatas
                ADD "doc_{doc_id}" to ids
                INCREMENT doc_id
            END IF
        END IF
        
        // Process questions for this subtopic
        IF subtopic has questions:
            CALL _process_questions(questions, meta_base, doc_id)
            APPEND returned documents, metadatas, ids
            UPDATE doc_id
        END IF
        
        // Process activity contexts for this subtopic
        IF subtopic has activity_contexts:
            CONCATENATE all activity_contexts into activity_text
            IF activity_text is not empty:
                ADD activity_text to documents
                COPY meta_base to meta
                ADD type='activity' to meta
                ADD meta to metadatas
                ADD "doc_{doc_id}" to ids
                INCREMENT doc_id
            END IF
        END IF
        
        // Process nested subtopics (recursive)
        IF subtopic has nested subtopics:
            CALL _process_subtopics(nested_subtopics, meta_base, doc_id, level+1)
            APPEND returned documents, metadatas, ids
            UPDATE doc_id
        END IF
    END FOR
    
    RETURN documents, metadatas, ids, doc_id
END FUNCTION
```

---

### 3. RAG QUERY PROCESSING

#### A. _retrieve_context(query, marks, n_results)
```
FUNCTION _retrieve_context(query, marks=None, n_results=5):
    // Create filter for marks if specified
    IF marks is provided:
        SET where_filter = {"marks": marks}
    ELSE:
        SET where_filter = None
    END IF
    
    // Query ChromaDB for relevant documents
    SET results = self.collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    RETURN results
END FUNCTION
```

#### B. _generate_text_answer(query, marks)
```
FUNCTION _generate_text_answer(query, marks=None):
    // Retrieve relevant context from database
    SET results = CALL _retrieve_context(query, marks, n_results=5)
    
    // Check if results exist
    IF results is empty OR results.documents is empty:
        RETURN "No relevant information found in the database."
    END IF
    
    // Build context from retrieved documents
    CREATE empty list: context_parts
    
    FOR EACH document, metadata IN results:
        EXTRACT chapter info from metadata
        EXTRACT topic info from metadata
        EXTRACT subtopic info from metadata (if exists)
        
        FORMAT context_entry = "[{chapter} | {topic} | {subtopic}]\n{document}"
        ADD context_entry to context_parts
    END FOR
    
    SET context = JOIN context_parts with "\n\n"
    
    // Build prompt based on marks
    IF marks is provided:
        SET marks_info = " ({marks} marks)"
        SET marks_instruction = "Write an answer suitable for a {marks}-mark question..."
    ELSE:
        SET marks_info = ""
        SET marks_instruction = "Write a general concise answer..."
    END IF
    
    // Create full prompt
    SET prompt = """
    You are a science teacher helping students prepare for exams.
    
    Context from textbook:
    {context}
    
    Student's Question{marks_info}: {query}
    
    {marks_instruction}
    
    Answer:
    """
    
    // Generate response using Ollama LLM
    SET response = ollama.generate(model=self.model_name, prompt=prompt)
    
    RETURN response['response']
END FUNCTION
```

---

### 4. IMAGE MAPPING SYSTEM INITIALIZATION

#### A. _init_image_mapping()
```
FUNCTION _init_image_mapping():
    PRINT "Loading image embeddings from: {embeddings_path}"
    
    // Load sentence transformer model for images
    SET self.image_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    // Load pre-computed embeddings from disk
    TRY:
        SET saved_data = torch.load(self.embeddings_path)
        SET self.image_filenames = saved_data['filenames']
        SET self.image_embeddings = saved_data['embeddings']
        PRINT "Loaded {count} image embeddings"
    CATCH FileNotFoundError:
        PRINT "Error: embeddings file not found"
        PRINT "Please run 'create_embeddings.py' first"
        THROW ERROR
    END TRY
    
    // Initialize EasyOCR reader
    PRINT "Loading EasyOCR..."
    SET self.reader = easyocr.Reader(['en'])
    PRINT "EasyOCR loaded successfully"
END FUNCTION
```

---

### 5. IMAGE PROCESSING METHODS

#### A. _find_most_relevant_image(query)
```
FUNCTION _find_most_relevant_image(query):
    // Encode the query into an embedding
    SET query_embedding = self.image_model.encode(query, convert_to_tensor=True)
    
    // Calculate cosine similarity with all pre-computed image embeddings
    SET cosine_scores = util.cos_sim(query_embedding, self.image_embeddings)
    
    // Find the best match
    SET best_match_index = argmax(cosine_scores)
    SET confidence_score = cosine_scores[0][best_match_index]
    
    // Get the filename of the best matching image
    SET best_image_filename = self.image_filenames[best_match_index]
    
    RETURN best_image_filename, confidence_score
END FUNCTION
```

#### B. _extract_words_from_query(query)
```
FUNCTION _extract_words_from_query(query):
    // Extract all alphabetic words from query
    SET words = EXTRACT all words matching pattern [a-zA-Z]+ from query.lowercase()
    
    // Define common stop words to exclude
    SET stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'what', 'which',
                      'who', 'where', 'when', 'why', 'how', 'this', 'that', 'these', 'those'}
    
    // Filter out stop words and short words (length <= 2)
    SET meaningful_words = FILTER words WHERE:
        word NOT IN stop_words AND
        length(word) > 2
    
    RETURN meaningful_words
END FUNCTION
```

#### C. _normalize_word(word)
```
FUNCTION _normalize_word(word):
    // Convert to lowercase
    SET word = word.lowercase()
    
    // Remove all non-alphanumeric characters
    SET normalized = REMOVE all characters NOT matching [a-z0-9] from word
    
    RETURN normalized
END FUNCTION
```

#### D. _find_matching_words(query_words, detected_words)
```
FUNCTION _find_matching_words(query_words, detected_words):
    /*
    Simple exact word matching algorithm:
    1. Check if exact word from query matches word in OCR text
    2. Handle multi-word OCR text by splitting and checking each word
    3. Also check substring matches for words >= 4 characters
    4. If NO matches found, returns empty list (signals to show all words)
    */
    
    CREATE empty list: matches
    
    // Normalize all query words for comparison
    CREATE dictionary: normalized_query_words
    FOR EACH query_word IN query_words:
        SET normalized = CALL _normalize_word(query_word)
        ADD to normalized_query_words: normalized → original query_word
    END FOR
    
    // Check each detected word for matches
    FOR EACH detected IN detected_words:
        SET detected_normalized = CALL _normalize_word(detected)
        SET match_found = False
        
        // Check 1: Direct exact match
        IF detected_normalized IN normalized_query_words:
            SET original_query_word = normalized_query_words[detected_normalized]
            ADD to matches: (detected, original_query_word, confidence=1.0)
            CONTINUE to next detected word
        END IF
        
        // Check 2: Substring match (query word in detected word or vice versa)
        FOR EACH norm_qw, original_qw IN normalized_query_words:
            // Query word is substring of detected word (min 4 chars)
            IF length(norm_qw) >= 4 AND norm_qw IN detected_normalized:
                ADD to matches: (detected, original_qw, confidence=1.0)
                SET match_found = True
                BREAK
            END IF
            
            // Detected word is substring of query word (min 4 chars)
            IF length(detected_normalized) >= 4 AND detected_normalized IN norm_qw:
                ADD to matches: (detected, original_qw, confidence=1.0)
                SET match_found = True
                BREAK
            END IF
        END FOR
        
        IF match_found:
            CONTINUE to next detected word
        END IF
        
        // Check 3: For multi-word detected text, check each word separately
        SET detected_words_list = SPLIT detected by spaces, convert to lowercase
        
        FOR EACH detected_word IN detected_words_list:
            SET detected_word_normalized = CALL _normalize_word(detected_word)
            
            // Exact match with any query word
            IF detected_word_normalized IN normalized_query_words:
                SET original_query_word = normalized_query_words[detected_word_normalized]
                ADD to matches: (detected, original_query_word, confidence=1.0)
                BREAK
            END IF
            
            // Substring match (min 4 chars)
            FOR EACH norm_qw, original_qw IN normalized_query_words:
                IF length(norm_qw) >= 4 AND norm_qw IN detected_word_normalized:
                    ADD to matches: (detected, original_qw, confidence=1.0)
                    BREAK
                END IF
                
                IF length(detected_word_normalized) >= 4 AND detected_word_normalized IN norm_qw:
                    ADD to matches: (detected, original_qw, confidence=1.0)
                    BREAK
                END IF
            END FOR
        END FOR
    END FOR
    
    RETURN matches
END FUNCTION
```

#### E. _process_image_with_blanking(image_path, query_words, output_path)
```
FUNCTION _process_image_with_blanking(image_path, query_words, output_path):
    /*
    Intelligent word highlighting logic:
    - If exact words from query match OCR text → show ONLY those (green), blank the rest
    - If NO words match → show ALL words (orange), no blanking
    */
    
    // Check if image exists
    IF NOT file_exists(image_path):
        RETURN {
            'error': "Image not found at {image_path}",
            'all_detected_words': [],
            'matches': []
        }
    END IF
    
    // Run OCR on the image
    SET result = self.reader.readtext(image_path)
    
    // Load the image using OpenCV
    SET image = cv2.imread(image_path)
    
    CREATE empty lists: all_detected_words, matches
    
    IF result is not empty:
        // Extract all detected words
        FOR EACH (bbox, text, probability) IN result:
            ADD text to all_detected_words
        END FOR
        
        // Find matching words using exact matching algorithm
        SET matches = CALL _find_matching_words(query_words, all_detected_words)
        
        // INTELLIGENT DECISION LOGIC:
        IF length(matches) > 0:
            // Matches found → show only matched words, blank the rest
            CREATE set: words_to_keep_set from matched words in matches
            SET show_all = False
        ELSE:
            // No matches → show all words, no blanking
            CREATE set: words_to_keep_set from all_detected_words
            SET show_all = True
        END IF
        
        // Process each detected text region
        FOR EACH (bbox, text, probability) IN result:
            SET box_points = CONVERT bbox to numpy array of integers
            
            IF text IN words_to_keep_set:
                // Keep this word visible with outline
                IF show_all:
                    // Show all mode - use ORANGE outline
                    DRAW polyline on image at box_points with color=(255,165,0), thickness=2
                ELSE:
                    // Matched word - use GREEN outline
                    DRAW polyline on image at box_points with color=(0,255,0), thickness=3
                END IF
            ELSE:
                // Blank out this word
                FILL polygon on image at box_points with WHITE color=(255,255,255)
                DRAW polyline on image at box_points with color=(220,220,220), thickness=1
            END IF
        END FOR
        
        // Save the processed image
        CALL cv2.imwrite(output_path, image)
    ELSE:
        // No OCR results, just save the original image
        CALL cv2.imwrite(output_path, image)
    END IF
    
    // Return processing results
    RETURN {
        'all_detected_words': all_detected_words,
        'matches': matches,
        'output_path': output_path,
        'show_all_mode': (length(matches) == 0 AND length(all_detected_words) > 0)
    }
END FUNCTION
```

---

### 6. PARALLEL PROCESSING

#### A. _cleanup_old_images(keep_latest)
```
FUNCTION _cleanup_old_images(keep_latest=5):
    /*
    Clean up old processed images to prevent accumulation
    */
    
    TRY:
        // Find all output images
        SET output_images = FIND all files matching pattern "output_*"
        
        IF length(output_images) > keep_latest:
            // Sort by modification time (oldest first)
            SORT output_images by modification_time
            
            // Delete older images, keep only the latest ones
            FOR EACH image IN output_images[0 : -keep_latest]:
                TRY:
                    DELETE file image
                CATCH any error:
                    IGNORE error  // Continue cleanup even if deletion fails
                END TRY
            END FOR
        END IF
    CATCH any error:
        IGNORE error  // Ignore all cleanup errors
    END TRY
END FUNCTION
```

#### B. process_question(query, marks)
```
FUNCTION process_question(query, marks=None):
    /*
    Main entry point: Process question using both RAG and image mapping in parallel
    */
    
    // Clean up old processed images first
    CALL _cleanup_old_images(keep_latest=5)
    
    // Parse marks from query if present in format "[X marks]"
    SET marks_match = SEARCH for pattern '\[(\w+)\s*mark' in query
    IF marks_match found AND marks is None:
        EXTRACT marks value from match
        REMOVE marks notation from query
        
        // Convert number to word (e.g., "2" → "two")
        SET marks_map = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
        IF marks IN marks_map:
            SET marks = marks_map[marks]
        END IF
    END IF
    
    // Initialize result storage
    SET text_answer = None
    SET image_result = None
    SET text_error = None
    SET image_error = None
    
    // Define text generation task
    DEFINE FUNCTION generate_text():
        TRY:
            SET text_answer = CALL _generate_text_answer(query, marks)
        CATCH error:
            SET text_error = error message
        END TRY
    END FUNCTION
    
    // Define image processing task
    DEFINE FUNCTION process_image():
        TRY:
            // Find most relevant image
            SET mapped_image, score = CALL _find_most_relevant_image(query)
            
            // Extract words from query
            SET query_words = CALL _extract_words_from_query(query)
            
            // Build image paths with timestamp (prevent browser caching)
            SET timestamp = CURRENT_TIME_IN_MILLISECONDS()
            SET image_path = JOIN(self.images_folder, mapped_image)
            SPLIT mapped_image into base_name and extension
            SET output_path = "output_{base_name}_{timestamp}{extension}"
            
            // Process image with OCR and word matching
            SET processing_result = CALL _process_image_with_blanking(
                image_path, query_words, output_path
            )
            
            // Store image result
            SET image_result = {
                'mapped_image': mapped_image,
                'confidence_score': score,
                'query_words': query_words,
                'image_path': image_path,
                'output_path': processing_result['output_path'],
                'all_detected_words': processing_result['all_detected_words'],
                'matches': processing_result['matches'],
                'show_all_mode': processing_result['show_all_mode'],
                'error': processing_result.get('error')
            }
        CATCH error:
            SET image_error = error message
        END TRY
    END FUNCTION
    
    // Execute both tasks in parallel using ThreadPoolExecutor
    CREATE ThreadPoolExecutor with max_workers=2:
        SET text_future = SUBMIT generate_text task
        SET image_future = SUBMIT process_image task
        
        // Wait for both tasks to complete
        WAIT FOR [text_future, image_future]
    END CREATE
    
    // Return combined results
    RETURN {
        'query': query,
        'marks': marks,
        'text_answer': text_answer,
        'text_error': text_error,
        'image_result': image_result,
        'image_error': image_error
    }
END FUNCTION
```

---

### 7. INTERACTIVE MODE

#### A. interactive_mode()
```
FUNCTION interactive_mode():
    /*
    Run the interactive command-line Q&A interface
    */
    
    // Display welcome banner
    PRINT "=" * 70
    PRINT "COMBINED SCIENCE Q&A SYSTEM WITH IMAGE MAPPING"
    PRINT "=" * 70
    PRINT ""
    PRINT "How it works:"
    PRINT "  1. Enter your science question"
    PRINT "  2. System generates text answer (using RAG)"
    PRINT "  3. System finds and processes relevant diagram (in parallel)"
    PRINT "  4. Both results are displayed together"
    PRINT ""
    PRINT "Commands:"
    PRINT "  - Type your question directly"
    PRINT "  - Add '[X marks]' to specify mark value"
    PRINT "  - Example: 'What is ecosystem? [2 marks]'"
    PRINT "  - Type 'quit' or 'exit' to stop"
    PRINT "=" * 70
    
    // Main interaction loop
    LOOP FOREVER:
        TRY:
            // Get user input
            SET user_input = INPUT "Your Question: "
            TRIM whitespace from user_input
            
            // Skip empty input
            IF user_input is empty:
                CONTINUE loop
            END IF
            
            // Check for exit commands
            IF user_input.lowercase() IN ['quit', 'exit', 'q']:
                PRINT "Goodbye! Happy studying!"
                BREAK loop
            END IF
            
            PRINT "Processing (running text generation and image mapping in parallel)..."
            PRINT "-" * 70
            
            // Process the question with both systems
            SET result = CALL process_question(user_input)
            
            // ========== DISPLAY TEXT ANSWER ==========
            PRINT ""
            PRINT "TEXT ANSWER:"
            PRINT "-" * 70
            
            IF result['text_error'] exists:
                PRINT "Error generating text answer: {text_error}"
            ELSE:
                PRINT result['text_answer']
            END IF
            PRINT "-" * 70
            
            // ========== DISPLAY IMAGE INFORMATION ==========
            PRINT ""
            PRINT "IMAGE MAPPING:"
            PRINT "-" * 70
            
            IF result['image_error'] exists:
                PRINT "Error processing image: {image_error}"
            ELSE IF result['image_result'] exists:
                SET img_res = result['image_result']
                
                IF img_res has 'error':
                    PRINT img_res['error']
                ELSE:
                    PRINT "Best Match: {mapped_image}"
                    PRINT "Confidence Score: {confidence_score}"
                    PRINT ""
                    PRINT "Words extracted from query: {query_words}"
                    PRINT ""
                    PRINT "Detection Summary:"
                    PRINT "  • Total words detected in image: {count}"
                    PRINT "  • Exact matches found: {matches_count}"
                    
                    IF img_res['matches'] is not empty:
                        PRINT ""
                        PRINT "Exact Word Matches Found:"
                        FOR EACH (detected, query, score) IN img_res['matches']:
                            PRINT "  ✓ '{detected}' ← matched with '{query}'"
                        END FOR
                        PRINT ""
                        PRINT "Result: Showing ONLY matched words (green), rest blanked"
                    ELSE:
                        PRINT ""
                        PRINT "No exact matches found with query words"
                        PRINT "Result: Showing ALL detected words (orange outline)"
                        PRINT "All words: {all_detected_words}"
                    END IF
                    
                    PRINT ""
                    PRINT "Processed image saved to: {output_path}"
                END IF
            ELSE:
                PRINT "No image result generated"
            END IF
            
            PRINT "-" * 70
            
        CATCH KeyboardInterrupt:
            PRINT ""
            PRINT "Goodbye! Happy studying!"
            BREAK loop
        
        CATCH any other error:
            PRINT "Unexpected error: {error}"
        END TRY
    END LOOP
END FUNCTION
```

---

### 8. MAIN ENTRY POINT

#### A. main()
```
FUNCTION main():
    /*
    Main entry point for command-line execution
    */
    
    // Check command-line arguments
    IF number of arguments == 1 (only script name):
        PRINT "Usage:"
        PRINT "  python combined_qa_system.py <path_to_json_file> [model_name]"
        PRINT "  python combined_qa_system.py --chat [model_name]"
        PRINT ""
        PRINT "Examples:"
        PRINT "  python combined_qa_system.py text/syllabus_map_1.json"
        PRINT "  python combined_qa_system.py --chat llama3.2"
        EXIT program with code 1
    END IF
    
    // Handle --chat mode (use existing database)
    IF first argument == "--chat":
        SET model_name = second argument IF provided, ELSE "llama3.2"
        
        // Initialize system without loading JSON (use existing DB)
        SET system = NEW CombinedScienceQA(model_name=model_name)
        
        // Start interactive mode
        CALL system.interactive_mode()
    
    // Handle normal mode (load from JSON file)
    ELSE:
        SET json_file = first argument
        
        // Check if file exists, try 'text/' folder if not found
        IF NOT file_exists(json_file) AND json_file does not start with 'text':
            SET potential_path = JOIN('text', json_file)
            IF file_exists(potential_path):
                SET json_file = potential_path
            END IF
        END IF
        
        SET model_name = second argument IF provided, ELSE "llama3.2"
        
        // Initialize system with JSON file
        SET system = NEW CombinedScienceQA(
            json_file_path=json_file,
            model_name=model_name
        )
        
        // Start interactive mode
        CALL system.interactive_mode()
    END IF
END FUNCTION

// Execute main if running as script
IF this file is being run directly (not imported):
    CALL main()
END IF
```

---

## Data Flow Summary

### Question Processing Flow:
```
1. User enters question
   ↓
2. Parse marks notation if present [X marks]
   ↓
3. PARALLEL EXECUTION:
   ├─→ RAG Text Generation:
   │   ├─ Extract embeddings from query
   │   ├─ Search ChromaDB for relevant documents
   │   ├─ Build context from top results
   │   ├─ Generate prompt with context
   │   └─ Get answer from Ollama LLM
   │
   └─→ Image Processing:
       ├─ Encode query to embedding
       ├─ Find most similar image (cosine similarity)
       ├─ Extract meaningful words from query
       ├─ Run OCR on selected image
       ├─ Find exact word matches
       ├─ Decide: show matched only OR show all
       ├─ Draw outlines (green/orange) and blank unmatched
       └─ Save processed image
   ↓
4. Combine both results
   ↓
5. Display to user (text + image)
```

### Image Matching Logic:
```
Query: "What is photosynthesis?"
   ↓
Extract words: ["photosynthesis"]
   ↓
OCR detects: ["Photosynthesis", "Light", "Chlorophyll", "Oxygen", "Glucose"]
   ↓
Match algorithm:
   - "Photosynthesis" ← EXACT MATCH with query word "photosynthesis"
   ↓
Decision: MATCHES FOUND (1 match)
   ↓
Result:
   - Show "Photosynthesis" with GREEN outline
   - Blank out: "Light", "Chlorophyll", "Oxygen", "Glucose"
```

### No Match Scenario:
```
Query: "Explain ecosystem?"
   ↓
Extract words: ["explain", "ecosystem"]
   ↓
OCR detects: ["Food", "Chain", "Producer", "Consumer"]
   ↓
Match algorithm:
   - No exact matches found
   ↓
Decision: NO MATCHES
   ↓
Result:
   - Show ALL words with ORANGE outline
   - No blanking (all words visible)
```

---

## Key Features

1. **Parallel Processing**: Text generation and image processing run simultaneously for faster response
2. **Intelligent Image Highlighting**: 
   - Shows only matched words when found (green outline)
   - Shows all words when no matches (orange outline)
3. **Persistent Database**: ChromaDB stores embeddings for fast retrieval
4. **Mark-Based Filtering**: Can filter questions by mark value (1-5 marks)
5. **Cache Prevention**: Uses timestamps in image filenames to prevent browser caching
6. **Cleanup Management**: Automatically removes old processed images
7. **Flexible Input**: Accepts marks notation in query or as parameter

---

## Dependencies

- **ollama**: LLM inference
- **chromadb**: Vector database for RAG
- **sentence-transformers**: Text embeddings
- **easyocr**: Optical character recognition
- **opencv-python (cv2)**: Image processing
- **torch**: Deep learning framework
- **numpy**: Numerical operations
- **concurrent.futures**: Parallel task execution

---

## File I/O Operations

### Input Files:
- **JSON file** (`text/syllabus_map_1.json`): Science curriculum data
- **embeddings.pt**: Pre-computed image embeddings
- **chapter5/*.jpg/png**: Image files for diagrams

### Output Files:
- **science_db/**: ChromaDB persistent storage
- **output_*.jpg/png**: Processed images with highlighted labels

### Temporary Files:
- Automatically cleaned up to keep only latest 5 processed images

---

This pseudocode provides a complete understanding of the system's logic, data flow, and decision-making processes.
