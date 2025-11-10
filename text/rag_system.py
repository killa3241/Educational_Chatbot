"""
Science Q&A System using Ollama + RAG
Handles multi-chapter textbooks with nested subtopics
Requires: pip install ollama chromadb sentence-transformers
"""

import json
import ollama
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class ScienceRAG:
    def __init__(self, json_file_path, model_name="llama3.2"):
        """
        Initialize the RAG system
        
        Args:
            json_file_path: Path to your JSON dataset
            model_name: Ollama model to use (default: llama3.2)
        """
        self.model_name = model_name
        self.client = chromadb.Client()
        
        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.create_collection(
            name="science_qa",
            embedding_function=self.embedding_function,
            get_or_create=True
        )
        
        # Load and index data
        print("Loading dataset...")
        self.load_data(json_file_path)
        print(f"✅ Indexed {self.collection.count()} documents from all chapters")
    
    def process_questions(self, questions_list, metadata_base, doc_id_start):
        """
        Process questions from a topic or subtopic
        
        Args:
            questions_list: List of question sets
            metadata_base: Base metadata dict
            doc_id_start: Starting document ID
            
        Returns:
            Tuple of (documents, metadatas, ids, next_doc_id)
        """
        documents = []
        metadatas = []
        ids = []
        doc_id = doc_id_start
        
        for q_set in questions_list:
            for mark_type in ['one_mark_questions', 'two_mark_questions', 
                            'three_mark_questions', 'four_mark_questions', 
                            'five_mark_questions']:
                if mark_type in q_set:
                    for qa in q_set[mark_type]:
                        question = qa.get('question', '')
                        answer = qa.get('answer', '')
                        
                        if not question or not answer:
                            continue
                        
                        # Create document with Q&A
                        doc_text = f"Question: {question}\nAnswer: {answer}"
                        documents.append(doc_text)
                        
                        # Extract marks
                        marks = mark_type.split('_')[0]
                        
                        # Create metadata
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
    
    def process_subtopics(self, subtopics_list, metadata_base, doc_id_start, level=1):
        """
        Recursively process subtopics at any nesting level
        
        Args:
            subtopics_list: List of subtopics
            metadata_base: Base metadata dict
            doc_id_start: Starting document ID
            level: Nesting level (1, 2, 3...)
            
        Returns:
            Tuple of (documents, metadatas, ids, next_doc_id)
        """
        documents = []
        metadatas = []
        ids = []
        doc_id = doc_id_start
        
        for subtopic in subtopics_list:
            subtopic_name = subtopic.get('subtopic_name', 'Unknown')
            
            # Create metadata for this subtopic level
            meta_base = metadata_base.copy()
            meta_base[f'subtopic_level_{level}'] = subtopic_name
            
            # Process key concepts
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
            
            # Process questions
            questions = subtopic.get('questions', [])
            if questions:
                q_docs, q_metas, q_ids, doc_id = self.process_questions(
                    questions, meta_base, doc_id
                )
                documents.extend(q_docs)
                metadatas.extend(q_metas)
                ids.extend(q_ids)
            
            # Process activity contexts
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
            
            # Recursively process nested subtopics
            nested_subtopics = subtopic.get('subtopics', [])
            if nested_subtopics:
                nested_docs, nested_metas, nested_ids, doc_id = self.process_subtopics(
                    nested_subtopics, meta_base, doc_id, level + 1
                )
                documents.extend(nested_docs)
                metadatas.extend(nested_metas)
                ids.extend(nested_ids)
        
        return documents, metadatas, ids, doc_id
    
    def load_data(self, json_file_path):
        """Load JSON data and create vector embeddings"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Could be list of chapters or list with single dict
            if len(data) > 0 and isinstance(data[0], dict):
                chapters = data
            else:
                print("❌ Error: Unexpected JSON structure")
                return
        elif isinstance(data, dict):
            # Single chapter object
            chapters = [data]
        else:
            print("❌ Error: JSON must be a dict or list")
            return
        
        documents = []
        metadatas = []
        ids = []
        doc_id = 0
        
        print(f"Found {len(chapters)} chapter(s) to process\n")
        
        for chapter in chapters:
            # Validate chapter structure
            if not isinstance(chapter, dict):
                print(f"⚠️  Skipping invalid chapter: {chapter}")
                continue
            
            chapter_num = chapter.get('chapter_number', 'Unknown')
            chapter_name = chapter.get('chapter_name', 'Unknown')
            
            print(f"  Processing Chapter {chapter_num}: {chapter_name}")
            
            topics = chapter.get('topics', [])
            if not topics:
                print(f"    ⚠️  No topics found in this chapter")
                continue
            
            for topic in topics:
                # Validate topic structure
                if not isinstance(topic, dict):
                    print(f"    ⚠️  Skipping invalid topic: {topic}")
                    continue
                
                topic_name = topic.get('topic_name', 'Unknown')
                
                # Base metadata for this topic
                metadata_base = {
                    'chapter': str(chapter_num),
                    'chapter_name': chapter_name,
                    'topic': topic_name
                }
                
                # Process key concepts
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
                
                # Process questions at topic level
                questions = topic.get('questions', [])
                if questions:
                    q_docs, q_metas, q_ids, doc_id = self.process_questions(
                        questions, metadata_base, doc_id
                    )
                    documents.extend(q_docs)
                    metadatas.extend(q_metas)
                    ids.extend(q_ids)
                
                # Process activity contexts at topic level
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
                
                # Process subtopics (handles nested subtopics recursively)
                subtopics = topic.get('subtopics', [])
                if subtopics:
                    sub_docs, sub_metas, sub_ids, doc_id = self.process_subtopics(
                        subtopics, metadata_base, doc_id
                    )
                    documents.extend(sub_docs)
                    metadatas.extend(sub_metas)
                    ids.extend(sub_ids)
        
        # Add all documents to ChromaDB
        if documents:
            # Add in batches to avoid any size limits
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                print(f"  Indexed {end_idx}/{len(documents)} documents")
    
    def retrieve_context(self, query, marks=None, n_results=5):
        where_filter = None

        if marks:
            where_filter = {"marks": marks}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        return results

    
    def generate_answer(self, query, marks=None, chapter=None):
        """
        Generate answer using RAG pipeline
        
        Args:
            query: User's question
            marks: Optional mark specification
            chapter: (Ignored now, kept only for compatibility)
        """
        # Retrieve relevant context (ignore chapter filter)
        results = self.retrieve_context(query, marks=marks, n_results=5)
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant information found in the database."
        
        # Prepare context with metadata
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            
            chapter_info = f"Chapter {metadata.get('chapter', '')}: {metadata.get('chapter_name', '')}"
            topic_info = f"Topic: {metadata.get('topic', '')}"
            
            subtopic_levels = [k for k in metadata.keys() if k.startswith('subtopic_level_')]
            if subtopic_levels:
                subtopic_info = " > ".join([metadata[k] for k in sorted(subtopic_levels)])
                topic_info += f" > {subtopic_info}"
            
            context_parts.append(f"[{chapter_info} | {topic_info}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Ollama
        marks_info = f" ({marks} marks)" if marks else ""
        
        prompt = f"""You are a science teacher helping students prepare for exams.

    Context from textbook:
    {context}

    Student's Question{marks_info}: {query}

    Please provide a clear, concise, and accurate answer based on the context provided.
    Ensure the answer length matches the mark value (e.g., short for 1 mark, detailed for 5 marks).

    Answer:"""
        
        # Generate response using Ollama
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )
        
        return response['response']

    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("Science Q&A System (Ollama + RAG)")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question directly")
        print("  - Add '[X marks]' to specify mark value")
        print("  - Example: 'What is ecosystem? [2 marks]'")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n🎓 Your Question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Happy studying!")
                    break
                
                # Parse marks if specified
                marks = None
                import re
                
                marks_match = re.search(r'\[(\w+)\s*mark', user_input, re.IGNORECASE)
                if marks_match:
                    marks = marks_match.group(1).lower()
                    user_input = re.sub(r'\[.*?mark.*?\]', '', user_input, flags=re.IGNORECASE).strip()

                    # Convert numeric marks to text format used in dataset
                    marks_map = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
                    if marks in marks_map:
                        marks = marks_map[marks]

                print("\n⏳ Generating answer...")
                answer = self.generate_answer(user_input, marks)
                
                print("\n" + "-"*60)
                print("📝 Answer:")
                print("-"*60)
                print(answer)
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy studying!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")



def main():
    import sys
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python rag_system.py <path_to_json_file> [model_name]")
        print("\nExamples:")
        print("  python rag_system.py science_data.json")
        print("  python rag_system.py science_data.json llama3.1")
        sys.exit(1)
    
    json_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
    
    # Initialize RAG system
    try:
        print(f"\n🚀 Starting Science RAG System with model: {model_name}\n")
        rag = ScienceRAG(json_file, model_name)
        
        # Run interactive mode
        rag.interactive_mode()
        
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file}' not found!")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in '{json_file}'!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()