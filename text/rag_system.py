"""
Science Q&A System using Ollama + RAG (Persistent Version)
---------------------------------------------------------
✅ Uses persistent ChromaDB storage to avoid reloading JSON each time
✅ Can directly start chatbot after first run
Requires:
    pip install ollama chromadb sentence-transformers
"""

import json
import ollama
import chromadb
from chromadb.utils import embedding_functions


class ScienceRAG:
    def __init__(self, json_file_path=None, model_name="llama3.2", db_path="./science_db"):
        """
        Initialize the RAG system with persistent ChromaDB storage.
        If the DB already contains data, skips reloading from JSON.
        """
        self.model_name = model_name
        self.db_path = db_path

        # Persistent storage for embeddings
        print(f"🔗 Using persistent ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)

        # SentenceTransformer for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Create or get existing collection
        self.collection = self.client.get_or_create_collection(
            name="science_qa",
            embedding_function=self.embedding_function
        )

        existing_count = self.collection.count()

        if existing_count == 0 and json_file_path:
            print("🧩 No data found — indexing from JSON...")
            self.load_data(json_file_path)
            print(f"✅ Indexed {self.collection.count()} documents in total.")
        elif existing_count > 0:
            print(f"📦 Loaded existing database with {existing_count} documents.")
        else:
            raise ValueError("❌ No database found and no JSON file provided!")

    # -------------------- Internal Processing --------------------

    def process_questions(self, questions_list, metadata_base, doc_id_start):
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

    def process_subtopics(self, subtopics_list, metadata_base, doc_id_start, level=1):
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
                q_docs, q_metas, q_ids, doc_id = self.process_questions(
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
                nested_docs, nested_metas, nested_ids, doc_id = self.process_subtopics(
                    nested_subtopics, meta_base, doc_id, level + 1)
                documents.extend(nested_docs)
                metadatas.extend(nested_metas)
                ids.extend(nested_ids)

        return documents, metadatas, ids, doc_id

    def load_data(self, json_file_path):
        """Load JSON data and create vector embeddings"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chapters = data if isinstance(data, list) else [data]
        documents, metadatas, ids, doc_id = [], [], [], 0

        print(f"Found {len(chapters)} chapter(s) to process\n")

        for chapter in chapters:
            chapter_num = chapter.get('chapter_number', 'Unknown')
            chapter_name = chapter.get('chapter_name', 'Unknown')
            print(f"  Processing Chapter {chapter_num}: {chapter_name}")

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
                    q_docs, q_metas, q_ids, doc_id = self.process_questions(
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
                    sub_docs, sub_metas, sub_ids, doc_id = self.process_subtopics(
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
                print(f"  Indexed {end_idx}/{len(documents)} documents")

    # -------------------- Query + Generation --------------------

    def retrieve_context(self, query, marks=None, n_results=5):
        where_filter = {"marks": marks} if marks else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results

    def generate_answer(self, query, marks=None, chapter=None):
        results = self.retrieve_context(query, marks=marks, n_results=5)

        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant information found in the database."

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

        # Generate prompt depending on marks
        if marks:
            marks_info = f" ({marks} marks)"
            marks_instruction = f"Write an answer suitable for a {marks}-mark question. Keep the answer length and depth appropriate for {marks} marks."
        else:
            marks_info = ""
            marks_instruction = "Write a general concise answer suitable for revision."

        prompt = f"""You are a science teacher helping students prepare for exams.

    Context from textbook:
    {context}

    Student's Question{marks_info}: {query}

    {marks_instruction}

    Answer:"""

        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response']


    # -------------------- Interactive CLI --------------------

    def interactive_mode(self):
        print("\n" + "="*60)
        print("Science Q&A System (Ollama + RAG, Persistent DB)")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question directly")
        print("  - Add '[X marks]' to specify mark value")
        print("  - Example: 'What is ecosystem? [2 marks]'")
        print("  - Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")

        import re
        while True:
            try:
                user_input = input("\n🎓 Your Question: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Happy studying!")
                    break

                marks = None
                marks_match = re.search(r'\[(\w+)\s*mark', user_input, re.IGNORECASE)
                if marks_match:
                    marks = marks_match.group(1).lower()
                    user_input = re.sub(r'\[.*?mark.*?\]', '', user_input, flags=re.IGNORECASE).strip()
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
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python rag_system.py <path_to_json_file> [model_name]")
        print("  python rag_system.py --chat [model_name]")
        print()
        sys.exit(1)

    if sys.argv[1] == "--chat":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
        rag = ScienceRAG(model_name=model_name)
        rag.interactive_mode()
    else:
        json_file = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
        rag = ScienceRAG(json_file_path=json_file, model_name=model_name)
        rag.interactive_mode()


if __name__ == "__main__":
    main()
