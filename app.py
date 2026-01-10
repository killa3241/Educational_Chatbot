"""
Flask Web Application for Combined Science Q&A System
-----------------------------------------------------
A user-friendly chatbot interface for the Educational Chatbot system.
Features:
- Chat-like interface for asking questions
- Displays text answers followed by relevant diagrams
- Shows system rules and instructions on startup
- Real-time question processing with loading indicators
"""

from flask import Flask, render_template, request, jsonify, send_file
from combined_qa_system import CombinedScienceQA
import os
import traceback
from datetime import datetime

app = Flask(__name__)

# Initialize the QA system (it will use existing database)
print("Initializing Combined Science QA System...")
qa_system = None

try:
    qa_system = CombinedScienceQA(model_name="llama3.2")
    print("✅ System initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing system: {str(e)}")
    traceback.print_exc()


@app.route('/')
def index():
    """Render the main chatbot interface."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """Process a question and return both text and image results."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if qa_system is None:
            return jsonify({'error': 'System not initialized. Please check server logs.'}), 500
        
        # Process the question using the combined system
        result = qa_system.process_question(question)
        
        # Prepare response
        response = {
            'question': result['query'],
            'marks': result['marks'],
            'timestamp': datetime.now().strftime('%I:%M %p')
        }
        
        # Add text answer
        if result['text_error']:
            response['text_answer'] = f"Error: {result['text_error']}"
            response['text_success'] = False
        else:
            response['text_answer'] = result['text_answer']
            response['text_success'] = True
        
        # Add image information
        if result['image_error']:
            response['image_error'] = result['image_error']
            response['image_success'] = False
        elif result['image_result']:
            img_res = result['image_result']
            
            # Check for low confidence
            if img_res.get('low_confidence'):
                response['image_success'] = False
                response['low_confidence'] = True
                response['confidence'] = f"{img_res['confidence_score']:.2f}"
                response['image_error'] = img_res['message']
            elif img_res.get('error'):
                response['image_error'] = img_res['error']
                response['image_success'] = False
            else:
                response['image_success'] = True
                response['image_path'] = img_res['output_path']
                response['image_name'] = img_res['mapped_image']
                response['confidence'] = f"{img_res['confidence_score']:.2f}"
                response['query_words'] = img_res['query_words']
                response['detected_words_count'] = len(img_res['all_detected_words'])
                response['matched_words_count'] = len(img_res['matches'])
                response['show_all_mode'] = img_res.get('show_all_mode', False)
                
                # Format matches for display
                if img_res['matches']:
                    matches_formatted = []
                    for detected, query, score in img_res['matches']:
                        matches_formatted.append({
                            'detected': detected,
                            'query': query
                        })
                    response['matches'] = matches_formatted
                    response['match_mode'] = 'exact_matches'
                else:
                    response['matches'] = []
                    response['all_words'] = img_res['all_detected_words']
                    response['match_mode'] = 'show_all'
        else:
            response['image_error'] = "No image result generated"
            response['image_success'] = False
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serve processed images with no-cache headers."""
    try:
        # Check if file exists in current directory
        if os.path.exists(filename):
            response = send_file(filename, mimetype='image/jpeg')
            # Add cache-busting headers
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Check if the system is ready."""
    if qa_system is None:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 500
    return jsonify({'status': 'ok', 'message': 'System is ready'})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("   EDUCATIONAL CHATBOT - WEB INTERFACE")
    print("="*70)
    print("\n🌐 Starting Flask server...")
    print("📍 Open your browser and navigate to: http://localhost:5000")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
