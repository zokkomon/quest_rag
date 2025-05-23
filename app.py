from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import logging
import markdown
from typing import List
from pdfminer.high_level import extract_text
from pathlib import Path

from query_praśna import QueryPraśna

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGSystem:
    def __init__(self, pdf_folder="legal_documents"):
        self.query_prasna = QueryPraśna(layout_data=None)
        self.pdf_folder = pdf_folder
        self.documents_content = ""
        self.documents_processed = False
        self.loaded_documents = []
        
        # Initialize with predefined PDFs
        self.initialize_system()
    
    def extract_text_from_pdf(self, pdf_path):
        try:
            text_content = extract_text(pdf_path)
            return text_content.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def initialize_system(self):
        try:
            # Create documents folder if it doesn't exist
            pdf_folder_path = Path(self.pdf_folder)
            if not pdf_folder_path.exists():
                pdf_folder_path.mkdir(exist_ok=True)
                logger.warning(f"Created {self.pdf_folder} folder. Please add your legal PDFs there.")
                return
            
            # Look for PDF files in the folder
            pdf_files = list(pdf_folder_path.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.pdf_folder} folder.")
                return
            
            logger.info(f"Found {len(pdf_files)} PDF files. Processing...")
            
            all_content = []
            for pdf_file in pdf_files:
                logger.info(f"Processing: {pdf_file.name}")
                content = self.extract_text_from_pdf(pdf_file)
                
                if content:
                    all_content.append(content)
                    self.loaded_documents.append({
                        'filename': pdf_file.name,
                        'word_count': len(content.split()),
                        'char_count': len(content)
                    })
                    logger.info(f"✓ Processed {pdf_file.name}: {len(content.split())} words")
                else:
                    logger.warning(f"✗ Failed to extract content from {pdf_file.name}")
            
            if all_content:
                self.documents_content = "\n\n---DOCUMENT_SEPARATOR---\n\n".join(all_content)
                self.documents_processed = True
                
                total_words = sum(doc['word_count'] for doc in self.loaded_documents)
                logger.info(f"✓ System initialized with {len(self.loaded_documents)} documents ({total_words} total words)")
            else:
                logger.error("No content could be extracted from any PDF files.")
                
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
    
    def get_system_info(self):
        return {
            'documents_processed': self.documents_processed,
            'total_documents': len(self.loaded_documents),
            'loaded_documents': self.loaded_documents,
            'total_words': sum(doc['word_count'] for doc in self.loaded_documents),
            'total_characters': sum(doc['char_count'] for doc in self.loaded_documents)
        }
    
    def answer_query(self, query: str):
        if not self.documents_processed:
            return "System is not properly initialized. Please check if legal documents are available."
        
        try:
            # Pass the document content to QueryPrasna and get the response
            response = self.query_prasna(query, self.documents_content)
            return response
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
    
    def reload_documents(self):
        """Reload documents from the PDF folder."""
        self.__init__(self.pdf_folder)

# Initialize the RAG system
rag_system = LegalRAGSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/system_info', methods=['GET'])
def get_system_info():
    """Get detailed information about the system and loaded documents."""
    try:
        info = rag_system.get_system_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"System info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_legal():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        if not rag_system.documents_processed:
            return jsonify({
                'error': 'System not ready. Please ensure legal documents are loaded.',
                'system_ready': False
            }), 400
        
        # Get response from RAG system
        response_md = rag_system.answer_query(query)
        response_html = markdown.markdown(response_md)  
        
        return jsonify({
            'query': query,
            'response_markdown': response_md,
            'response_html': response_html,
            'system_info': rag_system.get_system_info()
        })
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status."""
    info = rag_system.get_system_info()
    return jsonify({
        'system_ready': info['documents_processed'],
        'documents_loaded': info['total_documents'],
        'total_words': info['total_words'],
        'initialization_complete': info['documents_processed']
    })

@app.route('/reload', methods=['POST'])
def reload_documents():
    """Reload documents from the PDF folder."""
    try:
        logger.info("Reloading documents...")
        rag_system.reload_documents()
        info = rag_system.get_system_info()
        
        return jsonify({
            'message': f'Documents reloaded successfully. Loaded {info["total_documents"]} documents.',
            'system_info': info
        })
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    info = rag_system.get_system_info()
    return jsonify({
        'status': 'healthy',
        'system_ready': info['documents_processed'],
        'documents_count': info['total_documents']
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not found in environment variables")
    
    logger.info("Starting Legal RAG System...")
    logger.info("Place your legal PDF documents in the 'legal_documents' folder")
    
    app.run(debug=True, host='0.0.0.0', port=5000)