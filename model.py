import os
import openai
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Ensure OpenAI API key is set
openai.api_key = ''
embeddings = OpenAIEmbeddings(api_key=openai.api_key)


# Flask setup
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
CORS(app, supports_credentials=True)
CORS(app, resources={r"/upload": {"origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://127.0.0.1:5000/upload",]}})
# CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx'}

vector_store = None  # Global FAISS store

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('model.html') 

@app.route('/upload', methods=['POST'])
def upload_files():
    global vector_store

    if 'files' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    files = request.files.getlist('files')
    documents = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif filename.endswith('.xlsx'):
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    return jsonify({'error': f'Unsupported file type: {filename}'}), 400

                documents.extend(loader.load())
            except Exception as e:
                return jsonify({'error': f'Error processing file {filename}: {str(e)}'}), 500
        else:
            return jsonify({'error': f'File {file.filename} has an invalid extension'}), 400

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Use OpenAI Embeddings to embed the documents
    try:
        embeddings = OpenAIEmbeddings(api_key=openai.api_key)  # Update this line
        vector_store = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        return jsonify({'error': f'Error creating embeddings: {str(e)}'}), 500

    return jsonify({'message': 'Files uploaded and processed successfully'}), 200


@app.route('/query', methods=['POST'])
def query():
    global vector_store

    if vector_store is None:
        return jsonify({'error': 'No files uploaded yet'}), 400

    query_text = request.form.get('query', None)
    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    # Use OpenAI LLM for querying
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAIEmbeddings(model_name="text-davinci-003"),  # Using OpenAI LLM
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Get the answer by running the query through the QA chain
    answer = qa_chain.run(query_text)

    return jsonify({'answer': answer}), 200

if __name__ == '__main__':
    app.run(debug=True)
