import re
import spacy
import json
import pickle
import logging
import numpy as np
from typing import Optional, List, Literal

import infinity_embedded
from infinity_embedded.common import SparseVector
from infinity_embedded.common import ConflictType
from infinity_embedded.index import IndexInfo, IndexType

from google.genai import types

from hashlib import sha256
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from file_utils import llm, embedding, client

logger = logging.getLogger(__name__)

NLP_MODEL = spacy.load("en_core_web_sm")


def clean_json_response(text):
    """
    Basic cleaning: removes markdown code fences and strips whitespace.
    Handles potential ```json ... ``` or ``` ... ``` blocks.
    """
    text = text.strip()
    # Remove potential starting ```json marker
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    # Remove potential starting ``` marker
    elif text.startswith("```"):
         text = text[len("```"):].strip()
    # Remove potential ending ``` marker
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text

class RecursiveChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Recursively split text into chunks using different separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    # If adding this part would exceed chunk size, save current chunk
                    if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Start new chunk with overlap from previous chunk
                            if len(current_chunk) > self.overlap:
                                current_chunk = current_chunk[-self.overlap:] + separator + part
                            else:
                                current_chunk = part
                        else:
                            # If single part is too long, recursively split it
                            if len(part) > self.chunk_size:
                                chunks.extend(self.split_text(part))
                            else:
                                chunks.append(part.strip())
                            current_chunk = ""
                    else:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                
                # Add remaining chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [chunk for chunk in chunks if chunk.strip()]
        
        # If no separator worked, force split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

class SpacyBM25Handler:
    def __init__(self):
        self.nlp = NLP_MODEL
        self.bm25 = None
        self.corpus_tokens = []
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy and return filtered tokens."""
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop]
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            sub_tokens = re.findall(r"\w+", token)
            if len(sub_tokens) > 1:
                expanded_tokens.extend(sub_tokens)
        return expanded_tokens

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0,1] range."""
        max_score = max(scores) if scores.size > 0 else 1.0
        return [score / max_score for score in scores]


    def get_sparse_vector(self, text: str, vocab_size: int = 3072, min_score: float = 0.01) -> SparseVector:
        """Convert text to BM25 sparse vector."""
        tokens = self.tokenize(text)

        # Initialize BM25 only once
        if self.bm25 is None:
            self.corpus_tokens = [tokens]
            self.bm25 = BM25Okapi(self.corpus_tokens)
        elif tokens not in self.corpus_tokens:  # Avoid adding duplicate tokens
            self.corpus_tokens.append(tokens)
            self.bm25 = BM25Okapi(self.corpus_tokens)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokens)
        scores = self.normalize_scores(scores)

        indices = [i for i, score in enumerate(scores) if score > min_score]
        values = [float(scores[i]) for i in indices]

        if len(indices) > vocab_size:
            sorted_pairs = sorted(zip(values, indices), reverse=True)[:vocab_size]
            values, indices = zip(*sorted_pairs)
            values = list(values)
            indices = list(indices)

        return SparseVector(indices, values)


class CacheEmbedding:
    def __init__(self, inf_db_name: str, sparse_method: Literal["tfidf", "bm25"] = "bm25", embedding_instance=None, infinity_object=None) -> None:
        self._embeddings_instance = embedding_instance
        self._infinity_object = infinity_object
        self._sparse_method = sparse_method
        self._inf_db_name = inf_db_name
        
        if sparse_method == "tfidf":
            self._vectorizer = TfidfVectorizer(max_features=3072)
            self._spacy_handler = None
        else:  
            self._vectorizer = None
            self._spacy_handler = SpacyBM25Handler()

    # def __del__(self):
    #     if hasattr(self, '_infinity_object'):
    #         try:
    #             self._infinity_object.disconnect()
    #         except:
    #             pass

    def _get_sparse_vector(self, text: str) -> SparseVector:
        """Get sparse vector based on selected method."""
        if self._sparse_method == "tfidf":
            sparse_matrix = self._vectorizer.fit_transform([text])
            indices = sparse_matrix.indices.tolist()
            values = sparse_matrix.data.tolist()
            return SparseVector(indices, values)
        else:  
            return self._spacy_handler.get_sparse_vector(text)

    def inf_obj(self, create_table: bool = True):
        tbl_name = self._inf_db_name+'_tbl'
        id_idx = self._inf_db_name+'_id_idx'
        vec_idx = self._inf_db_name+'_vec_idx'
        sparse_idx = self._inf_db_name+'_sparse_idx'

        # Ensure the database exists
        self._infinity_object.create_database(self._inf_db_name, conflict_type=ConflictType.Ignore)
        db_object = self._infinity_object.get_database(self._inf_db_name)
        
        # Define table schema
        columns_definition = {
            "id": {"type": "varchar", "unique": True}, 
            "text": {"type": "varchar"},
            "vec": {"type": "vector, 3072, float"}, 
            "sparse": {"type": "sparse,3072,float,int"}
        }
        

        # db_object.drop_table(tbl_name, conflict_type = ConflictType.Ignore) if create_table else db_object
        table_object = db_object.create_table(tbl_name, columns_definition, conflict_type=ConflictType.Ignore)
        table_object.create_index(id_idx, IndexInfo("text", IndexType.FullText), conflict_type=ConflictType.Ignore)
        table_object.create_index(vec_idx, IndexInfo("vec", IndexType.Hnsw, {"metric": "cosine"}), conflict_type=ConflictType.Ignore)
        table_object.create_index(sparse_idx, IndexInfo("sparse", IndexType.BMP), conflict_type=ConflictType.Ignore)

        return table_object

    def embed_documents(self, db_name: str, texts: list[str]) -> list[list[float]]:
        table_object = self.inf_obj()
        """Embed search docs with both dense and sparse embeddings."""
        # Fetch existing IDs to avoid duplication
        df_result = table_object.output(["id"]).to_pl()
        existing_ids = []
        if df_result[0] is not None:  # Check if the DataFrame is not None
            existing_ids = df_result[0]["id"].to_list()

        max_chunks = 1024
        text_embeddings = []
        for i in range(0, len(texts), max_chunks):
            batch_texts = texts[i:i + max_chunks]
            embedding_results = self._embeddings_instance.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-large"
            )
            
            # Process each text and its embedding
            for text, emb_data in zip(batch_texts, embedding_results.data):
                vector = emb_data.embedding
                normalized_embedding = (vector / np.linalg.norm(vector)).tolist()
                text_embeddings.append(normalized_embedding)
                
                hash_text = str(text) + "None"
                hash_value = sha256(hash_text.encode()).hexdigest()
                
                # Skip if the document already exists
                if hash_value in existing_ids:
                    #print(f"Document already exists in the database: {text}")
                    continue
                
                sparse_vec = self._get_sparse_vector(text)
                
                # Insert new document
                table_object.insert({
                    "id": hash_value,
                    "text": text,
                    "vec": normalized_embedding,
                    "sparse": sparse_vec
                })
                #print("hello")

        return table_object

    def embed_query(self, db_name: str, text: str) -> list[float]:
        """Embed query text with both dense and sparse embeddings."""
        hash_text = str(text) + "None"
        hash_value = sha256(hash_text.encode()).hexdigest()
        
        embedding_result = self._embeddings_instance.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        vector = embedding_result.data[0].embedding
        embedding_results = (vector / np.linalg.norm(vector)).tolist()
        
        sparse_vec = self._get_sparse_vector(text)  
        
        return sparse_vec, embedding_results

class QueryPraśna:
    def __init__(self, layout_data):
        self.layout_data = layout_data
        self.chat_history = []

    # Example usage
    def store_infinity_vec(self, texts, db_name):
        embedding_instance = embedding
        infinity_object = infinity_embedded.connect("/var/infinity")
        
        # Create cache embedding instance
        cache_embedding = CacheEmbedding(
            inf_db_name=db_name,
            sparse_method="bm25",
            embedding_instance=embedding_instance,
            infinity_object=infinity_object
        )

        return cache_embedding.embed_documents(db_name, texts)
        
    def get_infinity_vec(self, query, db_name, n):
        #print(query)
        embedding_instance = embedding
        infinity_object = infinity_embedded.connect("/var/infinity")
        
        # Create cache embedding instance
        cache_embedding = CacheEmbedding(
            inf_db_name=db_name,
            sparse_method="bm25",
            embedding_instance=embedding_instance,
            infinity_object=infinity_object
        )
        sparse, dense = cache_embedding.embed_query(db_name, query)
        tb_obj = cache_embedding.inf_obj(create_table=False)
        
        op = tb_obj.output(["id", "text", "vec", "sparse", "_score"]) \
                .match_dense("vec", dense, "float", "cosine", n) \
                .match_sparse("sparse", sparse, "ip", n) \
                .match_text("text", query, 10) \
                .fusion("weighted_sum", n, {"weights": "2,1,0.5"}) \
                .to_pl()

        #print(op)
        return op[0]

    def process_document_json(self, layout_data):
        chunks = []
        current_text = ""
        chunk_count = 1
        max_words = 200

        for pages in layout_data[0]:
            elements = sorted(pages, key=lambda x: x['top'])
            current_header = None
            
            for element in elements:
                element_type = element['type'].lower()
                text = element['text'].strip()
                
                if not text:
                    continue
                    
                if element_type == 'header':
                    if current_text:
                        chunks.append(f"{current_text.strip()}")
                        chunk_count += 1
                        current_text = ""
                    current_header = text
                    current_text = text + "\n"
                else:
                    words = len(text.split())
                    if len((current_text + " " + text).split()) > max_words and current_text:
                        chunks.append(f"{current_text.strip()}")
                        chunk_count += 1
                        current_text = text + " "
                    else:
                        current_text += text + " "
        
        # Add remaining text if exists
        if current_text:
            chunks.append(f"{current_text.strip()}")
            current_text = ""
        
        return chunks

    def _get_retriever_agent(self, prompt, query):
        model = "gemini-2.5-pro-preview-05-06"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{prompt}\n{query}"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response =  client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        print(response.text)
        return response.text

    def _get_summarization_agent(self, prompt, query):
        model = "gemini-2.5-flash-preview-05-20"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"{prompt}\n{query}"),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        full_response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            full_response_text += chunk.text

        #print("\n--- Streaming End ---")
        return full_response_text

    def query_agent(self, query, initial_context):
        evaluation_prompt = """
        You are a legal information specialist tasked with determining if the retrieved information 
        is sufficient to answer a user query about legal matters.
        
        User Query: {query}
        
        Retrieved Information:
        {context}
        
        Carefully analyze the information above and determine:
        1. Is the retrieved information relevant to the query?
        2. Is the information sufficient to provide a complete answer?
        3. Are there any specific aspects of the query that are not addressed in the retrieved information?
        
        Respond with JSON in this format:
        {{
        "information_sufficient": true/false,
        "missing_aspects": ["list any missing aspects"],
        "rephrased_query": "a rephrased query to find the missing information (if needed)"
        }}
        
        Be thorough in your analysis and precise in identifying gaps.
        """
        
        evaluation = self._get_retriever_agent(evaluation_prompt, f"Query: {query}\nContext: {initial_context}")
        
        try:
            evaluation = clean_json_response(evaluation)
            evaluation_result = json.loads(evaluation)
            
            if not evaluation_result["information_sufficient"] and evaluation_result["rephrased_query"]:
                additional_pl_data = self.get_infinity_vec(
                    evaluation_result["rephrased_query"], 'legal_prasna', 5
                )
                additional_context = "\n\n".join(additional_pl_data['text'].to_list())
                print(additional_context)
                
                combined_context = initial_context
                for text in additional_pl_data['text'].to_list():
                    if text not in combined_context:
                        combined_context += f"\n\n{text}"
                
                return combined_context
            else:
                return initial_context
        except:
            return initial_context

    def summarization_agent(self, query, context):
        summarization_prompt = """
            You are a legal information specialist whose goal is to explain complex legal concepts using **Markdown formatting**.

            User Query: {query}

            Context Information:
            {context}

            Respond by:

            1. **Adapting to the user's intent**:
            - Use short or long responses depending on the tone or phrasing.
            - If the user asks for *steps*, structure the answer using numbered steps.
            - If the user wants a summary, keep it concise.

            2. **Reflecting sentiment** – match the user's emotional tone (neutral, concerned, urgent, etc.).

            3. **Using Markdown**:
            - Use `**bold**` for key terms.
            - Use bullet points (`*`) or numbered lists (`1.`, `2.`) where appropriate.
            - Use paragraph breaks for clarity.

            4. **Remaining accurate** – rely only on the provided context and do not invent legal advice.

            5. **Simplifying legal language** – explain complex terms in plain English.

            6. **Ending with a helpful question**:
            > Would you like more details or additional context on any part of this explanation?

            Make sure your response is suitable for direct rendering as Markdown or HTML.
        """
        
        initial_summary = self._get_summarization_agent(summarization_prompt, f"Query: {query}\nContext: {context}")
        
        verification_prompt = """
        You are a legal quality assurance specialist. Review this summary response to ensure it fully 
        answers the user's query and accurately represents the source information.
        
        User Query: {query}
        
        Source Information:
        {context}
        
        Generated Summary:
        {summary}
        
        Evaluate and respond with JSON in this format:
        {{
        "is_complete": true/false,
        "is_accurate": true/false, 
        "improvements_needed": ["specific improvements needed, if any"],
        "enhanced_summary": "an improved version of the summary that addresses any identified issues"
        }}
        
        Be thorough but fair in your assessment.
        """
        
        verification = self._get_retriever_agent(
            verification_prompt, 
            f"Query: {query}\nContext: {context}\nSummary: {initial_summary}"
        )
        
        try:
            verification = clean_json_response(verification)
            verification_result = json.loads(verification)
            
            if verification_result["is_complete"] and verification_result["is_accurate"]:
                return initial_summary
            else:
                return verification_result["enhanced_summary"]
        except:
            return initial_summary

    def __call__(self, query, text_content):
        chunker = RecursiveChunker(chunk_size=1024, overlap=50)
    
        # Assume text_content is a single large string or a list of documents
        if isinstance(text_content, str):
            texts = chunker.split_text(text_content)
        elif isinstance(text_content, list):
            texts = [chunk for t in text_content for chunk in chunker.split_text(t)]
        else:
            raise ValueError("text_content must be a string or a list of strings")

        # Store chunks in vector database
        for chunk in texts:
            if chunk.strip():
                self.store_infinity_vec([chunk], 'legal_prasna')
    
        pl_data = self.get_infinity_vec(query, 'legal_prasna', 3)
        initial_context = "\n\n".join(pl_data['text'].to_list())
        #print(f'context{initial_context}')
        context = self.query_agent(query, initial_context)
        return self.summarization_agent(query, context)