"""
RAG Engine for SkySense Copilot using LangChain

Implements retrieval-augmented generation with:
- Local embeddings (sentence-transformers)
- FAISS vector store
- LangChain retrieval chains
- Proper source attribution
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


class SkySenseRAG:
    """
    RAG engine for flight log analysis knowledge retrieval.
    Uses local embeddings and FAISS for efficient retrieval.
    """
    
    def __init__(self, knowledge_base_path: str = None, api_key: str = None):
        """
        Initialize RAG engine.
        
        Args:
            knowledge_base_path: Path to knowledge_base.json
            api_key: Groq API key (or read from env)
        """
        
        self.knowledge_base_path = knowledge_base_path or (
            Path(__file__).parent / "embeddings" / "knowledge_base.json"
        )
        
        self.vector_store_path = (
            Path(__file__).parent / "embeddings" / "vector_store"
        )
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        print("Initializing SkySense RAG engine...")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from JSON"""
        
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(
                f"Knowledge base not found: {self.knowledge_base_path}\n"
                "Run knowledge_builder.py first to create it."
            )
        
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        print(f"✓ Loaded knowledge base from {self.knowledge_base_path}")
        return knowledge
    
    def _create_documents(self, knowledge: Dict[str, Any]) -> List[Document]:
        """Convert knowledge base to LangChain documents with metadata"""
        
        documents = []
        
        # 1. Detector specifications
        for detector_name, spec in knowledge.get("detector_specs", {}).items():
            content = f"""
Detector: {detector_name}

Definition: {spec.get('definition', '')}

Detection Method: {spec.get('detection_method', '')}

Normal Behavior: {spec.get('normal_behavior', '')}

Thresholds:
{json.dumps(spec.get('thresholds', {}), indent=2)}

Common Causes:
{chr(10).join('- ' + cause for cause in spec.get('common_causes', []))}

Remediation Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(spec.get('remediation_steps', [])))}

Related Detectors: {', '.join(spec.get('related_detectors', []))}

Frequency: {spec.get('frequency', '')}
Severity Impact: {spec.get('severity_impact', '')}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "detector_specs",
                    "detector": detector_name,
                    "type": "specification"
                }
            )
            documents.append(doc)
        
        # 2. Terminology
        for term, definition in knowledge.get("terminology", {}).items():
            content = f"Term: {term}\n\nDefinition: {definition}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "terminology",
                    "term": term,
                    "type": "glossary"
                }
            )
            documents.append(doc)
        
        # 3. Normal ranges
        for metric, range_desc in knowledge.get("normal_ranges", {}).items():
            content = f"Metric: {metric}\n\nNormal Range: {range_desc}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "normal_ranges",
                    "metric": metric,
                    "type": "reference"
                }
            )
            documents.append(doc)
        
        # 4. Failure patterns
        for pattern in knowledge.get("failure_patterns", []):
            content = f"""
Failure Pattern: {pattern.get('pattern_name', '')}

Sequence:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(pattern.get('sequence', [])))}

Indicators: {', '.join(pattern.get('indicators', []))}

Prevention: {pattern.get('prevention', '')}

Criticality: {pattern.get('criticality', '')}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "failure_patterns",
                    "pattern": pattern.get('pattern_name', ''),
                    "type": "pattern",
                    "criticality": pattern.get('criticality', '')
                }
            )
            documents.append(doc)
        
        # 5. PX4 docs content (if available)
        for section, content in knowledge.get("px4_docs_content", {}).items():
            # Split large docs into chunks
            doc = Document(
                page_content=content,
                metadata={
                    "source": "px4_docs",
                    "section": section,
                    "type": "documentation",
                    "license": "CC BY 4.0"
                }
            )
            documents.append(doc)
        
        print(f"✓ Created {len(documents)} documents from knowledge base")
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(split_docs)} chunks")
        
        return split_docs
    
    def _initialize_embeddings(self):
        """Initialize local embedding model"""
        
        print("Loading embedding model (this may take a moment)...")
        
        # Use lightweight but effective model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("✓ Embedding model loaded (all-MiniLM-L6-v2, 80MB)")
    
    def build_vector_store(self, force_rebuild: bool = False):
        """Build or load FAISS vector store"""
        
        # Check if vector store already exists
        if self.vector_store_path.exists() and not force_rebuild:
            print(f"Loading existing vector store from {self.vector_store_path}...")
            self._initialize_embeddings()
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vector store loaded")
            return
        
        print("Building new vector store...")
        
        # Load knowledge base
        knowledge = self._load_knowledge_base()
        
        # Create documents
        documents = self._create_documents(knowledge)
        
        # Split into chunks
        split_docs = self._split_documents(documents)
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Create vector store
        print("Creating FAISS index...")
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        
        # Save vector store
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.vector_store_path))
        
        print(f"✓ Vector store saved to {self.vector_store_path}")
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,  # Low for factual accuracy
            max_tokens=1000
        )
        
        print("✓ Groq LLM initialized (llama-3.3-70b-versatile)")
    
    def _create_qa_chain(self):
        """Create RetrievalQA chain with custom prompt"""
        
        prompt_template = """You are SkySense Copilot, an expert drone flight log analyst for PX4 autopilot systems.

Answer concisely using the knowledge base context below. Keep responses under 150 words.

- For detectors/failures: List main causes and 2-3 fixes
- For terminology: Give clear definition with example
- For troubleshooting: Provide actionable steps
- If unsure, say "I need more specific flight data to answer that"

Context: {context}

Question: {question}

Answer (concise):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✓ QA chain created")
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize all components"""
        
        # Build/load vector store
        self.build_vector_store(force_rebuild=force_rebuild)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Create QA chain
        self._create_qa_chain()
        
        print("\n" + "=" * 60)
        print("SkySense RAG engine ready!")
        print("=" * 60)
    
    def build_knowledge_base(self, force_rebuild: bool = False):
        """Alias for initialize() - builds knowledge base and vector store"""
        return self.initialize(force_rebuild=force_rebuild)
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            include_sources: Whether to include source documents in response
        
        Returns:
            Dictionary with answer and optional sources
        """
        
        if self.qa_chain is None:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")
        
        # Use invoke() instead of __call__ to avoid deprecation warning
        result = self.qa_chain.invoke({"query": question})
        
        response = {
            "answer": result["result"],
            "question": question
        }
        
        if include_sources and "source_documents" in result:
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",  # Truncate for display
                    "metadata": doc.metadata
                })
            response["sources"] = sources
        
        return response
    
    def get_relevant_context(self, question: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents without LLM generation.
        Useful for debugging or custom processing.
        """
        
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        
        return self.vector_store.similarity_search(question, k=k)


if __name__ == "__main__":
    # Example usage
    
    print("Initializing SkySense RAG...")
    rag = SkySenseRAG()
    
    # Initialize (builds vector store if needed)
    rag.initialize(force_rebuild=False)
    
    # Test queries
    test_questions = [
        "What causes battery sag?",
        "How do I fix vibration issues?",
        "What is an EKF innovation spike?",
        "Why is my tracking error high?"
    ]
    
    print("\n" + "=" * 60)
    print("Testing RAG with sample questions...")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = rag.query(question, include_sources=False)
        print(f"A: {result['answer']}")
        print("-" * 60)
