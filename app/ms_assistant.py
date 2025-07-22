from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os
import re
from openai import OpenAI
from pydantic import EmailStr
from sqlalchemy.orm import Session
from app.models import AI_User, AI_ChatSession, AI_ChatMessage, AI_Document, AI_DocumentChunk
from app.database import SessionLocal
from sentence_transformers import SentenceTransformer
import chromadb
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import uuid
from fastapi import UploadFile
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAssistantError(Exception):
    """Base exception for AI Assistant errors"""
    pass

class AIAssistant:
    """Main AI class for AI health assistance using ChatGPT with Mycotoxin testing knowledge"""
    
    def __init__(self, db: Session):
        """Initialize AIAssistant with OpenAI client and database session"""
        try:
            self.db = db
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.mycotoxin_knowledge = self._load_mycotoxin_knowledge()
            self.system_prompt = self._create_enhanced_system_prompt()
            # Use a persistent uploads directory
            self.upload_dir = os.path.join("uploads")
            os.makedirs(self.upload_dir, exist_ok=True)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Use PersistentClient for ChromaDB persistence
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        except Exception as e:
            logger.error(f"Error initializing AIAssistant: {str(e)}")
            raise AIAssistantError("Failed to initialize AI Assistant")

    def _clean_text(self, text: str) -> str:
        try:
            text = text.replace('\x00', '')
            text = text.replace('\r', ' ')
            text = text.replace('\t', ' ')
            text = ' '.join(text.split())
            text = ''.join(char for char in text if char == '\n' or char >= ' ')
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text

    def _extract_pdf_text(self, file_path: str) -> tuple[str, int]:
        try:
            text_content = ""
            page_count = 0
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            cleaned_text = self._clean_text(page_text)
                            if cleaned_text.strip():
                                text_content += f"\n[Page {page_num}]\n{cleaned_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
            text_content = self._clean_text(text_content)
            if not text_content.strip():
                raise AIAssistantError("No text could be extracted from the PDF")
            return text_content, page_count
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise AIAssistantError(f"Failed to extract text from PDF: {str(e)}")

    def _extract_txt_text(self, file_path: str) -> tuple[str, int]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            text = self._clean_text(text)
            if not text.strip():
                raise AIAssistantError("No text could be extracted from the TXT file")
            return text, 1
        except Exception as e:
            logger.error(f"Error extracting TXT text: {str(e)}")
            raise AIAssistantError(f"Failed to extract text from TXT: {str(e)}")

    def get_or_create_collection(self, email: str = None):
        try:
            return self.chroma_client.get_or_create_collection(
                name="public_documents",
                metadata={"type": "public"}
            )
        except Exception as e:
            logger.error(f"Error getting/creating collection: {str(e)}")
            raise AIAssistantError(f"Failed to get/create collection: {str(e)}")

    async def upload_and_process_document(self, email: str, file: UploadFile, author: Optional[str] = None, description: Optional[str] = None, category: Optional[str] = None, is_public: bool = True) -> str:
        file_path = ""
        try:
            filename = file.filename
            document_id = str(uuid.uuid4())
            file_path = os.path.join(self.upload_dir, f"{document_id}_{filename}")
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext == ".pdf":
                text_content, page_count = self._extract_pdf_text(file_path)
            elif file_ext == ".txt":
                text_content, page_count = self._extract_txt_text(file_path)
            else:
                raise AIAssistantError(f"Unsupported file type: {file_ext}")
            doc_category = category or "Uncategorized"
            document = AI_Document(
                id=document_id,
                admin_email=email,
                document_name=filename,
                file_path=file_path,
                upload_on=datetime.utcnow(),
                processed=False,
                page_count=page_count,
                is_public=is_public,
                author=author,
                description=description,
                category=doc_category
            )
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            await self._process_and_store_chunks(document_id, email, text_content, filename, doc_category)
            document.processed = True
            self.db.commit()
            logger.info(f"Successfully processed document {filename} for admin {email}")
            return document_id
        except Exception as e:
            logger.error(f"Error processing document {getattr(file, 'filename', 'unknown')}: {e}")
            if file_path and os.path.exists(file_path):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except OSError as ose:
                    logger.error(f"Error removing file {file_path}: {ose}")
            self.db.rollback()
            raise AIAssistantError(f"Failed to process document: {e}")

    async def _process_and_store_chunks(self, document_id: str, email: str, text_content: str, document_name: str, category: Optional[str] = None):
        try:
            collection = self.get_or_create_collection()
            chunks = self._split_text_into_chunks(text_content)
            chunk_category = category or "Uncategorized"
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                chunk_embeddings = []
                chunk_texts = []
                chunk_metadata = []
                chunk_ids = []
                for j, chunk_text in enumerate(batch_chunks, i):
                    try:
                        chunk_text = self._clean_text(chunk_text)
                        if not chunk_text.strip():
                            continue
                        chunk_embedding_id = f"{document_id}_chunk_{j}"
                        metadata = {
                            "document_id": str(document_id),
                            "document_name": str(document_name),
                            "chunk_index": str(j),
                            "category": str(chunk_category),
                            "is_public": "true"
                        }
                        embedding = self.embedding_model.encode(chunk_text).tolist()
                        chunk_embeddings.append(embedding)
                        chunk_texts.append(chunk_text)
                        chunk_metadata.append(metadata)
                        chunk_ids.append(chunk_embedding_id)
                        chunk = AI_DocumentChunk(
                            document_id=document_id,
                            chunk_text=chunk_text,
                            chunk_index=j,
                            embedding_id=chunk_embedding_id,
                            category=chunk_category
                        )
                        self.db.add(chunk)
                    except Exception as e:
                        logger.warning(f"Error processing chunk {j}: {e}")
                        continue
                if chunk_embeddings:
                    collection.add(
                        ids=chunk_ids,
                        embeddings=chunk_embeddings,
                        documents=chunk_texts,
                        metadatas=chunk_metadata
                    )
                    self.db.commit()
            logger.info(f"Stored {len(chunks)} chunks for document {document_name}")
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            self.db.rollback()
            raise AIAssistantError(f"Failed to process chunks: {e}")

    def _split_text_into_chunks(self, text_content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        current_page = 1
        page_sections = re.split(r'\[Page (\d+)\]', text_content)
        for i in range(1, len(page_sections), 2):
            if i + 1 < len(page_sections):
                page_num = int(page_sections[i])
                page_text = page_sections[i + 1].strip()
                if not page_text:
                    continue
                sentences = sent_tokenize(page_text)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        if overlap > 0:
                            words = current_chunk.split()
                            current_chunk = ' '.join(words[-overlap:]) + ' '
                        else:
                            current_chunk = ""
                    current_chunk += sentence + " "
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        return chunks

    def _load_mycotoxin_knowledge(self) -> Dict[str, Any]:
        """Load mycotoxin testing knowledge base"""
        return {
            "tests": {
                "ochratoxin_a": {
                    "name": "Urine: Ochratoxin A",
                    "reference_ranges": {
                        "not_present": "<1.8",
                        "equivocal": "1.8 to <2",
                        "present": ">=2"
                    }
                },
                "aflatoxin_group": {
                    "name": "Urine: Aflatoxin Group (B1, G1, G2)",
                    "reference_ranges": {
                        "not_present": "<0.8",
                        "equivocal": "0.8 to <1",
                        "present": ">=1"
                    }
                },
                "trichothecene_group": {
                    "name": "Urine: Trichothecene Group (Macrocyclic)",
                    "components": ["Roridin A", "Roridin E", "Roridin H", "Roridin L-2", 
                                 "Verrucarin A", "Verrucarin J", "Satratoxin G", "Satratoxin H", "Isosatratoxin F"],
                    "reference_ranges": {
                        "not_present": "<0.07",
                        "equivocal": "0.07 to <0.09",
                        "present": ">=0.09"
                    }
                },
                "gliotoxin_derivative": {
                    "name": "Urine: Gliotoxin Derivative",
                    "reference_ranges": {
                        "not_present": "<0.5",
                        "equivocal": "0.5 to <1",
                        "present": ">=1"
                    }
                },
                "zearalenone": {
                    "name": "Urine: Zearalenone",
                    "reference_ranges": {
                        "not_present": "<0.5",
                        "equivocal": "0.5 to <0.7",
                        "present": ">=0.7"
                    }
                }
            },
            "symptoms": {
                "physical": [
                    "Headaches & Dizziness", "Nosebleeds", "Painful Lymph Nodes", "Asthma",
                    "Shortness Of Breath", "Gastrointestinal distress", "Decreased Libido",
                    "Hair Loss", "Brain Fog", "Sinusitis & Sinus Issues", "Hearing Problems",
                    "Cardiac Arrhythmias", "Abdominal Pain and Discomfort", 
                    "Numbness and Tingling In Hands", "Uncomfortable or Frequent Urination",
                    "Rashes & Hives", "Muscles and Joint Aches and Pains", "Fluid Retention",
                    "Numbness and Tingling In Feet"
                ],
                "systemic": [
                    "Depression", "Anxiety", "Chronic Fatigue", "Chronic Illness",
                    "General Weakness", "Immune Suppression", "Anemia", "Night Sweats"
                ]
            },
            "mycotoxins": {
                "aflatoxin_b1": {
                    "activity": "Binds DNA and proteins",
                    "symptoms": "Shortness of breath, weight loss, most potent and highly carcinogenic",
                    "disease_state": "Primarily attacks the liver, other organs include kidneys and lungs"
                },
                "aflatoxin_b2": {
                    "activity": "Inhibits DNA and RNA replication",
                    "symptoms": "Impaired fetal growth",
                    "disease_state": "Affects the liver and kidneys"
                },
                "aflatoxin_g1": {
                    "activity": "Cytotoxic, induces apoptosis in cells, DNA damage",
                    "symptoms": "A flavus is a leading cause of invasive aspergillus in immunocompromised patients",
                    "disease_state": "Cancer, neonatal jaundice"
                },
                "aflatoxin_g2": {
                    "activity": "Cancer, neonatal jaundice",
                    "symptoms": "Aflatoxicosis in humans and animals",
                    "disease_state": "Malnutrition, lung cancer"
                },
                "ochratoxin_a": {
                    "activity": "Inhibits mitochondrial ATP, potent teratogen, and immune suppressor",
                    "symptoms": "Fatigue, dermatitis, irritated bowel",
                    "disease_state": "Kidney disease and cancer"
                },
                "satratoxin_g": {
                    "activity": "DNA, RNA, and protein synthesis inhibition",
                    "symptoms": "Fatigue",
                    "disease_state": "Bleeding disorders, nervous system disorders"
                },
                "satratoxin_h": {
                    "activity": "Inhibits protein synthesis",
                    "symptoms": "Fatigue",
                    "disease_state": "Breathing issues"
                },
                "isosatratoxin": {
                    "activity": "Immunosuppression",
                    "symptoms": "Weakened immune system",
                    "disease_state": ""
                },
                "roridin_a": {
                    "activity": "Immunosuppression",
                    "symptoms": "Weakened immune system",
                    "disease_state": ""
                },
                "roridin_e": {
                    "activity": "DNA, RNA, and protein synthesis disruption",
                    "symptoms": "Weakened immune system",
                    "disease_state": "Lung and nasal olfactory problems"
                },
                "roridin_h": {
                    "activity": "Inhibits protein synthesis",
                    "symptoms": "Weakened immune system",
                    "disease_state": ""
                },
                "roridin_l2": {
                    "activity": "Immunosuppression",
                    "symptoms": "Weakened immune system",
                    "disease_state": ""
                },
                "verrucarin_a": {
                    "activity": "Immunosuppression",
                    "symptoms": "",
                    "disease_state": ""
                },
                "verrucarin_j": {
                    "activity": "Immunosuppression",
                    "symptoms": "",
                    "disease_state": ""
                },
                "gliotoxin": {
                    "activity": "Attacks intracellular function in immune system",
                    "symptoms": "Memory and breathing issues",
                    "disease_state": "Immune dysfunction disorders"
                },
                "zearalenone": {
                    "activity": "Estrogen mimic",
                    "symptoms": "Early puberty, low sperm counts, cancer",
                    "disease_state": "Cancer"
                }
            }
        }

    def _create_enhanced_system_prompt(self) -> str:
        """Create enhanced system prompt with mycotoxin knowledge"""
        return """You are an AI assistant specialized in Multiple Sclerosis (MS) and mycotoxin testing. 
Your role is to provide accurate, helpful, and supportive information about:

MS-related topics:
- Symptoms and their management
- Treatment options and medications
- Lifestyle recommendations
- Diagnostic procedures
- Research updates
- Support resources

Mycotoxin testing and environmental health:
- Mycotoxin exposure symptoms and health effects
- Urine mycotoxin testing interpretation
- Environmental sources of mycotoxins
- Relationship between mycotoxin exposure and MS symptoms
- Recommendations for mycotoxin exposure reduction

You have access to specialized knowledge about mycotoxin testing including:
- Reference ranges for 5 key mycotoxin tests (Ochratoxin A, Aflatoxin Group, Trichothecene Group, Gliotoxin Derivative, Zearalenone)
- Detailed information about specific mycotoxins and their health effects
- Comprehensive symptom lists for mycotoxin exposure

When discussing mycotoxin testing:
1. Always recommend consulting with healthcare providers for testing and interpretation
2. Suggest appropriate tests based on symptoms described
3. If test results are provided, analyze them against reference ranges
4. Provide specific recommendations based on test results
5. Explain the connection between mycotoxin exposure and various health conditions

Always maintain a professional, empathetic tone and encourage users to work with qualified healthcare providers for personalized medical advice."""

    def process_message(self, session_id: str, message: str, email: EmailStr) -> str:
        """Process a user message and return AI response"""
        try:
            # Validate inputs
            if not message or not email:
                raise AIAssistantError("Message and email are required")
            
            # Get or create session
            session = self._get_or_create_session(session_id, email)
            
            # Get chat history
            chat_history = self._get_chat_history(session_id)
            
            # Process with ChatGPT
            response = self._get_chatgpt_response(message, chat_history)
            
            # Store the message and response
            self._store_chat_message(session_id, message, response)
            
            # Update session title if needed
            self._update_session_title(session, message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AIAssistantError(f"Failed to process message: {str(e)}")

    def _is_mycotoxin_related(self, message: str) -> bool:
        """Check if message is related to mycotoxins"""
        mycotoxin_keywords = [
            "mycotoxin", "mold", "fungal", "ochratoxin", "aflatoxin", "trichothecene",
            "gliotoxin", "zearalenone", "satratoxin", "roridin", "verrucarin",
            "environmental testing", "urine test", "mold exposure", "fungal exposure"
        ]
        return any(keyword in message.lower() for keyword in mycotoxin_keywords)

    def _parse_test_results(self, message: str) -> Dict[str, float]:
        """Parse numerical test results from message"""
        results = {}
        
        # Look for numerical values that might be test results
        patterns = [
            r"ochratoxin.*?(\d+\.?\d*)",
            r"aflatoxin.*?(\d+\.?\d*)",
            r"trichothecene.*?(\d+\.?\d*)",
            r"gliotoxin.*?(\d+\.?\d*)",
            r"zearalenone.*?(\d+\.?\d*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                test_name = pattern.split(".*?")[0]
                value = float(match.group(1))
                results[test_name] = value
        
        return results

    def _create_mycotoxin_context(self, test_results: Dict[str, float] = None) -> str:
        """Create relevant mycotoxin context for the conversation"""
        context = "MYCOTOXIN TESTING KNOWLEDGE:\n\n"
        
        # Add test reference ranges
        context += "TEST REFERENCE RANGES:\n"
        for test_key, test_info in self.mycotoxin_knowledge["tests"].items():
            context += f"- {test_info['name']}:\n"
            for range_type, range_value in test_info["reference_ranges"].items():
                context += f"  {range_type.replace('_', ' ').title()}: {range_value}\n"
        
        # Add symptoms
        context += "\nMYCOTOXIN EXPOSURE SYMPTOMS:\n"
        context += "Physical: " + ", ".join(self.mycotoxin_knowledge["symptoms"]["physical"][:10]) + "...\n"
        context += "Systemic: " + ", ".join(self.mycotoxin_knowledge["symptoms"]["systemic"]) + "\n"
        
        # Add test result analysis if provided
        if test_results:
            context += "\nTEST RESULT ANALYSIS:\n"
            for test_name, value in test_results.items():
                interpretation = self._interpret_test_result(test_name, value)
                context += f"- {test_name.title()}: {value} ({interpretation})\n"
        
        return context

    def _interpret_test_result(self, test_name: str, value: float) -> str:
        """Interpret a test result based on reference ranges"""
        # Map test names to knowledge base keys
        test_mapping = {
            "ochratoxin": "ochratoxin_a",
            "aflatoxin": "aflatoxin_group",
            "trichothecene": "trichothecene_group",
            "gliotoxin": "gliotoxin_derivative",
            "zearalenone": "zearalenone"
        }
        
        test_key = test_mapping.get(test_name)
        if not test_key or test_key not in self.mycotoxin_knowledge["tests"]:
            return "Unable to interpret"
        
        test_info = self.mycotoxin_knowledge["tests"][test_key]
        ranges = test_info["reference_ranges"]
        
        # Parse reference ranges and compare
        not_present_threshold = float(ranges["not_present"].replace("<", ""))
        equivocal_lower = float(ranges["equivocal"].split(" to ")[0])
        present_threshold = float(ranges["present"].replace(">=", ""))
        
        if value < not_present_threshold:
            return "Not Present"
        elif equivocal_lower <= value < present_threshold:
            return "Equivocal"
        else:
            return "Present"

    def _get_or_create_session(self, session_id: str, email: str) -> AI_ChatSession:
        """Get existing session or create new one"""
        try:
            session = self.db.query(AI_ChatSession).filter(AI_ChatSession.id == session_id).first()
            
            if not session:
                # Create new user if needed
                user = self.db.query(AI_User).filter(AI_User.email == email).first()
                if not user:
                    user = AI_User(email=email)
                    self.db.add(user)
                    self.db.commit()
                
                # Create new session
                session = AI_ChatSession(
                    id=session_id,
                    email=email,
                    analysis_complete=False,
                    ai_state={},
                    title="New MS Consultation",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                self.db.add(session)
                self.db.commit()
                self.db.refresh(session)  # Refresh to get all fields populated
            
            return session
            
        except Exception as e:
            logger.error(f"Error in get_or_create_session: {str(e)}")
            raise AIAssistantError("Failed to get or create session")

    def _get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """
        Get recent chat history for a session.
        
        Args:
            session_id: The session ID to get history for
            limit: Maximum number of message pairs to return (default: 20)
            
        Returns:
            List of message dictionaries in chronological order
        """
        try:
            # Get messages ordered by timestamp
            messages = self.db.query(AI_ChatMessage).filter(
                AI_ChatMessage.session_id == session_id
            ).order_by(AI_ChatMessage.timestamp.asc()).all()
            
            # Convert to list of dicts
            history = []
            for msg in messages:
                # Add user message
                history.append({
                    "role": "user",
                    "content": msg.query_text
                })
                # Add assistant response
                history.append({
                    "role": "assistant",
                    "content": msg.response_text
                })
            
            # If we have too many messages, keep the most recent ones
            if len(history) > limit * 2:  # *2 because each exchange has 2 messages
                history = history[-limit * 2:]
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []

    def _interpret_mycotoxin_result(self, test_name: str, value: float) -> str:
        """
        Interpret mycotoxin test result based on reference ranges.
        Returns one of: 'Not Present', 'Equivocal', or 'Present'
        """
        try:
            # Define reference ranges for each test
            reference_ranges = {
                "Ochratoxin A": {
                    "not_present": 1.8,
                    "equivocal_min": 1.8,
                    "equivocal_max": 2.0
                },
                "Aflatoxin Group (Bl, G1, G2)": {
                    "not_present": 0.8,
                    "equivocal_min": 0.8,
                    "equivocal_max": 1.0
                },
                "Trichothecene Group (Macrocyclic)": {
                    "not_present": 0.07,
                    "equivocal_min": 0.07,
                    "equivocal_max": 0.09
                },
                "Giotoxin Derivative": {
                    "not_present": 0.5,
                    "equivocal_min": 0.5,
                    "equivocal_max": 1.0
                },
                "Zearalenone": {
                    "not_present": 0.5,
                    "equivocal_min": 0.5,
                    "equivocal_max": 0.7
                }
            }

            if test_name not in reference_ranges:
                return "Unable to interpret"

            ranges = reference_ranges[test_name]
            
            if value < ranges["not_present"]:
                return "Not Present"
            elif ranges["equivocal_min"] <= value < ranges["equivocal_max"]:
                return "Equivocal"
            else:
                return "Present"

        except Exception as e:
            logger.error(f"Error interpreting mycotoxin result: {str(e)}")
            return "Unable to interpret"

    def _extract_test_values(self, message: str) -> Dict[str, float]:
        """Extract test values from the message"""
        # This is a simple example - you might need more sophisticated parsing
        test_values = {}
        try:
            # Example message format: "Ochratoxin A: 2.1, Aflatoxin Group: 0.9, ..."
            parts = message.split(',')
            for part in parts:
                if ':' in part:
                    test_name, value = part.split(':')
                    test_name = test_name.strip()
                    value = float(value.strip())
                    test_values[test_name] = value
        except Exception as e:
            logger.error(f"Error extracting test values: {str(e)}")
        return test_values

    def _get_chatgpt_response(self, message: str, chat_history: List[Dict[str, str]]) -> str:
        """Get response from ChatGPT with beautiful markdown formatting"""
        try:
            # Check if the message contains mycotoxin test results
            if "mycotoxin" in message.lower() or "test results" in message.lower():
                # Extract test values from the message
                test_values = self._extract_test_values(message)
                if test_values:
                    return self._format_mycotoxin_response(test_values)

            # Regular chat response for non-mycotoxin queries
            messages = [
                {
                    "role": "system", 
                    "content": """You are an AI assistant specialized in Multiple Sclerosis (MS) and mycotoxin testing. 
                    Provide accurate, helpful information while maintaining a professional and empathetic tone.
                    
                    Format your responses using markdown with the following guidelines:
                    1. Use headers (##) for main topics
                    2. Use bullet points (•) for lists
                    3. Use bold (**) for important terms or key points
                    4. Use italics (*) for emphasis
                    5. Use code blocks (```) for any technical information
                    6. Use blockquotes (>) for important notes or warnings
                    7. Use tables when presenting structured information
                    8. Use horizontal rules (---) to separate different sections
                    9. Use emojis sparingly and professionally (e.g., ℹ️ for information, ⚠️ for warnings)
                    
                    Always structure your response in a clear, organized manner with proper spacing and formatting."""
                }
            ]
            
            messages.extend(chat_history)
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting ChatGPT response: {str(e)}")
            raise AIAssistantError("Failed to get AI response")

    def _format_mycotoxin_response(self, test_results: Dict[str, float]) -> str:
        """Format mycotoxin test results in a beautiful markdown format"""
        response = """## Mycotoxin Test Results Analysis

### Test Results Summary

| Test | Value | Result | Interpretation |
|------|-------|---------|----------------|
"""
        
        # Add each test result to the table
        for test_name, value in test_results.items():
            interpretation = self._interpret_mycotoxin_result(test_name, value)
            response += f"| {test_name} | {value} | {interpretation} | {self._get_interpretation_description(interpretation)} |\n"
        
        response += """
---

### Detailed Analysis

"""
        
        # Add detailed analysis for each test
        for test_name, value in test_results.items():
            interpretation = self._interpret_mycotoxin_result(test_name, value)
            response += f"#### {test_name}\n"
            response += f"**Value:** {value}\n"
            response += f"**Result:** {interpretation}\n\n"
            response += f"{self._get_detailed_analysis(test_name, interpretation)}\n\n"
        
        response += """
---

### Recommendations

"""
        
        # Generate and add recommendations
        recommendations = self._generate_recommendations(test_results)
        for i, recommendation in enumerate(recommendations, 1):
            response += f"{i}. {recommendation}\n"
        
        response += """
---

> ⚠️ **Important Note:** These interpretations are based on standard reference ranges. Please consult with a healthcare professional for personalized medical advice.

ℹ️ *For more detailed information about specific tests or recommendations, please ask follow-up questions.*
"""
        
        return response

    def _get_interpretation_description(self, interpretation: str) -> str:
        """Get a brief description of the interpretation"""
        descriptions = {
            "Not Present": "No significant levels detected",
            "Equivocal": "Borderline levels, may need monitoring",
            "Present": "Significant levels detected",
            "Unable to interpret": "Insufficient data for interpretation"
        }
        return descriptions.get(interpretation, "Unknown interpretation")

    def _get_detailed_analysis(self, test_name: str, interpretation: str) -> str:
        """Get detailed analysis for a specific test result"""
        analysis = {
            "Not Present": "No significant levels of this mycotoxin were detected in your system.",
            "Equivocal": "Borderline levels were detected. This may require monitoring and follow-up testing.",
            "Present": "Significant levels were detected. This may indicate exposure to this mycotoxin."
        }
        return analysis.get(interpretation, "Unable to provide detailed analysis.")

    def _generate_recommendations(self, test_results: Dict[str, float]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Add general recommendations
        recommendations.append("Consult with a healthcare professional for personalized medical advice")
        
        # Add specific recommendations based on results
        for test_name, value in test_results.items():
            interpretation = self._interpret_mycotoxin_result(test_name, value)
            if interpretation == "Present":
                recommendations.append(f"Monitor {test_name} levels - consider retesting and environmental assessment")
            elif interpretation == "Equivocal":
                recommendations.append(f"Monitor {test_name} levels - consider retesting in 3-6 months")
        
        return recommendations

    def _store_chat_message(self, session_id: str, message: str, response: str) -> None:
        """Store chat message and response in database"""
        try:
            chat_message = AI_ChatMessage(
                session_id=session_id,
                query_text=message,
                response_text=response,
                timestamp=datetime.utcnow()
            )
            self.db.add(chat_message)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error storing chat message: {str(e)}")
            raise AIAssistantError("Failed to store chat message")

    def _update_session_title(self, session: AI_ChatSession, message: str) -> None:
        """Update session title based on conversation context"""
        try:
            # Get recent messages for context
            recent_messages = self._get_chat_history(str(session.id), limit=5)
            
            # Prepare context for title generation
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])
            
            # Get title suggestion from ChatGPT
            title_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate a concise, descriptive title (max 50 chars) for this MS consultation based on the conversation context."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # Update session title
            new_title = title_response.choices[0].message.content.strip()
            if len(new_title) > 50:
                new_title = new_title[:47] + "..."
            
            session.title = new_title
            session.last_updated = datetime.utcnow()
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating session title: {str(e)}")
            # Don't raise error, just log it and keep existing title

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a session"""
        try:
            session = self.db.query(AI_ChatSession).filter(AI_ChatSession.id == session_id).first()
            if not session:
                return None
                
            return {
                "session_id": str(session.id),
                "email": session.email,
                "stage": session.stage,
                "analysis_complete": session.analysis_complete,
                "title": session.title,
                "created_at": session.created_at,
                "last_updated": session.last_updated
            }
            
        except Exception as e:
            logger.error(f"Error getting session state: {str(e)}")
            return None

    def clear_session(self, session_id: str) -> bool:
        try:
            # First, delete all messages explicitly (safe & reliable)
            self.db.query(AI_ChatMessage).filter(AI_ChatMessage.session_id == session_id).delete(
                synchronize_session=False
            )

            session = self.db.query(AI_ChatSession).filter(AI_ChatSession.id == session_id).first()
            if not session:
                return False

            self.db.delete(session)
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting session {session_id}: {str(e)}", exc_info=True)
            raise AIAssistantError("Failed to delete session")

    def suggest_mycotoxin_testing(self, symptoms: List[str]) -> Dict[str, Any]:
        """Suggest mycotoxin tests based on reported symptoms"""
        try:
            all_symptoms = (self.mycotoxin_knowledge["symptoms"]["physical"] + 
                          self.mycotoxin_knowledge["symptoms"]["systemic"])
            
            matching_symptoms = [s for s in symptoms if s in all_symptoms]
            
            if matching_symptoms:
                return {
                    "recommend_testing": True,
                    "matching_symptoms": matching_symptoms,
                    "suggested_tests": list(self.mycotoxin_knowledge["tests"].keys()),
                    "message": "Based on your symptoms, mycotoxin testing may be beneficial."
                }
            else:
                return {
                    "recommend_testing": False,
                    "message": "Your symptoms don't strongly suggest mycotoxin exposure, but testing could still be considered."
                }
                
        except Exception as e:
            logger.error(f"Error suggesting mycotoxin testing: {str(e)}")
            return {"recommend_testing": False, "message": "Unable to analyze symptoms for testing recommendations."}

    def _contains_symptoms(self, message: str) -> bool:
        """Check if message contains any known symptoms"""
        all_symptoms = (self.mycotoxin_knowledge["symptoms"]["physical"] + 
                       self.mycotoxin_knowledge["symptoms"]["systemic"])
        return any(symptom.lower() in message.lower() for symptom in all_symptoms)

    def add_document_to_knowledge_base(self, filename: str, text: str, public: bool = True):
        """
        Ingest a document: chunk, embed, and store in ChromaDB and SQL DB.
        Args:
            filename: Name of the file
            text: Extracted text content
            public: If True, document is accessible to all users
        """
        try:
            # Chunk text into ~500 word chunks
            sentences = sent_tokenize(text)
            chunks = []
            chunk = []
            word_count = 0
            for sent in sentences:
                words = sent.split()
                if word_count + len(words) > 500 and chunk:
                    chunks.append(' '.join(chunk))
                    chunk = []
                    word_count = 0
                chunk.append(sent)
                word_count += len(words)
            if chunk:
                chunks.append(' '.join(chunk))

            # Initialize embedding model and ChromaDB client
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            chroma_client = chromadb.Client()
            collection = chroma_client.get_or_create_collection('ms_knowledge_base')

            # Embed and store each chunk
            for i, chunk_text in enumerate(chunks):
                embedding = embedder.encode(chunk_text).tolist()
                # Store in ChromaDB
                collection.add(
                    documents=[chunk_text],
                    metadatas=[{"filename": filename, "public": public, "chunk_id": i}],
                    ids=[f"{filename}_{i}"]
                )
                # Optionally, store metadata in SQL DB (not implemented here, but can be added)
        except Exception as e:
            logger.error(f"Error ingesting document {filename}: {str(e)}")
            raise AIAssistantError(f"Failed to ingest document: {filename}")