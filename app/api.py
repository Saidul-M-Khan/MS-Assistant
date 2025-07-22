from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Annotated, Union
from pydantic import BaseModel, EmailStr
import logging
from datetime import datetime
import os
from PyPDF2 import PdfReader
from app.models import Base, AI_User, AI_ChatSession as DBSession, AI_ChatMessage, AI_Document, AI_DocumentChunk
from app.ms_assistant import AIAssistant, AIAssistantError
from app.database import get_db, engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="MS Assistant API with Mycotoxin Testing")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class MessageRequest(BaseModel):
    session_id: Optional[str] = None  # Make session_id optional
    message: str
    email: EmailStr

class MessageResponse(BaseModel):
    session_id: str
    message: str
    response: str
    timestamp: datetime

class SessionInfo(BaseModel):
    session_id: str
    email: str
    title: str
    created_at: datetime
    last_updated: datetime

class ChatMessageInfo(BaseModel):
    id: str
    session_id: str
    message: str
    response: str
    timestamp: datetime

class MycotoxinTestResult(BaseModel):
    test_name: str
    value: float
    unit: Optional[str] = None

class MycotoxinTestRequest(BaseModel):
    session_id: str
    email: EmailStr
    test_results: List[MycotoxinTestResult]

class MycotoxinTestResponse(BaseModel):
    session_id: str
    analysis: str
    recommendations: List[str]
    test_interpretations: Dict[str, str]

class SymptomAnalysisRequest(BaseModel):
    symptoms: List[str]

class SymptomAnalysisResponse(BaseModel):
    recommend_testing: bool
    matching_symptoms: List[str]
    suggested_tests: List[str]
    message: str

class EmailRequest(BaseModel):
    email: EmailStr

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    email: str
    title: str

class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]

def normalize_files(files):
    """Normalize files input to always return a list of UploadFile objects"""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="'files' is required (multipart/form-data)"
        )
    
    # If it's a single UploadFile, convert to list
    if isinstance(files, UploadFile):
        return [files]
    
    # If it's already a list of UploadFile objects, return as is
    if isinstance(files, list) and all(isinstance(f, UploadFile) for f in files):
        return files
    
    # If none of the above, raise error
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, 
        detail="'files' must be UploadFile or list of UploadFile (multipart/form-data)"
    )

@app.post("/session/create", response_model=SessionResponse)
async def create_session(request: EmailRequest, db: Session = Depends(get_db)):
    """
    Create a new session for a user.
    
    Args:
        request: EmailRequest containing the user's email
        db: Database session
        
    Returns:
        SessionResponse containing the created session details
    """
    try:
        # Check if user exists, if not create
        user = db.query(AI_User).filter(AI_User.email == request.email).first()
        if not user:
            try:
                user = AI_User(email=request.email)
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Created new user with email: {request.email}")
            except Exception as e:
                logger.error(f"Error creating user: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
        
        # Create new session
        try:
            session = DBSession(
                email=request.email,
                title="New MS Consultation"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info(f"Created new session with ID: {session.id}")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
        
        # Convert session to response model
        try:
            return SessionResponse(
                session_id=str(session.id),
                created_at=session.created_at,
                email=session.email,
                title=session.title
            )
        except Exception as e:
            logger.error(f"Error creating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to create response: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat", response_model=MessageResponse)
async def process_message(request: MessageRequest, db: Session = Depends(get_db)):
    """
    Process a user message and return the AI's response.
    If no session_id is provided, creates a new session.
    
    Args:
        request: MessageRequest containing session_id (optional), message, and email
        db: Database session
        
    Returns:
        MessageResponse containing the AI's response, session_id, message, and timestamp
    """
    try:
        ai_assistant = AIAssistant(db)
        
        # If no session_id provided, create new session
        if not request.session_id:
            # Check if user exists, if not create
            user = db.query(AI_User).filter(AI_User.email == request.email).first()
            if not user:
                user = AI_User(email=request.email)
                db.add(user)
                db.commit()
                db.refresh(user)
            
            # Create new session
            session = DBSession(
                email=request.email,
                title="New MS Consultation"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            request.session_id = str(session.id)
            logger.info(f"Created new session with ID: {request.session_id}")
        
        # Process message
        response = ai_assistant.process_message(
            session_id=request.session_id,
            message=request.message,
            email=request.email
        )
        logger.info(f"AI response: {response}")
        if not response or not isinstance(response, str):
            logger.error("AI assistant returned an empty or invalid response.")
            raise HTTPException(status_code=500, detail="AI assistant failed to generate a response.")
        # Get current timestamp
        current_time = datetime.utcnow()
        
        return MessageResponse(
            response=response,
            session_id=request.session_id,
            message=request.message,
            timestamp=current_time
        )
        
    except AIAssistantError as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze-mycotoxin-tests", response_model=MycotoxinTestResponse)
async def analyze_mycotoxin_tests(request: MycotoxinTestRequest, db: Session = Depends(get_db)):
    """
    Analyze mycotoxin test results and provide recommendations.
    
    Args:
        request: MycotoxinTestRequest containing test results
        db: Database session
        
    Returns:
        MycotoxinTestResponse with analysis and recommendations
    """
    try:
        ai_assistant = AIAssistant(db)
        
        # Format test results for analysis
        test_results_text = "Mycotoxin Test Results:\n"
        test_interpretations = {}
        
        for result in request.test_results:
            interpretation = ai_assistant._interpret_test_result(result.test_name.lower(), result.value)
            test_interpretations[result.test_name] = interpretation
            test_results_text += f"- {result.test_name}: {result.value} ({interpretation})\n"
        
        # Process through the main chat system for comprehensive analysis
        analysis_response = ai_assistant.process_message(
            session_id=request.session_id,
            message=f"Please analyze these mycotoxin test results and provide detailed recommendations:\n\n{test_results_text}",
            email=request.email
        )
        
        # Generate specific recommendations based on results
        recommendations = []
        for result in request.test_results:
            interpretation = test_interpretations[result.test_name]
            if interpretation == "Present":
                recommendations.append(f"Address {result.test_name} exposure - consider environmental remediation and detoxification support")
            elif interpretation == "Equivocal":
                recommendations.append(f"Monitor {result.test_name} levels - consider retesting and environmental assessment")
        
        if not recommendations:
            recommendations.append("Continue monitoring environmental health and consider preventive measures")
        
        return MycotoxinTestResponse(
            session_id=request.session_id,
            analysis=analysis_response,
            recommendations=recommendations,
            test_interpretations=test_interpretations
        )
        
    except AIAssistantError as e:
        logger.error(f"Error analyzing mycotoxin tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze-symptoms", response_model=SymptomAnalysisResponse)
async def analyze_symptoms(request: SymptomAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyze symptoms to determine if mycotoxin testing is recommended.
    
    Args:
        request: SymptomAnalysisRequest containing list of symptoms
        db: Database session
        
    Returns:
        SymptomAnalysisResponse with testing recommendations
    """
    try:
        ai_assistant = AIAssistant(db)
        result = ai_assistant.suggest_mycotoxin_testing(request.symptoms)
        
        return SymptomAnalysisResponse(
            recommend_testing=result.get("recommend_testing", False),
            matching_symptoms=result.get("matching_symptoms", []),
            suggested_tests=result.get("suggested_tests", []),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"Error analyzing symptoms: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ai/upload", summary="Upload multiple PDF or TXT files for MS knowledge base")
async def upload_knowledge_files(
    files: List[UploadFile] = File(...),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    email: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process documents for the MS knowledge base
    
    Args:
        files: One or more PDF/TXT files
        author: Optional author name
        description: Optional description
        category: Optional category
        email: User email (required)
        db: Database session
    
    Returns:
        Dict with upload results
    """
    try:
        # Validate email
        if not email or not isinstance(email, str) or not email.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="'email' is required and must be a non-empty string"
            )
        
        # Use files directly (no normalization needed)
        ai_assistant = AIAssistant(db)
        
        processed_files = []
        errors = []
        
        for file in files:
            try:
                # Validate file type
                if not file.filename:
                    errors.append({"file": "unknown", "error": "Filename is required"})
                    continue
                
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ['.pdf', '.txt']:
                    errors.append({
                        "file": file.filename, 
                        "error": f"Unsupported file type: {file_ext}. Only PDF and TXT files are allowed."
                    })
                    continue
                
                # Process the document
                document_id = await ai_assistant.upload_and_process_document(
                    email=email,
                    file=file,
                    author=author,
                    description=description,
                    category=category,
                    is_public=True
                )
                
                processed_files.append({
                    "filename": file.filename, 
                    "document_id": document_id,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                errors.append({
                    "file": file.filename, 
                    "error": str(e)
                })
        
        return {
            "uploaded": processed_files, 
            "errors": errors,
            "total_files": len(files),
            "successful": len(processed_files),
            "failed": len(errors)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_knowledge_files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/mycotoxin-info")
async def get_mycotoxin_info(db: Session = Depends(get_db)):
    """
    Get comprehensive mycotoxin testing information.
    
    Returns:
        Dict containing mycotoxin knowledge base information
    """
    try:
        ai_assistant = AIAssistant(db)
        return {
            "tests": ai_assistant.mycotoxin_knowledge["tests"],
            "symptoms": ai_assistant.mycotoxin_knowledge["symptoms"],
            "mycotoxins": ai_assistant.mycotoxin_knowledge["mycotoxins"]
        }
    except Exception as e:
        logger.error(f"Error getting mycotoxin info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    """
    Get information about a specific session.
    
    Args:
        session_id: The session ID to retrieve
        db: Database session
        
    Returns:
        SessionInfo containing session details
    """
    try:
        session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/session/{session_id}/chats", response_model=List[ChatMessageInfo])
async def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """
    Get all messages for a specific session.
    """
    try:
        messages = db.query(AI_ChatMessage).filter(
            AI_ChatMessage.session_id == session_id
        ).order_by(AI_ChatMessage.timestamp.asc()).all()
        # Convert to list of dicts for Pydantic validation
        return [
            {
                "id": msg.id,
                "session_id": msg.session_id,
                "message": msg.query_text,
                "response": msg.response_text,
                "timestamp": msg.timestamp
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# @app.delete("/sessions/{session_id}")
# async def clear_session(session_id: str, db: Session = Depends(get_db)):
#     """
#     Clear a session's state.
    
#     Args:
#         session_id: The session ID to clear
#         db: Database session
#     """
#     try:
#         ai_assistant = AIAssistant(db)
#         ai_assistant.clear_session(session_id)
#         return {"message": "Session cleared successfully"}
#     except AIAssistantError as e:
#         logger.error(f"Error clearing session: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: Session = Depends(get_db)):
    """
    Clear a session's state — deletes the session and all its messages.
    """
    ai_assistant = AIAssistant(db)
    try:
        deleted = ai_assistant.clear_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session cleared successfully"}
    except AIAssistantError as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/user/{email}/sessions")
async def get_sessions_by_email(email: EmailStr, db: Session = Depends(get_db)):
    """
    Get all sessions for a specific email address.
    
    Args:
        email: Email address to get sessions for
        db: Database session
        
    Returns:
        List of sessions with session details
    """
    try:
        # Query sessions and order by created_at in descending order (newest first)
        sessions = db.query(DBSession).filter(
            DBSession.email == email
        ).order_by(DBSession.created_at.desc()).all()
        
        # Format the response
        return [
            {
                "session_id": str(session.id),
                "created_at": session.created_at.isoformat(),
                "email": session.email,
                "title": session.title
            }
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Error getting sessions by email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get sessions")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)