# API Contracts for Physical AI & Humanoid Robotics Platform

## 1. Authentication Endpoints

### POST /api/auth/register
**Description**: Register a new user with background questions

**Request**:
```json
{
  "email": "user@example.com",
  "password": "secure_password_123",
  "name": "John Doe",
  "experience_level": "Intermediate",
  "has_rtx_gpu": false,
  "has_jetson": true,
  "preferred_language": "English"
}
```

**Response (201 Created)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "name": "John Doe",
    "experience_level": "Intermediate",
    "has_rtx_gpu": false,
    "has_jetson": true,
    "preferred_language": "English",
    "created_at": "2025-12-10T10:00:00Z"
  },
  "session_token": "session-token-string"
}
```

**Validation**:
- Email must be valid format
- Password must be at least 8 characters
- experience_level must be "Beginner", "Intermediate", or "Advanced"
- preferred_language must be "English", "Urdu", or "Roman Urdu"

### POST /api/auth/login
**Description**: Authenticate user and return session token

**Request**:
```json
{
  "email": "user@example.com",
  "password": "secure_password_123"
}
```

**Response (200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "name": "John Doe",
    "experience_level": "Intermediate",
    "has_rtx_gpu": false,
    "has_jetson": true,
    "preferred_language": "English"
  },
  "session_token": "session-token-string"
}
```

**Response (401 Unauthorized)**:
```json
{
  "error": "Invalid credentials"
}
```

### GET /api/auth/me
**Description**: Get current authenticated user info

**Headers**:
```
Authorization: Bearer session-token-string
```

**Response (200 OK)**:
```json
{
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "name": "John Doe",
    "experience_level": "Intermediate",
    "has_rtx_gpu": false,
    "has_jetson": true,
    "preferred_language": "English",
    "created_at": "2025-12-10T10:00:00Z"
  }
}
```

## 2. RAG Query Endpoints

### POST /api/query
**Description**: Query the RAG system with optional selected text context

**Headers**:
```
Authorization: Bearer session-token-string (optional)
Content-Type: application/json
```

**Request**:
```json
{
  "query": "Explain how ROS 2 handles message passing between nodes",
  "selected_text": "ROS 2 uses a publish-subscribe pattern for message passing between nodes",
  "user_id": "uuid-string (optional if not authenticated)"
}
```

**Response (200 OK)**:
```json
{
  "response": "ROS 2 handles message passing between nodes using a publish-subscribe pattern...",
  "sources": [
    {
      "title": "ROS 2 Architecture Deep Dive",
      "chapter": "01-introduction",
      "url": "/docs/01-introduction",
      "excerpt": "ROS 2 uses DDS (Data Distribution Service) as the underlying middleware..."
    }
  ],
  "conversation_id": "uuid-string",
  "message_id": "uuid-string"
}
```

### POST /api/ingest
**Description**: Trigger document ingestion to Qdrant (admin only)

**Headers**:
```
Authorization: Bearer admin-token-string
Content-Type: application/json
```

**Request**:
```json
{
  "source_url": "https://example-book.github.io/docs",
  "force_reindex": false
}
```

**Response (200 OK)**:
```json
{
  "status": "ingestion_started",
  "documents_processed": 12,
  "chunks_created": 245,
  "estimated_completion": "2025-12-10T10:15:00Z"
}
```

## 3. Personalization Endpoints

### GET /api/personalize/{chapter_id}
**Description**: Get personalized version of a chapter based on user profile

**Headers**:
```
Authorization: Bearer session-token-string
```

**Response (200 OK)**:
```json
{
  "chapter_id": "01-introduction",
  "original_content": "Full chapter content in standard format...",
  "personalized_content": "Chapter content adapted to user's experience level...",
  "personalization_reasoning": "Simplified explanations for Beginner level user, added advanced examples for Advanced hardware user",
  "last_updated": "2025-12-10T10:00:00Z"
}
```

### POST /api/personalize/{chapter_id}
**Description**: Request personalization of a chapter

**Headers**:
```
Authorization: Bearer session-token-string
Content-Type: application/json
```

**Request**:
```json
{
  "preference": "simplify",
  "additional_context": "Focus on practical examples"
}
```

**Response (200 OK)**:
```json
{
  "chapter_id": "01-introduction",
  "personalized_content": "Personalized chapter content...",
  "personalization_reasoning": "Content simplified based on user preferences"
}
```

## 4. Translation Endpoints

### GET /api/translate/{chapter_id}/{language}
**Description**: Get translated version of a chapter

**Headers**:
```
Authorization: Bearer session-token-string (optional)
```

**Response (200 OK)**:
```json
{
  "chapter_id": "01-introduction",
  "target_language": "urdu",
  "original_content_hash": "sha256-hash-of-original-content",
  "translated_content": "مطابق کردہ مواد یہاں...",
  "translation_metadata": {
    "translated_at": "2025-12-10T10:00:00Z",
    "cached": true,
    "quality_score": 0.95
  }
}
```

## 5. Quiz Endpoints

### POST /api/quizzes/{chapter_id}/submit
**Description**: Submit answers for a chapter quiz

**Headers**:
```
Authorization: Bearer session-token-string
Content-Type: application/json
```

**Request**:
```json
{
  "answers": {
    "q1": "option_b",
    "q2": "option_a",
    "q3": "true",
    "q4": "multiple options here"
  }
}
```

**Response (200 OK)**:
```json
{
  "chapter_id": "01-introduction",
  "score": 8,
  "total_questions": 10,
  "percentage": 80,
  "detailed_results": [
    {
      "question_id": "q1",
      "correct": true,
      "explanation": "Detailed explanation of why option B is correct..."
    }
  ],
  "quiz_attempt_id": "uuid-string"
}
```

### GET /api/quizzes/{chapter_id}/results
**Description**: Get results for a specific chapter quiz

**Headers**:
```
Authorization: Bearer session-token-string
```

**Response (200 OK)**:
```json
{
  "chapter_id": "01-introduction",
  "latest_attempt": {
    "score": 8,
    "total_questions": 10,
    "percentage": 80,
    "completed_at": "2025-12-10T09:30:00Z",
    "quiz_attempt_id": "uuid-string"
  },
  "historical_scores": [
    {
      "score": 8,
      "percentage": 80,
      "completed_at": "2025-12-10T09:30:00Z"
    }
  ]
}
```

## 6. Chatbot Configuration Endpoints

### GET /api/chatbot/config
**Description**: Get chatbot configuration and available tools

**Response (200 OK)**:
```json
{
  "config": {
    "model": "gpt-4o-mini",
    "features": {
      "selected_text_mode": true,
      "agent_skills": [
        "explain_concept",
        "generate_quiz",
        "translate_to_urdu",
        "translate_to_roman_urdu",
        "simplify_for_beginner",
        "add_advanced_robotics_code",
        "explain_diagram_with_vision",
        "retrieve_and_answer_from_selection",
        "generate_ros2_node",
        "generate_urdf_from_description",
        "debug_gazebo_launch"
      ]
    }
  }
}
```

## Error Response Format

All error responses follow this format:

```json
{
  "error": "Error message describing the issue",
  "error_code": "ERROR_CODE",
  "details": {
    "field": "specific field if applicable",
    "value": "problematic value if applicable"
  }
}
```

## Common Error Codes

- `VALIDATION_ERROR`: Request validation failed
- `AUTHENTICATION_ERROR`: Authentication required or failed
- `AUTHORIZATION_ERROR`: User not authorized for this action
- `NOT_FOUND`: Requested resource not found
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `INTERNAL_ERROR`: Internal server error occurred