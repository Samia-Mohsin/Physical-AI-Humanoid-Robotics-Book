# Data Model for Physical AI & Humanoid Robotics Platform

## 1. User Entity

**Table: users**
- id (UUID, Primary Key, NOT NULL) - Unique identifier for the user
- email (VARCHAR(255), NOT NULL, UNIQUE) - User's email address
- name (VARCHAR(255), NOT NULL) - User's display name
- password_hash (VARCHAR(255), NOT NULL) - Hashed password
- experience_level (VARCHAR(50), NOT NULL) - User's experience level (enum: Beginner, Intermediate, Advanced)
- has_rtx_gpu (BOOLEAN, DEFAULT false) - Whether user has RTX GPU
- has_jetson (BOOLEAN, DEFAULT false) - Whether user has Jetson device
- preferred_language (VARCHAR(20), DEFAULT 'English') - Preferred language (enum: English, Urdu, Roman Urdu)
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Account creation time
- updated_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Last update time
- email_verified (BOOLEAN, DEFAULT false) - Whether email is verified

**Validation Rules**:
- Email must be valid email format
- Experience level must be one of the defined enum values
- Preferred language must be one of the defined enum values

## 2. Session Entity

**Table: sessions**
- id (UUID, Primary Key, NOT NULL) - Unique session identifier
- user_id (UUID, Foreign Key to users.id, NOT NULL) - Reference to user
- token (VARCHAR(255), NOT NULL, UNIQUE) - Session token
- expires_at (TIMESTAMP, NOT NULL) - Session expiration time
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Session creation time

## 3. Conversation Entity

**Table: conversations**
- id (UUID, Primary Key, NOT NULL) - Unique conversation identifier
- user_id (UUID, Foreign Key to users.id, NOT NULL) - Reference to user who owns conversation
- title (VARCHAR(255), NOT NULL) - Auto-generated title based on first message
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Conversation creation time
- updated_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Last interaction time

## 4. Message Entity

**Table: messages**
- id (UUID, Primary Key, NOT NULL) - Unique message identifier
- conversation_id (UUID, Foreign Key to conversations.id, NOT NULL) - Reference to conversation
- role (VARCHAR(20), NOT NULL) - Message role (enum: user, assistant)
- content (TEXT, NOT NULL) - The message content
- sources (JSONB) - JSON array of source references for assistant responses
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Message creation time

## 5. Quiz Attempt Entity

**Table: quiz_attempts**
- id (UUID, Primary Key, NOT NULL) - Unique quiz attempt identifier
- user_id (UUID, Foreign Key to users.id, NOT NULL) - Reference to user
- chapter_id (VARCHAR(100), NOT NULL) - ID of the chapter being quizzed
- score (INTEGER, NOT NULL) - Number of correct answers
- total_questions (INTEGER, NOT NULL) - Total number of questions in quiz
- answers (JSONB) - JSON object of {question_id: {selected_answer, is_correct}}
- completed_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - When quiz was completed

**Validation Rules**:
- Score must be between 0 and total_questions
- answers JSON must have valid structure

## 6. Translation Cache Entity

**Table: translation_cache**
- id (UUID, Primary Key, NOT NULL) - Unique cache entry identifier
- chapter_id (VARCHAR(100), NOT NULL) - ID of the chapter being cached
- target_language (VARCHAR(20), NOT NULL) - Target language (enum: urdu, roman_urdu)
- content_hash (VARCHAR(64), NOT NULL) - SHA256 hash of original content
- translated_content (TEXT, NOT NULL) - The translated content
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Cache creation time
- updated_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Last update time

**Validation Rules**:
- Unique constraint on (chapter_id, target_language, content_hash)
- Target language must be one of the defined enum values

## 7. User Preferences Entity

**Table: user_preferences**
- id (UUID, Primary Key, NOT NULL) - Unique preference identifier
- user_id (UUID, Foreign Key to users.id, NOT NULL) - Reference to user
- preference_key (VARCHAR(100), NOT NULL) - Key for the preference
- preference_value (TEXT, NOT NULL) - Value of the preference
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Creation time
- updated_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Last update time

**Validation Rules**:
- Unique constraint on (user_id, preference_key)

## 8. Personalized Content Entity

**Table: personalized_content**
- id (UUID, Primary Key, NOT NULL) - Unique identifier
- user_id (UUID, Foreign Key to users.id, NOT NULL) - Reference to user
- chapter_id (VARCHAR(100), NOT NULL) - ID of the original chapter
- personalized_content (TEXT, NOT NULL) - The personalized version of the content
- personalization_reasoning (TEXT) - Explanation of how content was personalized
- created_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Creation time
- updated_at (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP) - Last update time

**Validation Rules**:
- Unique constraint on (user_id, chapter_id)

## Relationships

1. **User → Sessions**: One-to-Many (one user has many sessions)
2. **User → Conversations**: One-to-Many (one user has many conversations)
3. **User → Quiz Attempts**: One-to-Many (one user has many quiz attempts)
4. **User → Translation Cache**: One-to-Many (one user associated with many cached translations)
5. **User → User Preferences**: One-to-Many (one user has many preferences)
6. **User → Personalized Content**: One-to-Many (one user has many personalized chapters)
7. **Conversation → Messages**: One-to-Many (one conversation has many messages)

## Indexes

1. **users.email**: B-tree index for efficient login
2. **conversations.user_id**: B-tree index for fetching user conversations
3. **messages.conversation_id**: B-tree index for fetching conversation messages
4. **quiz_attempts.user_id**: B-tree index for fetching user quiz attempts
5. **quiz_attempts.chapter_id**: B-tree index for chapter-based analysis
6. **translation_cache.chapter_id**: B-tree index for content lookup
7. **translation_cache.target_language**: B-tree index for language-based queries
8. **user_preferences.user_id**: B-tree index for user preference lookups

## State Transitions

### User Session States
- Active: Session token is valid and not expired
- Expired: Session token has expired
- Revoked: Session token was manually invalidated

### Quiz Attempt States
- Started: Quiz was initiated but not completed
- Completed: Quiz was finished and score is recorded
- Abandoned: Quiz was started but not completed within timeout period

## Database Constraints

1. **Referential Integrity**: Foreign key constraints ensure data consistency
2. **Unique Constraints**: Prevent duplicate entries where inappropriate
3. **Check Constraints**: Validate enum values and value ranges
4. **Not Null Constraints**: Ensure required fields are always populated