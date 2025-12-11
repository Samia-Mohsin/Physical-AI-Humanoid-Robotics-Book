# Research Findings for Physical AI & Humanoid Robotics Platform

## 1. OpenAI Agents SDK Integration

**Decision**: Use OpenAI Assistant API for RAG chatbot functionality
**Rationale**: The Assistant API provides managed thread state, tool calling capabilities, and streaming responses which are perfect for our chatbot requirements. It handles conversation memory automatically and allows for custom tools that can integrate with our Qdrant vector store.

**Alternatives Considered**:
- OpenAI Chat Completions API: Requires manual thread management and context handling
- LangChain: Adds complexity without significant benefits for our use case
- Custom solution: Would require significant development time

## 2. Better-Auth Implementation

**Decision**: Use Better-Auth with Neon Postgres adapter
**Rationale**: Better-Auth provides a modern, secure authentication solution with built-in support for custom fields, social login options, and session management. It integrates well with React applications and supports database adapters including Neon Postgres.

**Alternatives Considered**:
- NextAuth.js: Primarily designed for Next.js, not ideal for Docusaurus
- Supabase Auth: Good alternative but requires different data modeling
- Custom JWT implementation: Security risks and development overhead

## 3. Qdrant Vector Storage Strategy

**Decision**: Use Qdrant Cloud Free Tier with document chunking strategy
**Rationale**: Qdrant provides efficient vector similarity search, good Python client library, and cloud hosting. The free tier supports our initial requirements. Chunking strategy will use 1000-token chunks with 200-token overlap to maintain context.

**Alternatives Considered**:
- Pinecone: Good but requires credit card for free tier
- Weaviate: Feature-rich but steeper learning curve
- ChromaDB: Open-source but requires self-hosting

## 4. Docusaurus Chatbot Integration

**Decision**: Embed React chatbot component in Docusaurus layout
**Rationale**: This approach provides the most flexibility for creating a rich chat interface while maintaining Docusaurus benefits. The chatbot can be implemented as a React component that's included in the Docusaurus layout.

**Alternatives Considered**:
- Iframe embedding: Limited styling and communication capabilities
- External widget: Less integration with Docusaurus features
- Native Docusaurus plugin: Would require significant custom development

## 5. Text Selection API Implementation

**Decision**: Use browser Selection API with custom event handlers
**Rationale**: The browser's built-in Selection API provides reliable text selection detection across all modern browsers. Combined with custom event handlers, it allows for the "Explain Selection" functionality required.

**Alternatives Considered**:
- Custom selection libraries: Add unnecessary complexity
- Mutation observers: Overkill for simple text selection
- CSS-based solutions: Don't provide the necessary JavaScript access

## 6. Claude Code Subagents Architecture

**Decision**: Implement subagents as specialized OpenAI Assistant tools
**Rationale**: OpenAI Assistant tools provide a natural way to implement subagents with specific capabilities. Each subagent can be implemented as a custom tool that the assistant can call when needed.

**Subagent Implementations**:
- ContentGenerator: Tool for generating educational content
- Personalizer: Tool for adapting content to user profile
- UrduTranslator: Tool for translating to Urdu
- RomanUrduConverter: Tool for translating to Roman Urdu
- QuizMaster: Tool for generating quizzes
- DiagramExplainer: Tool for explaining visual content
- RagIngester: Tool for indexing new content

## 7. Live Code Blocks Implementation

**Decision**: Use static code highlighting with no execution capability (as per clarifications)
**Rationale**: Based on clarification decisions, live code blocks should be static highlighted code only, with no execution capability. This reduces security risks and complexity while maintaining visual presentation.

**Alternatives Considered**:
- Code sandbox execution: Security risks and complexity
- Server-side execution: Performance and cost concerns
- WASM-based execution: Still complex and potentially limited

## 8. Translation Quality Strategy

**Decision**: Machine-only GPT-4o translation with caching (as per clarifications)
**Rationale**: Based on clarification decisions, translation should be machine-only GPT-4o with no human review. Caching in Neon Postgres will improve performance and reduce API costs.

**Alternatives Considered**:
- Human review process: Would add significant cost and delay
- Multiple service comparison: Would increase complexity and costs
- Static pre-translated content: Would reduce freshness and adaptability

## 9. Quiz Interactivity Implementation

**Decision**: Basic interactive forms with immediate feedback and Neon score tracking (as per clarifications)
**Rationale**: Based on clarification decisions, quizzes should be basic interactive forms with immediate feedback and Neon score tracking. This provides user engagement while maintaining simplicity.

**Alternatives Considered**:
- Complex adaptive quizzes: Would add significant complexity
- Rich media quizzes: Would increase development time
- Offline-capable quizzes: Would require additional caching strategies

## 10. Personalization Engine Approach

**Decision**: Rule-based personalization with GPT-4o enhancement
**Rationale**: Combine rule-based personalization based on user profile (experience level, hardware) with GPT-4o for content adaptation. This provides both predictable behavior and intelligent adaptation.

**Alternatives Considered**:
- ML-based personalization: Would require significant training data
- Static personalization: Would be less adaptive to user needs
- No personalization: Would not meet constitution requirements