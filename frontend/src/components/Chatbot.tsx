import React, { useState, useEffect, useRef } from 'react';
import { useUser } from '../contexts/UserContext';
import TextSelectionPopup from './TextSelectionPopup';
import './Chatbot.css';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Array<{
    content: string;
    metadata: any;
  }>;
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useUser();

  // Function to handle text selection actions
  const handleTextSelectionAction = (action: 'explain' | 'translate' | 'save', text: string) => {
    if (action === 'explain') {
      handleExplainSelection(text);
    }
    // For other actions, we could implement additional functionality
  };

  const handleExplainSelection = async (text: string = selectedText || '') => {
    if (!text || isLoading) return;

    // Add user message indicating selected text explanation request
    const userMessage: Message = {
      id: Date.now().toString(),
      content: `Explain this selected text: "${text}"`,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Prepare the request payload specifically for explaining selected text
      const requestBody = {
        query: `Explain the following text: ${text}`,
        selected_text: text, // This will trigger the selected-text priority mode
        user_id: user?.id?.toString() || null,
      };

      // Try streaming endpoint first, fallback to regular endpoint
      const streamResponse = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (streamResponse.ok && streamResponse.body) {
        // Handle streaming response
        const reader = streamResponse.body.getReader();
        if (!reader) throw new Error('No response body');

        // Create a temporary assistant message for streaming content
        const streamingMessageId = `stream_${Date.now()}`;
        const initialMessage: Message = {
          id: streamingMessageId,
          content: '',
          role: 'assistant',
          timestamp: new Date(),
          sources: [],
        };

        setMessages(prev => [...prev, initialMessage]);

        const decoder = new TextDecoder();
        let buffer = '';
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;

          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            // Process the buffer for complete messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6); // Remove 'data: ' prefix
                if (data === '[DONE]') {
                  done = true;
                  break;
                }

                try {
                  const parsed = JSON.parse(data);
                  if (parsed.type === 'content') {
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === streamingMessageId
                          ? { ...msg, content: msg.content + parsed.content }
                          : msg
                      )
                    );
                  } else if (parsed.type === 'sources') {
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === streamingMessageId
                          ? { ...msg, sources: parsed.sources }
                          : msg
                      )
                    );
                  }
                } catch (e) {
                  // Skip malformed JSON
                  continue;
                }
              }
            }
          }
        }
      } else {
        // Fallback to regular endpoint
        const regularResponse = await fetch('/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!regularResponse.ok) {
          throw new Error(`API request failed with status ${regularResponse.status}`);
        }

        const data = await regularResponse.json();

        // Add assistant response to chat
        const assistantMessage: Message = {
          id: `resp_${Date.now()}`,
          content: data.response,
          role: 'assistant',
          timestamp: new Date(),
          sources: data.sources || [],
        };

        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error explaining selection:', error);

      // Add error message to chat
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        content: 'Sorry, I encountered an error while explaining the selected text. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setSelectedText(null); // Clear selected text after sending
    }
  };

  // Function to get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection()?.toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare the request payload
      const requestBody = {
        query: inputValue,
        selected_text: selectedText, // Include selected text if available
        user_id: user?.id?.toString() || null,
      };

      // Try streaming endpoint first, fallback to regular endpoint
      const streamResponse = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (streamResponse.ok && streamResponse.body) {
        // Handle streaming response
        const reader = streamResponse.body.getReader();
        if (!reader) throw new Error('No response body');

        // Create a temporary assistant message for streaming content
        const streamingMessageId = `stream_${Date.now()}`;
        const initialMessage: Message = {
          id: streamingMessageId,
          content: '',
          role: 'assistant',
          timestamp: new Date(),
          sources: [],
        };

        setMessages(prev => [...prev, initialMessage]);

        const decoder = new TextDecoder();
        let buffer = '';
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;

          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            // Process the buffer for complete messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6); // Remove 'data: ' prefix
                if (data === '[DONE]') {
                  done = true;
                  break;
                }

                try {
                  const parsed = JSON.parse(data);
                  if (parsed.type === 'content') {
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === streamingMessageId
                          ? { ...msg, content: msg.content + parsed.content }
                          : msg
                      )
                    );
                  } else if (parsed.type === 'sources') {
                    setMessages(prev =>
                      prev.map(msg =>
                        msg.id === streamingMessageId
                          ? { ...msg, sources: parsed.sources }
                          : msg
                      )
                    );
                  }
                } catch (e) {
                  // Skip malformed JSON
                  continue;
                }
              }
            }
          }
        }
      } else {
        // Fallback to regular endpoint
        const regularResponse = await fetch('/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!regularResponse.ok) {
          throw new Error(`API request failed with status ${regularResponse.status}`);
        }

        const data = await regularResponse.json();

        // Add assistant response to chat
        const assistantMessage: Message = {
          id: `resp_${Date.now()}`,
          content: data.response,
          role: 'assistant',
          timestamp: new Date(),
          sources: data.sources || [],
        };

        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setSelectedText(null); // Clear selected text after sending
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Text Selection Popup - appears when text is selected */}
      <TextSelectionPopup onSelectAction={handleTextSelectionAction} />

      {/* Chatbot toggle button */}
      <button
        className={`chatbot-toggle ${isOpen ? 'open' : ''}`}
        onClick={toggleChat}
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chatbot widget */}
      {isOpen && (
        <div className="chatbot-widget">
          <div className="chatbot-header">
            <h3>Physical AI & Humanoid Robotics Assistant</h3>
            <div className="header-buttons">
              {selectedText && (
                <button
                  className="explain-button"
                  onClick={() => handleExplainSelection(selectedText)}
                  disabled={isLoading}
                  title={`Explain selected text: "${selectedText.substring(0, 30)}${selectedText.length > 30 ? '...' : ''}"`}
                >
                  Explain Selection
                </button>
              )}
            </div>
          </div>

          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your Physical AI & Humanoid Robotics assistant.</p>
                <p>Ask me anything about the book content, or select text and click "Explain Selection".</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.role}`}
                >
                  <div className="message-content">
                    {message.content}
                  </div>
                  <div className="message-timestamp">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>

                  {/* Show sources if available */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <details>
                        <summary>Sources</summary>
                        <ul>
                          {message.sources.map((source, idx) => (
                            <li key={idx}>
                              <small>{source.content.substring(0, 100)}{source.content.length > 100 ? '...' : ''}</small>
                            </li>
                          ))}
                        </ul>
                      </details>
                    </div>
                  )}
                </div>
              ))
            )}
            {isLoading && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            {selectedText && (
              <div className="selected-text-preview">
                <small>Selected: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"</small>
              </div>
            )}
            <div className="input-container">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about Physical AI & Humanoid Robotics..."
                disabled={isLoading}
                rows={1}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputValue.trim()}
                className="send-button"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;