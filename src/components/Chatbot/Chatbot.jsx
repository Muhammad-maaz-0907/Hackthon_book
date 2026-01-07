import React, { useState, useRef, useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import './Chatbot.css';

const Chatbot = ({ backendUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const { colorMode } = useColorMode();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Enable text selection listener
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare the request payload
      const requestBody = {
        message: inputValue,
        conversation_id: sessionId || null,
        selected_text: isSelectionMode && selectedText ? selectedText : null,
        max_tokens: 1000
      };

      const response = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Update session ID if it was returned
      if (data.conversation_id && !sessionId) {
        setSessionId(data.conversation_id);
      }

      const botMessage = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        sources: data.sources || [],
        confidence: data.confidence,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error while processing your request. Please try again.',
        sender: 'bot',
        error: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      if (isSelectionMode) {
        setIsSelectionMode(false);
        setSelectedText('');
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode);
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className={`chatbot-container ${colorMode}`}>
      <div className="chatbot-header">
        <h3>Humanoid Robotics Book Assistant</h3>
        <div className="chatbot-controls">
          <button
            className={`selection-mode-btn ${isSelectionMode ? 'active' : ''}`}
            onClick={toggleSelectionMode}
            title={isSelectionMode ? "Exit Selected Text Mode" : "Enter Selected Text Mode"}
          >
            {isSelectionMode ? "📝 Selected Text Mode" : "📝 Select Text Mode"}
          </button>
          <button className="clear-chat-btn" onClick={clearChat}>
            🗑️ Clear
          </button>
        </div>
      </div>

      {isSelectionMode && selectedText && (
        <div className="selected-text-preview">
          <p><strong>Using selected text:</strong></p>
          <div className="selected-text-content">
            "{selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}"
          </div>
        </div>
      )}

      <div className="chatbot-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Hello! I'm your Humanoid Robotics Book assistant.</p>
            <p>
              {isSelectionMode
                ? "Select text on the page and ask questions about it in 'Selected Text Mode', or ask general questions about the book."
                : "Ask me questions about the Humanoid Robotics Book. I can help explain concepts, find information, and provide citations."}
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender} ${message.error ? 'error' : ''}`}
            >
              <div className="message-content">
                {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <strong>Sources:</strong>
                    <ul>
                      {message.sources.map((source, index) => (
                        <li key={index}>
                          <span className="doc-path">{source.doc_path}</span>
                          {source.heading && <span className="heading"> → {source.heading}</span>}
                          {source.chunk_id && <span className="chunk-id"> (Chunk: {source.chunk_id})</span>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="text">{message.text}</div>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="message bot">
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
        {isSelectionMode && !selectedText && (
          <p className="selection-instruction">
            Please select text on the page first, then ask your question.
          </p>
        )}
        <div className="input-container">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isSelectionMode && !selectedText
                ? "Select text on the page first..."
                : isSelectionMode
                ? `Ask about the selected text: "${selectedText.substring(0, 50)}${selectedText.length > 50 ? '...' : ''}"`
                : "Ask a question about the Humanoid Robotics Book..."
            }
            disabled={isLoading || (isSelectionMode && !selectedText)}
            rows={3}
          />
          <button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isLoading || (isSelectionMode && !selectedText)}
            className="send-button"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;