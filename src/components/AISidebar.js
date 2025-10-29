import React, { useState, useRef, useEffect } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import './AISidebar.css';

/**
 * AI Assistant Sidebar Component
 * Provides a chat interface for asking questions about derivation content
 * In edit mode, helps edit and improve derivation content
 * Integrates with Claude API via backend endpoints
 * Updated: Removed paste handler to fix Ctrl-A
 */
const AISidebar = ({
  isOpen,
  onClose,
  selectedText,
  derivationData,
  currentStep,
  editMode = false,
  onDerivationUpdate = null,
  derivationId = null
}) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachedImages, setAttachedImages] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when sidebar opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Initialize with selected text if available
  useEffect(() => {
    if (isOpen && selectedText && messages.length === 0) {
      setInputValue(`Can you explain: "${selectedText}"?`);
    }
  }, [isOpen, selectedText, messages.length]);

  /**
   * Handle file selection from file input
   */
  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    await processImageFiles(files);
    // Reset input so same file can be selected again
    e.target.value = '';
  };

  const processImageFiles = async (files) => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));

    if (imageFiles.length === 0) return;

    // Convert images to base64
    const imagePromises = imageFiles.map(file => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const base64Data = e.target.result.split(',')[1]; // Remove data URL prefix
          resolve({
            type: 'image',
            source: {
              type: 'base64',
              media_type: file.type,
              data: base64Data
            },
            preview: e.target.result, // Keep full data URL for preview
            name: file.name
          });
        };
        reader.readAsDataURL(file);
      });
    });

    const newImages = await Promise.all(imagePromises);
    setAttachedImages([...attachedImages, ...newImages]);
  };

  const removeImage = (index) => {
    setAttachedImages(attachedImages.filter((_, i) => i !== index));
  };

  /**
   * Send question to Claude API (or edit command in edit mode)
   */
  const handleSendMessage = async () => {
    if ((!inputValue.trim() && attachedImages.length === 0) || isLoading) return;

    // Create user message with images if attached
    const userMessage = {
      role: 'user',
      content: inputValue,
      images: attachedImages.map(img => ({ preview: img.preview, name: img.name })), // For display only
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentImages = [...attachedImages]; // Save current images before clearing
    setInputValue('');
    setAttachedImages([]); // Clear attached images after sending
    setIsLoading(true);

    try {
      if (editMode) {
        // Edit mode: call edit endpoint
        const response = await fetch('/api/edit-derivation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            command: inputValue,
            currentData: derivationData,
            conversationHistory: messages,
            attachedImages: currentImages.map(img => ({
              type: img.type,
              source: img.source
            })),
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update the derivation data
        if (onDerivationUpdate && data.modifiedData) {
          onDerivationUpdate(data.modifiedData);
        }

        const assistantMessage = {
          role: 'assistant',
          content: `✅ ${data.explanation}\n\nThe preview on the left has been updated. Review the changes and click "Save Changes" when ready.`,
          timestamp: new Date().toISOString(),
          usage: data.usage,
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        // Regular mode: ask questions
        const response = await fetch('/api/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: inputValue,
            selectedText: selectedText,
            derivationContext: derivationData,
            conversationHistory: messages,
            attachedImages: currentImages.map(img => ({
              type: img.type,
              source: img.source
            })),
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        const assistantMessage = {
          role: 'assistant',
          content: data.answer,
          timestamp: new Date().toISOString(),
          usage: data.usage,
        };

        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`,
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Request improvement for current step
   */
  const handleImproveAnswer = async () => {
    if (!currentStep || isLoading) return;

    setIsLoading(true);

    const userMessage = {
      role: 'user',
      content: '🔧 Requesting improved version of this step...',
      timestamp: new Date().toISOString(),
      isSystem: true,
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch('/api/improve-answer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          currentStep: currentStep,
          derivationContext: derivationData,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: `I've improved this step! Here's the enhanced version:\n\n**Improved Explanation:**\n${data.improvedStep.explanation}\n\n**Equations:**\n${data.improvedStep.equation.map(eq => `$$${eq}$$`).join('\n\n')}`,
        timestamp: new Date().toISOString(),
        improvedStep: data.improvedStep,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error improving answer:', error);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I couldn't improve the answer: ${error.message}`,
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Render message content with LaTeX and Markdown support
   */
  const renderMessage = (content) => {
    const elements = [];
    let currentIndex = 0;
    let key = 0;

    // Split content by lines first to handle headings and paragraphs
    const lines = content.split('\n');

    lines.forEach((line, lineIdx) => {
      // Handle markdown headings
      if (line.startsWith('## ')) {
        elements.push(
          <h3 key={`h3-${lineIdx}`} style={{ fontSize: '1.1em', fontWeight: 'bold', marginTop: '0.8em', marginBottom: '0.5em' }}>
            {renderTextWithLatex(line.substring(3), `h3-${lineIdx}`)}
          </h3>
        );
      } else if (line.startsWith('# ')) {
        elements.push(
          <h2 key={`h2-${lineIdx}`} style={{ fontSize: '1.2em', fontWeight: 'bold', marginTop: '1em', marginBottom: '0.5em' }}>
            {renderTextWithLatex(line.substring(2), `h2-${lineIdx}`)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        elements.push(
          <h4 key={`h4-${lineIdx}`} style={{ fontSize: '1em', fontWeight: 'bold', marginTop: '0.6em', marginBottom: '0.4em' }}>
            {renderTextWithLatex(line.substring(4), `h4-${lineIdx}`)}
          </h4>
        );
      } else if (line.trim() === '') {
        // Empty line - add spacing
        elements.push(<br key={`br-${lineIdx}`} />);
      } else {
        // Regular paragraph
        elements.push(
          <p key={`p-${lineIdx}`} style={{ marginTop: '0.3em', marginBottom: '0.3em' }}>
            {renderTextWithLatex(line, `p-${lineIdx}`)}
          </p>
        );
      }
    });

    return <div className="message-content">{elements}</div>;
  };

  /**
   * Helper function to render text with inline/display LaTeX
   */
  const renderTextWithLatex = (text, baseKey) => {
    const parts = [];
    let remaining = text;
    let idx = 0;

    while (remaining.length > 0) {
      // Check for display math $$...$$
      const displayMatch = remaining.match(/\$\$(.*?)\$\$/);
      if (displayMatch && displayMatch.index === 0) {
        parts.push(<BlockMath key={`${baseKey}-dm-${idx}`} math={displayMatch[1]} />);
        remaining = remaining.substring(displayMatch[0].length);
        idx++;
        continue;
      }

      // Check for inline math $...$
      const inlineMatch = remaining.match(/\$(.*?)\$/);
      if (inlineMatch && inlineMatch.index === 0) {
        parts.push(<InlineMath key={`${baseKey}-im-${idx}`} math={inlineMatch[1]} />);
        remaining = remaining.substring(inlineMatch[0].length);
        idx++;
        continue;
      }

      // Find next LaTeX delimiter
      const nextDisplay = remaining.indexOf('$$');
      const nextInline = remaining.indexOf('$');

      let nextDelim = -1;
      if (nextDisplay !== -1 && (nextInline === -1 || nextDisplay < nextInline)) {
        nextDelim = nextDisplay;
      } else if (nextInline !== -1) {
        nextDelim = nextInline;
      }

      if (nextDelim === -1) {
        // No more LaTeX, render rest as text with markdown formatting
        parts.push(<span key={`${baseKey}-text-${idx}`}>{renderMarkdownText(remaining)}</span>);
        break;
      } else {
        // Render text before next LaTeX
        const textBefore = remaining.substring(0, nextDelim);
        if (textBefore.length > 0) {
          parts.push(<span key={`${baseKey}-text-${idx}`}>{renderMarkdownText(textBefore)}</span>);
        }
        remaining = remaining.substring(nextDelim);
        idx++;
      }
    }

    return parts;
  };

  /**
   * Helper to render basic markdown formatting (bold, italic)
   */
  const renderMarkdownText = (text) => {
    // Handle **bold**
    const boldParts = text.split(/\*\*(.*?)\*\*/g);
    return boldParts.map((part, idx) => {
      if (idx % 2 === 1) {
        return <strong key={idx}>{part}</strong>;
      }
      // Handle *italic*
      const italicParts = part.split(/\*(.*?)\*/g);
      return italicParts.map((iPart, iIdx) => {
        if (iIdx % 2 === 1) {
          return <em key={`${idx}-${iIdx}`}>{iPart}</em>;
        }
        return iPart;
      });
    });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setInputValue('');
  };

  return (
    <div className={`ai-sidebar ${isOpen ? 'open' : ''} ${editMode ? 'edit-mode' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="header-content">
          <svg className="ai-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            {editMode ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            )}
          </svg>
          <div>
            <h3 className="sidebar-title">{editMode ? 'AI Editor' : 'AI Assistant'}</h3>
            <p className="sidebar-subtitle">{editMode ? 'Edit and improve this derivation' : 'Ask about the derivation'}</p>
          </div>
        </div>
        {!editMode && (
          <button onClick={onClose} className="close-button" aria-label="Close sidebar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Selected Text Display */}
      {selectedText && (
        <div className="selected-text-banner">
          <span className="banner-label">Selected:</span>
          <span className="banner-text">
            {selectedText.startsWith('$$') && selectedText.endsWith('$$') ? (
              <BlockMath math={selectedText.slice(2, -2)} />
            ) : selectedText.startsWith('$') && selectedText.endsWith('$') ? (
              <InlineMath math={selectedText.slice(1, -1)} />
            ) : (
              selectedText.substring(0, 100) + (selectedText.length > 100 ? '...' : '')
            )}
          </span>
        </div>
      )}

      {/* Messages Container */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <svg className="empty-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              {editMode ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              )}
            </svg>
            <p className="empty-text">{editMode ? 'AI Editor Ready!' : 'Ask me anything about this derivation!'}</p>
            <p className="empty-hint" style={{fontSize: editMode ? '0.8rem' : '0.875rem', textAlign: 'left', maxWidth: editMode ? '100%' : 'auto'}}>
              {editMode ? (
                <>
                  <strong>Try these commands:</strong><br/>
                  • "Fix the equation in step 3.2"<br/>
                  • "Add an intermediate step between 5.1 and 5.2"<br/>
                  • "Improve the explanation for step 8.5"<br/>
                  • "Add a new section about [topic]"<br/>
                  • "Change the title to..."<br/>
                </>
              ) : (
                'Select text to ask about specific equations, or type a general question below.'
              )}
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role} ${message.isError ? 'error' : ''} ${message.isSystem ? 'system' : ''}`}>
            <div className="message-avatar">
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-bubble">
              {message.images && message.images.length > 0 && (
                <div className="message-images">
                  {message.images.map((img, imgIdx) => (
                    <img
                      key={imgIdx}
                      src={img.preview}
                      alt={img.name}
                      className="message-image-thumbnail"
                      title={img.name}
                    />
                  ))}
                </div>
              )}
              {renderMessage(message.content)}
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message assistant loading">
            <div className="message-avatar">🤖</div>
            <div className="message-bubble">
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Action Buttons */}
      {messages.length > 0 && (
        <div className="action-buttons">
          <button
            onClick={handleClearConversation}
            className="action-button secondary"
            disabled={isLoading}
          >
            Clear Chat
          </button>
          {currentStep && (
            <button
              onClick={handleImproveAnswer}
              className="action-button primary"
              disabled={isLoading}
            >
              ✨ Improve This Step
            </button>
          )}
        </div>
      )}

      {/* Input Area */}
      <div className="input-container">

        {/* Image Previews */}
        {attachedImages.length > 0 && (
          <div className="attached-images-preview">
            {attachedImages.map((img, idx) => (
              <div key={idx} className="image-preview-item">
                <img src={img.preview} alt={img.name} />
                <button
                  onClick={() => removeImage(idx)}
                  className="remove-image-button"
                  aria-label="Remove image"
                >
                  ×
                </button>
                <span className="image-name">{img.name}</span>
              </div>
            ))}
          </div>
        )}

        <div className="input-row">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="image/*"
            multiple
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="attach-button"
            aria-label="Attach image"
            disabled={isLoading}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
          </button>
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={editMode ? "Tell me what to edit... (e.g., 'Fix step 3.2')" : "Ask a question about this derivation..."}
            className="message-input"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={(!inputValue.trim() && attachedImages.length === 0) || isLoading}
            className="send-button"
            aria-label="Send message"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default AISidebar;
