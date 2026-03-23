'use client';

import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Mic, MicOff, Paperclip, TrendingUp, Bot, Sparkles, FileText, Image, File } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// ─── Types ───────────────────────────────────────────────────
interface Message {
  id:        string;
  type:      'ai' | 'user' | 'service-card' | 'thinking';
  content?:  string;
  resolved?: boolean;
  attachment?: { name: string; type: string };
  serviceCard?: {
    title:     string;
    riskLevel: string;
    riskColor: string;
    buttons:   { label: string; action: string }[];
  };
}

// ─── Web Speech API type declarations ────────────────────────
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

// ─── Thinking animation ───────────────────────────────────────
const ThinkingBubble: React.FC = () => (
  <div className="cw-thinking-row">
    <div className="cw-avatar">
      <Sparkles size={11} color="white" />
    </div>
    <div className="cw-thinking-box">
      <div className="cw-thinking-label">
        <Sparkles size={10} color="#10b981" />
        <span>Analyzing</span>
      </div>
      <div className="cw-bar cw-bar-full"><div className="cw-bar-fill cw-bar-d0" /></div>
      <div className="cw-bar cw-bar-70"  ><div className="cw-bar-fill cw-bar-d1" /></div>
      <div className="cw-bar cw-bar-50"  ><div className="cw-bar-fill cw-bar-d2" /></div>
    </div>
  </div>
);

// ─── File icon helper ─────────────────────────────────────────
const FileIcon: React.FC<{ mime: string }> = ({ mime }) => {
  if (mime.startsWith('image/')) return <Image size={14} />;
  if (mime === 'application/pdf') return <FileText size={14} />;
  return <File size={14} />;
};

// ─── Main Widget ──────────────────────────────────────────────
export default function ChatWidget() {
  const [isOpen,      setIsOpen]      = useState(false);
  const [messages,    setMessages]    = useState<Message[]>([{
    id: '1', type: 'ai',
    content: 'Hello! I am your **Pro Finance AI Analyst**. My secure neural link is active.\n\nAsk me about clients, portfolios, meetings, or any financial topic.\n\n🎤 You can also use **voice input** or **upload a file** to ask questions.',
  }]);
  const [input,       setInput]       = useState('');
  const [isLoading,   setIsLoading]   = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [micError,    setMicError]    = useState('');
  const [attachment,  setAttachment]  = useState<{ name: string; type: string; text: string } | null>(null);

  const bottomRef    = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isOpen]);

  // ── Speech-to-Text ─────────────────────────────────────────
  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setMicError('Voice input not supported in this browser. Try Chrome.');
      setTimeout(() => setMicError(''), 3000);
      return;
    }

    if (isListening) {
      // Stop if already listening
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;

    recognition.lang            = 'en-US';
    recognition.continuous      = false;
    recognition.interimResults  = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setMicError('');
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(prev => prev ? `${prev} ${transcript}` : transcript);
      setIsListening(false);
    };

    recognition.onerror = (event: any) => {
      setIsListening(false);
      if (event.error === 'not-allowed') {
        setMicError('Microphone access denied. Allow mic in browser settings.');
      } else if (event.error === 'no-speech') {
        setMicError('No speech detected. Try again.');
      } else {
        setMicError(`Voice error: ${event.error}`);
      }
      setTimeout(() => setMicError(''), 3000);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  // ── File Upload ─────────────────────────────────────────────
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input so same file can be re-selected
    e.target.value = '';

    const MAX_SIZE = 5 * 1024 * 1024; // 5MB
    if (file.size > MAX_SIZE) {
      setMicError('File too large. Max 5MB.');
      setTimeout(() => setMicError(''), 3000);
      return;
    }

    try {
      let extractedText = '';

      const textExtensions = ['.txt','.csv','.md','.json','.xml','.yaml','.yml','.html','.htm','.log'];
      const isTextFile = file.type.startsWith('text/') || textExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      if (isTextFile) {
        // Plain text / CSV — read directly
        extractedText = await file.text();
        if (extractedText.length > 3000) {
          extractedText = extractedText.slice(0, 3000) + '\n...[truncated]';
        }
      } else if (file.type.startsWith('image/')) {
        // Image — tell the AI the filename and ask it to analyse based on name
        extractedText = `[Image uploaded: ${file.name}]`;
      } else if (file.type === 'application/pdf') {
        // PDF — use FileReader to get raw text (basic extraction)
        extractedText = await extractPdfText(file);
      } else {
        // Unsupported — try to read as text anyway
        try {
          extractedText = await file.text();
          if (extractedText.length > 3000) {
            extractedText = extractedText.slice(0, 3000) + '\n...[truncated]';
          }
        } catch {
          extractedText = `[File: ${file.name} — could not read content]`;
        }
      }

      setAttachment({ name: file.name, type: file.type, text: extractedText });

      // Smart auto-prompt based on file category
      const ext      = file.name.toLowerCase().split('.').pop() || '';
      const isImage  = file.type.startsWith('image/');
      const isSpread = ['xls','xlsx','csv'].includes(ext);
      const isSlide  = ['ppt','pptx'].includes(ext);
      const isWord   = ['doc','docx'].includes(ext);
      const isCode   = ['json','xml','yaml','yml','html','htm','log'].includes(ext);
      const isMd     = ext === 'md';

      const autoPrompt =
        isImage  ? `I've uploaded the image "${file.name}". Based on its filename, what type of image might this be and what context can you provide?`
        : isSpread ? `I've uploaded the spreadsheet "${file.name}". Please analyse the data and give me a summary of the key figures and trends.`
        : isSlide  ? `I've uploaded the presentation "${file.name}". Please summarise the key points from the content.`
        : isWord   ? `I've uploaded the document "${file.name}". Please give me a clear summary of the key information.`
        : isCode   ? `I've uploaded the file "${file.name}". Please analyse the structure and content and explain what it represents.`
        : isMd     ? `I've uploaded the markdown file "${file.name}". Please summarise the content and highlight the main points.`
        : `I've uploaded "${file.name}". Please analyse the content and give me a brief summary.`;

      setInput(prev => prev || autoPrompt);
    } catch (err) {
      setMicError('Could not read file. Try a .txt or .pdf file.');
      setTimeout(() => setMicError(''), 3000);
    }
  };

  // Basic PDF text extraction using FileReader
  const extractPdfText = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const binary = e.target?.result as string;
          // Extract readable text from PDF binary stream
          const textMatches = binary.match(/\(([^)]{1,200})\)/g) || [];
          const extracted   = textMatches
            .map(s => s.slice(1, -1))
            .filter(s => /[a-zA-Z]{2,}/.test(s))   // keep only human-readable strings
            .join(' ')
            .replace(/\\n/g, '\n')
            .replace(/\\/g, '');

          const result = extracted.length > 100
            ? (extracted.length > 3000 ? extracted.slice(0, 3000) + '\n...[truncated]' : extracted)
            : `[PDF: ${file.name} — text extraction limited. Content may be image-based.]`;

          resolve(result);
        } catch {
          resolve(`[PDF: ${file.name}]`);
        }
      };
      reader.onerror = () => resolve(`[PDF: ${file.name} — read error]`);
      reader.readAsBinaryString(file);
    });
  };

  // ── Remove attachment ───────────────────────────────────────
  const removeAttachment = () => {
    setAttachment(null);
    setInput('');
  };

  // ── Send message ────────────────────────────────────────────
  const send = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    // Build message — trim file content so Ollama doesn't time out
    const fileContent = attachment?.text
      ? attachment.text.slice(0, 1500) + (attachment.text.length > 1500 ? '\n...[truncated for speed]' : '')
      : '';
    const fullMessage = attachment
      ? `${text}\n\n---\nFILE: ${attachment.name}\nCONTENT:\n${fileContent}`
      : text;

    const userMsg: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: text,  // display only the user's text, not the full file
      ...(attachment ? { attachment: { name: attachment.name, type: attachment.type } } : {}),
    };
    const thinkId = `think-${Date.now()}`;

    setMessages(prev => [...prev, userMsg, { id: thinkId, type: 'thinking' }]);
    setInput('');
    setAttachment(null);
    setIsLoading(true);

    try {
      const token      = localStorage.getItem('token') || '';
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort(), 90000); // 90s timeout
      const res = await fetch('http://localhost:8000/api/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body:    JSON.stringify({ message: fullMessage }),
        signal:  controller.signal,
      });
      clearTimeout(timeoutId);
      const data  = await res.json();
      const reply = data.reply || data.detail || '⚠️ No response received.';

      setMessages(prev => prev.map(m => m.id === thinkId ? { ...m, resolved: true } : m));
      await new Promise(r => setTimeout(r, 200));
      setMessages(prev => [
        ...prev.filter(m => m.id !== thinkId),
        { id: thinkId, type: 'ai', content: reply },
      ]);

      if (text.toLowerCase().includes('portfolio')) {
        setTimeout(() => setMessages(prev => [...prev, {
          id: Date.now().toString(), type: 'service-card',
          serviceCard: {
            title: 'Premium Wealth Management', riskLevel: 'Moderate', riskColor: '#f59e0b',
            buttons: [{ label: 'View Portfolio', action: 'view' }, { label: 'Book Meeting', action: 'meeting' }],
          },
        }]), 400);
      }
    } catch (err: any) {
      const isTimeout = err?.name === 'AbortError';
      setMessages(prev => [
        ...prev.filter(m => m.id !== thinkId),
        {
          id: thinkId, type: 'ai',
          content: isTimeout
            ? '⚠️ **AI is taking too long.** The model is busy or the file is too large.\n\nTry:\n- Ask a shorter question\n- Upload a smaller file\n- Run `ollama run llama3.2:1b` in terminal'
            : '⚠️ **Connection Error** — ensure the backend is running on port 8000.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Render ──────────────────────────────────────────────────
  return (
    <>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.csv,.pdf,.md,.png,.jpg,.jpeg,.webp,.gif,.bmp,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.json,.xml,.yaml,.yml,.html,.htm,.log"
        style={{ display: 'none' }}
        onChange={handleFileSelect}
      />

      {/* ── Chat window ─────────────────────────────────────── */}
      {isOpen && (
        <div className="cw-window">

          {/* Header */}
          <div className="cw-header">
            <div className="cw-header-left">
              <div className="cw-header-avatar">
                <TrendingUp size={16} color="white" />
              </div>
              <div>
                <p className="cw-header-title">Finance AI Analyst</p>
                <div className="cw-header-status">
                  <span className="cw-status-dot" />
                  <span className="cw-status-text">
                    {isListening ? 'Listening…' : 'Online'}
                  </span>
                </div>
              </div>
            </div>
            <button type="button" className="cw-close-btn" onClick={() => setIsOpen(false)}>
              <X size={16} />
            </button>
          </div>

          {/* Messages */}
          <div className="cw-messages">
            {messages.map(m => {
              if (m.type === 'thinking' && !m.resolved) return <ThinkingBubble key={m.id} />;
              if (m.type === 'thinking') return null;

              if (m.type === 'service-card' && m.serviceCard) {
                const sc = m.serviceCard;
                return (
                  <div key={m.id} className="cw-sc-row">
                    <div className="cw-sc-card">
                      <p className="cw-sc-title">{sc.title}</p>
                      <span className="cw-sc-badge" style={{ background: `${sc.riskColor}22`, color: sc.riskColor }}>
                        {sc.riskLevel} Risk
                      </span>
                      <div className="cw-sc-buttons">
                        {sc.buttons.map(btn => (
                          <button key={btn.action} type="button" className="cw-sc-btn">{btn.label}</button>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              }

              const isUser = m.type === 'user';
              return (
                <div key={m.id} className={`cw-msg-row ${isUser ? 'cw-msg-row-user' : 'cw-msg-row-ai'}`}>
                  {!isUser && (
                    <div className="cw-avatar cw-avatar-sm">
                      <Bot size={12} color="white" />
                    </div>
                  )}
                  <div className={`cw-bubble ${isUser ? 'cw-bubble-user' : 'cw-bubble-ai'}`}>
                    {/* Show attachment badge if message has a file */}
                    {m.attachment && (
                      <div className="cw-attach-badge">
                        <FileIcon mime={m.attachment.type} />
                        <span>{m.attachment.name}</span>
                      </div>
                    )}
                    <ReactMarkdown components={{
                      p:      ({ children }) => <p      style={{ margin: '0 0 0.35rem' }}>{children}</p>,
                      strong: ({ children }) => <strong style={{ color: isUser ? '#fff' : '#e2e8f0', fontWeight: 700 }}>{children}</strong>,
                      ul:     ({ children }) => <ul     style={{ margin: '0.3rem 0', paddingLeft: '1.2rem' }}>{children}</ul>,
                      ol:     ({ children }) => <ol     style={{ margin: '0.3rem 0', paddingLeft: '1.2rem' }}>{children}</ol>,
                      li:     ({ children }) => <li     style={{ marginBottom: '0.2rem' }}>{children}</li>,
                      h3:     ({ children }) => <h3     style={{ color: '#10b981', fontSize: '0.85rem', fontWeight: 700, margin: '0.5rem 0 0.25rem' }}>{children}</h3>,
                      code:   ({ children }) => <code   style={{ background: 'rgba(255,255,255,0.1)', padding: '0.1rem 0.3rem', borderRadius: '0.25rem', fontSize: '0.78rem', fontFamily: 'monospace' }}>{children}</code>,
                    }}>
                      {m.content || ''}
                    </ReactMarkdown>
                  </div>
                </div>
              );
            })}
            <div ref={bottomRef} />
          </div>

          {/* Error / mic status banner */}
          {(micError || isListening) && (
            <div className={`cw-status-bar ${isListening ? 'cw-status-bar-listening' : 'cw-status-bar-error'}`}>
              {isListening ? (
                <>
                  <span className="cw-listening-dot" />
                  Listening — speak now, click mic to stop
                </>
              ) : micError}
            </div>
          )}

          {/* Attachment preview */}
          {attachment && (
            <div className="cw-attach-preview">
              <FileIcon mime={attachment.type} />
              <span className="cw-attach-name">{attachment.name}</span>
              <button type="button" className="cw-attach-remove" onClick={removeAttachment}>
                <X size={12} />
              </button>
            </div>
          )}

          {/* Input */}
          <div className="cw-input-area">
            <div className="cw-input-row">

              {/* Mic button */}
              <button
                type="button"
                className={`cw-icon-btn ${isListening ? 'cw-icon-btn-active' : ''}`}
                onClick={startListening}
                title={isListening ? 'Stop listening' : 'Voice input'}
              >
                {isListening ? <MicOff size={16} /> : <Mic size={16} />}
              </button>

              <input
                className="cw-input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder={isListening ? 'Listening…' : 'Ask me anything…'}
              />

              {/* File upload button */}
              <button
                type="button"
                className={`cw-icon-btn ${attachment ? 'cw-icon-btn-active' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                title="Upload: PDF, Word, Excel, PPT, TXT, CSV, MD, JSON, XML, YAML, HTML, Images, Logs"
              >
                <Paperclip size={16} />
              </button>

              {/* Send button */}
              <button
                type="button"
                onClick={send}
                disabled={!input.trim() || isLoading}
                className={`cw-send-btn ${input.trim() && !isLoading ? 'cw-send-btn-active' : 'cw-send-btn-idle'}`}
              >
                <Send size={15} />
              </button>
            </div>
            <p className="cw-footer-text">Secured · Voice & File upload supported</p>
          </div>
        </div>
      )}

      {/* ── FAB ─────────────────────────────────────────────── */}
      <div className="cw-fab-wrap">
        {!isOpen && <span className="cw-ping" />}
        <button
          type="button"
          onClick={() => setIsOpen(o => !o)}
          className={`cw-fab ${isOpen ? 'cw-fab-open' : 'cw-fab-closed'}`}
        >
          {isOpen ? <X size={20} /> : <MessageCircle size={20} />}
        </button>
      </div>

      {/* ── CSS ─────────────────────────────────────────────── */}
      <style>{`
        /* Window */
        .cw-window {
          position: fixed; bottom: 5.5rem; right: 2rem;
          width: 390px; height: 620px; max-height: calc(100vh - 8rem);
          z-index: 99998; display: flex; flex-direction: column;
          border-radius: 1.5rem; overflow: hidden;
          box-shadow: 0 32px 80px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.07);
          background: #0f1c2e; border: 1px solid rgba(255,255,255,0.08);
          font-family: inherit;
        }

        /* Header */
        .cw-header {
          background: linear-gradient(135deg,#0f2234,#0d1f30);
          border-bottom: 1px solid rgba(255,255,255,0.07);
          padding: 1rem 1.25rem; display: flex; align-items: center;
          justify-content: space-between; flex-shrink: 0;
        }
        .cw-header-left    { display: flex; align-items: center; gap: 0.75rem; }
        .cw-header-avatar  {
          width: 2.5rem; height: 2.5rem; border-radius: 50%;
          background: linear-gradient(135deg,#10b981,#059669);
          display: flex; align-items: center; justify-content: center;
          box-shadow: 0 0 0 3px rgba(16,185,129,0.2); flex-shrink: 0;
        }
        .cw-header-title   { color: #fff; font-weight: 700; font-size: 0.875rem; margin: 0; }
        .cw-header-status  { display: flex; align-items: center; gap: 0.375rem; margin-top: 0.2rem; }
        .cw-status-dot     { width: 0.45rem; height: 0.45rem; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981; }
        .cw-status-text    { color: #10b981; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .cw-close-btn      {
          background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
          border-radius: 0.5rem; padding: 0.375rem; cursor: pointer; color: #94a3b8;
          display: flex; align-items: center; justify-content: center;
        }
        .cw-close-btn:hover { color: #fff; }

        /* Messages */
        .cw-messages {
          flex: 1; overflow-y: auto; padding: 1.25rem 1rem;
          display: flex; flex-direction: column; gap: 0.9rem;
          scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
        }

        /* Avatar */
        .cw-avatar {
          width: 1.75rem; height: 1.75rem; border-radius: 50%;
          background: linear-gradient(135deg,#10b981,#059669);
          display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        }
        .cw-avatar-sm { margin-bottom: 2px; }

        /* Thinking */
        .cw-thinking-row { display: flex; align-items: flex-end; gap: 0.5rem; }
        .cw-thinking-box {
          background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
          border-radius: 1.1rem 1.1rem 1.1rem 0.25rem;
          padding: 0.75rem 1rem; display: flex; flex-direction: column; gap: 0.45rem; min-width: 10rem;
        }
        .cw-thinking-label { display: flex; align-items: center; gap: 0.4rem; color: #10b981; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
        .cw-bar      { height: 0.25rem; border-radius: 9999px; background: rgba(16,185,129,0.15); overflow: hidden; }
        .cw-bar-full { width: 100%; }
        .cw-bar-70   { width: 70%; }
        .cw-bar-50   { width: 50%; }
        .cw-bar-fill { height: 100%; border-radius: 9999px; background: linear-gradient(90deg,#10b981,#34d399); }
        .cw-bar-d0   { animation: cwShimmer 1.4s ease-in-out 0s    infinite; }
        .cw-bar-d1   { animation: cwShimmer 1.4s ease-in-out 0.15s infinite; }
        .cw-bar-d2   { animation: cwShimmer 1.4s ease-in-out 0.3s  infinite; }

        /* Message rows */
        .cw-msg-row      { display: flex; align-items: flex-end; gap: 0.5rem; }
        .cw-msg-row-user { justify-content: flex-end; }
        .cw-msg-row-ai   { justify-content: flex-start; }

        /* Bubbles */
        .cw-bubble {
          max-width: 78%; padding: 0.65rem 0.9rem; word-break: break-word;
          line-height: 1.65; font-size: 0.82rem; text-align: left;
        }
        .cw-bubble-user {
          border-radius: 1.1rem 1.1rem 0.25rem 1.1rem;
          background: linear-gradient(135deg,#10b981,#059669);
          color: #fff; box-shadow: 0 4px 14px rgba(16,185,129,0.2);
        }
        .cw-bubble-ai {
          border-radius: 1.1rem 1.1rem 1.1rem 0.25rem;
          background: rgba(255,255,255,0.055);
          border: 1px solid rgba(255,255,255,0.09); color: #cbd5e1;
        }

        /* Attachment badge inside bubble */
        .cw-attach-badge {
          display: flex; align-items: center; gap: 0.35rem;
          background: rgba(255,255,255,0.15); border-radius: 0.4rem;
          padding: 0.25rem 0.5rem; margin-bottom: 0.4rem;
          font-size: 0.72rem; font-weight: 600; color: #fff;
          width: fit-content;
        }

        /* Service card */
        .cw-sc-row  { display: flex; }
        .cw-sc-card {
          background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
          border-radius: 1rem; padding: 0.875rem 1rem; max-width: 90%;
        }
        .cw-sc-title   { color: #fff; font-weight: 700; font-size: 0.8rem; margin: 0 0 0.35rem; }
        .cw-sc-badge   { font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 999px; display: inline-block; margin-bottom: 0.7rem; }
        .cw-sc-buttons { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .cw-sc-btn     { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #10b981; font-size: 0.72rem; font-weight: 600; padding: 0.35rem 0.75rem; border-radius: 0.5rem; cursor: pointer; }

        /* Status bar (mic listening / error) */
        .cw-status-bar {
          padding: 0.4rem 1rem; font-size: 0.72rem; font-weight: 600;
          display: flex; align-items: center; gap: 0.4rem; flex-shrink: 0;
        }
        .cw-status-bar-listening { background: rgba(16,185,129,0.12); color: #10b981; border-top: 1px solid rgba(16,185,129,0.2); }
        .cw-status-bar-error     { background: rgba(239,68,68,0.1);   color: #f87171; border-top: 1px solid rgba(239,68,68,0.2); }
        .cw-listening-dot {
          width: 0.5rem; height: 0.5rem; border-radius: 50%; background: #10b981;
          animation: cwPulse 1s ease-in-out infinite; flex-shrink: 0;
        }

        /* Attachment preview bar */
        .cw-attach-preview {
          display: flex; align-items: center; gap: 0.5rem;
          padding: 0.4rem 1rem; background: rgba(16,185,129,0.08);
          border-top: 1px solid rgba(16,185,129,0.2); flex-shrink: 0;
          color: #10b981; font-size: 0.75rem; font-weight: 600;
        }
        .cw-attach-name   { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .cw-attach-remove {
          background: none; border: none; cursor: pointer; color: #64748b;
          display: flex; align-items: center; padding: 0.1rem;
          border-radius: 0.25rem;
        }
        .cw-attach-remove:hover { color: #f87171; }

        /* Input area */
        .cw-input-area { padding: 0.875rem 1rem; border-top: 1px solid rgba(255,255,255,0.07); background: #0b1826; flex-shrink: 0; }
        .cw-input-row  { display: flex; align-items: center; gap: 0.5rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 0.875rem; padding: 0.3rem 0.5rem 0.3rem 0.75rem; }
        .cw-icon-btn   { background: none; border: none; cursor: pointer; color: #64748b; display: flex; align-items: center; padding: 0.25rem; border-radius: 0.35rem; transition: all 0.15s; }
        .cw-icon-btn:hover       { color: #10b981; }
        .cw-icon-btn-active      { color: #10b981 !important; background: rgba(16,185,129,0.15); }
        .cw-input                { flex: 1; background: transparent; border: none; outline: none; color: #e2e8f0; font-size: 0.82rem; padding: 0.5rem 0; caret-color: #10b981; }
        .cw-input::placeholder   { color: #475569; }
        .cw-send-btn             { border: none; border-radius: 0.6rem; padding: 0.45rem 0.55rem; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all 0.2s; cursor: pointer; }
        .cw-send-btn-active      { background: linear-gradient(135deg,#10b981,#059669); color: #fff; box-shadow: 0 2px 8px rgba(16,185,129,0.3); }
        .cw-send-btn-idle        { background: rgba(255,255,255,0.06); color: #475569; cursor: default; }
        .cw-footer-text          { color: #1e3a5f; font-size: 0.62rem; text-align: center; margin-top: 0.45rem; }

        /* FAB */
        .cw-fab-wrap { position: fixed; bottom: 2rem; right: 2rem; z-index: 99999; }
        .cw-fab {
          width: 3.25rem; height: 3.25rem; border-radius: 50%; cursor: pointer; color: #fff;
          display: flex; align-items: center; justify-content: center;
          transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
        }
        .cw-fab-open   { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); box-shadow: none; }
        .cw-fab-closed { background: linear-gradient(135deg,#10b981,#059669); border: none; box-shadow: 0 8px 24px rgba(16,185,129,0.45); }
        .cw-fab:hover  { transform: scale(1.08); }
        .cw-ping {
          position: absolute; inset: -3px; border-radius: 50%;
          border: 2px solid rgba(16,185,129,0.35);
          animation: cwPing 2s ease-out infinite; pointer-events: none;
        }

        /* Keyframes */
        @keyframes cwShimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(200%)} }
        @keyframes cwPing    { 0%{transform:scale(1);opacity:0.5} 100%{transform:scale(1.6);opacity:0} }
        @keyframes cwPulse   { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </>
  );
}