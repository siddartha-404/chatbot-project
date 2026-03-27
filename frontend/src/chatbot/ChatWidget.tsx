
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MessageCircle, X, Send, Mic, MicOff, Paperclip,
  TrendingUp, Bot, Sparkles, FileText, Image, File,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// ─── Types ────────────────────────────────────────────────────
interface Message {
  id:        string;
  type:      'ai' | 'user' | 'thinking';
  content?:  string;
  resolved?: boolean;
  attachment?: { name: string; type: string };
}

// ─── Web Speech API shim ──────────────────────────────────────
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

// ─── Thinking bubble ──────────────────────────────────────────
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
  if (mime.startsWith('image/'))       return <Image    size={14} />;
  if (mime === 'application/pdf')      return <FileText size={14} />;
  return <File size={14} />;
};

// ─── Main widget ──────────────────────────────────────────────
export default function ChatWidget() {
  const [isOpen,      setIsOpen]      = useState(false);
  const navigate = useNavigate();
  const [messages,    setMessages]    = useState<Message[]>([{
    id: '1', type: 'ai',
    content:
      'Hello! I am your **Pro Finance AI Analyst**. My secure neural link is active.\n\n' +
      'Ask me about clients, portfolios, meetings, or any financial topic.\n\n' +
      '🎤 You can also use **voice input** or **upload a file** to ask questions.',
  }]);
  const [input,       setInput]       = useState('');
  const [isLoading,   setIsLoading]   = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [micError,    setMicError]    = useState('');
  const [attachment,  setAttachment]  = useState<{ name: string; type: string; text: string } | null>(null);

  const bottomRef      = useRef<HTMLDivElement>(null);
  const fileInputRef   = useRef<HTMLInputElement>(null);
  const recRef         = useRef<any>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isOpen]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recRef.current) {
        try { recRef.current.abort(); } catch { /* ignore */ }
      }
    };
  }, []);

  // ── EXACT WORKING Mic toggle from your previous code ──────────
  const toggleMic = () => {
    // ── STOP ─────────────────────────────────────────────────
    if (isListening) {
      if (recRef.current) {
        try { recRef.current.stop(); } catch { /* ignore */ }
        recRef.current = null;
      }
      setIsListening(false);
      return;
    }

    // ── START ─────────────────────────────────────────────────
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setMicError('Voice input not supported. Use Chrome or Edge.');
      setTimeout(() => setMicError(''), 4000);
      return;
    }

    const rec = new SR();
    recRef.current = rec;

    // This is the working logic: wait for the user to finish speaking, 
    // then process all text perfectly at once.
    rec.lang            = 'en-US';
    rec.continuous      = false;
    rec.interimResults  = false;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      setIsListening(true);
      setMicError('');
    };

    rec.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript.trim();
      if (transcript) {
        // Append to existing input (in case user already typed something)
        setInput(prev => {
          const existing = prev.trim();
          return existing ? `${existing} ${transcript}` : transcript;
        });
      }
      setIsListening(false);
      recRef.current = null;
    };

    rec.onerror = (event: any) => {
      setIsListening(false);
      recRef.current = null;
      if (event.error === 'aborted') return; // user stopped — no error needed
      if (event.error === 'not-allowed' || event.error === 'permission-denied') {
        setMicError('Microphone access denied. Allow mic in browser settings.');
      } else if (event.error === 'no-speech') {
        setMicError('No speech detected. Please try again.');
      } else if (event.error === 'network') {
        setMicError('Network error with speech service. Check your connection.');
      } else if (event.error === 'audio-capture') {
        setMicError('No microphone found. Connect one and try again.');
      } else {
        setMicError(`Voice error: ${event.error}`);
      }
      setTimeout(() => setMicError(''), 4000);
    };

    rec.onend = () => {
      // Always reset state when recognition ends for any reason
      setIsListening(false);
      recRef.current = null;
    };

    try {
      rec.start();
    } catch (err: any) {
      setMicError('Could not start microphone. Check browser permissions.');
      setTimeout(() => setMicError(''), 4000);
      setIsListening(false);
      recRef.current = null;
    }
  };

  // ── File upload ───────────────────────────────────────────────
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    if (file.size > 5 * 1024 * 1024) {
      setMicError('File too large. Max 5 MB.');
      setTimeout(() => setMicError(''), 3000);
      return;
    }

    try {
      let extractedText = '';
      const textExts = ['.txt','.csv','.md','.json','.xml','.yaml','.yml','.html','.htm','.log'];
      const isText   = file.type.startsWith('text/') || textExts.some(x => file.name.toLowerCase().endsWith(x));

      if (isText) {
        extractedText = await file.text();
        if (extractedText.length > 3000) extractedText = extractedText.slice(0, 3000) + '\n...[truncated]';
      } else if (file.type.startsWith('image/')) {
        extractedText = `[Image uploaded: ${file.name}]`;
      } else if (file.type === 'application/pdf') {
        extractedText = await extractPdfText(file);
      } else {
        try {
          extractedText = await file.text();
          if (extractedText.length > 3000) extractedText = extractedText.slice(0, 3000) + '\n...[truncated]';
        } catch {
          extractedText = `[File: ${file.name} — could not read content]`;
        }
      }

      setAttachment({ name: file.name, type: file.type, text: extractedText });

      const ext = file.name.toLowerCase().split('.').pop() || '';
      const autoPrompt =
        file.type.startsWith('image/')
          ? `I've uploaded the image "${file.name}". What context can you provide based on the filename?`
        : ['xls','xlsx','csv'].includes(ext)
          ? `I've uploaded the spreadsheet "${file.name}". Please analyse the data and summarise key figures and trends.`
        : ['ppt','pptx'].includes(ext)
          ? `I've uploaded the presentation "${file.name}". Please summarise the key points.`
        : ['doc','docx'].includes(ext)
          ? `I've uploaded the document "${file.name}". Please give me a clear summary.`
        : ['json','xml','yaml','yml','html','htm','log'].includes(ext)
          ? `I've uploaded "${file.name}". Please analyse the structure and explain what it represents.`
        : ext === 'md'
          ? `I've uploaded the markdown file "${file.name}". Please summarise and highlight the main points.`
        : `I've uploaded "${file.name}". Please analyse the content and give a brief summary.`;

      setInput(prev => prev || autoPrompt);
    } catch {
      setMicError('Could not read file. Try a .txt or .pdf file.');
      setTimeout(() => setMicError(''), 3000);
    }
  };

  const extractPdfText = (file: File): Promise<string> =>
    new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = e => {
        try {
          const binary  = e.target?.result as string;
          const matches = binary.match(/\(([^)]{1,200})\)/g) || [];
          const text    = matches
            .map(s => s.slice(1, -1))
            .filter(s => /[a-zA-Z]{2,}/.test(s))
            .join(' ')
            .replace(/\\n/g, '\n')
            .replace(/\\/g, '');
          resolve(
            text.length > 100
              ? (text.length > 3000 ? text.slice(0, 3000) + '\n...[truncated]' : text)
              : `[PDF: ${file.name} — text extraction limited. Content may be image-based.]`
          );
        } catch { resolve(`[PDF: ${file.name}]`); }
      };
      reader.onerror = () => resolve(`[PDF: ${file.name} — read error]`);
      reader.readAsBinaryString(file);
    });

  const removeAttachment = () => { setAttachment(null); setInput(''); };

  // ── Send ──────────────────────────────────────────────────────
  const send = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    // Stop mic if still active
    if (isListening) {
      if (recRef.current) { try { recRef.current.stop(); } catch { /* ignore */ } recRef.current = null; }
      setIsListening(false);
    }

    const fileContent = attachment?.text
      ? attachment.text.slice(0, 1500) + (attachment.text.length > 1500 ? '\n...[truncated for speed]' : '')
      : '';
    const fullMessage = attachment
      ? `${text}\n\n---\nFILE: ${attachment.name}\nCONTENT:\n${fileContent}`
      : text;

    const userMsg: Message = {
      id: Date.now().toString(), type: 'user', content: text,
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
      
      // Increased timeout to 150 seconds
      const tid        = setTimeout(() => controller.abort(), 150000); 
      
      const res = await fetch('http://localhost:8000/api/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body:    JSON.stringify({ message: fullMessage }),
        signal:  controller.signal,
      });
      clearTimeout(tid);
      const data  = await res.json();
      const reply = data.reply || data.detail || '⚠️ No response received.';

      setMessages(prev => prev.map(m => m.id === thinkId ? { ...m, resolved: true } : m));
      await new Promise(r => setTimeout(r, 200));

      // --- NEW: Handle Navigation Marker ---
      let finalReply = reply;
      const navMatch = finalReply.match(/🧭NAV:([a-z]+)/);
      if (navMatch) {
        const page = navMatch[1];
        const paths: Record<string, string> = { dashboard: '/', clients: '/clients', portfolios: '/portfolios', services: '/services', meetings: '/meetings' };
        if (paths[page]) {
          navigate(paths[page]);
        }
        // Remove the ugly tag from the text the user sees
        finalReply = finalReply.replace(/🧭NAV:[a-z]+/, '').trim();
      }

      setMessages(prev => [
        ...prev.filter(m => m.id !== thinkId),
        { id: thinkId, type: 'ai', content: finalReply },
      ]);

      // Dispatch event to trigger frontend reload if AI confirms action
      if (finalReply.includes('✅') || finalReply.includes('🗑️')) {
        window.dispatchEvent(new Event('db-updated'));
      }

    } catch (err: any) {
      const isTimeout = err?.name === 'AbortError';
      setMessages(prev => [
        ...prev.filter(m => m.id !== thinkId),
        {
          id: thinkId, type: 'ai',
          content: isTimeout
            ? '⚠️ **AI is taking too long.**\n\nTry:\n- Ask a shorter question\n- Upload a smaller file'
            : '⚠️ **Connection Error** — ensure the backend is running on port 8000.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────
  return (
    <>
      <input
        ref={fileInputRef} type="file"
        accept=".txt,.csv,.pdf,.md,.png,.jpg,.jpeg,.webp,.gif,.bmp,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.json,.xml,.yaml,.yml,.html,.htm,.log"
        style={{ display: 'none' }}
        onChange={handleFileSelect}
      />

      {/* ── Chat window ──────────────────────────────────────── */}
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
                  <span className={`cw-status-dot ${isListening ? 'cw-status-dot-mic' : ''}`} />
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

              const isUser = m.type === 'user';
              return (
                <div key={m.id} className={`cw-msg-row ${isUser ? 'cw-msg-row-user' : 'cw-msg-row-ai'}`}>
                  {!isUser && (
                    <div className="cw-avatar cw-avatar-sm">
                      <Bot size={12} color="white" />
                    </div>
                  )}
                  <div className={`cw-bubble ${isUser ? 'cw-bubble-user' : 'cw-bubble-ai'}`}>
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

          {/* Status bar */}
          {(micError || isListening) && (
            <div className={`cw-status-bar ${isListening ? 'cw-status-bar-listening' : 'cw-status-bar-error'}`}>
              {isListening ? (
                <>
                  <span className="cw-listening-dot" />
                  Listening — speak now, text will appear when you finish...
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

          {/* Input row */}
          <div className="cw-input-area">
            <div className={`cw-input-row ${isListening ? 'cw-input-row-active' : ''}`}>

              {/* Mic toggle */}
              <button
                type="button"
                className={`cw-icon-btn ${isListening ? 'cw-icon-btn-mic-active' : ''}`}
                onClick={toggleMic}
                title={isListening ? 'Click to stop recording' : 'Click to start voice input'}
              >
                {isListening ? <MicOff size={16} /> : <Mic size={16} />}
              </button>

              <input
                className="cw-input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                placeholder={isListening ? '🔴 Listening — speak now…' : 'Ask me anything…'}
              />

              {/* File upload */}
              <button
                type="button"
                className={`cw-icon-btn ${attachment ? 'cw-icon-btn-active' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                title="Upload file (PDF, Word, Excel, image, text…)"
              >
                <Paperclip size={16} />
              </button>

              {/* Send */}
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

      {/* ── FAB ──────────────────────────────────────────────── */}
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

      {/* ── Styles ───────────────────────────────────────────── */}
      <style>{`
        .cw-window {
          position:fixed; bottom:5.5rem; right:2rem;
          width:390px; height:620px; max-height:calc(100vh - 8rem);
          z-index:99998; display:flex; flex-direction:column;
          border-radius:1.5rem; overflow:hidden;
          box-shadow:0 32px 80px rgba(0,0,0,.65),0 0 0 1px rgba(255,255,255,.07);
          background:#0f1c2e; border:1px solid rgba(255,255,255,.08);
          font-family:inherit;
        }

        /* Header */
        .cw-header {
          background:linear-gradient(135deg,#0f2234,#0d1f30);
          border-bottom:1px solid rgba(255,255,255,.07);
          padding:1rem 1.25rem; display:flex; align-items:center;
          justify-content:space-between; flex-shrink:0;
        }
        .cw-header-left   { display:flex; align-items:center; gap:.75rem; }
        .cw-header-avatar {
          width:2.5rem; height:2.5rem; border-radius:50%;
          background:linear-gradient(135deg,#10b981,#059669);
          display:flex; align-items:center; justify-content:center;
          box-shadow:0 0 0 3px rgba(16,185,129,.2); flex-shrink:0;
        }
        .cw-header-title  { color:#fff; font-weight:700; font-size:.875rem; margin:0; }
        .cw-header-status { display:flex; align-items:center; gap:.375rem; margin-top:.2rem; }
        .cw-status-dot    { width:.45rem; height:.45rem; border-radius:50%; background:#10b981; box-shadow:0 0 6px #10b981; transition:background .3s,box-shadow .3s; }
        .cw-status-dot-mic { background:#f43f5e !important; box-shadow:0 0 8px #f43f5e !important; animation:cwPulse .8s ease-in-out infinite; }
        .cw-status-text   { color:#10b981; font-size:.65rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .cw-close-btn {
          background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
          border-radius:.5rem; padding:.375rem; cursor:pointer; color:#94a3b8;
          display:flex; align-items:center; justify-content:center;
        }
        .cw-close-btn:hover { color:#fff; }

        /* Messages */
        .cw-messages {
          flex:1; overflow-y:auto; padding:1.25rem 1rem;
          display:flex; flex-direction:column; gap:.9rem;
          scrollbar-width:thin; scrollbar-color:rgba(255,255,255,.08) transparent;
        }

        /* Avatar */
        .cw-avatar {
          width:1.75rem; height:1.75rem; border-radius:50%;
          background:linear-gradient(135deg,#10b981,#059669);
          display:flex; align-items:center; justify-content:center; flex-shrink:0;
        }
        .cw-avatar-sm { margin-bottom:2px; }

        /* Thinking */
        .cw-thinking-row { display:flex; align-items:flex-end; gap:.5rem; }
        .cw-thinking-box {
          background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.25);
          border-radius:1.1rem 1.1rem 1.1rem .25rem;
          padding:.75rem 1rem; display:flex; flex-direction:column; gap:.45rem; min-width:10rem;
        }
        .cw-thinking-label { display:flex; align-items:center; gap:.4rem; color:#10b981; font-size:.62rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .cw-bar      { height:.25rem; border-radius:9999px; background:rgba(16,185,129,.15); overflow:hidden; }
        .cw-bar-full { width:100%; }
        .cw-bar-70   { width:70%; }
        .cw-bar-50   { width:50%; }
        .cw-bar-fill { height:100%; border-radius:9999px; background:linear-gradient(90deg,#10b981,#34d399); }
        .cw-bar-d0   { animation:cwShimmer 1.4s ease-in-out 0s    infinite; }
        .cw-bar-d1   { animation:cwShimmer 1.4s ease-in-out .15s  infinite; }
        .cw-bar-d2   { animation:cwShimmer 1.4s ease-in-out .3s   infinite; }

        /* Message rows */
        .cw-msg-row      { display:flex; align-items:flex-end; gap:.5rem; }
        .cw-msg-row-user { justify-content:flex-end; }
        .cw-msg-row-ai   { justify-content:flex-start; }

        /* Bubbles */
        .cw-bubble {
          max-width:78%; padding:.65rem .9rem; word-break:break-word;
          line-height:1.65; font-size:.82rem; text-align:left;
        }
        .cw-bubble-user {
          border-radius:1.1rem 1.1rem .25rem 1.1rem;
          background:linear-gradient(135deg,#10b981,#059669);
          color:#fff; box-shadow:0 4px 14px rgba(16,185,129,.2);
        }
        .cw-bubble-ai {
          border-radius:1.1rem 1.1rem 1.1rem .25rem;
          background:rgba(255,255,255,.055);
          border:1px solid rgba(255,255,255,.09); color:#cbd5e1;
        }

        /* Attach badge */
        .cw-attach-badge {
          display:flex; align-items:center; gap:.35rem;
          background:rgba(255,255,255,.15); border-radius:.4rem;
          padding:.25rem .5rem; margin-bottom:.4rem;
          font-size:.72rem; font-weight:600; color:#fff; width:fit-content;
        }

        /* Status bar */
        .cw-status-bar {
          padding:.4rem 1rem; font-size:.72rem; font-weight:600;
          display:flex; align-items:center; gap:.4rem; flex-shrink:0;
        }
        .cw-status-bar-listening { background:rgba(244,63,94,.1); color:#fb7185; border-top:1px solid rgba(244,63,94,.25); }
        .cw-status-bar-error     { background:rgba(239,68,68,.1); color:#f87171; border-top:1px solid rgba(239,68,68,.2); }
        .cw-listening-dot {
          width:.5rem; height:.5rem; border-radius:50%; background:#f43f5e; flex-shrink:0;
          animation:cwPulse .8s ease-in-out infinite;
        }

        /* Attachment preview */
        .cw-attach-preview {
          display:flex; align-items:center; gap:.5rem;
          padding:.4rem 1rem; background:rgba(16,185,129,.08);
          border-top:1px solid rgba(16,185,129,.2); flex-shrink:0;
          color:#10b981; font-size:.75rem; font-weight:600;
        }
        .cw-attach-name   { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .cw-attach-remove {
          background:none; border:none; cursor:pointer; color:#64748b;
          display:flex; align-items:center; padding:.1rem; border-radius:.25rem;
        }
        .cw-attach-remove:hover { color:#f87171; }

        /* Input area */
        .cw-input-area { padding:.875rem 1rem; border-top:1px solid rgba(255,255,255,.07); background:#0b1826; flex-shrink:0; }
        .cw-input-row {
          display:flex; align-items:center; gap:.5rem;
          background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
          border-radius:.875rem; padding:.3rem .5rem .3rem .75rem;
          transition:border-color .2s,background .2s;
        }
        .cw-input-row-active { border-color:rgba(244,63,94,.5) !important; background:rgba(244,63,94,.04) !important; }
        .cw-icon-btn              { background:none; border:none; cursor:pointer; color:#64748b; display:flex; align-items:center; padding:.25rem; border-radius:.35rem; transition:all .15s; }
        .cw-icon-btn:hover        { color:#10b981; }
        .cw-icon-btn-active       { color:#10b981 !important; background:rgba(16,185,129,.15); }
        .cw-icon-btn-mic-active   { color:#f43f5e !important; background:rgba(244,63,94,.15) !important; animation:cwMicGlow 1s ease-in-out infinite; }
        .cw-input                 { flex:1; background:transparent; border:none; outline:none; color:#e2e8f0; font-size:.82rem; padding:.5rem 0; caret-color:#10b981; }
        .cw-input::placeholder    { color:#475569; }
        .cw-send-btn              { border:none; border-radius:.6rem; padding:.45rem .55rem; display:flex; align-items:center; justify-content:center; flex-shrink:0; transition:all .2s; cursor:pointer; }
        .cw-send-btn-active       { background:linear-gradient(135deg,#10b981,#059669); color:#fff; box-shadow:0 2px 8px rgba(16,185,129,.3); }
        .cw-send-btn-idle         { background:rgba(255,255,255,.06); color:#475569; cursor:default; }
        .cw-footer-text           { color:#1e3a5f; font-size:.62rem; text-align:center; margin-top:.45rem; }

        /* FAB */
        .cw-fab-wrap { position:fixed; bottom:2rem; right:2rem; z-index:99999; }
        .cw-fab {
          width:3.25rem; height:3.25rem; border-radius:50%; cursor:pointer; color:#fff;
          display:flex; align-items:center; justify-content:center;
          transition:all .25s cubic-bezier(.34,1.56,.64,1);
        }
        .cw-fab-open   { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.15); box-shadow:none; }
        .cw-fab-closed { background:linear-gradient(135deg,#10b981,#059669); border:none; box-shadow:0 8px 24px rgba(16,185,129,.45); }
        .cw-fab:hover  { transform:scale(1.08); }
        .cw-ping {
          position:absolute; inset:-3px; border-radius:50%;
          border:2px solid rgba(16,185,129,.35);
          animation:cwPing 2s ease-out infinite; pointer-events:none;
        }

        /* Keyframes */
        @keyframes cwShimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(200%)} }
        @keyframes cwPing    { 0%{transform:scale(1);opacity:.5} 100%{transform:scale(1.6);opacity:0} }
        @keyframes cwPulse   { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes cwMicGlow { 0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,.5)} 50%{box-shadow:0 0 0 5px rgba(244,63,94,0)} }
      `}</style>
    </>
  );
}