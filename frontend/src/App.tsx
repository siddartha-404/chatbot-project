'use client';

import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import ChatWidget from './chatbot/ChatWidget';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import {
  LayoutDashboard, Users, Briefcase, Settings, Calendar, TrendingUp,
  Search, Bell, ChevronDown, DollarSign, X, Clock, Shield, FileText,
  CheckCircle, Lock, User, Eye, EyeOff, AlertCircle, ArrowRight
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────
// GLOBAL API FETCH — auto-logout on 401, always sends token
// ─────────────────────────────────────────────────────────────
let _forceLogout: (() => void) | null = null;

const apiFetch = async (url: string, options: RequestInit = {}): Promise<Response> => {
  const token = localStorage.getItem('token') || '';
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${token}`,
      ...(options.headers || {}),
    },
  });
  if (res.status === 401) {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    if (_forceLogout) _forceLogout();
  }
  return res;
};

// ─────────────────────────────────────────────────────────────
// PUBLIC BUSINESS LANDING PAGE (NO LOGIN REQUIRED)
// ─────────────────────────────────────────────────────────────
const PublicLandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#020617] text-white flex flex-col relative overflow-hidden font-sans">
      {/* Background Glow Effects */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 blur-[120px] rounded-full" />
      </div>

      {/* Header */}
      <header className="p-6 md:px-12 flex justify-between items-center relative z-10 border-b border-white/5 bg-slate-900/50 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/30">
            <TrendingUp className="text-white w-6 h-6" />
          </div>
          <span className="text-2xl font-bold tracking-tight">Pro Finance</span>
        </div>
        <Link to="/login" className="text-sm font-bold text-slate-300 hover:text-white transition-colors bg-white/5 px-5 py-2.5 rounded-xl border border-white/10 hover:bg-white/10 flex items-center gap-2">
          Admin Portal <ArrowRight size={16} />
        </Link>
      </header>

      {/* Hero Content */}
      <main className="flex-1 flex flex-col items-center justify-center text-center px-6 relative z-10">
        <div className="max-w-4xl">
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold text-sm tracking-wide uppercase">
            AI-Powered Wealth Management
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent leading-tight">
            Secure Your Financial Future Today.
          </h1>
          <p className="text-lg md:text-xl text-slate-400 mb-10 leading-relaxed max-w-2xl mx-auto">
            Expert wealth management, tax planning, and retirement strategies tailored to your unique goals. Talk to our AI assistant to get a personalized consultation.
          </p>
          <div className="flex items-center justify-center gap-4 animate-bounce">
            <p className="text-emerald-400 font-bold">Chat with our AI on the right 👉</p>
          </div>
        </div>
      </main>

      {/* The Chatbot acts as the Lead Gen tool here */}
      <ChatWidget />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// LOGIN PAGE
// ─────────────────────────────────────────────────────────────
const LoginPage: React.FC<{ onLogin: (t: string, u: string) => void }> = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw,   setShowPw]   = useState(false);
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const res = await fetch('http://localhost:8000/api/login', {
        method:  'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body:    `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
      });
      if (res.ok) {
        const d = await res.json();
        onLogin(d.access_token, d.username);
      } else {
        const d = await res.json().catch(() => ({}));
        setError(d.detail || 'Invalid credentials');
      }
    } catch {
      setError('Connection refused — start the backend first');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[#020617] relative overflow-hidden font-sans">
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 blur-[120px] rounded-full" />
      </div>
      
      {/* Back to Home Button */}
      <Link to="/" className="absolute top-6 left-6 text-slate-400 hover:text-white flex items-center gap-2 transition-colors z-20">
        <ArrowRight size={18} className="rotate-180" /> Back to Website
      </Link>

      <div className="w-full max-w-[420px] px-6 relative z-10">
        <div className="bg-slate-900/40 backdrop-blur-3xl border border-white/10 p-10 rounded-[2.5rem] shadow-2xl">
          <div className="flex flex-col items-center mb-10">
            <div className="w-16 h-16 bg-emerald-500 rounded-2xl flex items-center justify-center mb-4 shadow-lg shadow-emerald-500/30">
              <TrendingUp className="text-white w-8 h-8" />
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Admin Portal</h1>
            <p className="text-slate-400 text-sm mt-1">Internal System Access</p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
              <input required type="text" placeholder="Admin Username" value={username}
                onChange={e => setUsername(e.target.value)}
                className="w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-xl text-white outline-none focus:border-emerald-500/50 text-sm transition-colors" />
            </div>
            <div className="relative">
              <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
              <input required type={showPw ? 'text' : 'password'} placeholder="Password" value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full pl-12 pr-12 py-4 bg-white/5 border border-white/10 rounded-xl text-white outline-none focus:border-emerald-500/50 text-sm transition-colors" />
              <button type="button" onClick={() => setShowPw(p => !p)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-emerald-400 transition-colors">
                {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            {error && (
              <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs py-2.5 px-3 rounded-lg">
                <AlertCircle size={14} className="flex-shrink-0" />
                {error}
              </div>
            )}
            <button disabled={loading} type="submit"
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-white font-bold rounded-2xl shadow-lg shadow-emerald-500/20 mt-2 transition-all active:scale-[0.98]">
              {loading ? 'Authenticating…' : 'Login'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// MODAL PORTAL
// ─────────────────────────────────────────────────────────────
const Modal: React.FC<{ onClose: () => void; children: React.ReactNode; wide?: boolean }> = ({ onClose, children, wide }) =>
  ReactDOM.createPortal(
    <div
      onMouseDown={onClose}
      style={{ position:'fixed', inset:0, zIndex:9999, display:'flex', alignItems:'center', justifyContent:'center', padding:'1rem', background:'rgba(2,6,23,0.9)', backdropFilter:'blur(10px)' }}
    >
      <div onMouseDown={e => e.stopPropagation()} style={{ width:'100%', maxWidth: wide ? '42rem' : '28rem' }}>
        {children}
      </div>
    </div>,
    document.body,
  );

// ─────────────────────────────────────────────────────────────
// SIDEBAR (ADMIN ONLY)
// ─────────────────────────────────────────────────────────────
const Sidebar: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
  const loc = useLocation();
  const nav = [
    { icon: LayoutDashboard, label: 'Dashboard',  href: '/admin' },
    { icon: Users,           label: 'Clients',    href: '/admin/clients' },
    { icon: Briefcase,       label: 'Portfolios', href: '/admin/portfolios' },
    { icon: Settings,        label: 'Services',   href: '/admin/services' },
    { icon: Calendar,        label: 'Meetings',   href: '/admin/meetings' },
  ];
  return (
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col border-r border-white/10 z-30 bg-slate-900/80 backdrop-blur-2xl">
      <div className="px-6 py-6 border-b border-white/10 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
          <TrendingUp className="w-5 h-5 text-white" />
        </div>
        <div>
          <p className="text-lg font-bold text-white tracking-tight">Pro Finance</p>
          <p className="text-xs text-slate-400">Wealth Management</p>
        </div>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-1">
        {nav.map(item => {
          const active = loc.pathname === item.href || (item.href !== '/admin' && loc.pathname.startsWith(item.href));
          return (
            <Link key={item.href} to={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${active ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
              <item.icon className={`w-5 h-5 flex-shrink-0 ${active ? 'text-emerald-400' : ''}`} />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-white/10 space-y-2">
        <Link to="/" className="block w-full py-2 text-slate-400 hover:text-emerald-400 text-sm font-medium transition-colors text-left flex items-center gap-2">
          <ArrowRight size={16} className="rotate-180" /> View Live Site
        </Link>
        <button onClick={onLogout} className="w-full py-2 text-slate-500 hover:text-red-400 text-sm font-medium transition-colors text-left">
          Logout
        </button>
      </div>
    </aside>
  );
};

// ─────────────────────────────────────────────────────────────
// CLEAN HEADER
// ─────────────────────────────────────────────────────────────
const Header: React.FC = () => (
  <header className="sticky top-0 z-20 h-16 flex items-center px-8 border-b border-white/10 bg-slate-900/60 backdrop-blur-2xl shadow-sm">
    <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-3">
      <TrendingUp className="w-6 h-6 text-emerald-400" />
      Admin Dashboard
    </h1>
  </header>
);

// ─────────────────────────────────────────────────────────────
// INLINE ERROR BANNER
// ─────────────────────────────────────────────────────────────
const ErrBanner: React.FC<{ msg: string; onClose: () => void }> = ({ msg, onClose }) => (
  <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-4 py-3 rounded-xl mb-4">
    <AlertCircle size={16} className="flex-shrink-0" />
    <span className="flex-1">{msg}</span>
    <button type="button" onClick={onClose}><X size={14} /></button>
  </div>
);

// ─────────────────────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────────────────────
const DashboardHome: React.FC = () => {
  const [clientCount, setClientCount] = useState(0);
  const [reportData,  setReportData]  = useState<any>(null);
  const [showReport,  setShowReport]  = useState(false);

  const fetchClientCount = useCallback(() => {
    apiFetch('http://localhost:8000/api/clients')
      .then(r => r.json()).then(d => setClientCount(Array.isArray(d) ? d.length : 0)).catch(() => {});
  }, []);

  useEffect(() => {
    fetchClientCount();
    const handleUpdate = () => fetchClientCount();
    window.addEventListener('db-updated', handleUpdate);
    return () => window.removeEventListener('db-updated', handleUpdate);
  }, [fetchClientCount]);

  const fetchReport = (type: 'monthly' | 'quarterly') => {
    apiFetch(`http://localhost:8000/api/reports/${type}`)
      .then(r => r.json()).then(d => { setReportData(d); setShowReport(true); }).catch(() => {});
  };

  const chartData = [
    { month: 'Jan', value: 4.2 },  { month: 'Feb', value: 4.8 },
    { month: 'Mar', value: 5.1 },  { month: 'Apr', value: 5.9 },
    { month: 'May', value: 7.2 },  { month: 'Jun', value: 8.1 },
    { month: 'Jul', value: 8.9 },  { month: 'Aug', value: 9.8 },
    { month: 'Sep', value: 10.5 }, { month: 'Oct', value: 11.2 },
    { month: 'Nov', value: 11.8 }, { month: 'Dec', value: 12.5 },
  ];

  const isQ       = reportData?.report_type?.toLowerCase().includes('quarterly');
  const statAUM   = reportData?.stats?.total_aum   ?? reportData?.firm_stats?.total_aum   ?? '—';
  const statLabel = isQ ? 'Allocation Efficiency' : 'Firm Health Score';
  const statVal   = reportData?.stats?.allocation_efficiency ?? reportData?.firm_stats?.health_score ?? '—';
  const rawItems: string[] = reportData ? (reportData.risk_alerts || reportData.insights || []) : [];
  const fallback  = isQ
    ? ['Capital allocation is optimized for current market volatility.',
       'Investment yield outperformed benchmark by 2.4%.',
       'Diversification ratio is within healthy parameters (85%+).']
    : ['No urgent alerts detected across active portfolios.',
       'Liquidity ratios are stable across all active portfolios.'];
  const items = rawItems.length > 0 ? fallback.map((fb, i) => rawItems[i] || fb) : fallback;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        {([
          { icon: TrendingUp, label: 'Total AUM',       value: '$12.5M',    sub: null },
          { icon: Users,      label: 'Active Clients',  value: clientCount, sub: null },
          { icon: DollarSign, label: 'Monthly Revenue', value: '$84,200',   sub: null },
          { icon: Users,      label: 'Acquisition',     value: '+12%',      sub: 'Growth vs prev' },
        ] as const).map((card, i) => (
          <div key={i} className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md text-left">
            <card.icon className="w-5 h-5 text-emerald-400 mb-4" />
            <p className="text-sm text-slate-400 mb-1">{card.label}</p>
            <p className="text-3xl font-bold text-white">{card.value}</p>
            {card.sub && <div className="mt-1 text-[10px] text-emerald-400 font-bold uppercase bg-emerald-500/10 w-fit px-2 py-0.5 rounded">{card.sub}</div>}
          </div>
        ))}
      </div>

      <div className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-bold text-white">Portfolio Growth</h2>
          <div className="flex gap-2">
            <button onClick={() => fetchReport('monthly')}
              className="flex items-center gap-1.5 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-lg text-xs font-bold hover:bg-emerald-500/20 transition-all">
              <FileText className="w-3.5 h-3.5" /> Monthly
            </button>
            <button onClick={() => fetchReport('quarterly')}
              className="flex items-center gap-1.5 px-4 py-2 bg-emerald-500 text-white rounded-lg text-xs font-bold shadow-lg shadow-emerald-500/20 hover:bg-emerald-600 transition-all">
              <TrendingUp className="w-3.5 h-3.5" /> Quarterly
            </button>
          </div>
        </div>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="99%" height="99%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="clr" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} tickFormatter={v => `$${v}M`} />
              <Tooltip content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                return (
                  <div className="bg-slate-900 border border-emerald-500/40 p-4 rounded-2xl shadow-2xl">
                    <p className="text-xs font-black text-white uppercase tracking-widest mb-2 border-b border-white/10 pb-1">{label}</p>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      <p className="text-emerald-400 font-bold text-lg">${payload[0].value}M</p>
                    </div>
                  </div>
                );
              }} />
              <Area type="monotone" dataKey="value" stroke="#10b981" strokeWidth={3} fill="url(#clr)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {showReport && reportData && (
        <Modal onClose={() => setShowReport(false)} wide>
          <div className="bg-slate-900 border border-emerald-500/30 p-8 rounded-3xl shadow-2xl relative overflow-hidden text-white">
            <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
              <TrendingUp className="w-32 h-32 text-emerald-400" />
            </div>
            <div className="flex justify-between items-start mb-8 relative z-10">
              <div>
                <h3 className="text-2xl font-bold mb-1">{reportData.report_type}</h3>
                <p className="text-emerald-400 font-mono text-sm">{reportData.date}</p>
              </div>
              <button type="button" onClick={() => setShowReport(false)}
                className="p-2 text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-full transition-all">
                <X size={20} />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-8 relative z-10">
              <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
                <p className="text-xs text-slate-500 uppercase font-bold tracking-wider mb-2">Total Firm AUM</p>
                <p className="text-xl font-bold">{statAUM}</p>
              </div>
              <div className="p-4 rounded-2xl bg-white/5 border border-white/10">
                <p className="text-xs text-slate-500 uppercase font-bold tracking-wider mb-2">{statLabel}</p>
                <p className="text-xl font-bold text-emerald-400">{statVal}</p>
              </div>
            </div>
            <div className="space-y-3 mb-8 relative z-10">
              <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                {isQ ? 'Quarterly Strategy Insights' : 'Risk Analysis'} 
              </h4>
              {items.map((item, idx) => (
                <div key={idx} className="flex items-start gap-3 text-sm text-slate-300">
                  <CheckCircle className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
            <button type="button" onClick={() => setShowReport(false)}
              className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 text-white font-bold rounded-xl transition-colors relative z-10">
              Close Report
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// CLIENTS
// ─────────────────────────────────────────────────────────────
const ClientsPage: React.FC = () => {
  const [clients,   setClients]   = useState<any[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [saving,    setSaving]    = useState(false);
  const [formErr,   setFormErr]   = useState('');
  const [form, setForm] = useState({ name:'', email:'', phone:'', investment_profile:'Moderate' });

  const loadClients = useCallback(() => {
    apiFetch('http://localhost:8000/api/clients')
      .then(r => r.json()).then(d => setClients(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);

  useEffect(() => { 
    loadClients(); 
    const handleUpdate = () => loadClients();
    window.addEventListener('db-updated', handleUpdate);
    return () => window.removeEventListener('db-updated', handleUpdate);
  }, [loadClients]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (saving) return;
    setSaving(true); setFormErr('');
    try {
      const res = await apiFetch('http://localhost:8000/api/register', {
        method: 'POST',
        body:   JSON.stringify(form),
      });
      if (res.ok) {
        setModalOpen(false);
        setForm({ name:'', email:'', phone:'', investment_profile:'Moderate' });
        setTimeout(loadClients, 300);
      } else if (res.status !== 401) {
        const d = await res.json().catch(() => ({}));
        setFormErr(d.detail || `Server error ${res.status}`);
      }
    } catch {
      setFormErr('Network error — is the backend running?');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-6 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md text-white">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold">Client Directory</h2>
          <p className="text-sm text-slate-400">PostgreSQL Database System</p>
        </div>
        <button type="button" onClick={() => { setModalOpen(true); setFormErr(''); }}
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-lg text-sm font-bold shadow-lg shadow-emerald-500/20 transition-all">
          + Add Client
        </button>
      </div>

      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-white/10 text-slate-400 text-sm">
            <th className="pb-4 font-medium">Name</th>
            <th className="pb-4 font-medium">Contact</th>
            <th className="pb-4 font-medium">Profile</th>
            <th className="pb-4 font-medium text-right">ID No.</th>
          </tr>
        </thead>
        <tbody>
          {clients.length === 0 && (
            <tr><td colSpan={4} className="py-10 text-center text-slate-500 text-sm">No clients yet — add one above.</td></tr>
          )}
          {clients.map(c => (
            <tr key={c.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
              <td className="py-4 font-medium">{c.name}</td>
              <td className="py-4 text-sm font-mono">
                <div className="text-slate-300">{c.email}</div>
                <div className="text-xs text-slate-500 mt-0.5">{c.phone}</div>
              </td>
              <td className="py-4"><span className="px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 text-xs font-bold">{c.investment_profile}</span></td>
              <td className="py-4 text-right font-mono text-slate-500">#000{c.id}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {modalOpen && (
        <Modal onClose={() => setModalOpen(false)}>
          <div className="bg-slate-900 border border-white/10 p-8 rounded-[2rem] shadow-2xl text-white">
            <h3 className="text-xl font-bold mb-6">Add New Client</h3>
            {formErr && <ErrBanner msg={formErr} onClose={() => setFormErr('')} />}
            <form onSubmit={handleSubmit} className="space-y-4">
              <input required type="text" placeholder="Full Name" value={form.name}
                onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 outline-none focus:border-emerald-500/50" />
              <input required type="email" placeholder="Email Address" value={form.email}
                onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 outline-none focus:border-emerald-500/50" />
              <input required type="tel" placeholder="Phone Number" value={form.phone}
                onChange={e => setForm(p => ({ ...p, phone: e.target.value }))}
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 outline-none focus:border-emerald-500/50" />
              <div className="relative">
                <select value={form.investment_profile}
                  onChange={e => setForm(p => ({ ...p, investment_profile: e.target.value }))}
                  className="w-full p-4 bg-slate-800 border border-white/10 rounded-xl text-white outline-none appearance-none">
                  <option value="Conservative">Conservative</option>
                  <option value="Moderate">Moderate</option>
                  <option value="Aggressive Growth">Aggressive Growth</option>
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
              </div>
              <div className="flex gap-3 pt-2">
                <button type="button" onClick={() => setModalOpen(false)}
                  className="flex-1 py-4 text-white border border-white/10 rounded-2xl hover:bg-white/5 font-bold transition-all">Cancel</button>
                <button type="submit" disabled={saving}
                  className="flex-1 py-4 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-60 text-white rounded-2xl font-bold shadow-lg shadow-emerald-500/20 transition-all">
                  {saving ? 'Saving…' : 'Save Client'}
                </button>
              </div>
            </form>
          </div>
        </Modal>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// PORTFOLIOS
// ─────────────────────────────────────────────────────────────
const PortfoliosPage: React.FC = () => {
  const [clients,    setClients]    = useState<any[]>([]);
  const [portfolios, setPortfolios] = useState<any[]>([]);
  const [modalOpen,  setModalOpen]  = useState(false);
  const [saving,     setSaving]     = useState(false);
  const [formErr,    setFormErr]    = useState('');
  const [form, setForm] = useState({ client_id:'', assets:'60% Stocks, 40% Bonds', value:'' });

  const loadData = useCallback(() => {
    apiFetch('http://localhost:8000/api/clients').then(r => r.json()).then(d => setClients(Array.isArray(d) ? d : [])).catch(() => {});
    apiFetch('http://localhost:8000/api/portfolios').then(r => r.json()).then(d => setPortfolios(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);

  useEffect(() => { 
    loadData(); 
    const handleUpdate = () => loadData();
    window.addEventListener('db-updated', handleUpdate);
    return () => window.removeEventListener('db-updated', handleUpdate);
  }, [loadData]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (saving) return;
    setSaving(true); setFormErr('');
    try {
      const res = await apiFetch('http://localhost:8000/api/portfolios', {
        method: 'POST',
        body:   JSON.stringify({ client_id: parseInt(form.client_id), assets: form.assets, value: parseFloat(form.value), risk_score: 5.0 }),
      });
      if (res.ok) {
        setModalOpen(false);
        setForm({ client_id:'', assets:'60% Stocks, 40% Bonds', value:'' });
        setTimeout(loadData, 300);
      } else if (res.status !== 401) {
        const d = await res.json().catch(() => ({}));
        setFormErr(d.detail || `Server error ${res.status}`);
      }
    } catch { setFormErr('Network error'); }
    finally { setSaving(false); }
  };

  return (
    <div className="space-y-6 text-white">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Active Portfolios</h2>
        <button type="button" onClick={() => { setModalOpen(true); setFormErr(''); }}
          className="px-4 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-lg text-sm font-bold shadow-lg shadow-emerald-500/20 transition-all">
          + Assign Portfolio
        </button>
      </div>
      {portfolios.filter(p => clients.some(c => c.id === p.client_id)).length === 0 && (
        <div className="p-10 rounded-2xl border border-white/10 bg-white/5 text-center text-slate-500 text-sm">
          No portfolios yet — assign one to a client.
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {portfolios.filter(p => clients.some(c => c.id === p.client_id)).map(p => {
          const owner = clients.find(c => c.id === p.client_id);
          return (
            <div key={p.id} className="p-6 rounded-2xl border border-white/10 bg-white/5 relative overflow-hidden hover:border-emerald-500/30 transition-all">
              <Shield className="absolute top-4 right-4 w-12 h-12 text-white/5" />
              <p className="text-emerald-400 font-bold text-sm uppercase mb-1 tracking-wider">{owner?.name}</p>
              <p className="text-3xl font-bold mb-2">${Number(p.value).toLocaleString()}</p>
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Briefcase className="w-4 h-4 text-emerald-500/50 flex-shrink-0" /> {p.assets}
              </div>
            </div>
          );
        })}
      </div>

      {modalOpen && (
        <Modal onClose={() => setModalOpen(false)}>
          <div className="bg-slate-900 border border-white/10 p-8 rounded-[2rem] shadow-2xl text-white">
            <h3 className="text-xl font-bold mb-6">Assign Portfolio</h3>
            {formErr && <ErrBanner msg={formErr} onClose={() => setFormErr('')} />}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <select required value={form.client_id}
                  onChange={e => setForm(p => ({ ...p, client_id: e.target.value }))}
                  className="w-full p-4 bg-slate-800 border border-white/10 rounded-xl text-white outline-none appearance-none">
                  <option value="">— Select Client —</option>
                  {clients.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
              </div>
              <input required type="number" placeholder="Portfolio Value ($)" value={form.value}
                onChange={e => setForm(p => ({ ...p, value: e.target.value }))}
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 outline-none focus:border-emerald-500/50" />
              <input required placeholder="Asset Allocation (e.g. 70% Stocks)" value={form.assets}
                onChange={e => setForm(p => ({ ...p, assets: e.target.value }))}
                className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 outline-none focus:border-emerald-500/50" />
              <div className="flex gap-3 pt-2">
                <button type="button" onClick={() => setModalOpen(false)}
                  className="flex-1 py-4 text-white border border-white/10 rounded-2xl hover:bg-white/5 font-bold transition-all">Cancel</button>
                <button type="submit" disabled={saving}
                  className="flex-1 py-4 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-60 text-white rounded-2xl font-bold shadow-lg shadow-emerald-500/20 transition-all">
                  {saving ? 'Saving…' : 'Save Portfolio'}
                </button>
              </div>
            </form>
          </div>
        </Modal>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// SERVICES
// ─────────────────────────────────────────────────────────────
const ServicesPage: React.FC = () => {
  const [services, setServices] = useState<any[]>([]);
  useEffect(() => {
    fetch('http://localhost:8000/api/services')   // no auth needed
      .then(r => r.json()).then(d => setServices(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-white">
      {services.length === 0 && (
        <div className="col-span-3 p-10 rounded-2xl border border-white/10 bg-white/5 text-center text-slate-500 text-sm">Loading services…</div>
      )}
      {services.map((s, i) => (
        <div key={i} className="p-6 rounded-2xl border border-white/10 bg-white/5 hover:border-emerald-500/30 transition-all">
          <Settings className="w-8 h-8 text-emerald-400 mb-4" />
          <h3 className="text-xl font-bold mb-2">{s.title}</h3>
          <p className="text-sm text-slate-400 mb-4">{s.description}</p>
          <p className="text-emerald-400 font-bold">{s.pricing}</p>
        </div>
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// MEETINGS
// ─────────────────────────────────────────────────────────────
const MeetingsPage: React.FC = () => {
  const [clients,  setClients]  = useState<any[]>([]);
  const [meetings, setMeetings] = useState<any[]>([]);
  const [saving,   setSaving]   = useState(false);
  const [formErr,  setFormErr]  = useState('');
  const [form, setForm] = useState({ client_id:'', datetime:'', advisor:'Admin' });

  const loadData = useCallback(() => {
    apiFetch('http://localhost:8000/api/clients').then(r => r.json()).then(d => setClients(Array.isArray(d) ? d : [])).catch(() => {});
    apiFetch('http://localhost:8000/api/meetings').then(r => r.json()).then(d => setMeetings(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);

  useEffect(() => { 
    loadData(); 
    const handleUpdate = () => loadData();
    window.addEventListener('db-updated', handleUpdate);
    return () => window.removeEventListener('db-updated', handleUpdate);
  }, [loadData]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (saving) return;
    setSaving(true); setFormErr('');
    try {
      const res = await apiFetch('http://localhost:8000/api/meeting', {
        method: 'POST',
        body:   JSON.stringify({ client_id: parseInt(form.client_id), datetime: form.datetime, advisor: form.advisor }),
      });
      if (res.ok) {
        setForm({ client_id:'', datetime:'', advisor:'Admin' });
        setTimeout(loadData, 300);
      } else if (res.status !== 401) {
        const d = await res.json().catch(() => ({}));
        setFormErr(d.detail || `Server error ${res.status}`);
      }
    } catch { setFormErr('Network error'); }
    finally { setSaving(false); }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 text-white">
      <div className="lg:col-span-1 p-6 rounded-3xl border border-white/10 bg-white/5 h-fit">
        <h2 className="text-xl font-bold mb-6 flex items-center gap-3">
          <Calendar className="text-emerald-400 w-5 h-5 flex-shrink-0" /> Book Consultation
        </h2>
        {formErr && <ErrBanner msg={formErr} onClose={() => setFormErr('')} />}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <select required value={form.client_id}
              onChange={e => setForm(p => ({ ...p, client_id: e.target.value }))}
              className="w-full p-4 bg-slate-800 border border-white/10 rounded-xl text-sm text-white outline-none appearance-none">
              <option value="">— Select Client —</option>
              {clients.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
          </div>
          <input required type="datetime-local" value={form.datetime}
            onChange={e => setForm(p => ({ ...p, datetime: e.target.value }))}
            className="w-full p-4 bg-white/5 border border-white/10 rounded-xl text-sm text-white outline-none" />
          <div className="relative">
            <select value={form.advisor}
              onChange={e => setForm(p => ({ ...p, advisor: e.target.value }))}
              className="w-full p-4 bg-slate-800 border border-white/10 rounded-xl text-sm text-white outline-none appearance-none">
              <option value="Admin">Admin (Senior)</option>
              <option value="Sowmya">Sowmya (Advisor)</option>
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
          </div>
          <button type="submit" disabled={saving}
            className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-60 text-white font-bold rounded-xl transition-all shadow-lg shadow-emerald-500/20 active:scale-[0.98]">
            {saving ? 'Booking…' : 'Confirm Meeting'}
          </button>
        </form>
      </div>

      <div className="lg:col-span-2 p-6 rounded-3xl border border-white/10 bg-white/5">
        <h2 className="text-xl font-bold mb-6">Advisor Schedule</h2>
        <div className="space-y-3">
          {meetings.filter(m => clients.some(c => c.id === m.client_id)).length === 0
            ? <p className="text-slate-500 text-center py-10 text-sm">No meetings booked yet.</p>
            : meetings.filter(m => clients.some(c => c.id === m.client_id)).map(m => {
              const name = clients.find(c => c.id === m.client_id)?.name;
              return (
                <div key={m.id} className="p-4 rounded-xl border border-white/5 bg-white/5 flex items-center justify-between hover:border-emerald-500/30 transition-all group">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-400 flex-shrink-0">
                      <Users className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="font-bold text-white group-hover:text-emerald-400 transition-colors">{name}</p>
                      <p className="text-xs text-slate-400 flex items-center gap-1 font-mono mt-0.5">
                        <Clock className="w-3 h-3 flex-shrink-0" />
                        {new Date(m.datetime).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <span className="text-xs px-2 py-1 bg-white/5 border border-white/10 rounded-md text-slate-400">{m.advisor}</span>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// ADMIN PROTECTED LAYOUT
// ─────────────────────────────────────────────────────────────
const AdminLayout: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
  return (
    <div className="flex min-h-screen font-sans bg-[#0f172a]">
      <Sidebar onLogout={onLogout} />
      <div className="flex-1 flex flex-col ml-64">
        <Header />
        <main className="flex-1 p-6 overflow-y-auto">
          <Routes>
            <Route path="/"           element={<DashboardHome />} />
            <Route path="/clients"    element={<ClientsPage />} />
            <Route path="/portfolios" element={<PortfoliosPage />} />
            <Route path="/services"   element={<ServicesPage />} />
            <Route path="/meetings"   element={<MeetingsPage />} />
          </Routes>
        </main>
        {/* The Chatbot acts as the Admin Assistant here */}
        <ChatWidget />
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────
// ROOT APP ROUTER
// ─────────────────────────────────────────────────────────────
export default function App() {
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [user,  setUser]  = useState<string | null>(localStorage.getItem('user'));

  const handleLogin = (t: string, u: string) => {
    localStorage.setItem('token', t); localStorage.setItem('user', u);
    setToken(t); setUser(u);
    // Hard refresh to admin panel
    window.location.href = '/admin';
  };

  const handleLogout = useCallback(() => {
    localStorage.removeItem('token'); localStorage.removeItem('user');
    setToken(null); setUser(null);
    // Hard refresh back to public page
    window.location.href = '/';
  }, []);

  useEffect(() => {
    _forceLogout = handleLogout;
    return () => { _forceLogout = null; };
  }, [handleLogout]);

  return (
    <Router>
      <Routes>
        {/* 1. PUBLIC LANDING PAGE (NO AUTH) */}
        <Route path="/" element={<PublicLandingPage />} />
        
        {/* 2. LOGIN PAGE */}
        <Route path="/login" element={
          token ? <Navigate to="/admin" /> : <LoginPage onLogin={handleLogin} />
        } />

        {/* 3. PROTECTED ADMIN DASHBOARD (REQUIRES AUTH) */}
        <Route path="/admin/*" element={
          token ? <AdminLayout onLogout={handleLogout} /> : <Navigate to="/login" />
        } />
      </Routes>
    </Router>
  );
}