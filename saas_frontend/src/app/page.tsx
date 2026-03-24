"use client";

import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Activity,
  Database,
  AlertCircle,
  Clock,
  ChevronRight,
  Zap,
  Layers,
  BarChart3,
  Target,
  ShieldAlert
} from 'lucide-react';

export default function Home() {
  const [silverPrice, setSilverPrice] = useState<string>("---");
  const [lastCloseDate, setLastCloseDate] = useState<string>("");
  const [systemUptime, setSystemUptime] = useState<string>("00:00:00");
  const [forecast, setForecast] = useState<any>(null);

  useEffect(() => {
    // Fetch live silver price
    fetch("http://localhost:8000/api/latest-price/SI=F")
      .then(res => res.json())
      .then(data => {
        if (data.latest_close) {
          setSilverPrice(`$${data.latest_close.toFixed(2)}`);
          if (data.timestamp) {
            setLastCloseDate(`(${new Date(data.timestamp).toLocaleDateString()})`);
          }
        }
      })
      .catch(err => {
        console.error("API Error:", err);
        setSilverPrice("OFFLINE");
      });

    // Fetch CNN-LSTM Forecast Data
    fetch("http://localhost:8000/api/forecast/SI=F")
      .then(res => res.json())
      .then(data => {
        if (!data.detail) {
          setForecast(data);
        }
      })
      .catch(console.error);

    // Simple uptime counter mock
    const startTime = Date.now();
    const interval = setInterval(() => {
      const seconds = Math.floor((Date.now() - startTime) / 1000);
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      setSystemUptime(`${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Determine Signal Colors
  const getSignalUi = (signal: string) => {
    if (!signal) return { bg: "bg-slate-500/20", glow: "shadow-slate-500/20", text: "text-slate-400" };
    if (signal.includes("BUY")) return { bg: "bg-green-500/20 border-green-500/30", glow: "shadow-green-500/30", text: "text-green-400" };
    if (signal.includes("WAIT")) return { bg: "bg-orange-500/20 border-orange-500/30", glow: "shadow-orange-500/30", text: "text-orange-400" };
    return { bg: "bg-blue-500/20 border-blue-500/30", glow: "shadow-blue-500/20", text: "text-blue-400" };
  };

  const signalUi = getSignalUi(forecast?.signal);

  // Funciones Dinámicas de Lógica de Negocio (Tri-Dimensional)
  const getTradingSignal = (f: any) => {
    if (!f) return { text: "---", color: "" };
    if (f.signal.includes("BUY")) return { text: "LONG POSITION", color: "text-emerald-400" };
    return { text: "STAY FLAT / SHORT", color: "text-red-400" };
  };

  const getProcurementSignal = (f: any) => {
    if (!f) return { text: "---", color: "" };
    // Si esperamos que la plata suba mañana, comprar el inventario físico HOY.
    if (f.expected_return_pct > 0.0) return { text: "ACCELERATE INVENTORY", color: "text-amber-400" };
    // Si esperamos que baje, esperar para comprar más barato.
    return { text: "DELAY PURCHASES", color: "text-blue-400" };
  };

  const getHedgingSignal = (f: any) => {
    if (!f) return { text: "---", color: "" };
    if (f.urgency === "HIGH" && f.expected_return_pct > 0.5) return { text: "HEDGE IMMINENT RISK", color: "text-red-500 font-bold" };
    if (f.expected_return_pct > 0.0) return { text: "MAINTAIN HEDGE", color: "text-amber-400" };
    return { text: "REDUCE HEDGING", color: "text-blue-400" };
  };

  // Generamos los milisegundos exactos garantizados de grafana para forzar el marco visual de tiempo (10 días atrás, 5 adelante)
  const nowMs = Date.now();
  const fromMs = nowMs - (10 * 24 * 60 * 60 * 1000);
  const toMs = nowMs + (5 * 24 * 60 * 60 * 1000);
  const grafanaUrl = `http://localhost:3000/public-dashboards/f6855c415a214e059452e91bc77b944c`;

  return (
    <main className="min-h-screen p-6 md:p-12 max-w-7xl mx-auto">

      {/* Top Navigation / Status Bar */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 gap-4">
        <div className="flex items-center gap-4">
          <div className="h-12 w-12 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <TrendingUp className="text-white" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">MACRO FORECASTER</h1>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Procurement Engine</span>
              <span className="h-1 w-1 rounded-full bg-slate-700"></span>
              <span className="text-[10px] text-indigo-500 font-bold uppercase tracking-widest">CNN-LSTM V2</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-white/5 border border-white/10 px-4 py-2 rounded-full backdrop-blur-md">
          <div className="badge-pulse">
            <span></span>
            <span></span>
          </div>
          <span className="text-xs font-bold text-slate-300">SYSTEM ACTIVE: {systemUptime}</span>
        </div>
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <div className="glass-card p-6">
          <div className="stat-label flex items-center justify-between uppercase">
            Last Close {lastCloseDate}
            <Zap size={14} className="text-indigo-400 animate-pulse" />
          </div>
          <div className="stat-value pt-2 glow-orange text-white">{silverPrice}</div>
          <div className="mt-2 text-[10px] font-bold text-green-400 flex items-center gap-1">
            <ChevronRight size={10} /> YFINANCE API
          </div>
        </div>

        {/* Dynamic Probability Gauge */}
        <div className="glass-card p-6 border-indigo-500/20 group hover:border-indigo-500/40 transition-all">
          <div className="stat-label flex justify-between">
            Upward Prob.
            <Target size={14} className="text-indigo-400" />
          </div>
          <div className="stat-value pt-2 text-indigo-300">
            {forecast ? `${(forecast.upward_probability * 100).toFixed(1)}%` : "Loading..."}
          </div>
          <div className="mt-2 w-full bg-slate-800 rounded-full h-1.5 mb-1 overflow-hidden">
            <div
              className="bg-indigo-500 h-1.5 rounded-full transition-all duration-1000 ease-out"
              style={{ width: forecast ? `${forecast.upward_probability * 100}%` : "0%" }}
            ></div>
          </div>
          <div className="text-[9px] font-bold text-indigo-400/50 uppercase">
            Uncertainty: {forecast ? `±${(forecast.probability_uncertainty * 100).toFixed(2)}%` : "--"}
          </div>
        </div>

        <div className="glass-card p-6">
          <div className="stat-label">Expected Return</div>
          <div className={`stat-value pt-2 ${forecast && forecast.expected_return_pct > 0 ? "text-green-400" : "text-orange-400"}`}>
            {forecast ? `${forecast.expected_return_pct > 0 ? '+' : ''}${forecast.expected_return_pct.toFixed(2)}%` : "Loading..."}
          </div>
          <div className="mt-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider">1D Forecast</div>
        </div>

        <div className="glass-card p-6">
          <div className="stat-label uppercase tracking-widest text-[9px] font-black text-slate-400">
            Projected Price {forecast?.forecast_date ? `(${new Date(forecast.forecast_date).toLocaleDateString()})` : ""}
          </div>
          <div className="stat-value pt-2 text-white">
            {forecast ? `$${parseFloat(forecast.expected_price).toFixed(2)}` : "Loading..."}
          </div>
          <div className="mt-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider">Inference Magnitude</div>
        </div>
      </div>

      {/* Main Intelligence Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Visualization Canvas */}
        <div className="lg:col-span-2 space-y-8">
          <div className="glass-card overflow-hidden flex flex-col min-h-[500px] group border-indigo-500/10">
            <div className="p-5 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <BarChart3 size={18} className="text-indigo-400" />
                <h3 className="text-xs font-black uppercase tracking-[0.15em] text-slate-300">
                  Market Trend Analysis: Silver (USD/OZ)
                </h3>
              </div>
              <div className="flex gap-2">
                <div className="h-2 w-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]"></div>
                <div className="h-2 w-2 rounded-full bg-blue-500 opacity-50"></div>
              </div>
            </div>

            <div className="flex-1 bg-[#0a0c12] relative overflow-hidden flex items-center justify-center border-b border-white/5">
              <iframe
                src={grafanaUrl}
                className="absolute inset-0 w-full h-full border-0 opacity-80 mix-blend-screen"
                title="Grafana Dashboard"
              ></iframe>
            </div>

            <div className="p-3 bg-white/[0.01] flex justify-between px-6">
              <span className="text-[9px] font-bold text-slate-500 tracking-widest uppercase">Target DB: InfluxDB / MarketData</span>
              <span className="text-[9px] font-bold text-indigo-400/80 tracking-widest uppercase flex items-center gap-1">
                <Clock size={8} /> LIVE
              </span>
            </div>
          </div>
        </div>

        {/* Intelligence Sidebar */}
        <div className="space-y-6">

          {/* Master Procurement Signal */}
          <div className={`border ${signalUi.bg} p-8 rounded-2xl relative overflow-hidden group shadow-xl ${signalUi.glow} transition-colors duration-500`}>
            <div className="absolute -right-4 -top-4 opacity-10">
              <ShieldAlert size={100} />
            </div>
            <h3 className={`text-xs font-black mb-1 uppercase tracking-[0.2em] ${signalUi.text} flex items-center gap-2`}>
              <Activity size={14} /> Smart Signal
            </h3>
            <div className="text-4xl font-black text-white py-2 tracking-tighter shadow-sm">
              {forecast ? forecast.signal : "LOADING..."}
            </div>

            <p className="text-slate-300/80 text-xs leading-relaxed font-semibold mt-2">
              Based on <span className="text-white">100 stochastic simulation passes</span> of Monte Carlo Dropout, the model
              classifies the decision with <b>{forecast?.urgency || "..."} Urgency</b>.
            </p>

            <div className="mt-5 border-t border-white/10 pt-4 flex flex-col gap-3">
              <div className="flex justify-between items-center text-[10px] uppercase tracking-widest font-black">
                <span className="text-slate-500">Asset Trading</span>
                <span className={`${getTradingSignal(forecast).color}`}>{getTradingSignal(forecast).text}</span>
              </div>
              <div className="flex justify-between items-center text-[10px] uppercase tracking-widest font-black">
                <span className="text-slate-500">Procurement</span>
                <span className={`${getProcurementSignal(forecast).color}`}>{getProcurementSignal(forecast).text}</span>
              </div>
              <div className="flex justify-between items-center text-[10px] uppercase tracking-widest font-black">
                <span className="text-slate-500">Futures Hedging</span>
                <span className={`${getHedgingSignal(forecast).color}`}>{getHedgingSignal(forecast).text}</span>
              </div>
            </div>
          </div>

          {/* Infrastructure Health */}
          <div className="glass-card p-6 flex flex-col justify-center">
            <h3 className="stat-label text-white mb-5 flex items-center gap-2">
              <Layers size={14} className="text-indigo-400" />
              Core Infrastructure
            </h3>

            <div className="space-y-4">
              <div className="flex justify-between items-center group">
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-slate-200">Data Lake</span>
                  <span className="text-[10px] text-slate-600 font-bold uppercase tracking-wider">InfluxDB @ v2.7</span>
                </div>
                <span className="bg-green-500/10 text-green-500 px-2 py-1 rounded text-[9px] font-black uppercase tracking-widest border border-green-500/20 group-hover:bg-green-500 group-hover:text-black transition-colors">Active</span>
              </div>

              <div className="flex justify-between items-center group">
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-slate-200">Inference Engine</span>
                  <span className="text-[10px] text-slate-600 font-bold uppercase tracking-wider">FastAPI Backend</span>
                </div>
                <span className="bg-green-500/10 text-green-500 px-2 py-1 rounded text-[9px] font-black uppercase tracking-widest border border-green-500/20 group-hover:bg-green-500 group-hover:text-black transition-colors">Online</span>
              </div>

              <div className="flex justify-between items-center group">
                <div className="flex flex-col">
                  <span className="text-xs font-bold text-slate-200">ML Flow</span>
                  <span className="text-[10px] text-slate-600 font-bold uppercase tracking-wider">Model Registry</span>
                </div>
                <span className="bg-indigo-500/10 text-indigo-400 px-2 py-1 rounded text-[9px] font-black uppercase tracking-widest border border-indigo-500/30 group-hover:bg-indigo-500 group-hover:text-white transition-colors">Tracking</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}