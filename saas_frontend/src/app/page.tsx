"use client"; // Required for hooks like useEffect and useState

import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, Database, AlertCircle } from 'lucide-react';

export default function Home() {
  const [copperPrice, setCopperPrice] = useState<string>("Loading...");

  useEffect(() => {
    // Fetch live copper price from your FastAPI backend
    fetch("http://localhost:8000/api/latest-price/HG=F")
      .then(res => res.json())
      .then(data => {
        if (data.latest_close) {
          setCopperPrice(`$${data.latest_close.toFixed(2)}`);
        }
      })
      .catch(err => {
        console.error("API Error:", err);
        setCopperPrice("Offline");
      });
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Macro Forecaster Engine</h1>
          <p className="text-slate-500">Real-time Mining Intelligence</p>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">

        {/* Real Data Metric */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-sm font-medium text-slate-500">Live Copper (HG=F)</span>
            <Activity className="text-orange-500" size={18} />
          </div>
          <p className="text-2xl font-bold text-slate-900">{copperPrice}</p>
        </div>

        {/* Other mock metrics for now */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
          <span className="text-sm font-medium text-slate-500">USD/PEN</span>
          <p className="text-2xl font-bold text-slate-900">3.72</p>
        </div>

        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm opacity-50">
          <span className="text-sm font-medium text-slate-500">Forecast Status</span>
          <p className="text-xl font-bold text-slate-900 mt-1">Pending ML Data</p>
        </div>
      </div>

      {/* Main Visualization & Sidebars */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* The Live Grafana Integration */}
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col min-h-[450px]">
          <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
            <h3 className="text-slate-700 font-bold flex items-center gap-2 text-sm uppercase tracking-wider">
              <TrendingUp size={16} className="text-blue-600" />
              Market Intelligence: Copper Trends
            </h3>
            <span className="text-[10px] font-bold bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
              LIVE INFLUXDB FEED
            </span>
          </div>

          <div className="flex-1 w-full h-full bg-slate-50">
            <iframe
              src="http://localhost:3000/goto/efg87qbit10xsf?orgId=1&theme=light&kiosk"
              width="100%"
              height="100%"
              frameBorder="0"
              className="filter contrast-[1.05]"
            ></iframe>
          </div>
        </div>

        {/* Sidebar: System Health */}
        <div className="flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Database size={18} className="text-blue-500" />
              System Health
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">InfluxDB</span>
                <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs font-bold">ACTIVE</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">MLflow Models</span>
                <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded text-xs font-bold">ONLINE</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">FastAPI Backend</span>
                <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs font-bold">ACTIVE</span>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 p-6 rounded-2xl border border-orange-100">
            <h3 className="font-bold text-orange-900 mb-2 flex items-center gap-2 text-sm">
              <AlertCircle size={18} />
              Architecture Note
            </h3>
            <p className="text-orange-800 text-xs leading-relaxed">
              This dashboard successfully bridges data from a local Python FastAPI backend and a Dockerized Grafana instance running on port 3000.
            </p>
          </div>
        </div>

      </div>
    </main>
  );
}