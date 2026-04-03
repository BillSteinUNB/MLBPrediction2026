/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { 
  LayoutDashboard, 
  LineChart, 
  Star, 
  Settings, 
  BadgeCheck, 
  Activity, 
  BarChart2, 
  AlertTriangle, 
  ArrowRight 
} from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-[#0b0f11] text-[#e0e3e7] font-sans selection:bg-[#abc7ff]/30">
      {/* TopAppBar */}
      <header className="fixed top-0 w-full z-50 bg-[#0b0f11] flex justify-between items-center px-6 h-16 border-b border-[#43474f]/15">
        <div className="flex items-center gap-4">
          <span className="text-xl font-black text-[#abc7ff] tracking-tighter font-heading">
            The Precision Analyst
          </span>
          <nav className="hidden md:flex items-center ml-8 gap-6 h-full">
            <a href="#" className="text-[#abc7ff] border-b-2 border-[#abc7ff] h-16 flex items-center font-heading font-bold tracking-tight">
              Dashboard
            </a>
            <a href="#" className="text-[#c4c6d1] hover:text-[#abc7ff] transition-colors duration-200 h-16 flex items-center font-heading font-bold tracking-tight">
              Analytics
            </a>
            <a href="#" className="text-[#c4c6d1] hover:text-[#abc7ff] transition-colors duration-200 h-16 flex items-center font-heading font-bold tracking-tight">
              Market
            </a>
          </nav>
        </div>
      </header>

      {/* SideNavBar */}
      <aside className="fixed left-0 top-16 h-[calc(100vh-64px)] w-64 border-r border-[#43474f]/15 bg-[#0b0f11] flex-col py-6 hidden md:flex">
        <div className="px-4 mb-8">
          <div className="flex items-center gap-3 p-3 bg-[#1c2023]/50 rounded-xl">
            <div className="w-10 h-10 rounded-full bg-[#002d62] flex items-center justify-center text-[#abc7ff] font-bold">
              JD
            </div>
            <div>
              <p className="text-[#e0e3e7] font-bold text-sm">Analyst Pro</p>
              <p className="text-[#c4c6d1] text-[10px] font-bold uppercase tracking-widest">Personal Edition</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 flex flex-col">
          <a href="#" className="flex items-center gap-3 px-4 py-3 text-[#abc7ff] bg-[#1c2023] rounded-r-lg border-l-4 border-[#abc7ff] text-sm font-medium uppercase tracking-wider hover:translate-x-1 transition-transform duration-300">
            <LayoutDashboard size={18} />
            <span>Dashboard</span>
          </a>
          <a href="#" className="flex items-center gap-3 px-4 py-3 text-[#c4c6d1] hover:bg-[#1c2023]/50 text-sm font-medium uppercase tracking-wider hover:translate-x-1 transition-transform duration-300">
            <LineChart size={18} />
            <span>Model Insights</span>
          </a>
          <a href="#" className="flex items-center gap-3 px-4 py-3 text-[#c4c6d1] hover:bg-[#1c2023]/50 text-sm font-medium uppercase tracking-wider hover:translate-x-1 transition-transform duration-300">
            <Star size={18} className="fill-current" />
            <span>POTD</span>
          </a>
        </nav>
      </aside>

      {/* Main Content Canvas */}
      <main className="md:ml-64 pt-24 pb-24 md:pb-12 px-6 lg:px-12">
        
        {/* Hero Section: POTD */}
        <section className="mb-14 relative overflow-hidden rounded-[2rem] bg-[#0b0f11] border border-[#43474f]/20 shadow-2xl">
          <div className="absolute inset-0 opacity-10 grayscale">
            <img 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuAku_vCC29FcU7Kn6OklR-6EmAVe_zWHxVnPgoc063uG348NpQu9N0U48isaepMhYTcUAvkVJxECiGCQcIPvnu5nfGp31KDBmh0CznG_CCCB8rpeO1l-hCaJOeT3lGNMkcM_e-2fDU04LC62t8k_r3LkQwIEcpeQyPcTK_rJI4Ly_YPH-q6lWC7BXZ7Anq-gXPVHpztX-QT894UMw8GlfoVPjI7AO1-JkwQZLSy-YpffuvfAea_5rpxgHcz5_Wev3B86SVsDWVXIGM" 
              alt="Stadium background" 
              className="w-full h-full object-cover"
              referrerPolicy="no-referrer"
            />
          </div>
          <div className="absolute inset-0 bg-gradient-to-r from-[#0b0f11] via-[#0b0f11]/80 to-transparent"></div>
          
          <div className="relative z-10 p-8 lg:p-16 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-12">
            <div className="flex-1">
              <div className="flex flex-wrap items-center gap-4 mb-8">
                <span className="bg-[#abc7ff] text-[#032f64] px-4 py-1.5 rounded-full text-[11px] font-black uppercase tracking-[0.2em]">
                  Play of the Day
                </span>
                <span className="text-[#c4c6d1] text-[11px] font-bold uppercase tracking-widest">
                  MLB • SEP 14, 2023
                </span>
              </div>
              
              <h2 className="text-5xl lg:text-7xl font-black text-[#e0e3e7] tracking-tighter mb-8 leading-tight font-heading">
                NEW YORK<br/>YANKEES ML
              </h2>
              
              <div className="flex flex-wrap items-center gap-8 lg:gap-10">
                <div className="flex flex-col">
                  <span className="text-[#c4c6d1] text-[11px] font-bold uppercase tracking-widest mb-2">Matchup</span>
                  <span className="text-2xl font-bold text-[#e0e3e7]">NYY vs BOS</span>
                </div>
                <div className="w-px h-12 bg-[#43474f]/40 hidden sm:block"></div>
                <div className="flex flex-col">
                  <span className="text-[#c4c6d1] text-[11px] font-bold uppercase tracking-widest mb-2">Odds</span>
                  <span className="text-2xl font-bold text-[#58e07f]">-142</span>
                </div>
                <div className="w-px h-12 bg-[#43474f]/40 hidden sm:block"></div>
                <div className="flex flex-col">
                  <span className="text-[#c4c6d1] text-[11px] font-bold uppercase tracking-widest mb-2">Confidence Score</span>
                  <div className="flex items-center gap-3">
                    <span className="text-3xl font-black text-[#abc7ff]">88.4%</span>
                    <BadgeCheck className="text-[#58e07f] fill-[#58e07f]/20" size={32} />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="glass-panel p-8 rounded-[2rem] border border-white/5 w-full lg:w-80 shrink-0">
              <p className="text-[#c4c6d1] text-[11px] font-bold uppercase tracking-widest mb-6">Model Rationale</p>
              <p className="text-[15px] text-[#e0e3e7] leading-relaxed mb-8 font-medium">
                Gerrit Cole's spin rate against the Red Sox middle-order creates a significant EV edge at current market pricing levels.
              </p>
              <button className="w-full p-4 bg-[#abc7ff]/10 hover:bg-[#abc7ff]/20 transition-colors rounded-xl border border-[#abc7ff]/20 flex items-center justify-center">
                <span className="text-[#abc7ff] font-bold text-xs uppercase tracking-widest">Active Position</span>
              </button>
            </div>
          </div>
        </section>

        {/* Model Discrepancies */}
        <section className="mb-12">
          <div className="flex justify-between items-end mb-8">
            <div>
              <h3 className="text-2xl font-extrabold text-[#e0e3e7] tracking-tight font-heading">Model Discrepancies</h3>
              <p className="text-[#c4c6d1] text-sm mt-1">Direct comparison of internal proprietary model outputs.</p>
            </div>
            <button className="hidden sm:flex text-[#c4c6d1] text-xs font-bold uppercase tracking-widest hover:text-[#abc7ff] transition-colors items-center gap-2">
              Grid Data <ArrowRight size={16} />
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            
            {/* Model 1 */}
            <div className="bg-[#1c2023]/40 rounded-2xl p-6 border border-[#43474f]/30 hover:border-[#abc7ff]/30 transition-colors">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <span className="text-[10px] font-bold text-[#abc7ff] uppercase tracking-widest">Alpha-Stat</span>
                  <h4 className="text-lg font-bold text-[#e0e3e7] font-heading mt-1">Volume Optimized</h4>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#002d62]/40 flex items-center justify-center">
                  <Activity className="text-[#abc7ff]" size={20} />
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">ATL vs PHI</span>
                  <span className="text-xs font-bold text-[#58e07f]">ATL -1.5</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">LAD vs SD</span>
                  <span className="text-xs font-bold text-[#e0e3e7]">U 8.5</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">HOU vs TEX</span>
                  <span className="text-xs font-bold text-[#58e07f]">HOU ML</span>
                </div>
              </div>
              <div className="mt-6 pt-5 border-t border-[#43474f]/20 flex justify-between items-center">
                <span className="text-[10px] font-bold text-[#c4c6d1] uppercase tracking-widest">Performance</span>
                <span className="text-xs font-black text-[#58e07f]">+12.4U</span>
              </div>
            </div>

            {/* Model 2 */}
            <div className="bg-[#1c2023]/40 rounded-2xl p-6 border border-[#43474f]/30 hover:border-[#58e07f]/30 transition-colors">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <span className="text-[10px] font-bold text-[#58e07f] uppercase tracking-widest">Swing-Data</span>
                  <h4 className="text-lg font-bold text-[#e0e3e7] font-heading mt-1">Sabermetric Core</h4>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#03aa51]/20 flex items-center justify-center">
                  <BarChart2 className="text-[#58e07f]" size={20} />
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">ATL vs PHI</span>
                  <span className="text-xs font-bold text-[#e0e3e7]">PHI +140</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">LAD vs SD</span>
                  <span className="text-xs font-bold text-[#58e07f]">SD ML</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">HOU vs TEX</span>
                  <span className="text-xs font-bold text-[#58e07f]">HOU -110</span>
                </div>
              </div>
              <div className="mt-6 pt-5 border-t border-[#43474f]/20 flex justify-between items-center">
                <span className="text-[10px] font-bold text-[#c4c6d1] uppercase tracking-widest">Performance</span>
                <span className="text-xs font-black text-[#58e07f]">+8.1U</span>
              </div>
            </div>

            {/* Model 3 */}
            <div className="bg-[#1c2023]/40 rounded-2xl p-6 border border-[#43474f]/30 hover:border-[#ffb2b7]/30 transition-colors">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <span className="text-[10px] font-bold text-[#ffb2b7] uppercase tracking-widest">Bullpen-IQ</span>
                  <h4 className="text-lg font-bold text-[#e0e3e7] font-heading mt-1">Late Inning Focus</h4>
                </div>
                <div className="w-10 h-10 rounded-lg bg-[#640019]/40 flex items-center justify-center">
                  <AlertTriangle className="text-[#ffb2b7]" size={20} />
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">ATL vs PHI</span>
                  <span className="text-xs font-bold text-[#58e07f]">PHI F5 ML</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">LAD vs SD</span>
                  <span className="text-xs font-bold text-[#e0e3e7]">O 7.5</span>
                </div>
                <div className="flex justify-between items-center bg-[#181c1f]/80 p-3 rounded-xl">
                  <span className="text-xs text-[#c4c6d1]">HOU vs TEX</span>
                  <span className="text-xs font-bold text-[#c4c6d1]">No Action</span>
                </div>
              </div>
              <div className="mt-6 pt-5 border-t border-[#43474f]/20 flex justify-between items-center">
                <span className="text-[10px] font-bold text-[#c4c6d1] uppercase tracking-widest">Performance</span>
                <span className="text-xs font-black text-[#ffb4ab]">-2.2U</span>
              </div>
            </div>

          </div>
        </section>
      </main>

      {/* BottomNavBar (Mobile) */}
      <nav className="md:hidden fixed bottom-0 w-full h-16 bg-[#0b0f11] flex justify-around items-center z-50 border-t border-[#43474f]/15 pb-safe">
        <button className="flex flex-col items-center gap-1 text-[#abc7ff]">
          <LayoutDashboard size={20} />
          <span className="text-[10px] font-bold uppercase">Home</span>
        </button>
        <button className="flex flex-col items-center gap-1 text-[#c4c6d1]">
          <LineChart size={20} />
          <span className="text-[10px] font-bold uppercase">Models</span>
        </button>
        <button className="flex flex-col items-center gap-1 text-[#c4c6d1]">
          <Star size={20} className="fill-current" />
          <span className="text-[10px] font-bold uppercase">POTD</span>
        </button>
        <button className="flex flex-col items-center gap-1 text-[#c4c6d1]">
          <Settings size={20} />
          <span className="text-[10px] font-bold uppercase">More</span>
        </button>
      </nav>
    </div>
  );
}
