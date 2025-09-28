import React, { useState } from "react";
import MathComparison from "./components/MathComparison";
import LightGCNComparison from "./components/LightGCNComparison";
import DeepRLComparison from "./components/DeepRLComparison";
import LLMHGComparison from "./components/LLMHGComparison";
import HGNNComparison from "./components/HGNNComparison";
import SRGNNComparison from "./components/SRGNNComparison";
import GCHGNNComparison from "./components/GCHGNNComparison";
import DHCNComparison from "./components/DHCNComparison";
import SHAREComparison from "./components/SHAREComparison";
import BertModel from "./components/BertModel";
import LLMHGModel from "./components/LLMHGModel";
import LRMComplete from "./components/LRMComplete";
import MathDerivation from "./components/MathDerivation";
import Sidebar from "./components/Sidebar";

export default function App() {
  const [view, setView] = useState("MathComparison");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Add keyboard shortcut for sidebar (Cmd/Ctrl + K)
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSidebarOpen(prev => !prev);
      }
      // ESC to close sidebar
      if (e.key === 'Escape' && sidebarOpen) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sidebarOpen]);

  return (
    <div className="App">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        currentView={view}
        onNavigate={setView}
      />

      {/* Top Navigation Bar */}
      <nav className="bg-gray-800 p-4 text-white">
        <div className="max-w-7xl mx-auto flex items-center">
          {/* Menu Button */}
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-gray-700 transition-colors mr-4"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </button>
          
          {/* Title */}
          <h1 className="text-xl font-semibold">Paper Visualizations</h1>
          
          {/* Keyboard Shortcut Hint */}
          <div className="ml-4 text-xs text-gray-400">
            Press <kbd className="px-1 py-0.5 bg-gray-700 rounded">⌘K</kbd> to open menu
          </div>
          
          {/* Current Page */}
          <div className="ml-auto text-sm text-gray-300">
            {view.replace(/([A-Z])/g, ' $1').trim()}
          </div>
        </div>
      </nav>

      {/* Dynamic Rendering */}
      <main className="p-6">
        {view === "MathComparison" && <MathComparison />}
        {view === "LightGCNComparison" && <LightGCNComparison />}
        {view === "DeepRLComparison" && <DeepRLComparison />}
        {view === "LLMHGComparison" && <LLMHGComparison />}
        {view === "HGNNComparison" && <HGNNComparison />}
        {view === "SRGNNComparison" && <SRGNNComparison />}
        {view === "GCHGNNComparison" && <GCHGNNComparison />}
        {view === "DHCNComparison" && <DHCNComparison />}
        {view === "SHAREComparison" && <SHAREComparison />}
        {view === "BertModel" && <BertModel />}
        {view === "LLMHGModel" && <LLMHGModel />}
        {view === "LRMComplete" && <LRMComplete />}
        {view === "DiffusionModels" && <MathDerivation derivationId="diffusion-models" />}
      </main>
    </div>
  );
}