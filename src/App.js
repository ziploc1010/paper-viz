import React, { useState } from "react";
import GenericComparison from "./components/GenericComparison";
import BertModel from "./components/BertModel";
import LLMHGModel from "./components/LLMHGModel";
import LRMComplete from "./components/LRMComplete";
import QwenModel from "./components/QwenModel";
import DeepseekV3Model from "./components/DeepseekV3Model";
import Llama32Model from "./components/Llama32Model";
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
        {view === "MathComparison" && <GenericComparison comparisonId="math" />}
        {view === "LightGCNComparison" && <GenericComparison comparisonId="lightgcn" />}
        {view === "DeepRLComparison" && <GenericComparison comparisonId="deeprl" />}
        {view === "LLMHGComparison" && <GenericComparison comparisonId="llmhg" />}
        {view === "HGNNComparison" && <GenericComparison comparisonId="hgnn" />}
        {view === "SRGNNComparison" && <GenericComparison comparisonId="srgnn" />}
        {view === "GCHGNNComparison" && <GenericComparison comparisonId="gchgnn" />}
        {view === "DHCNComparison" && <GenericComparison comparisonId="dhcn" />}
        {view === "SHAREComparison" && <GenericComparison comparisonId="share" />}
        {view === "BertModel" && <BertModel />}
        {view === "LLMHGModel" && <LLMHGModel />}
        {view === "LRMComplete" && <LRMComplete />}
        {view === "QwenModel" && <QwenModel />}
        {view === "DeepseekV3Model" && <DeepseekV3Model />}
        {view === "Llama32Model" && <Llama32Model />}
        {view === "DiffusionModels" && <MathDerivation derivationId="diffusion-models" />}
        {view === "SchrodingerEquation" && <MathDerivation derivationId="schrodinger-equation" />}
      </main>
    </div>
  );
}