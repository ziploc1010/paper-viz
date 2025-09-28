import React, { useState } from 'react';

const Sidebar = ({ isOpen, onClose, currentView, onNavigate }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState(new Set(['Comparisons', 'Model Architectures', 'Mathematical Proofs']));

  // Icons for different categories
  const categoryIcons = {
    'Comparisons': (
      <svg className="w-5 h-5" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
      </svg>
    ),
    'Model Architectures': (
      <svg className="w-5 h-5" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
      </svg>
    ),
    'Mathematical Proofs': (
      <svg className="w-5 h-5" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path>
      </svg>
    ),
    'default': (
      <svg className="w-5 h-5" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
      </svg>
    )
  };

  const pages = [
    // Comparisons
    { id: 'MathComparison', name: 'Math Comparison', category: 'Comparisons' },
    { id: 'LightGCNComparison', name: 'Light GCN Comparison', category: 'Comparisons' },
    { id: 'DeepRLComparison', name: 'Deep RL Comparison', category: 'Comparisons' },
    { id: 'LLMHGComparison', name: 'LLMHG Comparison', category: 'Comparisons' },
    { id: 'HGNNComparison', name: 'HGNN Comparison', category: 'Comparisons' },
    { id: 'SRGNNComparison', name: 'SR-GNN Comparison', category: 'Comparisons' },
    { id: 'GCHGNNComparison', name: 'GC-HGNN Comparison', category: 'Comparisons' },
    { id: 'DHCNComparison', name: 'DHCN Comparison', category: 'Comparisons' },
    { id: 'SHAREComparison', name: 'SHARE Comparison', category: 'Comparisons' },
    
    // Model Architectures
    { id: 'BertModel', name: 'BERT Model', category: 'Model Architectures' },
    { id: 'LLMHGModel', name: 'LLMHG Model', category: 'Model Architectures' },
    { id: 'LRMComplete', name: 'LRM Complete Hierarchy', category: 'Model Architectures' },
    
    // Mathematical Proofs
    { id: 'DiffusionModels', name: 'Diffusion Models Derivation', category: 'Mathematical Proofs' },
    
    // Placeholder for future pages
    // You can add more categories like:
    // - Training Techniques
    // - Optimization Methods
    // - Evaluation Metrics
    // - Dataset Visualizations
    // - Algorithm Comparisons
    // - etc.
  ];

  // Filter pages based on search query
  const filteredPages = pages.filter(page => 
    page.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    page.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Group filtered pages by category
  const groupedPages = filteredPages.reduce((acc, page) => {
    if (!acc[page.category]) {
      acc[page.category] = [];
    }
    acc[page.category].push(page);
    return acc;
  }, {});

  const toggleCategory = (category) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full w-80 max-w-[85vw] bg-white shadow-xl z-50 transform transition-transform duration-300 ease-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-xl font-semibold text-gray-800">Paper Visualizations</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <svg
              className="w-5 h-5 text-gray-600"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>

        {/* Search Bar (for future use when you have 100+ pages) */}
        <div className="p-4 border-b">
          <div className="relative">
            <input
              type="text"
              placeholder="Search visualizations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-2 pl-10 pr-4 text-gray-700 bg-gray-100 rounded-lg focus:outline-none focus:bg-white focus:ring-2 focus:ring-blue-400"
            />
            <svg
              className="absolute left-3 top-2.5 w-5 h-5 text-gray-400"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="overflow-y-auto h-[calc(100vh-8rem)] pb-20">
          {Object.keys(groupedPages).length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              No visualizations found
            </div>
          ) : (
            Object.entries(groupedPages).map(([category, pages]) => (
              <div key={category} className="border-b border-gray-200 last:border-0">
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-gray-500">
                      {categoryIcons[category] || categoryIcons.default}
                    </span>
                    <h3 className="text-sm font-semibold text-gray-700">
                      {category}
                    </h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded-full">
                      {pages.length}
                    </span>
                    <svg
                      className={`w-4 h-4 text-gray-400 transform transition-transform ${
                        expandedCategories.has(category) ? 'rotate-90' : ''
                      }`}
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path d="M9 5l7 7-7 7"></path>
                    </svg>
                  </div>
                </button>
                {expandedCategories.has(category) && (
                  <div className="px-4 pb-2">
                    {pages.map((page) => (
                      <button
                        key={page.id}
                        onClick={() => {
                          onNavigate(page.id);
                          onClose();
                          setSearchQuery('');
                        }}
                        className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                          currentView === page.id
                            ? 'bg-blue-100 text-blue-700 font-medium'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        {page.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
};

export default Sidebar;