import React, { useState, useEffect, useRef } from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import EquationCanvas from './EquationCanvas';
import bertQuizData from '../data/bertmodel_updated.json';

export default function BertModel() {
  const [quizData, setQuizData] = useState(null);
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [activeTab, setActiveTab] = useState('equations');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPanPosition, setStartPanPosition] = useState({ x: 0, y: 0 });
  const [splitPosition, setSplitPosition] = useState(50); // percentage
  const [isDraggingDivider, setIsDraggingDivider] = useState(false);
  const [touchStartDistance, setTouchStartDistance] = useState(0);
  const [touchStartZoom, setTouchStartZoom] = useState(1);
  const [activePane, setActivePane] = useState(null); // 'top' or 'bottom'
  const [visibility, setVisibility] = useState({});
  const [userCode, setUserCode] = useState({});
  const [drawingData, setDrawingData] = useState({});
  const svgRef = useRef(null);
  const svgContainerRef = useRef(null);
  const containerRef = useRef(null);
  const zoomLevelRef = useRef(zoomLevel);
  const panPositionRef = useRef(panPosition);
  
  // Disable browser zoom on mount for this component
  useEffect(() => {
    // Add meta viewport tag to prevent zoom on mobile/touch devices
    const metaViewport = document.querySelector('meta[name="viewport"]');
    const originalContent = metaViewport?.getAttribute('content');
    
    if (metaViewport) {
      metaViewport.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no');
    }
    
    // Add CSS to prevent zoom
    const style = document.createElement('style');
    style.innerHTML = `
      body.bert-model-active {
        touch-action: none !important;
        -webkit-user-select: none;
        user-select: none;
      }
    `;
    document.head.appendChild(style);
    document.body.classList.add('bert-model-active');
    
    return () => {
      // Restore original viewport
      if (metaViewport && originalContent) {
        metaViewport.setAttribute('content', originalContent);
      }
      document.body.classList.remove('bert-model-active');
      style.remove();
    };
  }, []);
  
  // Keep refs in sync with state
  useEffect(() => {
    zoomLevelRef.current = zoomLevel;
  }, [zoomLevel]);
  
  useEffect(() => {
    panPositionRef.current = panPosition;
  }, [panPosition]);

  // Custom highlight function for Python code
  const highlightCode = (code) => {
    try {
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code;
    }
  };

  // Toggle visibility for equations and code
  const toggleVisibility = (key) => {
    setVisibility(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Save drawing data
  const handleSaveDrawing = (canvasId, data) => {
    setDrawingData(prev => ({ ...prev, [canvasId]: data }));
  };

  // Load the BERT model data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Use the imported JSON data directly
        setQuizData(bertQuizData);
        setLoading(false);
      } catch (err) {
        console.error('Error loading data:', err);
        setError('Failed to load BERT model data.');
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Set up click handlers for SVG components
  useEffect(() => {
    if (!quizData || !svgRef.current) return;

    // Add style element to the document head for SVG styles
    const styleId = 'bert-model-styles';
    let styleElement = document.getElementById(styleId);
    if (!styleElement) {
      styleElement = document.createElement('style');
      styleElement.id = styleId;
      styleElement.innerHTML = `
        .component-selected rect,
        .component-selected circle,
        .component-selected ellipse,
        .component-selected path {
          stroke: red !important;
          stroke-width: 4px !important;
          filter: drop-shadow(0 0 10px rgba(255,0,0,0.6));
          animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.7; }
          100% { opacity: 1; }
        }
      `;
      document.head.appendChild(styleElement);
    }

    const handleComponentClick = (e) => {
      // Look for the closest g element with class 'component'
      const target = e.target.closest('g.component[data-component]');
      if (!target) return;
      
      e.preventDefault();
      e.stopPropagation();
      
      const componentId = target.getAttribute('data-component');
      
      // Inline selectComponent logic here
      
      // Remove previous selection
      if (svgRef.current) {
        svgRef.current.querySelectorAll('.component-selected').forEach(el => {
          el.classList.remove('component-selected');
        });
      }

      // Add new selection
      if (svgRef.current) {
        const component = svgRef.current.querySelector(`g.component[data-component="${componentId}"]`);
        if (component) {
          component.classList.add('component-selected');
        }
      }

      setSelectedComponent(componentId);
      setActiveTab('equations'); // Reset to equations tab
    };

    // Add click listeners to all components
    const svg = svgRef.current;
    const components = svg.querySelectorAll('g.component[data-component]');
    
    components.forEach(component => {
      component.addEventListener('click', handleComponentClick);
    });

    // Cleanup
    return () => {
      components.forEach(component => {
        component.removeEventListener('click', handleComponentClick);
      });
    };
  }, [quizData]);

  // Add mouse move handler to window for divider dragging
  useEffect(() => {
    const handleGlobalMouseMove = (e) => {
      if (!isDraggingDivider || !containerRef.current) return;
      
      const containerRect = containerRef.current.getBoundingClientRect();
      const newPosition = ((e.clientY - containerRect.top) / containerRect.height) * 100;
      
      // Limit the split position between 20% and 80%
      setSplitPosition(Math.min(Math.max(newPosition, 20), 80));
    };

    const handleGlobalMouseUp = () => {
      setIsDraggingDivider(false);
    };

    if (isDraggingDivider) {
      document.body.classList.add('dragging-divider');
      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
      return () => {
        document.body.classList.remove('dragging-divider');
        window.removeEventListener('mousemove', handleGlobalMouseMove);
        window.removeEventListener('mouseup', handleGlobalMouseUp);
      };
    }
  }, [isDraggingDivider]);

  // Add wheel event listener - focusing on Mac trackpad support
  useEffect(() => {
    const container = svgContainerRef.current;
    const topPanel = container?.closest('[style*="height"]'); // Get the top panel div
    if (!container) return;


    const handleWheelEvent = (e) => {
      // Always prevent default to stop browser zoom
      e.preventDefault();
      e.stopPropagation();
      
      // On macOS with trackpad:
      // - Pinch gesture: ctrlKey is true
      // - Two-finger scroll: ctrlKey is false
      const isMac = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
      
      if (e.ctrlKey || e.metaKey) {
        // Pinch-to-zoom with zoom-to-cursor
        const rect = container.getBoundingClientRect();
        const delta = e.deltaY;
        const zoomSpeed = 0.005; // Smooth zoom speed
        const scaleFactor = 1 - (delta * zoomSpeed);
        const currentZoom = zoomLevelRef.current;
        const newZoom = Math.min(Math.max(currentZoom * scaleFactor, 0.3), 5);
        
        // Calculate cursor position relative to container
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;
        
        // Calculate the point under cursor in the transformed space
        const currentPan = panPositionRef.current;
        const pointX = (cursorX - currentPan.x) / currentZoom;
        const pointY = (cursorY - currentPan.y) / currentZoom;
        
        // Calculate new pan position to keep the point under cursor
        const newPanX = cursorX - pointX * newZoom;
        const newPanY = cursorY - pointY * newZoom;
        
        setZoomLevel(newZoom);
        setPanPosition({ x: newPanX, y: newPanY });
      } else {
        // Two-finger scroll/pan with smoother movement
        const currentPan = panPositionRef.current;
        const sensitivity = 0.7; // Reduced for smoother scrolling
        const newPan = {
          x: currentPan.x - (e.deltaX * sensitivity),
          y: currentPan.y - (e.deltaY * sensitivity)
        };
        
        setPanPosition(newPan);
      }
      
      return false;
    };

    const handleMouseEnter = () => {
      setActivePane('top');
    };
    
    const handleMouseLeave = () => {
      setActivePane(null);
    };

    // Prevent browser zoom on the entire document when in top panel
    const handleDocumentWheel = (e) => {
      if (activePane === 'top' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        return false;
      }
    };

    // Add event listeners with capture phase to intercept before browser
    container.addEventListener('wheel', handleWheelEvent, { passive: false, capture: true });
    container.addEventListener('mouseenter', handleMouseEnter);
    container.addEventListener('mouseleave', handleMouseLeave);
    
    // Also add to the top panel if we found it
    if (topPanel) {
      topPanel.addEventListener('wheel', handleWheelEvent, { passive: false, capture: true });
    }
    
    // Add document-level handler to prevent browser zoom
    document.addEventListener('wheel', handleDocumentWheel, { passive: false, capture: true });
    
    
    return () => {
      container.removeEventListener('wheel', handleWheelEvent, { capture: true });
      container.removeEventListener('mouseenter', handleMouseEnter);
      container.removeEventListener('mouseleave', handleMouseLeave);
      if (topPanel) {
        topPanel.removeEventListener('wheel', handleWheelEvent, { capture: true });
      }
      document.removeEventListener('wheel', handleDocumentWheel, { capture: true });
    };
  }, [activePane]);

  // Component selection is now handled inside the useEffect

  // Zoom handlers
  const handleZoomIn = () => {
    const container = svgContainerRef.current;
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const currentZoom = zoomLevel;
    const newZoom = Math.min(currentZoom * 1.2, 5);
    
    // Calculate the point at center in the transformed space
    const currentPan = panPosition;
    const pointX = (centerX - currentPan.x) / currentZoom;
    const pointY = (centerY - currentPan.y) / currentZoom;
    
    // Calculate new pan to keep center point stable
    const newPanX = centerX - pointX * newZoom;
    const newPanY = centerY - pointY * newZoom;
    
    setZoomLevel(newZoom);
    setPanPosition({ x: newPanX, y: newPanY });
  };

  const handleZoomOut = () => {
    const container = svgContainerRef.current;
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const currentZoom = zoomLevel;
    const newZoom = Math.max(currentZoom / 1.2, 0.3);
    
    // Calculate the point at center in the transformed space
    const currentPan = panPosition;
    const pointX = (centerX - currentPan.x) / currentZoom;
    const pointY = (centerY - currentPan.y) / currentZoom;
    
    // Calculate new pan to keep center point stable
    const newPanX = centerX - pointX * newZoom;
    const newPanY = centerY - pointY * newZoom;
    
    setZoomLevel(newZoom);
    setPanPosition({ x: newPanX, y: newPanY });
  };

  const handleZoomReset = () => {
    setZoomLevel(1);
    setPanPosition({ x: 0, y: 0 });
  };

  // Pan handlers
  const handleMouseDown = (e) => {
    if (e.target.closest('.zoom-controls')) return;
    setIsPanning(true);
    setStartPanPosition({ x: e.clientX - panPosition.x, y: e.clientY - panPosition.y });
  };

  const handleMouseMove = (e) => {
    if (!isPanning) return;
    setPanPosition({
      x: e.clientX - startPanPosition.x,
      y: e.clientY - startPanPosition.y
    });
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsDraggingDivider(false);
  };

  // Touch handlers for mobile gestures
  const getTouchDistance = (touches) => {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const handleTouchStart = (e) => {
    if (e.target.closest('.zoom-controls')) return;
    
    if (e.touches.length === 1) {
      // Single touch - pan
      setIsPanning(true);
      setStartPanPosition({ 
        x: e.touches[0].clientX - panPosition.x, 
        y: e.touches[0].clientY - panPosition.y 
      });
    } else if (e.touches.length === 2) {
      // Two fingers - pinch to zoom
      const distance = getTouchDistance(e.touches);
      setTouchStartDistance(distance);
      setTouchStartZoom(zoomLevel);
    }
  };

  const handleTouchMove = (e) => {
    e.preventDefault(); // Prevent scrolling
    
    if (e.touches.length === 1 && isPanning) {
      // Single touch - pan
      setPanPosition({
        x: e.touches[0].clientX - startPanPosition.x,
        y: e.touches[0].clientY - startPanPosition.y
      });
    } else if (e.touches.length === 2 && touchStartDistance > 0) {
      // Two fingers - pinch to zoom
      const currentDistance = getTouchDistance(e.touches);
      const scale = currentDistance / touchStartDistance;
      const newZoom = touchStartZoom * scale;
      setZoomLevel(Math.min(Math.max(newZoom, 0.5), 3));
    }
  };

  const handleTouchEnd = () => {
    setIsPanning(false);
    setTouchStartDistance(0);
  };


  // Divider drag handlers
  const handleDividerMouseDown = (e) => {
    e.preventDefault();
    setIsDraggingDivider(true);
  };

  const getComponentData = () => {
    if (!selectedComponent || !quizData) return null;
    
    const quiz = quizData.componentQuizzes?.[selectedComponent];
    const explanation = quizData.componentExplanations?.[selectedComponent];
    
    if (!quiz || !explanation) return null;
    
    return {
      title: quiz.title || explanation.title || selectedComponent,
      explanation: explanation.explanation || '',
      equations: quiz.equations || [],
      initialization: quiz.initialization || {},
      forward: quiz.forward || {}
    };
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-lg p-6 text-center">
          <h1 className="text-3xl font-bold mb-4">Loading BERT Model...</h1>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-red-50 rounded-lg shadow-lg p-6 text-center">
          <h1 className="text-3xl font-bold mb-4 text-red-600">Error</h1>
          <p className="text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  const componentData = getComponentData();

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0">
        <h1 className="text-2xl font-bold">🎯 BERT Model - Interactive Quiz</h1>
        <p className="text-gray-600 text-sm mt-1">Click on any component in the diagram to explore its details</p>
      </div>

      {/* Split Panel Container */}
      <div ref={containerRef} className="flex-1 flex flex-col overflow-hidden relative">
        {/* Top Panel - SVG */}
        <div 
          className={`bg-gray-50 relative overflow-hidden transition-all ${
            activePane === 'top' ? 'ring-2 ring-blue-400' : ''
          }`} 
          style={{ height: `${splitPosition}%` }}
        >
          {/* Zoom Controls */}
          <div className="zoom-controls absolute top-4 right-4 z-20 bg-white rounded-lg shadow-lg p-2">
            <div className="flex gap-2 mb-2">
              <button
                onClick={handleZoomIn}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                title="Zoom In"
              >
                +
              </button>
              <button
                onClick={handleZoomOut}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                title="Zoom Out"
              >
                -
              </button>
              <button
                onClick={handleZoomReset}
                className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
                title="Reset"
              >
                Reset
              </button>
              <span className="px-2 py-1 text-sm text-gray-600">
                {Math.round(zoomLevel * 100)}%
              </span>
            </div>
            <div className="text-xs text-gray-500 border-t pt-1">
              {navigator.platform.includes('Mac') ? (
                <>
                  <div>🔍 Pinch: Zoom</div>
                  <div>👆 Two fingers: Pan</div>
                </>
              ) : (
                <>
                  <div>🔍 Ctrl+Scroll: Zoom</div>
                  <div>👆 Scroll: Pan</div>
                </>
              )}
            </div>
          </div>

          {/* SVG Container */}
          <div 
            ref={svgContainerRef}
            className="w-full h-full overflow-hidden cursor-move"
            style={{ 
              touchAction: 'none',
              WebkitUserSelect: 'none',
              userSelect: 'none',
              overflow: 'hidden'
            }}
            tabIndex={0}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onMouseEnter={() => {
              setActivePane('top');
              // Focus the container to ensure it receives wheel events
              if (svgContainerRef.current) {
                svgContainerRef.current.focus();
              }
            }}
            onMouseLeave={() => setActivePane(null)}
            onWheel={(e) => {
              // Direct wheel handler as a failsafe
              e.preventDefault();
              e.stopPropagation();
              
              const rect = svgContainerRef.current.getBoundingClientRect();
              const isMac = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
              
              if (e.ctrlKey || e.metaKey) {
                // Pinch zoom with zoom-to-cursor
                const delta = e.deltaY;
                const zoomSpeed = 0.005; // Reduced for smoother zoom
                const scaleFactor = 1 - (delta * zoomSpeed);
                const currentZoom = zoomLevelRef.current;
                const newZoom = Math.min(Math.max(currentZoom * scaleFactor, 0.3), 5);
                
                // Calculate cursor position relative to container
                const cursorX = e.clientX - rect.left;
                const cursorY = e.clientY - rect.top;
                
                // Calculate the point under cursor in the transformed space
                const currentPan = panPositionRef.current;
                const pointX = (cursorX - currentPan.x) / currentZoom;
                const pointY = (cursorY - currentPan.y) / currentZoom;
                
                // Calculate new pan position to keep the point under cursor
                const newPanX = cursorX - pointX * newZoom;
                const newPanY = cursorY - pointY * newZoom;
                
                setZoomLevel(newZoom);
                setPanPosition({ x: newPanX, y: newPanY });
              } else {
                // Pan with reduced sensitivity for smoother movement
                const currentPan = panPositionRef.current;
                const sensitivity = 0.7; // Reduced for smoother pan
                setPanPosition({
                  x: currentPan.x - (e.deltaX * sensitivity),
                  y: currentPan.y - (e.deltaY * sensitivity)
                });
              }
            }}
          >
            <div
              ref={svgRef}
              className="svg-container"
              style={{
                transform: `translate(${panPosition.x}px, ${panPosition.y}px) scale(${zoomLevel})`,
                transformOrigin: '0 0',
                transition: 'none', // Remove transition for smoother performance
                willChange: 'transform'
              }}
              dangerouslySetInnerHTML={{ __html: quizData?.svgDiagram || '' }}
            />
          </div>
        </div>

        {/* Divider */}
        <div 
          className="h-2 bg-gray-300 cursor-ns-resize flex-shrink-0 hover:bg-gray-400 transition-colors relative"
          onMouseDown={handleDividerMouseDown}
          style={{ userSelect: 'none' }}
        >
          <div className="absolute inset-x-0 top-1/2 transform -translate-y-1/2 h-1 bg-gray-400 opacity-50"></div>
        </div>

        {/* Bottom Panel - Quiz */}
        <div 
          className="bg-white overflow-y-auto" 
          style={{ height: `calc(${100 - splitPosition}% - 0.5rem)` }}
          onWheel={(e) => {
            // Let the bottom panel scroll normally
            e.stopPropagation();
          }}
          onMouseEnter={() => setActivePane('bottom')}
          onMouseLeave={() => setActivePane(null)}
        >
          <div className="p-6">
            {!selectedComponent ? (
          <div className="text-center py-12">
            <p className="text-lg text-gray-600">👆 Click on any component in the diagram to see its details!</p>
          </div>
        ) : componentData && (
          <>
            <h2 className="text-2xl font-bold mb-4">📚 {componentData.title}</h2>
            {componentData.explanation && (
              <p className="text-gray-600 mb-6">{componentData.explanation}</p>
            )}

            {/* Tabs */}
            <div className="flex border-b border-gray-200 mb-6">
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                  activeTab === 'equations' 
                    ? 'text-blue-600 border-blue-600' 
                    : 'text-gray-600 border-transparent hover:text-gray-800'
                }`}
                onClick={() => setActiveTab('equations')}
              >
                📐 Equations
              </button>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                  activeTab === 'initialization' 
                    ? 'text-blue-600 border-blue-600' 
                    : 'text-gray-600 border-transparent hover:text-gray-800'
                }`}
                onClick={() => setActiveTab('initialization')}
              >
                ⚙️ Initialization
              </button>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                  activeTab === 'forward' 
                    ? 'text-blue-600 border-blue-600' 
                    : 'text-gray-600 border-transparent hover:text-gray-800'
                }`}
                onClick={() => setActiveTab('forward')}
              >
                ➡️ Forward Pass
              </button>
            </div>

            {/* Tab Content */}
            <div className="tab-content">
              {activeTab === 'equations' && (
                <div className="space-y-6">
                  {componentData.equations.length > 0 ? (
                    componentData.equations.slice(0, 5).map((eq, index) => {
                      const eqKey = `${selectedComponent}_eq_${index}`;
                      return (
                        <div key={index} className="bg-gray-50 rounded-lg p-6">
                          <h3 className="text-lg font-semibold mb-2">
                            {eq.title || `Equation ${index + 1}`}
                          </h3>
                          {eq.explanation && (
                            <p className="text-gray-600 italic mb-4">{eq.explanation}</p>
                          )}
                          <div className="relative">
                            <button 
                              onClick={() => toggleVisibility(eqKey)}
                              className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10 bg-white px-2 py-1 rounded border border-gray-300"
                            >
                              {visibility[eqKey] ? 'Hide' : 'Show'}
                            </button>
                            {visibility[eqKey] ? (
                              <div className="bg-white p-4 rounded border border-gray-200 overflow-x-auto">
                                <BlockMath>{eq.equation || ''}</BlockMath>
                              </div>
                            ) : (
                              <EquationCanvas 
                                canvasId={eqKey}
                                savedData={drawingData[eqKey]}
                                onSaveData={handleSaveDrawing}
                              />
                            )}
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-gray-600">No equations available for this component.</p>
                  )}
                </div>
              )}

              {activeTab === 'initialization' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    {componentData.initialization.title || 'Initialization'}
                  </h3>
                  {componentData.initialization.explanation && (
                    <p className="text-gray-600 italic mb-4">
                      {componentData.initialization.explanation}
                    </p>
                  )}
                  <div className="relative">
                    <button 
                      onClick={() => toggleVisibility(`${selectedComponent}_init_code`)}
                      className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10 bg-white px-2 py-1 rounded border border-gray-300"
                    >
                      {visibility[`${selectedComponent}_init_code`] ? 'Hide' : 'Show'}
                    </button>
                    {visibility[`${selectedComponent}_init_code`] ? (
                      <SyntaxHighlighter 
                        language="python" 
                        style={oneLight}
                        customStyle={{
                          fontSize: '14px',
                          backgroundColor: '#f6f8fa',
                          padding: '16px',
                          borderRadius: '6px',
                          border: '1px solid #e1e4e8',
                          overflowX: 'auto',
                          whiteSpace: 'pre'
                        }}
                        showLineNumbers={true}
                      >
                        {componentData.initialization.codeAnswer || '# No initialization code available'}
                      </SyntaxHighlighter>
                    ) : (
                      <Editor
                        value={userCode[`${selectedComponent}_init_code`] || ''}
                        onValueChange={code => setUserCode(prev => ({ ...prev, [`${selectedComponent}_init_code`]: code }))}
                        highlight={highlightCode}
                        padding={16}
                        style={{
                          fontFamily: '"Fira code", "Fira Mono", monospace',
                          fontSize: 14,
                          backgroundColor: '#f6f8fa',
                          border: '1px solid #e1e4e8',
                          borderRadius: '6px',
                          minHeight: '200px'
                        }}
                        textareaClassName="font-mono"
                        preClassName="font-mono"
                        placeholder="Type the initialization code here to test your memory..."
                      />
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'forward' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    {componentData.forward.title || 'Forward Pass'}
                  </h3>
                  {componentData.forward.explanation && (
                    <p className="text-gray-600 italic mb-4">
                      {componentData.forward.explanation}
                    </p>
                  )}
                  <div className="relative">
                    <button 
                      onClick={() => toggleVisibility(`${selectedComponent}_forward_code`)}
                      className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10 bg-white px-2 py-1 rounded border border-gray-300"
                    >
                      {visibility[`${selectedComponent}_forward_code`] ? 'Hide' : 'Show'}
                    </button>
                    {visibility[`${selectedComponent}_forward_code`] ? (
                      <SyntaxHighlighter 
                        language="python" 
                        style={oneLight}
                        customStyle={{
                          fontSize: '14px',
                          backgroundColor: '#f6f8fa',
                          padding: '16px',
                          borderRadius: '6px',
                          border: '1px solid #e1e4e8',
                          overflowX: 'auto',
                          whiteSpace: 'pre'
                        }}
                        showLineNumbers={true}
                      >
                        {componentData.forward.codeAnswer || '# No forward pass code available'}
                      </SyntaxHighlighter>
                    ) : (
                      <Editor
                        value={userCode[`${selectedComponent}_forward_code`] || ''}
                        onValueChange={code => setUserCode(prev => ({ ...prev, [`${selectedComponent}_forward_code`]: code }))}
                        highlight={highlightCode}
                        padding={16}
                        style={{
                          fontFamily: '"Fira code", "Fira Mono", monospace',
                          fontSize: 14,
                          backgroundColor: '#f6f8fa',
                          border: '1px solid #e1e4e8',
                          borderRadius: '6px',
                          minHeight: '200px'
                        }}
                        textareaClassName="font-mono"
                        preClassName="font-mono"
                        placeholder="Type the forward pass code here to test your memory..."
                      />
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
            )}
          </div>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        /* Ensure proper event handling on macOS */
        .svg-container {
          -webkit-user-select: none;
          user-select: none;
          touch-action: none;
          /* GPU acceleration for smoother performance */
          transform: translateZ(0);
          -webkit-transform: translateZ(0);
          backface-visibility: hidden;
          -webkit-backface-visibility: hidden;
          perspective: 1000;
          -webkit-perspective: 1000;
        }
        
        /* Prevent browser zoom globally when component is active */
        body.bert-model-active {
          touch-action: none !important;
        }
        
        /* Ensure the SVG container captures all events */
        div[tabindex="0"] {
          outline: none;
        }
        
        /* Override any global styles that might interfere */
        .overflow-hidden {
          -ms-touch-action: none;
          touch-action: none;
        }
        
        .component-selected rect,
        .component-selected circle,
        .component-selected ellipse,
        .component-selected path {
          stroke: red !important;
          stroke-width: 4px !important;
          filter: drop-shadow(0 0 10px rgba(255,0,0,0.6));
          animation: pulse 1.5s infinite;
        }
        
        .component-selected > rect,
        .component-selected > circle,
        .component-selected > ellipse,
        .component-selected > path {
          stroke: red !important;
          stroke-width: 4px !important;
          filter: drop-shadow(0 0 10px rgba(255,0,0,0.6));
          animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.7; }
          100% { opacity: 1; }
        }
        
        g.component:hover {
          opacity: 0.8;
          transition: opacity 0.2s ease;
        }
        
        g.component {
          cursor: pointer;
        }
        
        .svg-container svg {
          width: auto;
          height: auto;
        }
        
        .cursor-move {
          cursor: move;
        }
        
        .cursor-ns-resize {
          cursor: ns-resize;
        }
        
        body.dragging-divider {
          cursor: ns-resize !important;
          user-select: none !important;
        }
        
        body.dragging-divider * {
          cursor: ns-resize !important;
          user-select: none !important;
        }
        
        /* Ensure smooth gestures */
        .svg-container {
          will-change: transform;
          -webkit-user-select: none;
          user-select: none;
        }
        
        /* Mobile-friendly zoom controls */
        @media (max-width: 768px) {
          .zoom-controls {
            top: 2rem;
            right: 1rem;
          }
        }
      `}} />
    </div>
  );
}