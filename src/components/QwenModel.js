import React, { useState, useEffect, useRef } from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import qwenQuizData from '../data/models/qwen/model.json';

// Lazy load heavy components
const Editor = React.lazy(() => import('react-simple-code-editor'));
const EquationCanvas = React.lazy(() => import('./EquationCanvas'));

// Only load Prism when needed
let prismLoaded = false;
let highlight, languages;

const loadPrism = async () => {
  if (!prismLoaded) {
    const prismCore = await import('prismjs/components/prism-core');
    await import('prismjs/components/prism-python');
    await import('prismjs/themes/prism.css');
    highlight = prismCore.highlight;
    languages = prismCore.languages;
    prismLoaded = true;
  }
};

export default function QwenModel() {
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
  const [isInteractingWithSVG, setIsInteractingWithSVG] = useState(false);
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
  const highlightCode = async (code) => {
    try {
      await loadPrism();
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code;
    }
  };

  // Toggle visibility for equations and code
  const toggleVisibility = (key) => {
    setVisibility(prev => ({ 
      ...prev, 
      [key]: prev[key] === undefined ? false : !prev[key] 
    }));
  };

  // Save drawing data
  const handleSaveDrawing = (canvasId, data) => {
    setDrawingData(prev => ({ ...prev, [canvasId]: data }));
  };

  // Load the Qwen model data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Import SVG as a URL
        const svgUrl = (await import('../data/models/qwen/diagram.svg')).default;
        // Fetch the SVG content from the URL
        const response = await fetch(svgUrl);
        const svgContent = await response.text();

        // Combine JSON data with SVG
        setQuizData({
          ...qwenQuizData,
          svgDiagram: svgContent
        });
        setLoading(false);
      } catch (err) {
        console.error('Error loading data:', err);
        setError('Failed to load Qwen model data.');
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Set up click handlers for SVG components using componentMatchers
  useEffect(() => {
    console.log('=== QWEN CLICK HANDLER SETUP START ===');
    console.log('useEffect triggered. quizData:', quizData ? 'exists' : 'null');
    console.log('svgRef.current:', svgRef.current ? 'exists' : 'null');
    console.log('quizData.componentMatchers:', quizData?.componentMatchers ? 'exists' : 'undefined');

    if (quizData) {
      console.log('quizData keys:', Object.keys(quizData));
    }

    if (!quizData || !svgRef.current || !quizData.componentMatchers) {
      console.log('Early return - missing required data');
      return;
    }

    // Add style element to the document head for SVG styles
    const styleId = 'qwen-model-styles';
    let styleElement = document.getElementById(styleId);
    if (!styleElement) {
      styleElement = document.createElement('style');
      styleElement.id = styleId;
      styleElement.innerHTML = `
        .component-clickable {
          cursor: pointer !important;
        }

        rect.component-clickable:hover,
        circle.component-clickable:hover,
        ellipse.component-clickable:hover,
        path.component-clickable:hover {
          opacity: 0.8 !important;
          stroke: #4A90E2 !important;
          stroke-width: 3px !important;
          transition: all 0.2s ease;
        }

        g.component-clickable:hover {
          opacity: 0.8;
          transition: opacity 0.2s ease;
        }

        /* Target elements that have .component-selected class directly */
        rect.component-selected,
        circle.component-selected,
        ellipse.component-selected,
        path.component-selected,
        g.component-selected rect,
        g.component-selected circle,
        g.component-selected ellipse,
        g.component-selected path {
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

    const svg = svgRef.current;
    const componentMatchers = quizData.componentMatchers;
    const clickableElements = new Map(); // Map from element to componentId

    console.log('componentMatchers:', componentMatchers);
    console.log('componentMatchers keys:', Object.keys(componentMatchers || {}));

    // Find all text elements in the SVG and match them to components
    const textElements = svg.querySelectorAll('text');

    console.log('Total text elements found:', textElements.length);
    if (textElements.length > 0) {
      console.log('First 5 text contents:', Array.from(textElements).slice(0, 5).map(el => el.textContent.trim()));
    }

    // Log the first matcher to see the structure
    if (componentMatchers) {
      const firstKey = Object.keys(componentMatchers)[0];
      console.log('First matcher example:', firstKey, ':', componentMatchers[firstKey]);

      // Manual test: check if "Qwen3Model" matches "main-container"
      const testText = "Qwen3Model";
      const testPatterns = componentMatchers["main-container"];
      console.log('Test: Does "' + testText + '" match main-container patterns?', testPatterns);
      if (testPatterns) {
        const testMatch = testPatterns.some(pattern =>
          testText.includes(pattern) || pattern.includes(testText)
        );
        console.log('Test match result:', testMatch);
      }
    }

    let matchCount = 0;
    textElements.forEach(textEl => {
      const textContent = textEl.textContent.trim();

      // Check each component matcher to see if this text matches
      for (const [componentId, patterns] of Object.entries(componentMatchers)) {
        const matches = patterns.some(pattern =>
          textContent.includes(pattern) || pattern.includes(textContent)
        );

        if (matches) {
          matchCount++;
          console.log(`Match found! Text: "${textContent}" matched component: ${componentId}`);

          // Find the best click target for this text element
          let clickTargets = [];

          // First, try to find a parent group
          const parentGroup = textEl.closest('g');
          if (parentGroup) {
            clickTargets.push(parentGroup);
          } else {
            // If no parent group, find the nearest rect/shape that's positioned near this text
            const textRect = textEl.getBBox();
            const textX = parseFloat(textEl.getAttribute('x')) || textRect.x;
            const textY = parseFloat(textEl.getAttribute('y')) || textRect.y;

            // Find all shapes in the SVG
            const allShapes = svg.querySelectorAll('rect, circle, ellipse, path, polygon');
            let nearestShape = null;
            let minDistance = Infinity;

            allShapes.forEach(shape => {
              try {
                const shapeBox = shape.getBBox();
                // Check if text is inside or very close to this shape
                const isInside = textX >= shapeBox.x && textX <= (shapeBox.x + shapeBox.width) &&
                                 textY >= shapeBox.y && textY <= (shapeBox.y + shapeBox.height);

                if (isInside) {
                  // Calculate distance from text to shape center
                  const shapeCenterX = shapeBox.x + shapeBox.width / 2;
                  const shapeCenterY = shapeBox.y + shapeBox.height / 2;
                  const distance = Math.sqrt(Math.pow(textX - shapeCenterX, 2) + Math.pow(textY - shapeCenterY, 2));

                  if (distance < minDistance) {
                    minDistance = distance;
                    nearestShape = shape;
                  }
                }
              } catch (e) {
                // Skip shapes that can't be measured
              }
            });

            if (nearestShape) {
              clickTargets.push(nearestShape);
            }

            // Also add the text element itself as clickable
            clickTargets.push(textEl);
          }

          console.log('Click targets found:', clickTargets.map(t => t.tagName).join(', '), 'for text:', textContent);

          // Make all targets clickable
          clickTargets.forEach(target => {
            target.classList.add('component-clickable');
            target.setAttribute('data-component-id', componentId);
            clickableElements.set(target, componentId);
            console.log(`Added clickable element (${target.tagName}) for: ${componentId}`);
          });

          break; // Found a match, no need to check other patterns
        }
      }
    });

    console.log('Total matches found:', matchCount);

    const handleComponentClick = (e) => {
      const target = e.target.closest('[data-component-id]');
      if (!target) return;

      e.preventDefault();
      e.stopPropagation();

      const componentId = target.getAttribute('data-component-id');

      console.log('Component clicked:', componentId);
      console.log('Quiz data:', quizData.componentQuizzes?.[componentId]);
      console.log('Explanation data:', quizData.componentExplanations?.[componentId]);

      // Remove previous selection
      svg.querySelectorAll('.component-selected').forEach(el => {
        el.classList.remove('component-selected');
      });

      // Add new selection
      target.classList.add('component-selected');

      setSelectedComponent(componentId);
      setActiveTab('equations'); // Reset to equations tab
    };

    console.log('Found clickable elements:', clickableElements.size);
    console.log('Component IDs:', Array.from(clickableElements.values()));

    // Add click listeners to all clickable elements
    clickableElements.forEach((componentId, element) => {
      element.addEventListener('click', handleComponentClick);
    });

    // Cleanup
    return () => {
      clickableElements.forEach((componentId, element) => {
        element.removeEventListener('click', handleComponentClick);
        element.classList.remove('component-clickable', 'component-selected');
        element.removeAttribute('data-component-id');
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
        
        // Use requestAnimationFrame for smoother updates
        requestAnimationFrame(() => {
          setZoomLevel(newZoom);
          setPanPosition({ x: newPanX, y: newPanY });
        });
      } else {
        // Two-finger scroll/pan with smoother movement
        const currentPan = panPositionRef.current;
        const sensitivity = 0.7; // Reduced for smoother scrolling
        const newPan = {
          x: currentPan.x - (e.deltaX * sensitivity),
          y: currentPan.y - (e.deltaY * sensitivity)
        };
        
        // Use requestAnimationFrame for smoother updates
        requestAnimationFrame(() => {
          setPanPosition(newPan);
        });
      }
      
      return false;
    };

    const handleMouseEnter = () => {
      setActivePane('top');
      setIsInteractingWithSVG(true);
    };
    
    const handleMouseLeave = () => {
      setActivePane(null);
      setIsInteractingWithSVG(false);
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
    if (!selectedComponent || !quizData) {
      console.log('getComponentData: No selectedComponent or quizData');
      return null;
    }

    const quiz = quizData.componentQuizzes?.[selectedComponent];
    const explanation = quizData.componentExplanations?.[selectedComponent];

    console.log('getComponentData called for:', selectedComponent);
    console.log('Quiz:', quiz);
    console.log('Explanation:', explanation);

    if (!quiz && !explanation) {
      console.log('getComponentData: No quiz AND no explanation - returning null');
      return null;
    }

    const result = {
      title: quiz?.title || explanation?.title || selectedComponent,
      explanation: explanation?.explanation || '',
      equations: quiz?.equations || [],
      initialization: quiz?.initialization || {},
      forward: quiz?.forward || {},
      function: quiz?.function || null,
      isFunction: !!quiz?.function  // true if this is a function, false if class
    };

    console.log('getComponentData returning:', result);
    return result;
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-lg p-6 text-center">
          <h1 className="text-3xl font-bold mb-4">Loading Qwen3 Model...</h1>
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
        <h1 className="text-2xl font-bold">🎯 Qwen3 Model - Interactive Quiz</h1>
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
          onMouseEnter={() => {
            setActivePane('bottom');
            setIsInteractingWithSVG(false);
          }}
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
              {componentData.isFunction ? (
                <button
                  className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
                    activeTab === 'function'
                      ? 'text-blue-600 border-blue-600'
                      : 'text-gray-600 border-transparent hover:text-gray-800'
                  }`}
                  onClick={() => setActiveTab('function')}
                >
                  ⚡ Function
                </button>
              ) : (
                <>
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
                </>
              )}
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
                              {visibility[eqKey] === false ? 'Show' : 'Hide'}
                            </button>
                            {visibility[eqKey] !== false ? (
                              <div className="bg-white p-4 rounded border border-gray-200 overflow-x-auto">
                                <BlockMath>{eq.equation || ''}</BlockMath>
                              </div>
                            ) : (
                              isInteractingWithSVG ? (
                                <div className="bg-white p-4 rounded border border-gray-200 text-center text-gray-500">
                                  <p>Canvas paused while interacting with diagram</p>
                                  <p className="text-sm mt-2">Move cursor to bottom panel to draw</p>
                                </div>
                              ) : (
                                <React.Suspense fallback={<div className="bg-white p-4 rounded border border-gray-200 text-center">Loading canvas...</div>}>
                                  <EquationCanvas 
                                    canvasId={eqKey}
                                    savedData={drawingData[eqKey]}
                                    onSaveData={handleSaveDrawing}
                                  />
                                </React.Suspense>
                              )
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
                      {visibility[`${selectedComponent}_init_code`] === false ? 'Show' : 'Hide'}
                    </button>
                    {visibility[`${selectedComponent}_init_code`] !== false ? (
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
                      <React.Suspense fallback={<div className="bg-gray-100 p-4 rounded border border-gray-200 text-center" style={{minHeight: '200px'}}>Loading editor...</div>}>
                        <Editor
                          value={userCode[`${selectedComponent}_init_code`] || ''}
                          onValueChange={code => setUserCode(prev => ({ ...prev, [`${selectedComponent}_init_code`]: code }))}
                          highlight={code => code}
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
                      </React.Suspense>
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
                      {visibility[`${selectedComponent}_forward_code`] === false ? 'Show' : 'Hide'}
                    </button>
                    {visibility[`${selectedComponent}_forward_code`] !== false ? (
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
                      <React.Suspense fallback={<div className="bg-gray-100 p-4 rounded border border-gray-200 text-center" style={{minHeight: '200px'}}>Loading editor...</div>}>
                        <Editor
                          value={userCode[`${selectedComponent}_forward_code`] || ''}
                          onValueChange={code => setUserCode(prev => ({ ...prev, [`${selectedComponent}_forward_code`]: code }))}
                          highlight={code => code}
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
                      </React.Suspense>
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'function' && componentData.function && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    {componentData.function.title || 'Function'}
                  </h3>
                  {componentData.function.explanation && (
                    <p className="text-gray-600 italic mb-4">
                      {componentData.function.explanation}
                    </p>
                  )}
                  <div className="relative">
                    <button
                      onClick={() => toggleVisibility(`${selectedComponent}_function_code`)}
                      className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10 bg-white px-2 py-1 rounded border border-gray-300"
                    >
                      {visibility[`${selectedComponent}_function_code`] === false ? 'Show' : 'Hide'}
                    </button>
                    {visibility[`${selectedComponent}_function_code`] !== false ? (
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
                        {componentData.function.codeAnswer || '# No function code available'}
                      </SyntaxHighlighter>
                    ) : (
                      <React.Suspense fallback={<div className="bg-gray-100 p-4 rounded border border-gray-200 text-center" style={{minHeight: '200px'}}>Loading editor...</div>}>
                        <Editor
                          value={userCode[`${selectedComponent}_function_code`] || ''}
                          onValueChange={code => setUserCode(prev => ({ ...prev, [`${selectedComponent}_function_code`]: code }))}
                          highlight={code => code}
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
                          placeholder="Type the function code here to test your memory..."
                        />
                      </React.Suspense>
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