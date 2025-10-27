import React, { useState, useRef, useEffect } from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import '../styles/math-derivation.css';

const MathDerivation = ({ derivationId }) => {
  const [flippedSteps, setFlippedSteps] = useState(new Set());
  const [derivationData, setDerivationData] = useState(null);
  const [canvasRefs] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Reset state when derivation changes
    setFlippedSteps(new Set());
    setLoading(true);
    setError(null);

    // Map derivation IDs to their JSON files
    const derivationFiles = {
      'diffusion-models': 'diffusion.json',
      'schrodinger-equation': 'schrodinger.json',
      'euler-lagrange': 'euler-lagrange.json',
      'general-relativity': 'generalrelativity.json',
      // Add more derivations here as they're created
      // 'fourier-transform': 'fourier.json',
      // 'navier-stokes': 'navier-stokes.json',
    };

    const fileName = derivationFiles[derivationId];
    if (!fileName) {
      setError(`Derivation "${derivationId}" not found`);
      setLoading(false);
      return;
    }

    // Load derivation data from derivations/ directory
    import(`../data/derivations/${fileName}`)
      .then(module => {
        setDerivationData(module.default);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading derivation data:', err);
        setError('Error loading derivation data');
        setLoading(false);
      });
  }, [derivationId]);

  const toggleStep = (stepId) => {
    setFlippedSteps(prev => {
      const newSet = new Set(prev);
      if (newSet.has(stepId)) {
        newSet.delete(stepId);
      } else {
        newSet.add(stepId);
      }
      return newSet;
    });
  };

  const clearCanvas = (stepId) => {
    const canvas = canvasRefs[stepId];
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const clearAllCanvases = () => {
    Object.keys(canvasRefs).forEach(stepId => clearCanvas(stepId));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-xl text-gray-600">Loading derivation...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-xl text-red-600">{error}</div>
      </div>
    );
  }

  if (!derivationData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-xl text-red-600">Error loading derivation data</div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold mb-4">{derivationData.title}</h1>
        <h2 className="text-2xl text-gray-600 mb-2">{derivationData.subtitle}</h2>
        <p className="text-gray-500 max-w-3xl mx-auto">{derivationData.description}</p>
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8">
        <p className="text-sm text-blue-800 text-center">
          Write each equation by hand in the canvas. Click <span className="font-semibold">"Show"</span> to reveal the answer.
        </p>
      </div>

      {/* Sections */}
      {derivationData.sections.map((section, sectionIndex) => (
        <div key={sectionIndex} className="mb-12">
          <h3 className="text-2xl font-bold text-blue-600 mb-6">
            {renderExplanation(section.title)}
          </h3>
          
          {/* Steps */}
          {section.steps.map((step) => (
            <StepCard
              key={step.id}
              step={step}
              isFlipped={flippedSteps.has(step.id)}
              onToggle={() => toggleStep(step.id)}
              canvasRef={(ref) => { if (ref) canvasRefs[step.id] = ref; }}
              onClear={() => clearCanvas(step.id)}
            />
          ))}
        </div>
      ))}

      {/* Action Buttons */}
      <div className="mt-12 space-y-4">
        <button
          onClick={clearAllCanvases}
          className="w-full py-3 px-6 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          <span>Clear All</span>
        </button>
      </div>
    </div>
  );
};

// Helper function to render text with inline LaTeX
const renderExplanation = (text) => {
  // Split text by $ delimiters for inline math
  const parts = text.split(/(\$[^\$]+\$)/g);
  
  return parts.map((part, index) => {
    if (part.startsWith('$') && part.endsWith('$')) {
      // Remove $ delimiters and render as inline math
      const math = part.slice(1, -1);
      return <InlineMath key={index} math={math} />;
    }
    return part;
  });
};

const StepCard = ({ step, isFlipped, onToggle, canvasRef, onClear }) => {
  const canvasContainerRef = useRef(null);
  const localCanvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Parse equation array or string
  const equations = Array.isArray(step.equation) ? step.equation : [step.equation];

  // Calculate height based on number of equations
  // Use provided canvasHeight or auto-calculate with better scaling:
  // - Single equation: 80px (small, centered)
  // - Multiple equations: 40px base + 50px per equation (very tight fit)
  const calculatedHeight = step.canvasHeight || (
    equations.length === 1 ? 80 : 40 + equations.length * 50
  );

  useEffect(() => {
    const canvas = localCanvasRef.current;
    if (canvas && canvasContainerRef.current) {
      // Set canvas size to match container
      const rect = canvasContainerRef.current.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = calculatedHeight;

      // Set up drawing context
      const ctx = canvas.getContext('2d');
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2;

      // Pass ref up to parent
      canvasRef(canvas);
    }
  }, [calculatedHeight, canvasRef]);

  const startDrawing = (e) => {
    setIsDrawing(true);
    const canvas = localCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    
    const canvas = localCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  return (
    <div className="mb-8">
      {/* Step Title */}
      <h4 className="text-lg font-semibold mb-2">
        {renderExplanation(step.title)}
      </h4>

      {/* Explanation */}
      <p className="text-gray-600 mb-4">
        {renderExplanation(step.explanation)}
      </p>

      {/* Card Container */}
      <div className="relative" style={{ minHeight: `${calculatedHeight}px` }}>
        {/* Canvas */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${
          isFlipped ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`} style={{ height: `${calculatedHeight}px` }}>
          <div ref={canvasContainerRef} className="w-full h-full bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
            <canvas
              ref={localCanvasRef}
              className="w-full h-full cursor-crosshair"
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
            />
          </div>
        </div>
        
        {/* Answer */}
        <div className={`absolute inset-0 transition-opacity duration-300 ${
          isFlipped ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`} style={{ height: `${calculatedHeight}px` }}>
          <div className="w-full h-full bg-blue-50 rounded-lg shadow-md border-2 border-blue-300 p-2 flex items-center justify-center overflow-y-auto overflow-x-auto">
            <div className="w-full flex flex-col items-center">
              {equations.map((eq, index) => (
                <div key={index} className={`${index > 0 ? 'mt-2' : ''}`}>
                  <BlockMath math={eq} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      {/* Action Buttons */}
      <div className="flex items-center justify-between mt-4">
        <button
          onClick={onToggle}
          className={`px-6 py-2 rounded-full text-white font-medium transition-colors flex items-center space-x-2 ${
            isFlipped ? 'bg-green-500 hover:bg-green-600' : 'bg-blue-500 hover:bg-blue-600'
          }`}
        >
          {isFlipped ? (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
              </svg>
              <span>Hide</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <span>Show</span>
            </>
          )}
        </button>
        
        <button
          onClick={onClear}
          className="text-red-500 hover:text-red-600 transition-colors flex items-center space-x-1"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          <span>Clear</span>
        </button>
      </div>
    </div>
  );
};

export default MathDerivation;