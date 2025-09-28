import React, { useRef, useEffect, useState, useCallback } from 'react';

export default function SmoothCanvas({ canvasId, savedData, onSaveData }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isEraser, setIsEraser] = useState(false);
  const [paths, setPaths] = useState(savedData || []);
  const [currentPath, setCurrentPath] = useState([]);
  const saveTimeoutRef = useRef(null);
  const contextRef = useRef(null);
  const lastPointRef = useRef(null);

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Set actual size in memory
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale canvas to match device pixel ratio
    const ctx = canvas.getContext('2d', { 
      alpha: false,
      desynchronized: true, // Better performance
      willReadFrequently: false
    });
    ctx.scale(dpr, dpr);
    
    // Set drawing properties
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1.5;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // White background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    contextRef.current = ctx;

    // Redraw saved paths
    redrawPaths();
  }, []);

  // Redraw all paths
  const redrawPaths = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = contextRef.current;
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    // Redraw all paths
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1.5;
    
    paths.forEach(path => {
      if (path.length < 2) return;
      
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      
      // Use quadratic curves for smoother lines
      for (let i = 1; i < path.length - 1; i++) {
        const xc = (path[i].x + path[i + 1].x) / 2;
        const yc = (path[i].y + path[i + 1].y) / 2;
        ctx.quadraticCurveTo(path[i].x, path[i].y, xc, yc);
      }
      
      // Draw the last segment
      if (path.length > 1) {
        ctx.lineTo(path[path.length - 1].x, path[path.length - 1].y);
      }
      
      ctx.stroke();
    });
  }, [paths]);

  // Get mouse/touch position
  const getPosition = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }, []);

  // Drawing functions
  const startDrawing = useCallback((e) => {
    e.preventDefault();
    const pos = getPosition(e);
    if (!pos) return;

    setIsDrawing(true);
    lastPointRef.current = pos;
    setCurrentPath([pos]);

    // Draw a dot for single clicks
    const ctx = contextRef.current;
    if (ctx) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [getPosition]);

  const draw = useCallback((e) => {
    if (!isDrawing) return;
    e.preventDefault();

    const pos = getPosition(e);
    if (!pos) return;

    const ctx = contextRef.current;
    if (!ctx) return;

    // Use requestAnimationFrame for smoother drawing
    requestAnimationFrame(() => {
      if (isEraser) {
        // Eraser mode - draw white
        ctx.globalCompositeOperation = 'destination-out';
        ctx.lineWidth = 20;
      } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1.5;
      }

      // Draw smooth line using quadratic curve
      ctx.beginPath();
      ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
      
      const midPoint = {
        x: (lastPointRef.current.x + pos.x) / 2,
        y: (lastPointRef.current.y + pos.y) / 2
      };
      
      ctx.quadraticCurveTo(
        lastPointRef.current.x, 
        lastPointRef.current.y, 
        midPoint.x, 
        midPoint.y
      );
      ctx.stroke();

      lastPointRef.current = pos;
      setCurrentPath(prev => [...prev, pos]);
    });
  }, [isDrawing, isEraser, getPosition]);

  const stopDrawing = useCallback(() => {
    if (!isDrawing) return;
    
    setIsDrawing(false);
    
    if (!isEraser && currentPath.length > 0) {
      setPaths(prev => [...prev, currentPath]);
    }
    
    setCurrentPath([]);
    
    // Save after drawing stops
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveTimeoutRef.current = setTimeout(() => {
      if (onSaveData) {
        onSaveData(canvasId, isEraser ? [] : [...paths, currentPath].filter(p => p.length > 0));
      }
    }, 500);
  }, [isDrawing, isEraser, currentPath, paths, canvasId, onSaveData]);

  // Handle clear
  const handleClear = () => {
    setPaths([]);
    redrawPaths();
    if (onSaveData) {
      onSaveData(canvasId, []);
    }
  };

  // Handle undo
  const handleUndo = () => {
    if (paths.length > 0) {
      const newPaths = paths.slice(0, -1);
      setPaths(newPaths);
      redrawPaths();
      if (onSaveData) {
        onSaveData(canvasId, newPaths);
      }
    }
  };

  // Update paths when savedData changes
  useEffect(() => {
    if (savedData && JSON.stringify(savedData) !== JSON.stringify(paths)) {
      setPaths(savedData);
      redrawPaths();
    }
  }, [savedData]);

  // Redraw when paths change
  useEffect(() => {
    redrawPaths();
  }, [paths, redrawPaths]);

  return (
    <div className="relative bg-white border border-gray-300 rounded-lg overflow-hidden">
      <div className="absolute top-2 left-2 z-20 flex gap-2">
        <button
          onClick={() => setIsEraser(false)}
          className={`px-2 py-1 text-xs rounded border ${
            !isEraser 
              ? 'bg-blue-500 text-white border-blue-600' 
              : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
          }`}
          type="button"
          title="Pen mode"
        >
          ✏️
        </button>
        <button
          onClick={() => setIsEraser(true)}
          className={`px-2 py-1 text-xs rounded border ${
            isEraser 
              ? 'bg-blue-500 text-white border-blue-600' 
              : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
          }`}
          type="button"
          title="Eraser mode"
        >
          🧹
        </button>
        <div className="w-px h-6 bg-gray-300 mx-1" />
        <button
          onClick={handleUndo}
          className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border border-gray-300"
          type="button"
        >
          Undo
        </button>
        <button
          onClick={handleClear}
          className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded border border-gray-300"
          type="button"
        >
          Clear
        </button>
      </div>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        style={{
          display: 'block',
          width: '100%',
          height: '200px',
          cursor: isEraser ? 'grab' : 'crosshair',
          touchAction: 'none',
          userSelect: 'none',
          WebkitUserSelect: 'none',
          MozUserSelect: 'none',
          msUserSelect: 'none'
        }}
      />
      <div className="absolute bottom-2 left-2 text-xs text-gray-500 pointer-events-none select-none">
        Draw your equation here
      </div>
    </div>
  );
}