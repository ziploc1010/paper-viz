import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ReactSketchCanvas } from 'react-sketch-canvas';

export default function EquationCanvas({ canvasId, savedData, onSaveData }) {
  const canvasRef = useRef();
  const [isLoading, setIsLoading] = useState(true);
  const [isEraser, setIsEraser] = useState(false);
  const saveTimeoutRef = useRef(null);

  // Load saved data when component mounts
  useEffect(() => {
    const loadData = async () => {
      if (canvasRef.current && savedData && savedData.length > 0) {
        try {
          await canvasRef.current.loadPaths(savedData);
        } catch (error) {
          console.error('Error loading paths:', error);
        }
      }
      setIsLoading(false);
    };
    
    // Small delay to ensure canvas is ready
    const timer = setTimeout(loadData, 100);
    return () => clearTimeout(timer);
  }, []);

  // Debounced save function to avoid performance issues
  const debouncedSave = useCallback(() => {
    // Clear any existing timeout
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    // Set a new timeout to save after 1000ms of inactivity
    saveTimeoutRef.current = setTimeout(async () => {
      if (canvasRef.current && onSaveData) {
        try {
          const paths = await canvasRef.current.exportPaths();
          onSaveData(canvasId, paths);
        } catch (error) {
          console.error('Error saving paths:', error);
        }
      }
    }, 1000);
  }, [canvasId, onSaveData]);

  // Save canvas data after stroke ends (debounced)
  const handleStrokeEnd = () => {
    debouncedSave();
  };

  const handleClear = async () => {
    if (canvasRef.current) {
      try {
        await canvasRef.current.clearCanvas();
        if (onSaveData) {
          onSaveData(canvasId, []);
        }
      } catch (error) {
        console.error('Error clearing canvas:', error);
      }
    }
  };

  const handleUndo = async () => {
    if (canvasRef.current) {
      try {
        await canvasRef.current.undo();
        // Debounce the save after undo
        debouncedSave();
      } catch (error) {
        console.error('Error undoing:', error);
      }
    }
  };

  const handlePenMode = () => {
    if (canvasRef.current) {
      canvasRef.current.eraseMode(false);
      setIsEraser(false);
    }
  };

  const handleEraserMode = () => {
    if (canvasRef.current) {
      canvasRef.current.eraseMode(true);
      setIsEraser(true);
    }
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="relative bg-white border border-gray-300 rounded-lg overflow-hidden">
      <div className="absolute top-2 left-2 z-20 flex gap-2">
        <button
          onClick={handlePenMode}
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
          onClick={handleEraserMode}
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
      <div 
        style={{ 
          opacity: isLoading ? 0.5 : 1,
          touchAction: 'none',  // Prevent touch scrolling while drawing
          userSelect: 'none'    // Prevent text selection
        }}
      >
        <ReactSketchCanvas
          ref={canvasRef}
          width="100%"
          height="200px"
          strokeWidth={1.5}
          strokeColor="black"
          canvasColor="white"
          onStroke={handleStrokeEnd}
          exportWithBackgroundImage={false}
          allowOnlyPointerType="all"
          withViewBox={false}
          style={{ 
            border: 'none',
            borderRadius: '0.5rem',
            cursor: isEraser ? 'grab' : 'crosshair',
            touchAction: 'none',
            userSelect: 'none',
            WebkitUserSelect: 'none',
            MozUserSelect: 'none',
            msUserSelect: 'none'
          }}
          // Performance optimizations
          preserveBackgroundImageAspectRatio="none"
          withTimestamp={false}
        />
      </div>
      <div className="absolute bottom-2 left-2 text-xs text-gray-500 pointer-events-none select-none">
        Draw your equation here
      </div>
    </div>
  );
}