import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Editor from 'react-simple-code-editor';
import EquationCanvas from './EquationCanvas';
import SmoothCanvas from './SmoothCanvas';

// Use the smooth canvas for better performance
const CanvasComponent = SmoothCanvas;

// Individual equation block with its own hide/show toggle
export function EquationBlock({ 
  math, 
  description, 
  id, 
  visibility, 
  toggleVisibility, 
  drawingData, 
  onSaveDrawing 
}) {
  const isVisible = visibility?.[id] !== false; // Default to visible
  
  return (
    <div className="mb-4">
      <div className="relative min-h-[40px]">
        <button 
          onClick={() => toggleVisibility(id)}
          className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
        >
          {isVisible ? 'Hide' : 'Show'}
        </button>
        {isVisible ? (
          <>
            <BlockMath math={math} />
            {description && <p className="text-sm text-gray-600 mt-2">{description}</p>}
          </>
        ) : (
          <CanvasComponent 
            canvasId={id}
            savedData={drawingData?.[id] || []}
            onSaveData={(canvasId, paths) => onSaveDrawing(canvasId, paths)}
          />
        )}
      </div>
    </div>
  );
}

// Styled paragraph component for subheadings
export function SubHeader({ children, className = "" }) {
  return (
    <p className={`font-semibold mb-2 mt-6 ${className}`}>
      {children}
    </p>
  );
}

// Formula with description (similar to EquationBlock but different layout)
export function FormulaWithDescription({ 
  title, 
  math, 
  description, 
  id, 
  visibility, 
  toggleVisibility, 
  drawingData, 
  onSaveDrawing 
}) {
  const isVisible = visibility?.[id] !== false; // Default to visible
  
  return (
    <div className={title ? "mb-4 mt-6" : "mb-4"}>
      {title && <p className="font-semibold mb-2">{title}</p>}
      <div className="relative min-h-[40px]">
        <button 
          onClick={() => toggleVisibility(id)}
          className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
        >
          {isVisible ? 'Hide' : 'Show'}
        </button>
        {isVisible ? (
          <>
            <BlockMath math={math} />
            {description && <p className="text-sm text-gray-600 mt-2">{description}</p>}
          </>
        ) : (
          <CanvasComponent 
            canvasId={id}
            savedData={drawingData?.[id] || []}
            onSaveData={(canvasId, paths) => onSaveDrawing(canvasId, paths)}
          />
        )}
      </div>
    </div>
  );
}

// Code block with its own hide/show toggle
export function CodeBlock({
  code,
  id,
  visibility,
  toggleVisibility,
  userCode,
  setUserCode,
  highlightCode,
  placeholder,
  showEditor = true
}) {
  const isVisible = visibility?.[id] !== false; // Default to visible
  
  return (
    <div className="relative min-h-[40px]">
      <button 
        onClick={() => toggleVisibility(id)}
        className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
      >
        {isVisible ? 'Hide' : 'Show'}
      </button>
      {isVisible ? (
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
        >
          {code}
        </SyntaxHighlighter>
      ) : (
        showEditor && (
          <Editor
            value={userCode?.[id] || ''}
            onValueChange={code => setUserCode(prev => ({ ...prev, [id]: code }))}
            highlight={highlightCode}
            padding={16}
            style={{
              fontFamily: '"Fira code", "Fira Mono", monospace',
              fontSize: 14,
              backgroundColor: '#f6f8fa',
              borderRadius: '6px',
              border: '1px solid #e1e4e8',
              minHeight: '100px'
            }}
            textareaClassName="font-mono"
            preClassName="font-mono"
            placeholder={placeholder || "Type your code here..."}
          />
        )
      )}
    </div>
  );
}

// Reusable section component for equation-code pairs
export function ComparisonSection({ 
  title, 
  leftTitle = "Mathematical Definition",
  rightTitle = "Implementation",
  leftContent, 
  rightContent, 
  code,
  visibility,
  toggleVisibility,
  leftId,
  rightId,
  userCode,
  setUserCode,
  highlightCode,
  editorPlaceholder,
  showEditor = false,
  editorKey,
  fullWidth = false,
  drawingData,
  onSaveDrawing,
  canvasId
}) {
  // If fullWidth is true and there's no leftContent, render full-width code section
  if (fullWidth && !leftContent) {
    return (
      <div className="bg-gray-100 p-6 rounded-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">{title}</h2>
          <h3 className="text-lg font-semibold text-gray-600">{leftTitle || rightTitle}</h3>
        </div>
        <CodeBlock
          code={code}
          id={rightId}
          visibility={visibility}
          toggleVisibility={toggleVisibility}
          userCode={userCode}
          setUserCode={setUserCode}
          highlightCode={highlightCode}
          placeholder={editorPlaceholder}
          showEditor={showEditor}
        />
      </div>
    );
  }
  
  // Default two-column layout
  return (
    <div className="grid grid-cols-2 gap-8">
      {/* Equations/Theory (Left) */}
      <div className="bg-white p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-6">{title}</h2>
        <h3 className="text-lg font-semibold mb-4">{leftTitle}</h3>
        {React.Children.map(leftContent?.props?.children, (child, index) => {
          if (React.isValidElement(child)) {
            // Handle direct EquationBlock or FormulaWithDescription
            if (child.type === EquationBlock || child.type === FormulaWithDescription) {
              return React.cloneElement(child, {
                visibility,
                toggleVisibility,
                drawingData,
                onSaveDrawing,
                id: child.props.id || `eq-${index}`
              });
            }
            // Handle nested children (e.g., inside fragments or other elements)
            else if (child.props?.children) {
              return React.cloneElement(child, {
                children: React.Children.map(child.props.children, (nestedChild, nestedIndex) => {
                  if (React.isValidElement(nestedChild) && 
                      (nestedChild.type === EquationBlock || nestedChild.type === FormulaWithDescription)) {
                    return React.cloneElement(nestedChild, {
                      visibility,
                      toggleVisibility,
                      drawingData,
                      onSaveDrawing,
                      id: nestedChild.props.id || `eq-${index}-${nestedIndex}`
                    });
                  }
                  return nestedChild;
                })
              });
            }
          }
          return child;
        }) || leftContent}
      </div>

      {/* Code (Right) */}
      <div className="bg-gray-100 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">{rightTitle}</h3>
        <CodeBlock
          code={code}
          id={rightId || editorKey}
          visibility={visibility}
          toggleVisibility={toggleVisibility}
          userCode={userCode}
          setUserCode={setUserCode}
          highlightCode={highlightCode}
          placeholder={editorPlaceholder}
          showEditor={showEditor}
        />
      </div>
    </div>
  );
}

// Reusable layout wrapper
export function ComparisonLayout({ title, description, children }) {
  return (
    <div className="max-w-7xl mx-auto p-6 space-y-12">
      {title && (
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">{title}</h1>
          {description && <p className="text-gray-600">{description}</p>}
        </div>
      )}
      {children}
    </div>
  );
}