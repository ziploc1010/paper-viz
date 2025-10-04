import React, { useState, useEffect } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription } from './ComparisonLayout';

/**
 * Generic comparison component that loads data from JSON files
 * This component can render any comparison by loading its JSON configuration
 */
export default function GenericComparison({ comparisonId }) {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [visibility, setVisibility] = useState({});
  const [userCode, setUserCode] = useState({});
  const [drawingData, setDrawingData] = useState({});

  // Load comparison data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Load the comparison JSON file using dynamic import
        const data = await import(`../data/comparisons/${comparisonId}.json`)
          .then(module => module.default);

        setComparisonData(data);

        // Initialize visibility state for all sections
        const initialVisibility = {};
        data.sections?.forEach(section => {
          if (section.leftId) initialVisibility[section.leftId] = true;
          if (section.rightId) initialVisibility[section.rightId] = true;
        });
        setVisibility(initialVisibility);

        setLoading(false);
      } catch (err) {
        console.error('Error loading comparison data:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    loadData();
  }, [comparisonId]);

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const handleSaveDrawing = (canvasId, data) => {
    setDrawingData(prev => ({ ...prev, [canvasId]: data }));
  };

  const highlightCode = code => {
    try {
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code;
    }
  };

  // Render variables list
  const renderVariables = (variables) => {
    if (!variables || !Array.isArray(variables)) return null;

    return (
      <ul className="list-disc pl-5 space-y-1 mt-2">
        {variables.map((variable, idx) => (
          <li key={idx}>
            <InlineMath>{variable.symbol}</InlineMath>: {variable.description}
          </li>
        ))}
      </ul>
    );
  };

  // Render left content
  const renderLeftContent = (leftContent) => {
    if (!leftContent) return null;

    return (
      <>
        {leftContent.description && (
          <p className="mb-4">{leftContent.description}</p>
        )}

        {leftContent.equation && (
          <EquationBlock
            math={leftContent.equation}
            description={leftContent.equationDescription || ""}
          />
        )}

        {/* Handle equations array format */}
        {leftContent.equations && Array.isArray(leftContent.equations) && leftContent.equations.map((eq, idx) => (
          <div key={idx}>
            {eq.title && <p className="mt-8 mb-4">{eq.title}</p>}
            <EquationBlock
              math={eq.math || eq.equation}
              description={eq.description || ""}
            />
          </div>
        ))}

        {/* Handle formulas array format (HGNN structure) */}
        {leftContent.formulas && Array.isArray(leftContent.formulas) && leftContent.formulas.map((formula, idx) => (
          <div key={idx} className="mb-6">
            {formula.title && <h4 className="font-semibold mb-2 mt-6">{formula.title}</h4>}
            {formula.equation && (
              <EquationBlock math={formula.equation} description="" />
            )}
            {formula.variables && renderVariables(formula.variables)}
            {formula.subEquations && Array.isArray(formula.subEquations) && formula.subEquations.map((subEq, subIdx) => (
              <div key={subIdx} className="ml-4 mt-4">
                {subEq.title && <p className="font-medium mb-2">{subEq.title}</p>}
                {subEq.equation && <EquationBlock math={subEq.equation} description="" />}
              </div>
            ))}
          </div>
        ))}

        {/* Handle subSections array format (HGNN Vertex/Edge Degrees structure) */}
        {leftContent.subSections && Array.isArray(leftContent.subSections) && leftContent.subSections.map((subSection, idx) => (
          <div key={idx} className="mb-6">
            {subSection.title && <h4 className="font-semibold mb-2 mt-6">{subSection.title}</h4>}
            {subSection.equation && (
              <EquationBlock math={subSection.equation} description="" />
            )}
            {subSection.variables && renderVariables(subSection.variables)}
            {subSection.subEquations && Array.isArray(subSection.subEquations) && subSection.subEquations.map((subEq, subIdx) => (
              <div key={subIdx} className="ml-4 mt-4">
                {subEq.title && <p className="font-medium mb-2">{subEq.title}</p>}
                {subEq.equation && <EquationBlock math={subEq.equation} description="" />}
              </div>
            ))}
          </div>
        ))}

        {/* Handle additionalText1 + equation2 */}
        {leftContent.additionalText1 && (
          <p className="mt-8 mb-4">{leftContent.additionalText1}</p>
        )}
        {leftContent.equation2 && (
          <EquationBlock math={leftContent.equation2} description="" />
        )}

        {/* Handle additionalText2 + equation3 */}
        {leftContent.additionalText2 && (
          <p className="mt-8 mb-4">{leftContent.additionalText2}</p>
        )}
        {leftContent.equation3 && (
          <EquationBlock math={leftContent.equation3} description="" />
        )}

        {/* Handle equation1, equation2, equation3 object format */}
        {Object.keys(leftContent).filter(k => k.startsWith('equation')).map((key, idx) => {
          const eq = leftContent[key];
          if (eq && typeof eq === 'object' && eq.math) {
            return (
              <div key={idx} className="mb-4">
                {eq.title && <h4 className="font-semibold mb-2">{eq.title}</h4>}
                <EquationBlock math={eq.math} description="" />
              </div>
            );
          }
          return null;
        })}

        {leftContent.variables && renderVariables(leftContent.variables)}

        {leftContent.note && (
          <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-400">
            <p className="text-sm">
              <strong>Note:</strong> {leftContent.note}
            </p>
          </div>
        )}

        {leftContent.formulas && leftContent.formulas.map((formula, idx) => (
          <FormulaWithDescription
            key={idx}
            title={formula.title}
            math={formula.math}
          />
        ))}
      </>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full min-h-screen">
        <div className="text-xl text-gray-600">Loading comparison...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full min-h-screen">
        <div className="text-xl text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (!comparisonData) {
    return (
      <div className="flex items-center justify-center h-full min-h-screen">
        <div className="text-xl text-red-600">No comparison data loaded</div>
      </div>
    );
  }

  return (
    <ComparisonLayout
      title={comparisonData.title}
      description={comparisonData.description}
      subtitle={comparisonData.subtitle}
    >
      {comparisonData.sections?.map((section, index) => (
        <ComparisonSection
          key={section.id || index}
          title={section.title}
          leftTitle={section.leftTitle}
          leftContent={renderLeftContent(section.leftContent)}
          code={section.code}
          visibility={visibility}
          toggleVisibility={toggleVisibility}
          leftId={section.leftId}
          rightId={section.rightId}
          userCode={userCode}
          setUserCode={setUserCode}
          highlightCode={highlightCode}
          drawingData={drawingData}
          onSaveDrawing={handleSaveDrawing}
          canvasId={section.canvasId}
          showEditor={section.showEditor}
          editorKey={section.editorKey}
          editorPlaceholder={section.editorPlaceholder}
          fullWidth={section.fullWidth}
        />
      ))}
    </ComparisonLayout>
  );
}
