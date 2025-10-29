import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook to handle text selection on the page
 * Returns selected text and position for showing "Ask about this" button
 */
export const useTextSelection = () => {
  const [selection, setSelection] = useState({
    text: '',
    x: 0,
    y: 0,
    hasSelection: false,
  });

  const handleSelection = useCallback(() => {
    const selectedText = window.getSelection()?.toString().trim();

    if (selectedText && selectedText.length > 0) {
      const range = window.getSelection()?.getRangeAt(0);
      const rect = range?.getBoundingClientRect();

      if (rect) {
        // Check if selection is within AI sidebar - if so, ignore it
        const container = range.commonAncestorContainer;
        const element = container.nodeType === Node.TEXT_NODE
          ? container.parentElement
          : container;

        if (element && element.closest('.ai-sidebar')) {
          // Selection is inside AI sidebar, don't process it
          return;
        }

        // Extract LaTeX from selected KaTeX elements
        let latexText = selectedText;

        // Reuse container and element from above for LaTeX extraction
        const parentElement = element;

        // Strategy 1: Check if we're inside a single KaTeX element
        let currentElement = parentElement;
        while (currentElement && currentElement !== document.body) {
          if (currentElement.classList?.contains('katex')) {
            const annotation = currentElement.querySelector('annotation[encoding="application/x-tex"]');
            if (annotation) {
              // Check if it's display math or inline math
              const isDisplay = currentElement.classList.contains('katex-display');
              latexText = isDisplay
                ? '$$' + annotation.textContent + '$$'
                : '$' + annotation.textContent + '$';
              break;
            }
          }
          currentElement = currentElement.parentElement;
        }

        // Strategy 2: If not found, search for all KaTeX elements in the selection range
        if (latexText === selectedText && parentElement) {
          const katexElements = parentElement.querySelectorAll('.katex');
          const annotations = [];

          katexElements.forEach(katexEl => {
            const annotation = katexEl.querySelector('annotation[encoding="application/x-tex"]');
            if (annotation) {
              const isDisplay = katexEl.classList.contains('katex-display');
              const latex = isDisplay
                ? '$$' + annotation.textContent + '$$'
                : '$' + annotation.textContent + '$';
              annotations.push(latex);
            }
          });

          if (annotations.length > 0) {
            latexText = annotations.join(' ');
          }
        }

        setSelection({
          text: latexText,
          x: rect.left + rect.width / 2,
          y: rect.bottom + 10,
          hasSelection: true,
        });
      }
    } else {
      setSelection({
        text: '',
        x: 0,
        y: 0,
        hasSelection: false,
      });
    }
  }, []);

  useEffect(() => {
    // Listen for mouseup (end of text selection)
    document.addEventListener('mouseup', handleSelection);
    // Listen for selection changes
    document.addEventListener('selectionchange', handleSelection);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('selectionchange', handleSelection);
    };
  }, [handleSelection]);

  const clearSelection = useCallback(() => {
    window.getSelection()?.removeAllRanges();
    setSelection({
      text: '',
      x: 0,
      y: 0,
      hasSelection: false,
    });
  }, []);

  return { selection, clearSelection };
};
