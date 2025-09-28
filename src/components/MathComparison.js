// MathComparison.js

import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import EquationCanvas from './EquationCanvas';

export default function MathComparison() {
  // Initialize visibility state for all equations and code blocks (all visible by default)
  const [visibility, setVisibility] = useState({
    // RoPE Section
    ropeCode: true,
    ropeEquation1: true,
    ropeEquation2: true,
    ropeEquation3: true,
    // RMSNorm Section
    rmsNormCode: true,
    rmsNormEquation1: true,
    rmsNormEquation2: true,
    // SwiGLU Section
    swiGLUCode: true,
    swiGLUEquation1: true,
    swiGLUEquation2: true,
  });

  // Initialize userCode state for tracking user input
  const [userCode, setUserCode] = useState({
    ropeCode: '',
    rmsNormCode: '',
    swiGLUCode: ''
  });

  // State to store drawing data
  const [drawingData, setDrawingData] = useState({});

  // Handler function to save drawing data
  const handleSaveDrawing = (canvasId, paths) => {
    setDrawingData(prev => ({
      ...prev,
      [canvasId]: paths
    }));
  };

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Custom highlight function that ensures proper syntax highlighting
  const highlightCode = (code) => {
    try {
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code; // Return plain code if highlighting fails
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-12">
      {/* RoPE Section */}
      <div className="grid grid-cols-2 gap-8">
        <div className="bg-[#f5f5f5] p-6 rounded-lg">
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('ropeCode')}
              className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
            >
              {visibility.ropeCode ? 'Hide' : 'Show'}
            </button>
            {visibility.ropeCode ? (
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
{`def precompute_rope_cache(dim, seq_len, theta=10000.0):
    dim = dim // 2
    pos = torch.arange(0, seq_len)
    freqs = 1.0 / (theta ** (torch.arange(0, dim) / dim))
    emb = pos[:, None] * freqs[None, :]
    rope_cache = torch.cat([emb.cos(), emb.sin()], dim=-1)
    return rope_cache

def apply_rope(x, rope_cache):
    rope = rope_cache[:x.size(1), None, :]
    dim = x.shape[-1] // 2
    x_complex = x.float().view(*x.shape[:-1], -1, 2)
    
    cos, sin = rope.split(dim, dim=-1)
    cos = cos.unsqueeze(-1)
    sin = sin.unsqueeze(-1)
    
    x_out = torch.cat([
        x_complex[..., 0:1] * cos - x_complex[..., 1:2] * sin,
        x_complex[..., 0:1] * sin + x_complex[..., 1:2] * cos,
    ], dim=-1)
    
    return x_out.flatten(-2)`}
              </SyntaxHighlighter>
            ) : (
              <Editor
                value={userCode.ropeCode}
                onValueChange={code => setUserCode(prev => ({ ...prev, ropeCode: code }))}
                highlight={highlightCode}
                padding={16}
                style={{
                  fontFamily: '"Fira code", "Fira Mono", monospace',
                  fontSize: 14,
                  backgroundColor: '#f6f8fa',
                  border: '1px solid #e1e4e8',
                  borderRadius: '6px',
                  minHeight: '100px'
                }}
                textareaClassName="font-mono"
                preClassName="font-mono"
                placeholder="Write your RoPE implementation here..."
              />
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-6">Rotary Positional Encoding (RoPE)</h2>
          <p className="mb-4">RoPE applies a rotation matrix to each pair of dimensions in the embedding space:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('ropeEquation1')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.ropeEquation1 ? 'Hide' : 'Show'}
            </button>
            {visibility.ropeEquation1 ? (
              <BlockMath>{`R_{\\theta}(x) = \\begin{pmatrix} 
                \\cos(m\\theta) & -\\sin(m\\theta) \\\\[0.3em]
                \\sin(m\\theta) & \\cos(m\\theta)
              \\end{pmatrix}
              \\begin{pmatrix}
                x_1 \\\\[0.3em] x_2
              \\end{pmatrix}`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="ropeEquation1" 
                savedData={drawingData["ropeEquation1"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <p className="mt-8 mb-4">The frequency bands are computed as:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('ropeEquation2')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.ropeEquation2 ? 'Hide' : 'Show'}
            </button>
            {visibility.ropeEquation2 ? (
              <BlockMath>{`\\theta_m = m\\theta_b, \\quad \\theta_b = 10000^{-2(i-1)/d}`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="ropeEquation2" 
                savedData={drawingData["ropeEquation2"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <p className="mt-8 mb-4">The full transformation can be written as:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('ropeEquation3')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.ropeEquation3 ? 'Hide' : 'Show'}
            </button>
            {visibility.ropeEquation3 ? (
              <BlockMath>{`\\begin{aligned}
                \\begin{pmatrix} x'_1 \\\\ x'_2 \\end{pmatrix} &= 
                \\begin{pmatrix} 
                  \\cos(m\\theta) & -\\sin(m\\theta) \\\\
                  \\sin(m\\theta) & \\cos(m\\theta)
                \\end{pmatrix}
                \\begin{pmatrix} x_1 \\\\ x_2 \\end{pmatrix} \\\\[1em]
                &= \\begin{pmatrix}
                  x_1\\cos(m\\theta) - x_2\\sin(m\\theta) \\\\
                  x_1\\sin(m\\theta) + x_2\\cos(m\\theta)
                \\end{pmatrix}
              \\end{aligned}`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="ropeEquation3" 
                savedData={drawingData["ropeEquation3"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>
        </div>
      </div>

      {/* RMSNorm Section */}
      <div className="grid grid-cols-2 gap-8">
        <div className="bg-[#f5f5f5] p-6 rounded-lg">
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('rmsNormCode')}
              className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
            >
              {visibility.rmsNormCode ? 'Hide' : 'Show'}
            </button>
            {visibility.rmsNormCode ? (
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
{`class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True))
        # Normalize and scale
        x_normalized = x / (rms + self.eps)
        return self.weight * x_normalized`}
              </SyntaxHighlighter>
            ) : (
              <Editor
                value={userCode.rmsNormCode}
                onValueChange={code => setUserCode(prev => ({ ...prev, rmsNormCode: code }))}
                highlight={highlightCode}
                padding={16}
                style={{
                  fontFamily: '"Fira code", "Fira Mono", monospace',
                  fontSize: 14,
                  backgroundColor: '#f6f8fa',
                  border: '1px solid #e1e4e8',
                  borderRadius: '6px',
                  minHeight: '100px'
                }}
                textareaClassName="font-mono"
                preClassName="font-mono"
                placeholder="Write your RMSNorm implementation here..."
              />
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-6">Root Mean Square Normalization</h2>
          <p className="mb-4">RMSNorm differs from LayerNorm by only normalizing the scale, without centering the mean:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('rmsNormEquation1')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.rmsNormEquation1 ? 'Hide' : 'Show'}
            </button>
            {visibility.rmsNormEquation1 ? (
              <BlockMath>{`\\text{RMSNorm}(x) = \\gamma \\odot \\frac{x}{\\sqrt{\\frac{1}{n}\\sum_{i=1}^n x_i^2 + \\epsilon}}`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="rmsNormEquation1" 
                savedData={drawingData["rmsNormEquation1"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <p className="mt-8 mb-4">Compared to LayerNorm:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('rmsNormEquation2')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.rmsNormEquation2 ? 'Hide' : 'Show'}
            </button>
            {visibility.rmsNormEquation2 ? (
              <BlockMath>{`\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="rmsNormEquation2" 
                savedData={drawingData["rmsNormEquation2"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <div className="mt-8">
            <p className="font-semibold mb-2">Where:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>γ is the learned scale parameter</li>
              <li>ε is a small constant (typically 10⁻⁶)</li>
              <li>n is the dimension size</li>
            </ul>
          </div>
        </div>
      </div>

      {/* SwiGLU Section */}
      <div className="grid grid-cols-2 gap-8">
        <div className="bg-[#f5f5f5] p-6 rounded-lg">
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('swiGLUCode')}
              className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
            >
              {visibility.swiGLUCode ? 'Hide' : 'Show'}
            </button>
            {visibility.swiGLUCode ? (
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
{`class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Compute Swish gate
        swish_gate = self.linear_gate(x) * torch.sigmoid(
            self.beta * self.linear_gate(x)
        )
        return swish_gate * self.linear(x)`}
              </SyntaxHighlighter>
            ) : (
              <Editor
                value={userCode.swiGLUCode}
                onValueChange={code => setUserCode(prev => ({ ...prev, swiGLUCode: code }))}
                highlight={highlightCode}
                padding={16}
                style={{
                  fontFamily: '"Fira code", "Fira Mono", monospace',
                  fontSize: 14,
                  backgroundColor: '#f6f8fa',
                  border: '1px solid #e1e4e8',
                  borderRadius: '6px',
                  minHeight: '100px'
                }}
                textareaClassName="font-mono"
                preClassName="font-mono"
                placeholder="Write your SwiGLU implementation here..."
              />
            )}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-6">SwiGLU Activation</h2>
          <p className="mb-4">SwiGLU combines Swish activation with a gating mechanism:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('swiGLUEquation1')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.swiGLUEquation1 ? 'Hide' : 'Show'}
            </button>
            {visibility.swiGLUEquation1 ? (
              <BlockMath>{`\\text{SwiGLU}(x) = \\text{Swish}(W_1x) \\odot W_2x`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="swiGLUEquation1" 
                savedData={drawingData["swiGLUEquation1"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <p className="mt-8 mb-4">Where Swish is defined as:</p>
          <div className="relative min-h-[40px]">
            <button 
              onClick={() => toggleVisibility('swiGLUEquation2')}
              className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
            >
              {visibility.swiGLUEquation2 ? 'Hide' : 'Show'}
            </button>
            {visibility.swiGLUEquation2 ? (
              <BlockMath>{`\\text{Swish}(x) = x \\cdot \\sigma(\\beta x)`}</BlockMath>
            ) : (
              <EquationCanvas 
                canvasId="swiGLUEquation2" 
                savedData={drawingData["swiGLUEquation2"]}
                onSaveData={handleSaveDrawing}
              />
            )}
          </div>

          <div className="mt-8">
            <p className="font-semibold mb-2">Where:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>σ is the sigmoid function</li>
              <li>β is a learnable parameter</li>
              <li>W₁, W₂ are learnable weight matrices</li>
              <li>⊙ represents element-wise multiplication</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

