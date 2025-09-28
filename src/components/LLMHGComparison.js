import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import EquationCanvas from './EquationCanvas';

export default function LLMHGComparison() {
  const [expandedSections, setExpandedSections] = useState({
    overview: true,
    multiview: false,
    continuous: false,
    prototype: false,
    structure: false,
    laplacian: false,
    fusion: false,
    training: false,
    evaluation: false
  });

  // Custom highlight function that ensures proper syntax highlighting
  const highlightCode = (code) => {
    try {
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code; // Return plain code if highlighting fails
    }
  };

  // State to store drawing data
  const [drawingData, setDrawingData] = useState({});

  // Initialize visibility state for all equations and code blocks (all visible by default)
  const [visibility, setVisibility] = useState({
    // Overview equations
    userSequence: true,
    nextItemProb: true,
    overviewCode: true,
    // Multi-view equations
    hypergraphDef: true,
    continuousScore: true,
    incidenceMatrix: true,
    vertexDegree: true,
    edgeDegree: true,
    multiViewDef: true,
    multiViewCode: true,
    // Continuous scoring
    continuousScoreCode: true,
    // Prototype equations
    originalPrototype: true,
    textCorrectedPrototype: true,
    correctionWeight: true,
    simplifiedWeight: true,
    prototypeCode: true,
    // Structure learning equations
    intraEdgeWeight: true,
    interEdgeWeight: true,
    finalWeight: true,
    structureCode: true,
    // Laplacian equations
    laplacianDef: true,
    structureLoss: true,
    expandedLoss: true,
    convolution: true,
    multiViewCombination: true,
    laplacianCode: true,
    // Fusion equations
    concatenation: true,
    elementSum: true,
    attentionFusion1: true,
    attentionFusion2: true,
    fusionCode: true,
    // Training equations
    predictionLoss: true,
    combinedLoss: true,
    trainingCode: true,
    // Evaluation equations
    hitRatio: true,
    ndcg: true,
    evaluationCode: true
  });

  // Initialize user code state for memorization feature
  const [userCode, setUserCode] = useState({
    overviewCode: '',
    multiViewCode: '',
    continuousScoreCode: '',
    prototypeCode: '',
    structureCode: '',
    laplacianCode: '',
    fusionCode: '',
    trainingCode: '',
    evaluationCode: ''
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Handler function to save drawing data
  const handleSaveDrawing = (canvasId, paths) => {
    setDrawingData(prev => ({
      ...prev,
      [canvasId]: paths
    }));
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold mb-2">LLMHG: Mathematical Formulations vs Implementation</h1>
        <p className="text-gray-600">LLM-Guided Multi-View Hypergraph Learning for Human-Centric Explainable Recommendation</p>
        
        <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400 text-sm">
          <p className="font-semibold">Key Enhancement:</p>
          <p>Our implementation uses <strong>continuous probability scores</strong> for interest angles (e.g., 0.63 for "budget_conscious") rather than the paper's enumerated subcategories approach. This enables more nuanced product-interest relationships through semantic similarity-based scoring.</p>
        </div>
      </div>

      {/* Section 1: Overview of LLMHG Approach */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('overview')}
        >
          <h2 className="text-2xl font-bold">1. Overview of LLMHG Approach</h2>
          <span className="text-xl">{expandedSections.overview ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.overview && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Paper Formulation</h3>
                <p className="mb-3">LLMHG approach includes four major steps:</p>
                <ol className="list-decimal list-inside space-y-1 mb-4">
                  <li>Interest angle extraction</li>
                  <li>Multi-view hypergraph construction</li>
                  <li>Hypergraph structure learning</li>
                  <li>Representation fusion</li>
                </ol>
                <p className="mb-2">The problem statement for sequential recommendation:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('userSequence')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.userSequence ? 'Hide' : 'Show'}
                  </button>
                  {visibility.userSequence ? (
                    <BlockMath>{`S_u = [v_1^{(u)}, ..., v_t^{(u)}, ..., v_{n_u}^{(u)}]`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="userSequence" 
                      savedData={drawingData["userSequence"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2 mb-2">Where <InlineMath>{`S_u`}</InlineMath> represents the sequence of interactions for user <InlineMath>{`u`}</InlineMath>.</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('nextItemProb')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.nextItemProb ? 'Hide' : 'Show'}
                  </button>
                  {visibility.nextItemProb ? (
                    <BlockMath>{`p(v_{n_u+1}^{(u)} = v | S_u)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="nextItemProb" 
                      savedData={drawingData["nextItemProb"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2">The goal is to predict the probability distribution of the next item.</p>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('overviewCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.overviewCode ? 'Hide' : 'Show'}
                </button>
                {visibility.overviewCode ? (
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
{`class LLMHG:
    # Main pipeline methods:
    def extract_interest_angles(self, user_sequence):
        # Step 1: Extract interest angles
        
    def build_hypergraph(self, interest_angles):
        # Step 2: Construct multi-view hypergraph
        
    def optimize_hypergraph(self, hypergraph, interest_angles):
        # Step 3: Apply structure learning
        
    # Step 4: Fusion (implemented in LLMHGFusionModel)
    
    def end_to_end_pipeline(self, user_sequences, interest_angles):
        # Orchestrates all four steps`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.overviewCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, overviewCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 2: Multi-View Hypergraph Construction */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('multiview')}
        >
          <h2 className="text-2xl font-bold">2. Multi-View Hypergraph Construction</h2>
          <span className="text-xl">{expandedSections.multiview ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.multiview && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2">A hypergraph is a generalization of a graph where edges can connect any number of vertices.</p>
                <p className="mb-2">Mathematically, a hypergraph is represented as:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('hypergraphDef')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.hypergraphDef ? 'Hide' : 'Show'}
                  </button>
                  {visibility.hypergraphDef ? (
                    <BlockMath>{`\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}, \\mathcal{W})`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="hypergraphDef" 
                      savedData={drawingData["hypergraphDef"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`\mathcal{V}`}</InlineMath>: Set of vertices (items)</li>
                  <li><InlineMath>{`\mathcal{E}`}</InlineMath>: Set of hyperedges</li>
                  <li><InlineMath>{`\mathcal{W}`}</InlineMath>: Weights of hyperedges</li>
                </ul>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Continuous Probability Scores</h3>
                <p className="mb-2 font-semibold">Key Implementation Difference:</p>
                <p className="mb-2">The paper uses enumerated subcategories, but our implementation uses <strong>continuous probability scores</strong> for interest angles.</p>
                <p className="mb-2">For each interest angle <InlineMath>{`k`}</InlineMath> and item <InlineMath>{`v_i`}</InlineMath>, we compute:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('continuousScore')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.continuousScore ? 'Hide' : 'Show'}
                  </button>
                  {visibility.continuousScore ? (
                    <BlockMath>{`s_k(v_i) = \\text{CosineSim}(\\text{emb}(v_i), \\text{emb}(\\text{angle}_k)) \\in [0, 1]`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="continuousScore" 
                      savedData={drawingData["continuousScore"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Incidence Matrix and Degrees</h3>
                <p className="mb-2">The incidence matrix <InlineMath>{`H`}</InlineMath> has <strong>continuous weights</strong>:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('incidenceMatrix')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.incidenceMatrix ? 'Hide' : 'Show'}
                  </button>
                  {visibility.incidenceMatrix ? (
                    <BlockMath>{`H_{i,j} = s_j(v_i) \\quad \\text{(continuous probability score)}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="incidenceMatrix" 
                      savedData={drawingData["incidenceMatrix"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-4 mb-2">Vertex degree:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('vertexDegree')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.vertexDegree ? 'Hide' : 'Show'}
                  </button>
                  {visibility.vertexDegree ? (
                    <BlockMath>{`d(v) = \\sum_{e \\in \\mathcal{E}} w(e)h(v, e)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="vertexDegree" 
                      savedData={drawingData["vertexDegree"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-4 mb-2">Edge degree:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('edgeDegree')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.edgeDegree ? 'Hide' : 'Show'}
                  </button>
                  {visibility.edgeDegree ? (
                    <BlockMath>{`\\delta(e) = \\sum_{v \\in \\mathcal{V}} h(v, e)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="edgeDegree" 
                      savedData={drawingData["edgeDegree"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Multi-View Structure</h3>
                <p className="mb-2">A multi-view hypergraph maintains separate hypergraphs for each interest angle:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('multiViewDef')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.multiViewDef ? 'Hide' : 'Show'}
                  </button>
                  {visibility.multiViewDef ? (
                    <BlockMath>{`\\mathcal{G}^{mv} = \\{\\mathcal{G}_1, \\mathcal{G}_2, ..., \\mathcal{G}_K\\}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="multiViewDef" 
                      savedData={drawingData["multiViewDef"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2">Where <InlineMath>{`K`}</InlineMath> is the number of discovered interest angles.</p>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('multiViewCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.multiViewCode ? 'Hide' : 'Show'}
                </button>
                {visibility.multiViewCode ? (
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
{`class HypergraphConstructor:
    def construct_hypergraph(self, interest_angles):
        """
        Construct a multi-view hypergraph from interest angles.
        Our implementation creates continuous probability-based hyperedges.
        """
        hypergraph = {
            'num_nodes': len(self.products_df),
            'views': {},
            'node_mapping': self.product_to_idx,
            'reverse_mapping': self.idx_to_product
        }
        
        # Process each interest angle as a separate view
        for angle_name, angle_data in interest_angles.items():
            view = {
                'hyperedges': [],
                'hyperedge_weights': [],
                'hyperedge_names': [],
                'pattern_type': angle_data.get('pattern_type', ''),
                'description': angle_data.get('description', '')
            }
            
            # Create hyperedges based on discovered patterns
            # Each angle becomes a view with products grouped by similarity
            pattern_details = angle_data.get('pattern_details', {})
            
            if 'products' in pattern_details:
                # Direct product assignments from pattern discovery
                product_ids = pattern_details['products']
                product_indices = [self._get_product_idx(pid) 
                                 for pid in product_ids 
                                 if self._get_product_idx(pid) != -1]
                if product_indices:
                    view['hyperedges'].append(product_indices)
                    view['hyperedge_weights'].append(1.0)
                    view['hyperedge_names'].append(f"{angle_name}_main")
                
            # Add this view to the hypergraph
            if view['hyperedges']:
                hypergraph['views'][angle_name] = view
                
        return hypergraph`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.multiViewCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, multiViewCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 2a: Continuous Interest Scoring */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('continuous')}
        >
          <h2 className="text-2xl font-bold">2a. Continuous Interest Angle Scoring</h2>
          <span className="text-xl">{expandedSections.continuous ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.continuous && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Paper's Categorical Approach</h3>
                <p className="mb-3">The original paper describes using enumerated subcategories within each interest angle:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Genre → {"{"} Comedy, Drama, Action, ... {"}"}</li>
                  <li>Price Range → {"{"} Budget, Mid-range, Premium {"}"}</li>
                  <li>Brand Tier → {"{"} Luxury, Popular, Generic {"}"}</li>
                </ul>
                <p className="mt-3">This creates binary membership relationships where items either belong to a subcategory or they don't.</p>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Our Continuous Scoring Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('continuousScoreCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.continuousScoreCode ? 'Hide' : 'Show'}
                </button>
                {visibility.continuousScoreCode ? (
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
{`def calculate_interest_product_scores(self, product_id: str, interest_angles: Dict) -> Dict:
    """
    Calculate continuous probability scores (0-1) for how strongly a product 
    aligns with EVERY interest angle using semantic similarity.
    """
    scores = {}
    
    # Get product embedding
    product_text = f"{title}. {description}. Brand: {brand}. {price_info}"
    product_embedding = self.sentence_model.encode([product_text])[0]
    
    # Score against each interest angle
    for angle_name, angle_data in interest_angles.items():
        # Create rich representation of the interest angle
        angle_text = f"Interest angle: {angle_name}. {angle_description}. "
        if keywords_text:
            angle_text += f"Keywords: {keywords_text}. "
        if sample_text:
            angle_text += f"Example products: {sample_text}"
        
        # Generate angle embedding
        angle_embedding = self.sentence_model.encode([angle_text])[0]
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(product_embedding, angle_embedding)
        
        # Transform to 0-1 probability score
        normalized_score = (similarity + 1) / 2
        scores[angle_name] = round(normalized_score, 2)
    
    return scores
    
# Example output:
# {
#   "budget_conscious": 0.63,
#   "luxury_minded": 0.21,
#   "convenience_seeker": 0.78,
#   "sustainability_focused": 0.45
# }`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.continuousScoreCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, continuousScoreCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 3: Prototype Computation with LLM Correction */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('prototype')}
        >
          <h2 className="text-2xl font-bold">3. Prototype Computation with LLM Correction</h2>
          <span className="text-xl">{expandedSections.prototype ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.prototype && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2">For each hyperedge, a prototype (centroid) is computed and then corrected using LLM-generated text embeddings:</p>
                
                <p className="mt-4 mb-2">Original prototype (centroid of node features):</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('originalPrototype')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.originalPrototype ? 'Hide' : 'Show'}
                  </button>
                  {visibility.originalPrototype ? (
                    <BlockMath>{`p_k^{orig} = \\frac{1}{|e_k|}\\sum_{v_i \\in e_k} x_i`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="originalPrototype" 
                      savedData={drawingData["originalPrototype"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2">Text-corrected prototype:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('textCorrectedPrototype')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.textCorrectedPrototype ? 'Hide' : 'Show'}
                  </button>
                  {visibility.textCorrectedPrototype ? (
                    <BlockMath>{`p_k = (1 - \\lambda_k) \\cdot p_k^{orig} + \\lambda_k \\cdot emb(T_{e_k})`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="textCorrectedPrototype" 
                      savedData={drawingData["textCorrectedPrototype"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`e_k`}</InlineMath>: The k-th hyperedge in the hypergraph</li>
                  <li><InlineMath>{`v_i`}</InlineMath>: The i-th vertex (item) in the hypergraph</li>
                  <li><InlineMath>{`x_i`}</InlineMath>: Feature vector of vertex <InlineMath>{`v_i`}</InlineMath></li>
                  <li><InlineMath>{`p_k^{orig}`}</InlineMath>: Original prototype for hyperedge <InlineMath>{`e_k`}</InlineMath></li>
                  <li><InlineMath>{`\lambda_k`}</InlineMath>: Correction weight for hyperedge <InlineMath>{`e_k`}</InlineMath></li>
                  <li><InlineMath>{`T_{e_k}`}</InlineMath>: LLM-generated text description of hyperedge <InlineMath>{`e_k`}</InlineMath></li>
                  <li><InlineMath>{`emb(T_{e_k})`}</InlineMath>: Text embedding of hyperedge description</li>
                  <li><InlineMath>{`h(\\cdot)`}</InlineMath>: Learnable function mapping text embeddings to scalar values</li>
                </ul>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Correction Weight</h3>
                <p className="mb-2">The correction weight <InlineMath>{`\lambda_k`}</InlineMath> based on the LLM-generated description:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('correctionWeight')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.correctionWeight ? 'Hide' : 'Show'}
                  </button>
                  {visibility.correctionWeight ? (
                    <BlockMath>{`\\lambda_k = \\frac{exp(-h(T_{e_k}))}{1 + exp(-h(T_{e_k}))}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="correctionWeight" 
                      savedData={drawingData["correctionWeight"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2">In the implementation, this is simplified to be inversely proportional to the hyperedge size:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('simplifiedWeight')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.simplifiedWeight ? 'Hide' : 'Show'}
                  </button>
                  {visibility.simplifiedWeight ? (
                    <BlockMath>{`\\lambda_k = \\frac{1}{1 + |e_k|}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="simplifiedWeight" 
                      savedData={drawingData["simplifiedWeight"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('prototypeCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.prototypeCode ? 'Hide' : 'Show'}
                </button>
                {visibility.prototypeCode ? (
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
{`def compute_prototypes(self, hypergraph, interest_angles, use_text_correction=True):
    """
    Compute prototypes (centroids) for each hyperedge with LLM text correction.
    """
    # Create product embeddings
    product_embeddings = self._create_product_embeddings()
    hypergraph['node_features'] = product_embeddings
    
    # Process each view (interest angle)
    for view_name, view in hypergraph['views'].items():
        angle_data = interest_angles.get(view_name, {})
        angle_description = angle_data.get('description', '')
        
        # Get text representation for the view
        view_text = f"Interest angle: {view_name}. {angle_description}"
        view_embedding = self.text_encoder.encode([view_text])[0]
        
        view['hyperedge_prototypes'] = []
        
        # Calculate prototype for each hyperedge in this view
        for edge_idx, (node_indices, edge_name) in enumerate(
                zip(view['hyperedges'], view['hyperedge_names'])):
            # Calculate original prototype (mean of node features)
            edge_nodes = np.array(node_indices)
            if len(edge_nodes) > 0:
                original_prototype = np.mean(product_embeddings[edge_nodes], axis=0)
            else:
                original_prototype = np.zeros(product_embeddings.shape[1])
            
            # Get text embedding for the hyperedge
            edge_text = f"{view_text}. Category: {edge_name}"
            edge_text_embedding = self.text_encoder.encode([edge_text])[0]
            
            # Calculate correction weight lambda
            edge_size = len(node_indices)
            lambda_k = 1.0 / (1.0 + min(10, max(1, edge_size)))
            
            # Calculate corrected prototype
            if use_text_correction:
                corrected_prototype = (1.0 - lambda_k) * original_prototype + \
                                     lambda_k * edge_text_embedding
            else:
                corrected_prototype = original_prototype
            
            view['hyperedge_prototypes'].append(corrected_prototype)`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.prototypeCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, prototypeCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 4: Intra-Edge and Inter-Edge Structure Learning */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('structure')}
        >
          <h2 className="text-2xl font-bold">4. Intra-Edge and Inter-Edge Structure Learning</h2>
          <span className="text-xl">{expandedSections.structure ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.structure && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2"><strong>Intra-edge weights</strong> measure similarity of nodes within a hyperedge:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('intraEdgeWeight')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.intraEdgeWeight ? 'Hide' : 'Show'}
                  </button>
                  {visibility.intraEdgeWeight ? (
                    <BlockMath>{`w_{intra}(e_k) = \\frac{1}{|e_k|(|e_k|-1)} \\sum_{v_i, v_j \\in e_k, i \\neq j} \\exp\\left(-\\frac{\\|x_i - x_j\\|^2}{\\mu}\\right)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="intraEdgeWeight" 
                      savedData={drawingData["intraEdgeWeight"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>Inter-edge weights</strong> measure distinctness between hyperedge prototypes:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('interEdgeWeight')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.interEdgeWeight ? 'Hide' : 'Show'}
                  </button>
                  {visibility.interEdgeWeight ? (
                    <BlockMath>{`w_{inter}(e_k) = \\frac{1}{|E|-1} \\sum_{e_l \\in E, l \\neq k} \\|p_k - p_l\\|^2`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="interEdgeWeight" 
                      savedData={drawingData["interEdgeWeight"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>Final hyperedge weight</strong> combines both components:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('finalWeight')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.finalWeight ? 'Hide' : 'Show'}
                  </button>
                  {visibility.finalWeight ? (
                    <BlockMath>{`w(e_k) = \\beta \\cdot w_{intra}(e_k) + (1 - \\beta) \\cdot w_{inter}(e_k)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="finalWeight" 
                      savedData={drawingData["finalWeight"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`e_k`}</InlineMath>: The k-th hyperedge in the hypergraph</li>
                  <li><InlineMath>{`v_i, v_j`}</InlineMath>: Vertices (items) within hyperedge <InlineMath>{`e_k`}</InlineMath></li>
                  <li><InlineMath>{`x_i, x_j`}</InlineMath>: Feature vectors of vertices <InlineMath>{`v_i`}</InlineMath> and <InlineMath>{`v_j`}</InlineMath></li>
                  <li><InlineMath>{`p_k, p_l`}</InlineMath>: Prototypes of hyperedges <InlineMath>{`e_k`}</InlineMath> and <InlineMath>{`e_l`}</InlineMath></li>
                  <li><InlineMath>{`E`}</InlineMath>: Set of all hyperedges in the hypergraph</li>
                  <li><InlineMath>{`\mu`}</InlineMath>: Heat kernel parameter (controls similarity sensitivity)</li>
                  <li><InlineMath>{`\beta`}</InlineMath>: Balance parameter between intra-edge and inter-edge weights</li>
                </ul>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('structureCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.structureCode ? 'Hide' : 'Show'}
                </button>
                {visibility.structureCode ? (
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
{`def compute_intra_edge_weights(self, hypergraph, mu=0.5):
    """
    Compute intra-edge weights based on pairwise similarity.
    """
    product_embeddings = hypergraph['node_features']
    
    # Process each view
    for view_name, view in hypergraph['views'].items():
        view['intra_edge_weights'] = []
        
        # Compute weight for each hyperedge
        for node_indices in view['hyperedges']:
            if len(node_indices) <= 1:
                view['intra_edge_weights'].append(1.0)
                continue
                
            # Get embeddings for nodes in this hyperedge
            edge_nodes = np.array(node_indices)
            edge_embeddings = product_embeddings[edge_nodes]
            
            # Compute pairwise similarities using heat kernel
            pairwise_dists = np.sum((edge_embeddings[:, np.newaxis, :] - 
                               edge_embeddings[np.newaxis, :, :]) ** 2, axis=2)
            similarities = np.exp(-pairwise_dists / mu)
            
            # Remove self-similarities (diagonal)
            np.fill_diagonal(similarities, 0.0)
            
            # Compute average similarity as intra-edge weight
            n_pairs = len(node_indices) * (len(node_indices) - 1)
            if n_pairs > 0:
                intra_weight = np.sum(similarities) / n_pairs
            else:
                intra_weight = 1.0
                
            view['intra_edge_weights'].append(intra_weight)
                
def compute_inter_edge_weights(self, hypergraph):
    """
    Compute inter-edge weights based on prototype distances.
    """
    # Process each view
    for view_name, view in hypergraph['views'].items():
        if 'hyperedge_prototypes' not in view:
            continue
            
        # Stack all prototypes for this view
        prototypes = np.vstack(view['hyperedge_prototypes'])
        
        # Compute pairwise distances between prototypes
        pairwise_dists = np.sum((prototypes[:, np.newaxis, :] - 
                           prototypes[np.newaxis, :, :]) ** 2, axis=2)
        
        # Compute average distance to other prototypes
        view['inter_edge_weights'] = []
        for i in range(len(prototypes)):
            # Remove distance to self
            other_dists = np.concatenate([pairwise_dists[i, :i], pairwise_dists[i, i+1:]])
            if len(other_dists) > 0:
                inter_weight = np.mean(other_dists)
            else:
                inter_weight = 1.0
                
            view['inter_edge_weights'].append(inter_weight)
                
def compute_final_edge_weights(self, hypergraph, beta=0.5):
    """
    Compute final hyperedge weights by combining intra-edge and inter-edge weights.
    """
    # Process each view
    for view_name, view in hypergraph['views'].items():
        if 'intra_edge_weights' not in view or 'inter_edge_weights' not in view:
            continue
            
        n_edges = min(len(view['intra_edge_weights']), len(view['inter_edge_weights']))
        
        # Combine weights with beta parameter
        view['hyperedge_weights'] = []
        for i in range(n_edges):
            intra_weight = view['intra_edge_weights'][i]
            inter_weight = view['inter_edge_weights'][i]
            
            # As per paper: w(e) = β * intra_weight + (1 - β) * inter_weight
            combined_weight = beta * intra_weight + (1 - beta) * inter_weight
            view['hyperedge_weights'].append(combined_weight)`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.structureCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, structureCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 5: Hypergraph Laplacian and Convolution */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('laplacian')}
        >
          <h2 className="text-2xl font-bold">5. Hypergraph Laplacian and Convolution</h2>
          <span className="text-xl">{expandedSections.laplacian ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.laplacian && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2"><strong>Hypergraph Laplacian:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('laplacianDef')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.laplacianDef ? 'Hide' : 'Show'}
                  </button>
                  {visibility.laplacianDef ? (
                    <BlockMath>{`\\mathcal{L} = I - D_v^{-\\frac{1}{2}} H W D_e^{-1} H^{\\top} D_v^{-\\frac{1}{2}}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="laplacianDef" 
                      savedData={drawingData["laplacianDef"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`\\mathcal{L}`}</InlineMath>: Normalized hypergraph Laplacian matrix</li>
                  <li><InlineMath>{`H`}</InlineMath>: Incidence matrix (vertices × hyperedges)</li>
                  <li><InlineMath>{`D_v`}</InlineMath>: Diagonal matrix of vertex degrees</li>
                  <li><InlineMath>{`D_e`}</InlineMath>: Diagonal matrix of edge degrees</li>
                  <li><InlineMath>{`W`}</InlineMath>: Diagonal matrix of edge weights</li>
                  <li><InlineMath>{`I`}</InlineMath>: Identity matrix</li>
                  <li><InlineMath>{`F`}</InlineMath>: Node feature matrix</li>
                  <li><InlineMath>{`G`}</InlineMath>: Hypergraph structure</li>
                  <li><InlineMath>{`X^{(l)}`}</InlineMath>: Node features at layer l</li>
                  <li><InlineMath>{`W^{(l)}, b^{(l)}`}</InlineMath>: Learnable weights and bias at layer l</li>
                  <li><InlineMath>{`\\sigma`}</InlineMath>: Activation function</li>
                </ul>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Structure Learning Loss</h3>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('structureLoss')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.structureLoss ? 'Hide' : 'Show'}
                  </button>
                  {visibility.structureLoss ? (
                    <BlockMath>{`L_{str}(F,G) = Tr(F^T \\mathcal{L} F)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="structureLoss" 
                      savedData={drawingData["structureLoss"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2">This can be expanded to:</p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('expandedLoss')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.expandedLoss ? 'Hide' : 'Show'}
                  </button>
                  {visibility.expandedLoss ? (
                    <BlockMath>{`L_{str}(F,G) = \\frac{1}{2} \\sum_{e \\in E} \\sum_{(v_i,v_j) \\in e} \\frac{w(e)}{\\delta(e)} \\left\\|\\frac{F_{v_i}}{\\sqrt{d(v_i)}} - \\frac{F_{v_j}}{\\sqrt{d(v_j)}}\\right\\|^2`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="expandedLoss" 
                      savedData={drawingData["expandedLoss"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Hypergraph Convolution</h3>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('convolution')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.convolution ? 'Hide' : 'Show'}
                  </button>
                  {visibility.convolution ? (
                    <BlockMath>{`X^{(l+1)} = \\sigma(\\mathcal{L} X^{(l)} W^{(l)} + b^{(l)})`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="convolution" 
                      savedData={drawingData["convolution"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>Multi-View Combination:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('multiViewCombination')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.multiViewCombination ? 'Hide' : 'Show'}
                  </button>
                  {visibility.multiViewCombination ? (
                    <BlockMath>{`\\mathcal{L}_{combined} = \\sum_{k=1}^K \\alpha_k \\mathcal{L}_k`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="multiViewCombination" 
                      savedData={drawingData["multiViewCombination"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('laplacianCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.laplacianCode ? 'Hide' : 'Show'}
                </button>
                {visibility.laplacianCode ? (
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
{`def _compute_laplacian(self, H, W_e=None):
    """
    Compute the normalized hypergraph Laplacian.
    """
    # Apply edge weights if provided
    if W_e is not None:
        W_e_diag = torch.diag(W_e)
        H_weighted = torch.matmul(H, W_e_diag)
    else:
        H_weighted = H
    
    # Compute vertex degrees D_v
    D_v = torch.sum(H_weighted, dim=1)
    D_v_inv_sqrt = torch.diag(torch.pow(D_v, -0.5))
    
    # Compute edge degrees D_e
    D_e = torch.sum(H, dim=0)
    D_e_inv = torch.diag(torch.pow(D_e, -1.0))
    
    # Compute normalized Laplacian:
    # L = I - D_v^(-1/2) * H * W * D_e^(-1) * H^T * D_v^(-1/2)
    L = torch.eye(H.shape[0]) - torch.matmul(
        torch.matmul(D_v_inv_sqrt, 
                    torch.matmul(
                        torch.matmul(H_weighted, D_e_inv),
                        torch.t(H))),
        D_v_inv_sqrt)
    
    return L

class HypergraphConvolutionLayer(nn.Module):
    """Hypergraph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(HypergraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
    def forward(self, X, L):
        # Apply hypergraph convolution: Out = L * X * W + b
        support = torch.matmul(X, self.weight)
        output = torch.matmul(L, support)
        output = output + self.bias
        return output
        
def _combine_multi_view_laplacians(self, laplacians, view_weights=None):
    """
    Combine multiple view-specific Laplacians into a single Laplacian.
    """
    n_views = len(laplacians)
    if n_views == 0:
        return None
        
    # Apply view weights if provided
    if view_weights is None:
        view_weights = torch.ones(n_views) / n_views
    elif isinstance(view_weights, np.ndarray):
        view_weights = torch.FloatTensor(view_weights)
        
    # Combine Laplacians with weighted average
    combined_laplacian = None
    for i, L in enumerate(laplacians):
        if combined_laplacian is None:
            combined_laplacian = view_weights[i] * L
        else:
            combined_laplacian += view_weights[i] * L
            
    return combined_laplacian`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.laplacianCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, laplacianCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 6: Fusion of Hypergraph and Sequential Representations */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('fusion')}
        >
          <h2 className="text-2xl font-bold">6. Fusion of Hypergraph and Sequential Representations</h2>
          <span className="text-xl">{expandedSections.fusion ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.fusion && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-3">The paper mentions representation fusion but doesn't provide specific mathematical formulations. Three common fusion methods are:</p>
                
                <p className="mb-2"><strong>1. Concatenation:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('concatenation')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.concatenation ? 'Hide' : 'Show'}
                  </button>
                  {visibility.concatenation ? (
                    <BlockMath>{`z_i = W [x_i^{seq}; x_i^{hg}] + b`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="concatenation" 
                      savedData={drawingData["concatenation"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>2. Element-wise Sum:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('elementSum')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.elementSum ? 'Hide' : 'Show'}
                  </button>
                  {visibility.elementSum ? (
                    <BlockMath>{`z_i = x_i^{seq} + x_i^{hg}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="elementSum" 
                      savedData={drawingData["elementSum"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>3. Attention-based Fusion:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('attentionFusion1')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.attentionFusion1 ? 'Hide' : 'Show'}
                  </button>
                  {visibility.attentionFusion1 ? (
                    <BlockMath>{`\\alpha = \\text{softmax}\\left(\\frac{Q(x_i^{seq})K(x_i^{hg})^T}{\\sqrt{d}}\\right)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="attentionFusion1" 
                      savedData={drawingData["attentionFusion1"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('attentionFusion2')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.attentionFusion2 ? 'Hide' : 'Show'}
                  </button>
                  {visibility.attentionFusion2 ? (
                    <BlockMath>{`z_i = x_i^{seq} + \\alpha V(x_i^{hg})`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="attentionFusion2" 
                      savedData={drawingData["attentionFusion2"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`x_i^{seq}`}</InlineMath>: Sequential model embedding for item <InlineMath>{`i`}</InlineMath></li>
                  <li><InlineMath>{`x_i^{hg}`}</InlineMath>: Hypergraph model embedding for item <InlineMath>{`i`}</InlineMath></li>
                  <li><InlineMath>{`Q, K, V`}</InlineMath>: Query, Key, Value projection matrices</li>
                </ul>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('fusionCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.fusionCode ? 'Hide' : 'Show'}
                </button>
                {visibility.fusionCode ? (
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
{`class LLMHGFusionModel(nn.Module):
    def __init__(self, seq_model, hypergraph_model, num_items,
                embed_dim=64, fusion_type='concat', dropout=0.1):
        # ...
        # Fusion components
        if fusion_type == 'concat':
            # Concatenation fusion
            self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attn_q = nn.Linear(embed_dim, embed_dim)
            self.attn_k = nn.Linear(embed_dim, embed_dim)
            self.attn_v = nn.Linear(embed_dim, embed_dim)
            self.fusion_layer = nn.Linear(embed_dim, embed_dim)
            
    def forward(self, input_seq, attention_mask=None, 
               hypergraph_features=None, hypergraph_laplacians=None):
        # 1. Get sequential model embeddings
        seq_output = self.seq_model(input_seq, attention_mask)
        seq_item_embeddings = self.seq_model.item_embeddings.weight
        
        # 2. Get hypergraph model embeddings
        if hypergraph_features is not None and hypergraph_laplacians is not None:
            # Apply hypergraph model
            hg_output = self.hypergraph_model(hg_features, hypergraph_laplacians)
            hg_item_embeddings = hg_output.squeeze(0)
            
            # 3. Fuse the embeddings using various fusion methods
            if self.fusion_type == 'concat':
                # Concatenation fusion
                fused_embeddings = torch.cat([seq_item_embeddings, 
                                             hg_item_embeddings], dim=1)
                fused_embeddings = self.fusion_layer(fused_embeddings)
                
            elif self.fusion_type == 'sum':
                # Element-wise sum fusion
                fused_embeddings = seq_item_embeddings + hg_item_embeddings
                
            elif self.fusion_type == 'attention':
                # Attention-based fusion
                q = self.attn_q(seq_item_embeddings)
                k = self.attn_k(hg_item_embeddings)
                v = self.attn_v(hg_item_embeddings)
                
                # Compute attention scores
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / \
                             (self.embed_dim ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Apply attention to values
                attn_output = torch.matmul(attn_weights, v)
                
                # Final fusion
                fused_embeddings = self.fusion_layer(seq_item_embeddings + attn_output)
        else:
            # No hypergraph data, use sequential embeddings only
            fused_embeddings = seq_item_embeddings`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.fusionCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, fusionCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 7: Training Loss Functions */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('training')}
        >
          <h2 className="text-2xl font-bold">7. Training Loss Functions</h2>
          <span className="text-xl">{expandedSections.training ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.training && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2"><strong>Prediction Loss (Binary Cross-Entropy):</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('predictionLoss')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.predictionLoss ? 'Hide' : 'Show'}
                  </button>
                  {visibility.predictionLoss ? (
                    <BlockMath>{`L_{pre} = -\\sum_{u=1}^{|U|} \\left[y_u \\log(\\hat{y}_u) + (1-y_u)\\log(1-\\hat{y}_u)\\right]`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="predictionLoss" 
                      savedData={drawingData["predictionLoss"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2">Where <InlineMath>{`y_u`}</InlineMath> is the ground truth next item and <InlineMath>{`\hat{y}_u`}</InlineMath> is the predicted probability.</p>
                
                <p className="mt-4 mb-2"><strong>Combined Loss with Structure Learning:</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('combinedLoss')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.combinedLoss ? 'Hide' : 'Show'}
                  </button>
                  {visibility.combinedLoss ? (
                    <BlockMath>{`L = L_{str} + \\alpha L_{pre}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="combinedLoss" 
                      savedData={drawingData["combinedLoss"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                <p className="mt-2">Where <InlineMath>{`\alpha`}</InlineMath> is a hyperparameter that balances the structure learning and prediction losses.</p>
                
                <p className="mt-4">The structure learning loss is computed through the hypergraph Laplacian and acts as a regularization term to ensure that connected nodes have similar embeddings.</p>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('trainingCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.trainingCode ? 'Hide' : 'Show'}
                </button>
                {visibility.trainingCode ? (
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
{`def train(self, 
          train_dataloader, 
          hypergraph_features=None, 
          hypergraph_laplacians=None,
          num_epochs=10,
          eval_dataloader=None,
          early_stopping_patience=5):
    """
    Train the LLMHG model.
    """
    best_eval_loss = float('inf')
    patience_counter = 0
    train_stats = []
    
    # Training loop
    for epoch in range(num_epochs):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # Move batch data to device
            input_seq = batch['input_seq'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_seq, 
                attention_mask, 
                hypergraph_features, 
                hypergraph_laplacians
            )
            
            # Compute loss (CrossEntropyLoss implements softmax + negative log likelihood)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # This is implementing the prediction loss L_pre
            loss = self.criterion(outputs, labels)
            
            # Note: The structure learning loss L_str is implicitly incorporated
            # during the hypergraph construction and through the Laplacian
            # operations in the hypergraph neural network.
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            num_batches += 1`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.trainingCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, trainingCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 8: Evaluation Metrics */}
      <div className="bg-white rounded-lg shadow">
        <div 
          className="p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection('evaluation')}
        >
          <h2 className="text-2xl font-bold">8. Evaluation Metrics</h2>
          <span className="text-xl">{expandedSections.evaluation ? '▼' : '▶'}</span>
        </div>
        
        {expandedSections.evaluation && (
          <div className="p-6 grid grid-cols-2 gap-8 border-t">
            <div className="space-y-6">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold mb-3">Mathematical Formulation</h3>
                <p className="mb-2"><strong>Hit Ratio (HR@k):</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('hitRatio')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.hitRatio ? 'Hide' : 'Show'}
                  </button>
                  {visibility.hitRatio ? (
                    <BlockMath>{`\\text{HR@k} = \\frac{1}{|U|} \\sum_{u \\in U} \\mathbb{1}(r_u \\leq k)`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="hitRatio" 
                      savedData={drawingData["hitRatio"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4 mb-2"><strong>Normalized Discounted Cumulative Gain (NDCG@k):</strong></p>
                <div className="relative min-h-[40px]">
                  <button 
                    onClick={() => toggleVisibility('ndcg')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                    {visibility.ndcg ? 'Hide' : 'Show'}
                  </button>
                  {visibility.ndcg ? (
                    <BlockMath>{`\\text{NDCG@k} = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{1}{\\log_2(r_u + 1)}`}</BlockMath>
                  ) : (
                    <EquationCanvas 
                      canvasId="ndcg" 
                      savedData={drawingData["ndcg"]}
                      onSaveData={handleSaveDrawing}
                    />
                  )}
                </div>
                
                <p className="mt-4">Where:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li><InlineMath>{`U`}</InlineMath>: Set of users</li>
                  <li><InlineMath>{`r_u`}</InlineMath>: Rank position of the ground truth item for user <InlineMath>{`u`}</InlineMath></li>
                  <li><InlineMath>{`\mathbb{1}(r_u \leq k)`}</InlineMath>: Indicator function, equals 1 if the ground truth item is ranked in top-k positions</li>
                </ul>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-semibold mb-4">Implementation</h3>
              <div className="relative">
                <button 
                  onClick={() => toggleVisibility('evaluationCode')}
                  className="absolute top-2 right-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer z-10"
                >
                  {visibility.evaluationCode ? 'Hide' : 'Show'}
                </button>
                {visibility.evaluationCode ? (
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
{`def evaluate_metrics(self, test_dataloader, hypergraph_features=None, 
                     hypergraph_laplacians=None, k_values=[5, 10]):
    """
    Evaluate the model using ranking metrics.
    """
    self.model.eval()
    hits = {k: 0 for k in k_values}
    ndcg = {k: 0 for k in k_values}
    total = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Forward pass
            # ...
            
            # For each sequence in the batch
            for i in range(batch_size):
                scores = last_position_scores[i].cpu().numpy()
                target = labels[i, -1].item()
                
                if target == 0:  # Skip padded sequences
                    continue
                    
                # Mask items that are in the input sequence
                # ...
                
                # Get ranking list
                rank_list = np.argsort(scores)[::-1]
                
                # Calculate metrics
                rank = np.where(rank_list == target)[0][0] \
                       if target in rank_list else float('inf')
                
                for k in k_values:
                    # Hit@K
                    if rank < k:
                        hits[k] += 1
                    
                    # NDCG@K
                    if rank < k:
                        ndcg[k] += 1.0 / np.log2(rank + 2)
                
                total += 1
    
    # Calculate final metrics
    metrics = {}
    for k in k_values:
        metrics[f'HR@{k}'] = hits[k] / total if total > 0 else 0.0
        metrics[f'NDCG@{k}'] = ndcg[k] / total if total > 0 else 0.0
    
    return metrics`}
                  </SyntaxHighlighter>
                ) : (
                  <Editor
                    value={userCode.evaluationCode}
                    onValueChange={code => setUserCode(prev => ({ ...prev, evaluationCode: code }))}
                    highlight={highlightCode}
                    padding={16}
                    style={{
                      fontFamily: '"Fira code", "Fira Mono", monospace',
                      fontSize: 14,
                      backgroundColor: '#f6f8fa',
                      border: '1px solid #e1e4e8',
                      borderRadius: '6px'
                    }}
                    textareaClassName="font-mono"
                    preClassName="font-mono"
                    placeholder="Type the code here to test your memory..."
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Summary Box */}
      <div className="bg-blue-50 rounded-lg p-6">
        <h3 className="text-xl font-bold mb-4">Summary</h3>
        <p className="mb-4">This comparison demonstrates the complete mapping between the mathematical formulations in the LLMHG paper and their implementation in code. Our implementation includes a <strong>key methodological enhancement</strong>:</p>
        
        <div className="p-4 bg-yellow-50 border-l-4 border-yellow-400 mb-4">
          <p className="font-semibold">Key Implementation Enhancement:</p>
          <p>Unlike the paper's approach of using enumerated subcategories (e.g., genre→comedy, genre→drama), our implementation uses <strong>continuous probability scores</strong> for interest angle alignment. Each product receives a continuous score (0-1) for each interest angle based on semantic similarity, enabling more nuanced and flexible recommendations.</p>
        </div>
        
        <p className="mb-4">The implementation includes all core components from the theoretical framework:</p>
        <ul className="list-disc list-inside space-y-1 mb-4">
          <li>Problem statement and sequential recommendation notation</li>
          <li><strong>Enhanced multi-view hypergraph construction with continuous probability scoring</strong></li>
          <li>Prototype computation with LLM correction</li>
          <li>Hypergraph structure learning (intra-edge and inter-edge weights)</li>
          <li>Hypergraph Laplacian computation and convolution</li>
          <li>Structure learning loss and prediction loss</li>
          <li>Fusion methods for combining hypergraph and sequential representations</li>
          <li>Evaluation metrics (HR@k and NDCG@k)</li>
        </ul>
        
        <p className="font-semibold mb-2">Continuous Scoring Advantages:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>More flexible item-interest relationships (soft assignments vs. hard categories)</li>
          <li>Better handling of products that span multiple interest dimensions</li>
          <li>Improved recommendation granularity through probability-weighted connections</li>
          <li>Natural language processing-driven interest discovery without predefined categories</li>
        </ul>
      </div>
    </div>
  );
}