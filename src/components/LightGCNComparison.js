import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock } from './ComparisonLayout';

export default function LightGCNComparison() {
  // Initialize visibility state for all equations and code blocks (all visible by default)
  const [visibility, setVisibility] = useState({
    // LightGCN section
    lightGCNCode: true,
    lightGCNEq1: true,
    lightGCNEq2: true,
    // GAT Recommendation section
    gatRecommenderCode: true,
    gatMultiHeadEq: true,
    // GAT Attention Mechanism section
    gatConvCode: true,
    gatLinearTransformEq: true,
    gatAttentionScoresEq: true,
    gatNormalizationEq: true,
    gatFeatureAggregationEq: true
  });

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Initialize user code state
  const [userCode, setUserCode] = useState({
    lightGCNCode: '',
    gatRecommenderCode: '',
    gatConvCode: ''
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

  // Custom highlight function that ensures proper syntax highlighting
  const highlightCode = (code) => {
    try {
      return highlight(code, languages.python, 'python');
    } catch (e) {
      return code; // Return plain code if highlighting fails
    }
  };

  const lightGCNCode = `class LightGCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv = LGConv()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        x = self.embedding.weight
        layer_embeddings = [x]
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index)
            layer_embeddings.append(x)
        return torch.stack(layer_embeddings, dim=0).mean(dim=0)

    def decode(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=-1)`;

  const gatRecommenderCode = `class GATRecommender(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels=64, num_layers=2, heads=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)
        
        # Multiple GAT layers
        self.convs = torch.nn.ModuleList()
        
        # First layer: hidden_channels → hidden_channels * heads
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads))
        
        # Middle layers maintain dimension: hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, 
                                    hidden_channels * heads, 
                                    heads))
        
        # Last layer: hidden_channels * heads → hidden_channels
        self.convs.append(GATConv(hidden_channels * heads, 
                                hidden_channels, 
                                heads=1))

    def forward(self, edge_index):
        x = self.embedding.weight
        
        # Apply GAT layers with residual connections
        for i, conv in enumerate(self.convs):
            x_residual = x
            x = conv(x, edge_index)  # Apply attention
            if i != len(self.convs) - 1:
                x = F.elu(x)         # Non-linearity
            
            # Residual connection if dimensions match
            if x_residual.size(-1) == x.size(-1):
                x = x + x_residual
        
        return x`;

  const gatConvCode = `class GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.heads = heads
        # Linear transformation (Equation 1)
        self.lin = torch.nn.Linear(in_channels, heads * out_channels)
        # Attention mechanism (Equation 2)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

    def forward(self, x, edge_index):
        # 1. Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # 2. Compute attention scores
        x_i = x[edge_index[0]]  # Source nodes
        x_j = x[edge_index[1]]  # Target nodes
        alpha = torch.cat([x_i, x_j], dim=-1)  # Concatenate
        alpha = (alpha * self.att).sum(dim=-1)  # Attention scores
        
        # 3. Apply LeakyReLU and normalize
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[0])  # Normalize
        
        # 4. Weighted aggregation
        out = x_j * alpha.view(-1, self.heads, 1)
        out = scatter_add(out, edge_index[0], dim=0)

        return out.view(-1, self.heads * self.out_channels)`;

  return (
    <ComparisonLayout>
      {/* LightGCN Section */}
      <ComparisonSection
        title="LightGCN Architecture"
        leftContent={
          <>
            <p className="mb-4">
              LightGCN simplifies traditional GCNs by focusing on embedding propagation without feature transformations. The node embeddings are computed using:
            </p>
            <EquationBlock 
              math="H^{(k)} = A H^{(k-1)}"
              description=""
            />
            <p className="mt-8 mb-4">Aggregated embeddings are averaged across layers:</p>
            <EquationBlock 
              math="H = \frac{1}{K} \sum_{k=1}^{K} H^{(k)}"
              description=""
            />
          </>
        }
        code={lightGCNCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="lightGCNEq1"
        rightId="lightGCNCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="lightGCNCode"
        editorPlaceholder="Type the LightGCN code here..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="lightGCNCode"
      />

      {/* GAT Recommendation Section */}
      <ComparisonSection
        title="Multi-head Architecture"
        leftContent={
          <>
            <p className="mb-4">
              Each GAT layer transforms input features through multiple attention heads:
            </p>
            <EquationBlock 
              math="\mathbf{h}_i^{(l+1)} = \sigma \left( \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(l)} \right)"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"K"}</InlineMath>: Number of attention heads</li>
              <li><InlineMath>{"\\alpha_{ij}^k"}</InlineMath>: Attention coefficients from head <InlineMath>{"k"}</InlineMath></li>
              <li><InlineMath>{"\\mathbf{W}^k"}</InlineMath>: Weight matrix for head <InlineMath>{"k"}</InlineMath></li>
            </ul>
            
            <p className="mt-6 mb-4">Layer composition:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Input size: <InlineMath>{"d_{in}"}</InlineMath></li>
              <li>Hidden layer size: <InlineMath>{"d_{in} \\times \\text{heads}"}</InlineMath></li>
              <li>Output size: <InlineMath>{"d_{out}"}</InlineMath> (after concatenating or averaging heads)</li>
            </ul>
          </>
        }
        code={gatRecommenderCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="gatMultiHeadEq"
        rightId="gatRecommenderCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="gatRecommenderCode"
        editorPlaceholder="Type the GAT Recommender code here..."
        rightTitle="GAT Recommendation Implementation"
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="gatRecommenderCode"
      />

      {/* GAT Attention Mechanism Section */}
      <ComparisonSection
        title="Attention Mechanism"
        leftContent={
          <>
            <p className="mb-4">The attention mechanism follows four key steps:</p>

            <h3 className="text-lg font-semibold mt-4 mb-2">1. Linear Transformation</h3>
            <EquationBlock 
              math="\mathbf{h}_i' = W\mathbf{h}_i"
              description=""
            />

            <h3 className="text-lg font-semibold mt-4 mb-2">2. Attention Scores</h3>
            <EquationBlock 
              math="e_{ij} = \text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i || W\mathbf{h}_j])"
              description=""
            />

            <h3 className="text-lg font-semibold mt-4 mb-2">3. Normalization</h3>
            <EquationBlock 
              math="\alpha_{ij} = \text{softmax}_j(e_{ij})"
              description=""
            />

            <h3 className="text-lg font-semibold mt-4 mb-2">4. Feature Aggregation</h3>
            <EquationBlock 
              math="\mathbf{h}_i^{new} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}W\mathbf{h}_j"
              description=""
            />
          </>
        }
        code={gatConvCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="gatLinearTransformEq"
        rightId="gatConvCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="gatConvCode"
        editorPlaceholder="Type the GAT Conv code here..."
        rightTitle="GAT Attention Mechanism Implementation"
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="gatConvCode"
      />
    </ComparisonLayout>
  );
}