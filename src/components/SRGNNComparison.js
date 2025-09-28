import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription, SubHeader } from './ComparisonLayout';
import '../styles/ComparisonLayout.css';

export default function SRGNNComparison() {
  // Initialize visibility state for all equations and code blocks (all visible by default)
  const [visibility, setVisibility] = useState({
    // Session Graph Construction section
    sessionGraphCode: true,
    sessionGraphEq: true,
    connectionMatrixEq: true,
    // Gated Graph Neural Network section
    gatedGNNCode: true,
    nodeUpdateEq: true,
    updateGateEq: true,
    resetGateEq: true,
    candidateStateEq: true,
    finalStateEq: true,
    // Session Embeddings section
    sessionEmbeddingCode: true,
    localEmbeddingEq: true,
    globalEmbeddingEq: true,
    attentionEq: true,
    hybridEmbeddingEq: true,
    // Prediction section
    predictionCode: true,
    scoreEq: true,
    softmaxEq: true,
    lossEq: true,
    // Complete Model section
    completeModelCode: true,
    // Training section
    trainingCode: true
  });

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Initialize user code state
  const [userCode, setUserCode] = useState({
    sessionGraphCode: '',
    gatedGNNCode: '',
    sessionEmbeddingCode: '',
    predictionCode: '',
    completeModelCode: '',
    trainingCode: ''
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

  const highlightCode = code => highlight(code, languages.python);

  const sessionGraphCode = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SessionGraph:
    def __init__(self, session_sequence):
        """
        Convert session sequence to directed graph
        Paper: "each session sequence can be modeled as a directed graph"
        
        Args:
            session_sequence: list of item IDs in chronological order
        """
        self.sequence = session_sequence
        self.unique_items = list(set(session_sequence))
        self.n_nodes = len(self.unique_items)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.unique_items)}
        
        # Build adjacency matrices for incoming and outgoing edges
        self.build_graph()
    
    def build_graph(self):
        """Build directed graph from session sequence"""
        # Initialize adjacency matrices
        self.A_out = np.zeros((self.n_nodes, self.n_nodes))
        self.A_in = np.zeros((self.n_nodes, self.n_nodes))
        
        # Count transitions between items
        edge_counts = {}
        for i in range(len(self.sequence) - 1):
            src = self.item_to_idx[self.sequence[i]]
            dst = self.item_to_idx[self.sequence[i + 1]]
            edge = (src, dst)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        # Normalize by outdegree (as per paper)
        for (src, dst), count in edge_counts.items():
            # Calculate outdegree for source node
            src_item = self.unique_items[src]
            outdegree = sum(1 for i in range(len(self.sequence) - 1) 
                          if self.sequence[i] == src_item)
            
            # Normalized weight
            weight = count / outdegree if outdegree > 0 else 0
            self.A_out[src, dst] = weight
            self.A_in[dst, src] = weight
    
    def get_connection_matrix(self):
        """
        Get connection matrix A_s as concatenation of A_out and A_in
        Paper: "A_s is defined as the concatenation of two adjacency matrices"
        """
        return np.concatenate([self.A_out, self.A_in], axis=1)`;

  const gatedGNNCode = `class GatedGraphNeuralNetwork(nn.Module):
    def __init__(self, n_items, embedding_dim, n_layers=1):
        """
        Gated Graph Neural Network for session graphs
        Paper equations 1-5 implementation
        """
        super(GatedGraphNeuralNetwork, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # GRU parameters for graph propagation
        self.W_z = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.U_z = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.W_r = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.U_r = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.W_o = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.U_o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Transformation matrix H from paper
        self.H = nn.Linear(2 * embedding_dim, embedding_dim, bias=True)
        
    def forward(self, item_indices, A_s):
        """
        Forward propagation through gated GNN
        Implements equations 1-5 from the paper
        
        Args:
            item_indices: indices of items in the session graph
            A_s: connection matrix (n_nodes x 2*n_nodes)
        """
        # Get initial embeddings
        v = self.item_embeddings(item_indices)  # [n_nodes, embedding_dim]
        
        # Propagate through layers
        for t in range(self.n_layers):
            v = self.propagate(v, A_s)
        
        return v
    
    def propagate(self, v, A_s):
        """
        One step of propagation (equations 1-5)
        """
        n_nodes = v.shape[0]
        
        # Equation 1: Aggregate neighbor information
        # a_s,i = A_s,i: @ [v_1, ..., v_n]^T @ H + b
        v_concat = torch.cat([v, v], dim=0)  # [2*n_nodes, embedding_dim]
        a = torch.matmul(A_s, v_concat)  # [n_nodes, embedding_dim]
        a = self.H(torch.cat([a, v], dim=1))  # [n_nodes, embedding_dim]
        
        # Equation 2: Update gate
        # z_s,i = σ(W_z @ a_s,i + U_z @ v_i)
        z = torch.sigmoid(self.W_z(torch.cat([a, v], dim=1)) + self.U_z(v))
        
        # Equation 3: Reset gate  
        # r_s,i = σ(W_r @ a_s,i + U_r @ v_i)
        r = torch.sigmoid(self.W_r(torch.cat([a, v], dim=1)) + self.U_r(v))
        
        # Equation 4: Candidate state
        # ṽ_i = tanh(W_o @ a_s,i + U_o @ (r_s,i ⊙ v_i))
        v_tilde = torch.tanh(self.W_o(torch.cat([a, v], dim=1)) + 
                            self.U_o(r * v))
        
        # Equation 5: Final state
        # v_i = (1 - z_s,i) ⊙ v_i + z_s,i ⊙ ṽ_i
        v_new = (1 - z) * v + z * v_tilde
        
        return v_new`;

  const sessionEmbeddingCode = `class SessionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        """
        Generate session embeddings combining global and local preferences
        Paper Section 3.4: Generating Session Embeddings
        """
        super(SessionEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Attention mechanism parameters
        self.W_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.q = nn.Parameter(torch.randn(embedding_dim))
        self.c = nn.Parameter(torch.randn(embedding_dim))
        
        # Hybrid embedding transformation
        self.W_3 = nn.Linear(2 * embedding_dim, embedding_dim)
        
    def forward(self, node_vectors, session_items, last_item_idx):
        """
        Compute session embedding from node vectors
        
        Args:
            node_vectors: GNN output vectors for all nodes [n_nodes, embedding_dim]
            session_items: indices mapping nodes to session sequence
            last_item_idx: index of last clicked item
        """
        # Local embedding: simply the last clicked item
        # s_l = v_n
        s_l = node_vectors[last_item_idx]
        
        # Global embedding with soft attention
        # α_i = q^T σ(W_1 @ v_n + W_2 @ v_i + c)
        v_n = node_vectors[last_item_idx].unsqueeze(0)
        
        # Compute attention scores
        scores = []
        for i in range(node_vectors.shape[0]):
            v_i = node_vectors[i]
            score = torch.matmul(
                self.q,
                torch.sigmoid(self.W_1(v_n).squeeze() + self.W_2(v_i) + self.c)
            )
            scores.append(score)
        
        # Normalize attention scores
        alpha = F.softmax(torch.stack(scores), dim=0)
        
        # Global preference: weighted sum
        # s_g = Σ α_i @ v_i
        s_g = torch.sum(alpha.unsqueeze(1) * node_vectors, dim=0)
        
        # Hybrid embedding
        # s_h = W_3 @ [s_l; s_g]
        s_h = self.W_3(torch.cat([s_l, s_g]))
        
        return s_h, s_l, s_g`;

  const predictionCode = `class Predictor(nn.Module):
    def __init__(self, n_items, embedding_dim):
        """
        Make next-item predictions
        Paper Section 3.5: Making Recommendation and Model Training
        """
        super(Predictor, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Share embeddings with GNN
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
    def forward(self, session_embedding, candidate_items=None):
        """
        Compute scores for next item prediction
        
        Args:
            session_embedding: hybrid session representation s_h
            candidate_items: specific items to score (None = all items)
        """
        # Get all item embeddings or specific candidates
        if candidate_items is None:
            item_embs = self.item_embeddings.weight  # [n_items, embedding_dim]
        else:
            item_embs = self.item_embeddings(candidate_items)
        
        # Compute scores: ẑ_i = s_h^T @ v_i
        scores = torch.matmul(session_embedding, item_embs.t())
        
        # Apply softmax to get probabilities
        # ŷ = softmax(ẑ)
        probs = F.softmax(scores, dim=-1)
        
        return scores, probs
    
    def compute_loss(self, scores, target):
        """
        Cross-entropy loss for next-item prediction
        L(ŷ) = -Σ y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)
        
        Args:
            scores: unnormalized prediction scores
            target: ground truth next item (index)
        """
        return F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))`;

  const completeModelCode = `class SRGNN(nn.Module):
    def __init__(self, n_items, embedding_dim, n_layers=1):
        """
        Complete SR-GNN model
        Session-based Recommendation with Graph Neural Networks
        """
        super(SRGNN, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Components
        self.gnn = GatedGraphNeuralNetwork(n_items, embedding_dim, n_layers)
        self.session_embedding = SessionEmbedding(embedding_dim)
        self.predictor = Predictor(n_items, embedding_dim)
        
        # Share embeddings between GNN and Predictor
        self.predictor.item_embeddings = self.gnn.item_embeddings
        
    def forward(self, session_sequence):
        """
        Forward pass through SR-GNN
        
        Args:
            session_sequence: list of item IDs in chronological order
        
        Returns:
            scores: prediction scores for all items
            probs: probability distribution over items
        """
        # Step 1: Build session graph
        graph = SessionGraph(session_sequence)
        unique_items = graph.unique_items
        item_indices = torch.LongTensor([self.item_to_idx(item) 
                                        for item in unique_items])
        
        # Get connection matrix
        A_s = torch.FloatTensor(graph.get_connection_matrix())
        
        # Step 2: Get node vectors through GNN
        node_vectors = self.gnn(item_indices, A_s)
        
        # Step 3: Generate session embedding
        last_item_idx = graph.item_to_idx[session_sequence[-1]]
        session_emb, _, _ = self.session_embedding(
            node_vectors, 
            session_sequence,
            last_item_idx
        )
        
        # Step 4: Make predictions
        scores, probs = self.predictor(session_emb)
        
        return scores, probs
    
    def item_to_idx(self, item_id):
        """Map item ID to embedding index"""
        # In practice, you'd have a global item-to-index mapping
        return item_id`;

  const trainingCode = `# Training SR-GNN model
def train_srgnn(model, train_data, epochs=30, batch_size=100, lr=0.001, l2_reg=1e-5):
    """
    Train SR-GNN model on session data
    Paper uses: learning rate 0.001, batch size 100, L2 penalty 10^-5
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Process sessions in batches
        for batch_idx in range(0, len(train_data), batch_size):
            batch_sessions = train_data[batch_idx:batch_idx + batch_size]
            batch_loss = 0.0
            
            for session in batch_sessions:
                # Generate training sequences from session
                # For session [v1, v2, v3, v4], generate:
                # ([v1], v2), ([v1, v2], v3), ([v1, v2, v3], v4)
                for i in range(1, len(session)):
                    input_seq = session[:i]
                    target = session[i]
                    
                    # Forward pass
                    scores, _ = model(input_seq)
                    
                    # Compute loss
                    loss = model.predictor.compute_loss(scores, torch.tensor(target))
                    batch_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
        
        # Learning rate decay
        if (epoch + 1) % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}')

# Evaluation metrics
def evaluate_srgnn(model, test_data, k=20):
    """
    Evaluate using P@20 and MRR@20
    """
    model.eval()
    total_precision = 0.0
    total_mrr = 0.0
    
    with torch.no_grad():
        for session in test_data:
            # Use all but last item as input
            input_seq = session[:-1]
            target = session[-1]
            
            # Get predictions
            scores, _ = model(input_seq)
            
            # Get top-k items
            _, top_k_items = torch.topk(scores, k)
            
            # Calculate P@k
            if target in top_k_items:
                total_precision += 1.0
                
                # Calculate MRR
                rank = (top_k_items == target).nonzero()[0].item() + 1
                total_mrr += 1.0 / rank
    
    precision = total_precision / len(test_data)
    mrr = total_mrr / len(test_data)
    
    return precision, mrr

# Example usage
if __name__ == "__main__":
    # Dataset parameters (e.g., Yoochoose)
    n_items = 37483
    embedding_dim = 100
    n_gnn_layers = 1
    
    # Create model
    model = SRGNN(n_items, embedding_dim, n_gnn_layers)
    
    # Dummy data for illustration
    train_sessions = [
        [102, 243, 556, 102, 789],  # Example session
        [556, 102, 243, 987, 556],
        # ... more sessions
    ]
    
    # Train model
    train_srgnn(model, train_sessions, epochs=30)
    
    # Evaluate
    test_sessions = [
        [243, 556, 789, 102],
        [987, 102, 556, 243],
        # ... more test sessions
    ]
    
    p_at_20, mrr_at_20 = evaluate_srgnn(model, test_sessions)
    print(f"P@20: {p_at_20:.4f}, MRR@20: {mrr_at_20:.4f}")`;

  return (
    <ComparisonLayout
      title="Session-based Recommendation with Graph Neural Networks (SR-GNN)"
      description="Based on 'Session-based Recommendation with Graph Neural Networks' by Wu et al. (AAAI 2019). This component shows the exact equations and formulations from the paper alongside their implementations."
    >
      {/* Section 1: Session Graph Construction */}
      <ComparisonSection
        title="Session Graph Construction"
        leftContent={
          <>
            <p className="mb-4">A session sequence <InlineMath>{"s = [v_{s,1}, v_{s,2}, ..., v_{s,l}]"}</InlineMath> is transformed into a directed graph:</p>
            <EquationBlock 
              id="sessionGraphEq"
              math="\mathcal{G}_s = (\mathcal{V}_s, \mathcal{E}_s)"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{V}_s"}</InlineMath>: set of unique items in session <InlineMath>{"s"}</InlineMath></li>
              <li><InlineMath>{"\\mathcal{E}_s"}</InlineMath>: set of directed edges <InlineMath>{"(v_{s,i-1}, v_{s,i})"}</InlineMath></li>
              <li><InlineMath>{"n = |\\mathcal{V}_s|"}</InlineMath>: number of unique items in the session</li>
              <li><InlineMath>{"l"}</InlineMath>: length of the session sequence</li>
              <li>Edge weights: normalized by occurrence divided by outdegree</li>
            </ul>
            
            <p className="mt-6 mb-4">Initial node embeddings:</p>
            <EquationBlock 
              id="sessionGraphEq"
              math="\mathbf{v}_i^0 \in \mathbb{R}^d, \quad i = 1, ..., n"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{v}_i^0"}</InlineMath>: initial embedding vector for node/item <InlineMath>{"i"}</InlineMath></li>
              <li><InlineMath>{"d"}</InlineMath>: embedding dimension</li>
            </ul>
            
            <p className="mt-6 mb-4">Connection matrix <InlineMath>{"\\mathbf{A}_s"}</InlineMath>:</p>
            <EquationBlock 
              id="connectionMatrixEq"
              math="\mathbf{A}_s = [\mathbf{A}_s^{\text{(out)}}, \mathbf{A}_s^{\text{(in)}}] \in \mathbb{R}^{n \times 2n}"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{A}_s^{\\text{(out)}} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: outgoing edges adjacency matrix</li>
              <li><InlineMath>{"\\mathbf{A}_s^{\\text{(in)}} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: incoming edges adjacency matrix</li>
              <li><InlineMath>{"\\mathbf{A}_{s,i:}"}</InlineMath>: <InlineMath>{"i"}</InlineMath>-th row of <InlineMath>{"\\mathbf{A}_s"}</InlineMath></li>
            </ul>
            
            <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Example:</strong> Session [v₁, v₂, v₃, v₂, v₄] creates edges:
                (v₁→v₂), (v₂→v₃), (v₃→v₂), (v₂→v₄) with normalized weights.
              </p>
            </div>
          </>
        }
        leftTitle="Graph Structure Definition"
        code={sessionGraphCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="sessionGraphEq"
        rightId="sessionGraphCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        editorPlaceholder="Type the session graph code here..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="sessionGraph"
      />

      {/* Section 2: Gated Graph Neural Network */}
      <ComparisonSection
        title="Gated Graph Neural Network"
        leftContent={
          <>
            <p className="mb-4">The Gated GNN updates node embeddings through <InlineMath>{"T"}</InlineMath> layers:</p>
            
            <FormulaWithDescription
              id="nodeUpdateEq"
              title="Node representation update (Equation 1):"
              math="\mathbf{a}^t_{s,i} = \mathbf{A}_{s,i:} [\mathbf{v}^{t-1}_1, \dots ,\mathbf{v}^{t-1}_n]^\top \mathbf{H} + \mathbf{b}"
            />
            
            <FormulaWithDescription
              id="updateGateEq"
              title="Update gate (Equation 2):"
              math="\mathbf{z}^t_{s,i} = \sigma(\mathbf{W}_z\mathbf{a}^t_{s,i}+\mathbf{U}_z\mathbf{v}^{t-1}_{i})"
            />
            
            <FormulaWithDescription
              id="resetGateEq"
              title="Reset gate (Equation 3):"
              math="\mathbf{r}^t_{s,i} = \sigma(\mathbf{W}_r\mathbf{a}^t_{s,i}+\mathbf{U}_r\mathbf{v}^{t-1}_{i})"
            />
            
            <FormulaWithDescription
              id="candidateStateEq"
              title="Candidate state (Equation 4):"
              math="\widetilde{\mathbf{v}^t_{i}} = \tanh(\mathbf{W}_o \mathbf{a}^t_{s,i}+\mathbf{U}_o (\mathbf{r}^t_{s,i} \odot \mathbf{v}^{t-1}_{i}))"
            />
            
            <FormulaWithDescription
              id="finalStateEq"
              title="Final state (Equation 5):"
              math="\mathbf{v}^t_{i} = (1-\mathbf{z}^t_{s,i}) \odot \mathbf{v}^{t-1}_{i} + \mathbf{z}^t_{s,i} \odot \widetilde{\mathbf{v}^t_{i}}"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"t = 1, ..., T"}</InlineMath>: layer index (typically <InlineMath>{"T = 1"}</InlineMath>)</li>
              <li><InlineMath>{"\\mathbf{v}^{t}_{i} \\in \\mathbb{R}^d"}</InlineMath>: node <InlineMath>{"i"}</InlineMath> embedding at layer <InlineMath>{"t"}</InlineMath></li>
              <li><InlineMath>{"\\mathbf{a}^t_{s,i} \\in \\mathbb{R}^d"}</InlineMath>: aggregated neighbor information</li>
              <li><InlineMath>{"\\mathbf{H} \\in \\mathbb{R}^{2d \\times d}"}</InlineMath>: transformation matrix</li>
              <li><InlineMath>{"\\mathbf{b} \\in \\mathbb{R}^d"}</InlineMath>: bias vector</li>
              <li><InlineMath>{"\\mathbf{W}_z, \\mathbf{W}_r, \\mathbf{W}_o \\in \\mathbb{R}^{d \\times 2d}"}</InlineMath>: weight matrices for gates</li>
              <li><InlineMath>{"\\mathbf{U}_z, \\mathbf{U}_r, \\mathbf{U}_o \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: recurrent weight matrices</li>
              <li><InlineMath>{"\\mathbf{z}^t_{s,i}, \\mathbf{r}^t_{s,i} \\in \\mathbb{R}^d"}</InlineMath>: update and reset gates</li>
              <li><InlineMath>{"\\widetilde{\\mathbf{v}^t_{i}} \\in \\mathbb{R}^d"}</InlineMath>: candidate hidden state</li>
              <li><InlineMath>{"\\odot"}</InlineMath>: element-wise (Hadamard) multiplication</li>
              <li><InlineMath>{"\\sigma"}</InlineMath>: sigmoid activation function</li>
              <li><InlineMath>{"\\tanh"}</InlineMath>: hyperbolic tangent activation</li>
            </ul>
          </>
        }
        leftTitle="Node Update Equations"
        code={gatedGNNCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="nodeUpdateEq"
        rightId="gatedGNNCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        editorPlaceholder="Type the gated GNN code here..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="gatedGNN"
      />

      {/* Section 3: Session Embeddings */}
      <ComparisonSection
        title="Session Embeddings"
        leftContent={
          <>
            <p className="mb-4">After <InlineMath>{"T"}</InlineMath> GNN layers, we obtain final node embeddings <InlineMath>{"\\mathbf{v}_i = \\mathbf{v}_i^T"}</InlineMath> for each node.</p>
            
            <FormulaWithDescription
              id="localEmbeddingEq"
              title="Local embedding (last-clicked item):"
              math="\mathbf{s}_{\text{l}} = \mathbf{v}_{n}"
            />
            <p className="mt-2 text-sm">where <InlineMath>{"n"}</InlineMath> is the index of the last item in the session sequence</p>
            
            <FormulaWithDescription
              id="attentionEq"
              title="Attention weights:"
              math="\alpha_i = \mathbf{q}^{\top} \sigma(\mathbf{W}_1 \mathbf{v}_{n} + \mathbf{W}_2 \mathbf{v}_{i} + \mathbf{c})"
            />
            
            <FormulaWithDescription
              id="attentionEq"
              title="Normalized attention:"
              math="\alpha_i = \frac{\exp(\alpha_i)}{\sum_{j=1}^{n} \exp(\alpha_j)}"
            />
            
            <FormulaWithDescription
              id="globalEmbeddingEq"
              title="Global embedding (weighted sum):"
              math="\mathbf{s}_{\text{g}} = \sum_{i = 1}^{n} {\alpha_i \mathbf{v}_{i}}"
            />
            
            <FormulaWithDescription
              id="hybridEmbeddingEq"
              title="Hybrid embedding:"
              math="\mathbf{s}_{\text{h}} = \mathbf{W}_3 [\mathbf{s}_{\text{l}}; \mathbf{s}_{\text{g}}]"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{v}_i \\in \\mathbb{R}^d"}</InlineMath>: final embedding of node <InlineMath>{"i"}</InlineMath> from GNN</li>
              <li><InlineMath>{"\\mathbf{s}_{\\text{l}} \\in \\mathbb{R}^d"}</InlineMath>: local embedding (current interest)</li>
              <li><InlineMath>{"\\mathbf{s}_{\\text{g}} \\in \\mathbb{R}^d"}</InlineMath>: global embedding (general preference)</li>
              <li><InlineMath>{"\\mathbf{s}_{\\text{h}} \\in \\mathbb{R}^d"}</InlineMath>: hybrid embedding combining both</li>
              <li><InlineMath>{"\\mathbf{q} \\in \\mathbb{R}^d"}</InlineMath>: learnable attention query vector</li>
              <li><InlineMath>{"\\mathbf{c} \\in \\mathbb{R}^d"}</InlineMath>: learnable attention bias</li>
              <li><InlineMath>{"\\mathbf{W}_1, \\mathbf{W}_2 \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: attention weight matrices</li>
              <li><InlineMath>{"\\mathbf{W}_3 \\in \\mathbb{R}^{d \\times 2d}"}</InlineMath>: transformation matrix for hybrid embedding</li>
              <li><InlineMath>{"[\\cdot;\\cdot]"}</InlineMath>: concatenation operation</li>
              <li><InlineMath>{"\\alpha_i"}</InlineMath>: attention weight for node <InlineMath>{"i"}</InlineMath></li>
            </ul>
            
            <div className="mt-6 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Key Innovation:</strong> Combines immediate interest (last item) with overall session context 
                using attention mechanism to capture both short-term and session-level preferences.
              </p>
            </div>
          </>
        }
        leftTitle="Embedding Formulation"
        code={sessionEmbeddingCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="localEmbeddingEq"
        rightId="sessionEmbeddingCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        editorPlaceholder="Type the session embedding code here..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="sessionEmbedding"
      />

      {/* Section 4: Prediction and Training */}
      <ComparisonSection
        title="Making Predictions"
        leftContent={
          <>
            <FormulaWithDescription
              id="scoreEq"
              title="Recommendation score for each item:"
              math="\hat{z}_i = \mathbf{s}_{\text{h}}^{\top} \mathbf{v}_i"
            />
            <p className="mt-2 text-sm">where <InlineMath>{"\\mathbf{v}_i"}</InlineMath> is the embedding of candidate item <InlineMath>{"i"}</InlineMath> from the global item embedding matrix</p>
            
            <FormulaWithDescription
              id="scoreEq"
              title="Score vector for all items:"
              math="\hat{\mathbf{z}} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_m]^\top \in \mathbb{R}^m"
            />
            
            <FormulaWithDescription
              id="softmaxEq"
              title="Probability distribution:"
              math="\hat{\mathbf{y}} = \text{softmax}(\hat{\mathbf{z}}) = \frac{\exp(\hat{\mathbf{z}})}{\sum_{j=1}^{m} \exp(\hat{z}_j)}"
            />
            
            <FormulaWithDescription
              id="lossEq"
              title="Cross-entropy loss function:"
              math="\mathcal{L}(\hat{\mathbf{y}}) = -\sum_{i = 1}^{m} \mathbf{y}_i \log(\hat{\mathbf{y}_i}) + (1 - \mathbf{y}_i) \log(1 - \hat{\mathbf{y}_i})"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"m"}</InlineMath>: total number of items in the catalog</li>
              <li><InlineMath>{"\\hat{z}_i \\in \\mathbb{R}"}</InlineMath>: score for item <InlineMath>{"i"}</InlineMath></li>
              <li><InlineMath>{"\\hat{\\mathbf{z}} \\in \\mathbb{R}^m"}</InlineMath>: scores for all <InlineMath>{"m"}</InlineMath> items</li>
              <li><InlineMath>{"\\hat{\\mathbf{y}} \\in \\mathbb{R}^m"}</InlineMath>: predicted probability distribution</li>
              <li><InlineMath>{"\\mathbf{y} \\in \\{0, 1\\}^m"}</InlineMath>: one-hot encoded ground truth (next item)</li>
              <li><InlineMath>{"\\mathbf{v}_i \\in \\mathbb{R}^d"}</InlineMath>: item embedding from global item embedding matrix</li>
            </ul>
            
            <div className="mt-6 p-4 bg-purple-50 border-l-4 border-purple-400">
              <p className="text-sm">
                <strong>Evaluation Metrics:</strong>
                <ul className="list-disc pl-5 mt-2">
                  <li><strong>P@K:</strong> Precision at K = fraction of test sessions where the ground truth item appears in top-K predictions</li>
                  <li><strong>MRR@K:</strong> Mean Reciprocal Rank = average of 1/rank of the ground truth item (0 if not in top-K)</li>
                </ul>
              </p>
            </div>
          </>
        }
        leftTitle="Prediction Formulation"
        code={predictionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="scoreEq"
        rightId="predictionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        editorPlaceholder="Type the prediction code here..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="prediction"
      />

      {/* Section 5: Complete Model */}
      <ComparisonSection
        title="Complete SR-GNN Model"
        leftTitle="Architecture Overview"
        code={completeModelCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="completeModelCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="completeModelCode"
        editorPlaceholder="Type the complete model code here..."
        fullWidth={true}
      />

      {/* Section 6: Training */}
      <ComparisonSection
        title="Training SR-GNN"
        leftTitle="Training Strategy"
        code={trainingCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="trainingCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="trainingCode"
        editorPlaceholder="Type the training code here..."
        fullWidth={true}
      />

      {/* Summary */}
      <div className="mt-12 p-6 bg-gradient-to-r from-blue-100 to-purple-100 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Key Contributions from the Paper</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">1. Model Architecture</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Graph-based modeling of session sequences</li>
              <li>Gated GNN for capturing item transitions</li>
              <li>Attention-based session embedding combining local and global preferences</li>
              <li>No user representations needed (session-based only)</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">2. Experimental Results</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Yoochoose 1/64: P@20=70.57%, MRR@20=30.94%</li>
              <li>Yoochoose 1/4: P@20=71.36%, MRR@20=31.89%</li>
              <li>Diginetica: P@20=50.73%, MRR@20=17.59%</li>
              <li>Outperforms RNN-based methods (GRU4REC, NARM, STAMP)</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="font-semibold mb-2">3. Key Insights</h3>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>Complex item transitions captured through graph structure outperform sequential models</li>
            <li>Combining immediate interest with session context improves recommendations</li>
            <li>Graph neural networks can effectively model session-based recommendation without user profiles</li>
            <li>Normalized edge weights by outdegree improve model stability</li>
          </ul>
        </div>
        
        <p className="text-sm mt-4 text-center italic">
          "SR-GNN evidently outperforms the state-of-the-art session-based recommendation methods consistently."
        </p>
      </div>
    </ComparisonLayout>
  );
}