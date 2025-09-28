import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription, SubHeader } from './ComparisonLayout';
import '../styles/ComparisonLayout.css';

export default function HGNNComparison() {
  // Initialize visibility state - organized by equation-code pairs
  const [visibility, setVisibility] = useState({
    // Section 1: Incidence Matrix and Edge Weights
    incidenceMatrixEq: true,
    edgeWeightsEq: true,
    incidenceMatrixCode: true,
    // Section 2: Degrees
    vertexDegreeEq: true,
    edgeDegreeEq: true,
    degreeComputationCode: true,
    // Section 3: Laplacian
    thetaOperatorEq: true,
    laplacianDeltaEq: true,
    laplacianCode: true,
    // Section 4: Regularization (Equations 2-4)
    regularizationEq: true,
    regularizationCode: true,
    // Section 5: Spectral Theory (Equations 5-10)
    spectralConvEq: true,
    chebyshevApproxEq: true,
    firstOrderEq: true,
    chebyshevApproxCode: true,
    // Section 6: HGNN Layer (Equation 11)
    hgnnLayerEq: true,
    hgnnLayerCode: true,
    // Section 7: Session Construction
    sessionTheoryEq: true,
    sessionConstructCode: true,
    // Section 8: Attention
    standardAttentionEq1: true,
    standardAttentionEq2: true,
    lastClickAttentionEq: true,
    attentionCode: true,
    // Section 9: Complete Implementation
    completeArchitectureEq: true,
    completeModelCode: true,
    // Section 10: Training
    trainingCode: true
  });

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const [userCode, setUserCode] = useState({
    incidenceMatrixCode: '',
    degreeComputationCode: '',
    laplacianCode: '',
    regularizationCode: '',
    chebyshevApproxCode: '',
    hgnnLayerCode: '',
    sessionConstructCode: '',
    attentionCode: '',
    completeModelCode: '',
    trainingCode: ''
  });

  const [drawingData, setDrawingData] = useState({});

  const handleSaveDrawing = (canvasId, paths) => {
    setDrawingData(prev => ({
      ...prev,
      [canvasId]: paths
    }));
  };

  const highlightCode = code => highlight(code, languages.python);

  // Code implementations directly tied to equations
  const incidenceMatrixCode = `# Equation 1: Incidence Matrix and Edge Weights
def create_incidence_matrix(num_nodes, num_edges):
    """
    h(v,e) = {1 if v ∈ e, 0 if v ∉ e}
    """
    H = torch.zeros(num_nodes, num_edges)
    return H

def add_hyperedge(H, edge_idx, node_indices, weight=1.0):
    """Add hyperedge e containing vertices v with optional weight"""
    H[node_indices, edge_idx] = 1
    return H

def create_weight_matrix(edge_weights):
    """Create diagonal weight matrix W from edge weights"""
    W = torch.diag(edge_weights)
    return W`;

  const degreeComputationCode = `# Vertex and Edge Degree Formulas
def compute_degrees(H, W=None):
    """
    Vertex degree: d(v) = Σ_e∈E ω(e)h(v,e)
    Edge degree: δ(e) = Σ_v∈V h(v,e)
    """
    if W is None:
        W = torch.ones(H.size(1))
    
    # Vertex degrees: sum over hyperedges (weighted)
    D_v = torch.sum(H * W.unsqueeze(0), dim=1)
    
    # Edge degrees: sum over vertices
    D_e = torch.sum(H, dim=0)
    
    return D_v, D_e`;

  const laplacianCode = `# Hypergraph Laplacian Construction
def compute_laplacian(H, W=None):
    """
    Θ = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
    Δ = I - Θ
    """
    D_v, D_e = compute_degrees(H, W)
    
    # Compute D_v^(-1/2)
    D_v_sqrt_inv = torch.diag(1.0 / torch.sqrt(D_v + 1e-8))
    
    # Compute D_e^(-1)
    D_e_inv = torch.diag(1.0 / (D_e + 1e-8))
    
    # Weight matrix
    if W is None:
        W = torch.eye(H.size(1))
    else:
        W = torch.diag(W)
    
    # Θ = D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
    Theta = D_v_sqrt_inv @ H @ W @ D_e_inv @ H.t() @ D_v_sqrt_inv
    
    # Laplacian: Δ = I - Θ
    Delta = torch.eye(H.size(0)) - Theta
    
    return Delta, Theta`;

  const hgnnLayerCode = `# Equation 11: HGNN Layer Implementation
class HGNNLayer(nn.Module):
    """
    X^(l+1) = σ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X^(l) Θ^(l))
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.Theta = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.Theta)
    
    def forward(self, X, H, W=None):
        # Compute degrees
        D_v, D_e = compute_degrees(H, W)
        
        # Normalization terms
        D_v_sqrt_inv = 1.0 / torch.sqrt(D_v + 1e-8)
        D_e_inv = 1.0 / (D_e + 1e-8)
        
        # Apply hypergraph convolution
        # Step 1: X * Θ
        X = torch.matmul(X, self.Theta)
        
        # Step 2: D_v^(-1/2) * X
        X = X * D_v_sqrt_inv.unsqueeze(1)
        
        # Step 3: H^T * X (node to edge)
        X_e = torch.matmul(H.t(), X)
        
        # Step 4: D_e^(-1) * X_e * W
        if W is not None:
            X_e = X_e * D_e_inv.unsqueeze(1) * W.unsqueeze(1)
        else:
            X_e = X_e * D_e_inv.unsqueeze(1)
        
        # Step 5: H * X_e (edge to node)
        X = torch.matmul(H, X_e)
        
        # Step 6: D_v^(-1/2) * X
        X = X * D_v_sqrt_inv.unsqueeze(1)
        
        # Apply activation
        return F.relu(X)`;

  const chebyshevApproxCode = `# Equations 6-9: Chebyshev Approximation
def chebyshev_convolution(X, Delta, K=2, theta=None):
    """
    g★x ≈ Σ_{k=0}^K θ_k T_k(Δ̃)x
    
    First-order (K=1) simplification:
    g★x ≈ θ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2))x
    """
    if theta is None:
        theta = nn.Parameter(torch.Tensor(K+1))
        nn.init.normal_(theta)
    
    # Scaled Laplacian: Δ̃ = 2Δ/λ_max - I
    lambda_max = 2.0  # Approximation
    Delta_tilde = 2 * Delta / lambda_max - torch.eye(Delta.size(0))
    
    # Chebyshev polynomials
    T = [torch.eye(X.size(0)), Delta_tilde]
    for k in range(2, K+1):
        T.append(2 * Delta_tilde @ T[k-1] - T[k-2])
    
    # Apply filter
    output = torch.zeros_like(X)
    for k in range(K+1):
        output += theta[k] * (T[k] @ X)
    
    return output

# Simplified first-order version (Equation 9)
def simplified_hgnn_conv(X, H, W=None):
    """
    Simplified: g★x ≈ θ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2))x
    """
    D_v, D_e = compute_degrees(H, W)
    
    # Normalizations
    D_v_norm = 1.0 / torch.sqrt(D_v + 1e-8)
    D_e_inv = 1.0 / (D_e + 1e-8)
    
    # Apply convolution
    X = X * D_v_norm.unsqueeze(1)
    X = H.t() @ X  # Node to edge
    X = X * D_e_inv.unsqueeze(1)
    if W is not None:
        X = X * W.unsqueeze(1)
    X = H @ X  # Edge to node
    X = X * D_v_norm.unsqueeze(1)
    
    return X`;

  const sessionConstructCode = `# Session-as-Hyperedge Construction
def construct_session_hypergraph(sequences, lengths, n_items):
    """
    Each session forms a hyperedge connecting all its items.
    This implements the key insight: sessions are hyperedges!
    """
    batch_size = sequences.size(0)
    max_edges = batch_size
    
    # Initialize incidence matrix
    H = torch.zeros(n_items + 1, max_edges)
    edge_weights = []
    
    for session_idx in range(batch_size):
        seq_len = lengths[session_idx].item()
        if seq_len == 0:
            continue
        
        # Get items in this session
        items = sequences[session_idx, :seq_len]
        items = items[items > 0]  # Remove padding
        
        if len(items) > 0:
            # Create hyperedge for this session
            H[items, session_idx] = 1
            
            # Edge weight = 1/sqrt(|e|) for normalization
            edge_weights.append(1.0 / math.sqrt(len(items)))
    
    return H, torch.tensor(edge_weights)`;

  const regularizationCode = `# Equations 2-4: Regularization Framework
def hypergraph_regularizer(f, H, W, D_v, D_e):
    """
    Ω(f) = 1/2 Σ_e Σ_{u,v} w(e)h(u,e)h(v,e)/δ(e) * 
           (f(u)/√d(u) - f(v)/√d(v))²
           
    Matrix form: Ω(f) = f^T Δ f
    """
    # Normalized incidence
    D_v_sqrt_inv = 1.0 / torch.sqrt(D_v + 1e-8)
    D_e_inv = 1.0 / (D_e + 1e-8)
    
    # Compute Laplacian
    Theta = torch.diag(D_v_sqrt_inv) @ H @ torch.diag(W) @ \
            torch.diag(D_e_inv) @ H.t() @ torch.diag(D_v_sqrt_inv)
    Delta = torch.eye(len(D_v)) - Theta
    
    # Regularization term
    reg = 0.5 * torch.sum(f * (Delta @ f))
    return reg

def training_objective(predictions, labels, f, H, W, lambda_reg=0.01):
    """
    Equation 2: min_f {R_emp(f) + Ω(f)}
    """
    # Empirical risk (classification loss)
    R_emp = F.cross_entropy(predictions, labels)
    
    # Regularization
    D_v, D_e = compute_degrees(H, W)
    omega = hypergraph_regularizer(f, H, W, D_v, D_e)
    
    # Total objective
    return R_emp + lambda_reg * omega`;

  const attentionCode = `# Attention Mechanisms (from implementation)
class HGNNAttention(nn.Module):
    """Two types of attention for session modeling"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Standard attention
        self.W_att = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_att = nn.Linear(hidden_dim, 1, bias=False)
        
        # SR-GNN style last-click attention
        self.W_last = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_item = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_last_att = nn.Linear(hidden_dim, 1, bias=False)
    
    def standard_attention(self, item_reprs):
        """
        α_i = softmax(v^T tanh(W·h_i))
        s = Σ α_i · h_i
        """
        att_scores = self.v_att(torch.tanh(self.W_att(item_reprs)))
        att_weights = F.softmax(att_scores, dim=0)
        session_repr = torch.sum(att_weights * item_reprs, dim=0)
        return session_repr, att_weights
    
    def last_click_attention(self, item_reprs):
        """
        Use last item as query (SR-GNN style)
        α_i = softmax(v^T tanh(W_last·h_n + W_item·h_i))
        """
        last_item = item_reprs[-1]
        last_transformed = self.W_last(last_item).unsqueeze(0)
        items_transformed = self.W_item(item_reprs)
        
        att_scores = self.v_last_att(
            torch.tanh(last_transformed + items_transformed)
        )
        att_weights = F.softmax(att_scores, dim=0)
        session_repr = torch.sum(att_weights * item_reprs, dim=0)
        return session_repr, att_weights`;

  const completeModelCode = `# Complete HGNN Model with Prediction Pipeline
class HGNN(nn.Module):
    """
    Full HGNN for session-based recommendation
    Implements: Hypergraph construction → HGNN layers → Session embedding → Prediction
    """
    
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100, 
                 n_layers=2, dropout=0.25, use_attention=True):
        super().__init__()
        
        # Item embeddings with padding
        self.embedding = nn.Embedding(n_items + 1, embedding_dim, 
                                    padding_idx=0)
        
        # Stack of HGNN layers (Equation 11)
        self.layers = nn.ModuleList()
        dims = [embedding_dim] + [hidden_dim] * n_layers
        
        for i in range(n_layers):
            self.layers.append(
                HGNNLayer(dims[i], dims[i+1])
            )
        
        # Attention module for session aggregation
        if use_attention:
            self.attention = HGNNAttention(hidden_dim)
        self.use_attention = use_attention
        
        # Output transformation: s_h = W_out @ s
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        
        # For dimension matching in scoring
        if embedding_dim != hidden_dim:
            self.W_item_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.W_item_proj = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequences, lengths):
        """
        Complete forward pass:
        1. Session → Hypergraph construction
        2. HGNN propagation: X^(l+1) = σ(normalized_propagation(X^(l)Θ^(l)))
        3. Session embedding: s = Attention({x_i : i ∈ session})
        4. Prediction scores: ẑ = s_h @ E^T
        
        Returns:
            scores: [batch_size, n_items] unnormalized prediction scores
            probs: [batch_size, n_items] probability distribution
        """
        batch_size = sequences.size(0)
        
        # Step 1: Construct hypergraph from sessions
        # Each session becomes a hyperedge
        H, edge_weights = construct_session_hypergraph(
            sequences, lengths, self.embedding.num_embeddings - 1
        )
        
        # Step 2: Get initial item embeddings X^(0)
        X = self.embedding.weight  # [n_items+1, embedding_dim]
        
        # Step 3: Apply T layers of HGNN propagation (Equation 11)
        # X^(l+1) = σ(D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2) X^(l) Θ^(l))
        for i, layer in enumerate(self.layers):
            X_new = layer(X, H, edge_weights)
            
            # Residual connection after first layer
            if i > 0:
                X = X_new + X
            else:
                X = X_new
            
            # Dropout between layers
            if i < len(self.layers) - 1:
                X = self.dropout(X)
        
        # Step 4: Compute session representations
        session_embeds = []
        
        for idx in range(batch_size):
            seq_len = lengths[idx].item()
            if seq_len == 0:
                session_embeds.append(
                    torch.zeros(X.size(1), device=X.device)
                )
                continue
            
            # Get final embeddings for items in this session
            items = sequences[idx, :seq_len]
            items = items[items > 0]  # Remove padding
            item_reprs = X[items]  # [seq_len, hidden_dim]
            
            # Aggregate using attention or mean pooling
            if self.use_attention:
                # s = Σ α_i * x_i, where α_i depends on last item
                session_repr, _ = self.attention.last_click_attention(
                    item_reprs
                )
            else:
                # s = mean(x_i for i in session)
                session_repr = torch.mean(item_reprs, dim=0)
            
            session_embeds.append(session_repr)
        
        # Stack all session embeddings
        session_embeds = torch.stack(session_embeds)  # [batch_size, hidden_dim]
        
        # Step 5: Transform session embeddings
        # s_h = W_out @ s
        session_embeds = self.W_out(session_embeds)
        
        # Step 6: Compute prediction scores
        # Get all item embeddings (excluding padding)
        item_embeds = self.embedding.weight[1:]  # [n_items, embedding_dim]
        
        # Project item embeddings if dimensions don't match
        if self.W_item_proj is not None:
            item_embeds = self.W_item_proj(item_embeds)
        
        # Compute scores: ẑ = s_h @ E^T
        scores = torch.matmul(session_embeds, item_embeds.t())
        
        # Apply softmax to get probabilities
        probs = F.softmax(scores, dim=-1)
        
        return scores, probs
    
    def compute_loss(self, scores, targets):
        """
        Cross-entropy loss for next-item prediction
        L = -Σ y_i log(ŷ_i)
        
        Args:
            scores: [batch_size, n_items] unnormalized scores
            targets: [batch_size] ground truth item indices
        """
        return F.cross_entropy(scores, targets)`;

  const trainingCode = `# Training HGNN for Session-based Recommendation
def train_hgnn(model, train_data, epochs=30, batch_size=100, lr=0.001, l2_reg=1e-5):
    """
    Train HGNN model on session data
    
    Args:
        model: HGNN model instance
        train_data: list of session sequences
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        l2_reg: L2 regularization weight
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Process sessions in batches
        for batch_idx in range(0, len(train_data), batch_size):
            batch_sessions = train_data[batch_idx:batch_idx + batch_size]
            batch_loss = 0.0
            
            # Prepare batch data
            sequences = []
            targets = []
            lengths = []
            
            for session in batch_sessions:
                # Generate training sequences from session
                # For session [v1, v2, v3, v4], generate:
                # ([v1], v2), ([v1, v2], v3), ([v1, v2, v3], v4)
                for i in range(1, len(session)):
                    sequences.append(session[:i])
                    targets.append(session[i])
                    lengths.append(i)
            
            # Convert to tensors
            max_len = max(lengths)
            padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
            for i, seq in enumerate(sequences):
                padded_seqs[i, :len(seq)] = torch.tensor(seq)
            
            lengths = torch.tensor(lengths)
            targets = torch.tensor(targets)
            
            # Forward pass
            scores, probs = model(padded_seqs, lengths)
            
            # Compute loss
            loss = model.compute_loss(scores, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(sequences)
        
        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        avg_loss = total_loss / sum(len(s)-1 for s in train_data)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Evaluation function
def evaluate_hgnn(model, test_data, k=20):
    """
    Evaluate HGNN using P@K and MRR@K metrics
    
    Args:
        model: trained HGNN model
        test_data: list of test session sequences
        k: cutoff for metrics (default: 20)
    
    Returns:
        precision@k: fraction of sessions with correct item in top-k
        mrr@k: mean reciprocal rank
    """
    model.eval()
    total_precision = 0.0
    total_mrr = 0.0
    
    with torch.no_grad():
        for session in test_data:
            if len(session) < 2:
                continue
                
            # Use all but last item as input
            input_seq = torch.tensor(session[:-1]).unsqueeze(0)
            length = torch.tensor([len(session) - 1])
            target = session[-1]
            
            # Get predictions
            scores, _ = model(input_seq, length)
            
            # Get top-k items
            _, top_k_items = torch.topk(scores[0], k)
            
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
    # Dataset parameters (e.g., Diginetica)
    n_items = 43097  # number of unique items
    embedding_dim = 100
    hidden_dim = 100
    n_layers = 2
    
    # Create model
    model = HGNN(n_items, embedding_dim, hidden_dim, n_layers, 
                 dropout=0.25, use_attention=True)
    
    # Example training data (item IDs)
    train_sessions = [
        [214, 125, 589, 214, 875],  # Example session 1
        [589, 214, 125, 987, 589],  # Example session 2
        [125, 875, 214, 589],        # Example session 3
        # ... more sessions
    ]
    
    # Train model
    train_hgnn(model, train_sessions, epochs=30)
    
    # Test data
    test_sessions = [
        [125, 589, 875, 214],
        [987, 214, 589, 125],
        # ... more test sessions
    ]
    
    # Evaluate
    p_at_20, mrr_at_20 = evaluate_hgnn(model, test_sessions)
    print(f"P@20: {p_at_20:.4f}, MRR@20: {mrr_at_20:.4f}")`;

  return (
    <ComparisonLayout
      title="Hypergraph Neural Networks (HGNN) - Complete Implementation"
      description="This implementation follows the paper 'Hypergraph Neural Networks' (Feng et al., AAAI 2019). Each code section is directly tied to its corresponding mathematical equation from the paper."
    >
      {/* Section 1: Basic Definitions and Notation */}
      <ComparisonSection
        title="1. Basic Definitions and Notation"
        leftContent={
          <>
            <p className="mb-4">A hypergraph is defined as:</p>
            <EquationBlock 
              math="\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})"
              description=""
              id="hypergraphDefEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{V} = \\{v_1, v_2, ..., v_n\\}"}</InlineMath>: set of n vertices (items)</li>
              <li><InlineMath>{"\\mathcal{E} = \\{e_1, e_2, ..., e_m\\}"}</InlineMath>: set of m hyperedges (sessions)</li>
              <li><InlineMath>{"\\mathbf{W} \\in \\mathbb{R}^{m \\times m}"}</InlineMath>: diagonal weight matrix</li>
              <li><InlineMath>{"e \\subseteq \\mathcal{V}"}</InlineMath>: each hyperedge e is a subset of vertices</li>
              <li><InlineMath>{"n = |\\mathcal{V}|"}</InlineMath>: total number of vertices</li>
              <li><InlineMath>{"m = |\\mathcal{E}|"}</InlineMath>: total number of hyperedges</li>
            </ul>
            
            <SubHeader>The incidence matrix encodes vertex-hyperedge relationships:</SubHeader>
            <EquationBlock 
              math="h(v,e) = \begin{cases} 1, & \text{if } v \in e \\ 0, & \text{if } v \notin e \end{cases}"
              description=""
              id="incidenceMatrixEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"h(v,e) \\in \\{0,1\\}"}</InlineMath>: incidence function</li>
              <li><InlineMath>{"v \\in \\mathcal{V}"}</InlineMath>: a vertex in the hypergraph</li>
              <li><InlineMath>{"e \\in \\mathcal{E}"}</InlineMath>: a hyperedge in the hypergraph</li>
              <li><InlineMath>{"\\mathbf{H} \\in \\{0,1\\}^{n \\times m}"}</InlineMath>: full incidence matrix with entries <InlineMath>{"\\mathbf{H}_{ve} = h(v,e)"}</InlineMath></li>
            </ul>
            
            <FormulaWithDescription
              title="Edge weight matrix:"
              math="\mathbf{W} = \text{diag}(\omega(e_1), \omega(e_2), ..., \omega(e_m))"
              description=""
              id="edgeWeightsEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{W} \\in \\mathbb{R}^{m \\times m}"}</InlineMath>: diagonal weight matrix</li>
              <li><InlineMath>{"\\omega(e_i) \\in \\mathbb{R}^+"}</InlineMath>: positive weight for hyperedge <InlineMath>{"e_i"}</InlineMath></li>
              <li><InlineMath>{"\\text{diag}(\\cdot)"}</InlineMath>: diagonal matrix construction</li>
            </ul>
          </>
        }
        code={incidenceMatrixCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="incidenceMatrixEq"
        rightId="incidenceMatrixCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="incidenceMatrix"
      />

      {/* Section 2: Degrees */}
      <ComparisonSection
        title="2. Vertex and Edge Degrees"
        leftContent={
          <>
            <FormulaWithDescription
              title="Vertex degree:"
              math="d(v) = \sum_{e \in \mathcal{E}} \omega(e) h(v,e)"
              id="vertexDegreeEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"d(v) \\in \\mathbb{R}^+"}</InlineMath>: degree of vertex v</li>
              <li><InlineMath>{"\\omega(e)"}</InlineMath>: weight of hyperedge e</li>
              <li><InlineMath>{"h(v,e)"}</InlineMath>: incidence function (1 if v ∈ e, 0 otherwise)</li>
              <li><InlineMath>{"\\mathcal{E}"}</InlineMath>: set of all hyperedges</li>
            </ul>
            
            <FormulaWithDescription
              title="Edge degree:"
              math="\delta(e) = \sum_{v \in \mathcal{V}} h(v,e)"
              description=""
              id="edgeDegreeEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\delta(e) \\in \\mathbb{N}"}</InlineMath>: degree of hyperedge e (number of vertices it contains)</li>
              <li><InlineMath>{"\\mathcal{V}"}</InlineMath>: set of all vertices</li>
              <li><InlineMath>{"h(v,e)"}</InlineMath>: incidence function</li>
            </ul>
            
            <SubHeader>Degree matrices:</SubHeader>
            <FormulaWithDescription
              title="Vertex degree matrix:"
              math="\mathbf{D}_v = \text{diag}(d(v_1), d(v_2), ..., d(v_n))"
              id="vertexDegreeMatrixEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <FormulaWithDescription
              title="Edge degree matrix:"
              math="\mathbf{D}_e = \text{diag}(\delta(e_1), \delta(e_2), ..., \delta(e_m))"
              id="edgeDegreeMatrixEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{D}_v \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: diagonal vertex degree matrix</li>
              <li><InlineMath>{"\\mathbf{D}_e \\in \\mathbb{R}^{m \\times m}"}</InlineMath>: diagonal edge degree matrix</li>
            </ul>
          </>
        }
        leftTitle="Mathematical Formulas"
        code={degreeComputationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="degreeFormulasEq"
        rightId="degreeComputationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="degrees"
      />

      {/* Section 3: Hypergraph Laplacian */}
      <ComparisonSection
        title="3. Hypergraph Laplacian"
        leftContent={
          <>
            <p className="mb-4">The normalized hypergraph operator:</p>
            <EquationBlock 
              math="\Theta = \mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{H}^{\top} \mathbf{D}_v^{-1/2}"
              id="thetaOperatorEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\Theta \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: normalized hypergraph operator</li>
              <li><InlineMath>{"\\mathbf{D}_v^{-1/2} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: inverse square root of vertex degree matrix</li>
              <li><InlineMath>{"\\mathbf{H} \\in \\{0,1\\}^{n \\times m}"}</InlineMath>: incidence matrix</li>
              <li><InlineMath>{"\\mathbf{W} \\in \\mathbb{R}^{m \\times m}"}</InlineMath>: edge weight matrix</li>
              <li><InlineMath>{"\\mathbf{D}_e^{-1} \\in \\mathbb{R}^{m \\times m}"}</InlineMath>: inverse of edge degree matrix</li>
              <li><InlineMath>{"\\mathbf{H}^{\\top} \\in \\{0,1\\}^{m \\times n}"}</InlineMath>: transpose of incidence matrix</li>
            </ul>
            
            <SubHeader>The hypergraph Laplacian:</SubHeader>
            <EquationBlock 
              math="\Delta = \mathbf{I} - \Theta"
              description=""
              id="laplacianDeltaEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\Delta \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: hypergraph Laplacian matrix</li>
              <li><InlineMath>{"\\mathbf{I} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: identity matrix</li>
              <li><InlineMath>{"\\Theta"}</InlineMath>: normalized hypergraph operator from above</li>
            </ul>
          </>
        }
        code={laplacianCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="laplacianEq"
        rightId="laplacianCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="laplacian"
      />

      {/* Section 4: Regularization */}
      <ComparisonSection
        title="4. Regularization Framework (Equations 2-4)"
        leftContent={
          <>
            <FormulaWithDescription
              title="Optimization objective (Eq. 2):"
              math="\arg\min_f \{\mathcal{R}_{emp}(f) + \Omega(f)\}"
              id="optimizationEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"f: \\mathcal{V} \\rightarrow \\mathbb{R}^d"}</InlineMath>: function mapping vertices to d-dimensional features</li>
              <li><InlineMath>{"\\mathcal{R}_{emp}(f)"}</InlineMath>: empirical risk (e.g., classification loss)</li>
              <li><InlineMath>{"\\Omega(f)"}</InlineMath>: regularization term encouraging smoothness</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Regularization term (Eq. 3):"
              math="\Omega(f) = \frac{1}{2} \sum_{e \in \mathcal{E}} \sum_{\{u,v\} \subseteq e} \frac{\omega(e)h(u,e)h(v,e)}{\delta(e)} \left(\frac{f(u)}{\sqrt{d(u)}} - \frac{f(v)}{\sqrt{d(v)}}\right)^2"
              id="regularizationTermEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\{u,v\\} \\subseteq e"}</InlineMath>: pairs of vertices within hyperedge e</li>
              <li><InlineMath>{"f(u), f(v) \\in \\mathbb{R}^d"}</InlineMath>: feature vectors for vertices u and v</li>
              <li><InlineMath>{"d(u), d(v)"}</InlineMath>: vertex degrees from earlier definition</li>
              <li><InlineMath>{"\\delta(e)"}</InlineMath>: edge degree from earlier definition</li>
              <li><InlineMath>{"\\omega(e)"}</InlineMath>: weight of hyperedge e</li>
              <li><InlineMath>{"h(u,e), h(v,e)"}</InlineMath>: incidence values (both 1 since u,v ∈ e)</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Matrix form (Eq. 4):"
              math="\Omega(f) = \mathbf{f}^{\top} \Delta \mathbf{f}"
              description=""
              id="regularizationMatrixEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{f} \\in \\mathbb{R}^{n \\times d}"}</InlineMath>: matrix of all vertex features (stacked)</li>
              <li><InlineMath>{"\\Delta \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: hypergraph Laplacian from Section 3</li>
              <li><InlineMath>{"\\mathbf{f}^{\\top}"}</InlineMath>: transpose of feature matrix</li>
            </ul>
            
            <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Key insight:</strong> The regularizer encourages smoothness - vertices in the same hyperedge should have similar feature representations.
              </p>
            </div>
            </div>
          </>
        }
        leftTitle="Mathematical Formulation"
        code={regularizationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="regularizationEq"
        rightId="regularizationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="regularization"
      />

      {/* Section 5: Spectral Theory */}
      <ComparisonSection
        title="5. Spectral Convolution (Equations 5-10)"
        leftContent={
          <>
            <p className="mb-4">To define spectral convolution, we first need the eigendecomposition of the Laplacian:</p>
            <FormulaWithDescription
              title="Eigendecomposition of Laplacian:"
              math="\Delta = \mathbf{\Phi} \mathbf{\Lambda} \mathbf{\Phi}^{\top}"
              description=""
              id="eigenDecompDefEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{\\Phi} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: matrix of orthonormal eigenvectors</li>
              <li><InlineMath>{"\\mathbf{\\Lambda} = \\text{diag}(\\lambda_1, \\lambda_2, ..., \\lambda_n)"}</InlineMath>: diagonal matrix of eigenvalues</li>
              <li><InlineMath>{"\\lambda_i \\in \\mathbb{R}"}</InlineMath>: i-th eigenvalue of the Laplacian</li>
              <li><InlineMath>{"\\mathbf{\\Phi}^{\\top}"}</InlineMath>: transpose of eigenvector matrix</li>
            </ul>
            <FormulaWithDescription
              title="Spectral convolution (Eq. 5) - Full form:"
              math="\mathbf{g} \star \mathbf{x} = \mathbf{\Phi}((\mathbf{\Phi}^{\top}\mathbf{g}) \odot (\mathbf{\Phi}^{\top}\mathbf{x})) = \mathbf{\Phi}\mathbf{g}(\mathbf{\Lambda})\mathbf{\Phi}^{\top}\mathbf{x}"
              description=""
              id="spectralConvFullEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{g} \\in \\mathbb{R}^n"}</InlineMath>: filter in the vertex domain</li>
              <li><InlineMath>{"\\mathbf{x} \\in \\mathbb{R}^n"}</InlineMath>: signal (feature vector) on vertices</li>
              <li><InlineMath>{"\\star"}</InlineMath>: convolution operator</li>
              <li><InlineMath>{"\\odot"}</InlineMath>: element-wise multiplication</li>
              <li><InlineMath>{"\\mathbf{g}(\\mathbf{\\Lambda}) = \\text{diag}(g(\\lambda_1), g(\\lambda_2), ..., g(\\lambda_n))"}</InlineMath>: spectral filter</li>
              <li><InlineMath>{"g(\\lambda_i)"}</InlineMath>: filter function evaluated at eigenvalue <InlineMath>{"\\lambda_i"}</InlineMath></li>
            </ul>
            
            <FormulaWithDescription
              title="Fourier transform on hypergraph:"
              math="\hat{\mathbf{x}} = \mathbf{\Phi}^{\top}\mathbf{x}, \quad \hat{\mathbf{g}} = \mathbf{\Phi}^{\top}\mathbf{g}"
              description=""
              id="fourierTransformEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\hat{\\mathbf{x}}, \\hat{\\mathbf{g}} \\in \\mathbb{R}^n"}</InlineMath>: signals in the spectral domain</li>
              <li><InlineMath>{"\\mathbf{\\Phi}^{\\top}"}</InlineMath>: Fourier transform operator (eigenvector projection)</li>
            </ul>
            <FormulaWithDescription
              title="Chebyshev approximation (Eq. 6):"
              math="\mathbf{g} \star \mathbf{x} \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Delta}) \mathbf{x}"
              description=""
              id="chebyshevApproxEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"K \\in \\mathbb{N}"}</InlineMath>: order of Chebyshev approximation</li>
              <li><InlineMath>{"\\theta_k \\in \\mathbb{R}"}</InlineMath>: learnable Chebyshev coefficients</li>
              <li><InlineMath>{"T_k(\\cdot)"}</InlineMath>: Chebyshev polynomial of order k</li>
              <li><InlineMath>{"\\tilde{\\Delta}"}</InlineMath>: scaled Laplacian (defined below)</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Scaled Laplacian:"
              math="\tilde{\Delta} = \frac{2}{\lambda_{\max}}\Delta - \mathbf{I}"
              description=""
              id="scaledLaplacianEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\lambda_{\\max}"}</InlineMath>: largest eigenvalue of <InlineMath>{"\\Delta"}</InlineMath></li>
              <li><InlineMath>{"\\tilde{\\Delta} \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: scaled Laplacian with eigenvalues in [-1, 1]</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Chebyshev recurrence:"
              math="T_0(\tilde{\Delta}) = \mathbf{I}, \quad T_1(\tilde{\Delta}) = \tilde{\Delta}, \quad T_k(\tilde{\Delta}) = 2\tilde{\Delta}T_{k-1}(\tilde{\Delta}) - T_{k-2}(\tilde{\Delta})"
              id="chebyshevRecurrenceEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"T_k(\\tilde{\\Delta}) \\in \\mathbb{R}^{n \\times n}"}</InlineMath>: k-th order Chebyshev polynomial evaluated at <InlineMath>{"\\tilde{\\Delta}"}</InlineMath></li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="First-order approximation (K=1):"
              math="\mathbf{g} \star \mathbf{x} \approx \theta_0\mathbf{x} - \theta_1\mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{H}^{\top} \mathbf{D}_v^{-1/2} \mathbf{x}"
              id="firstOrderFullEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\theta_0, \\theta_1 \\in \\mathbb{R}"}</InlineMath>: two learnable parameters</li>
              <li>Using <InlineMath>{"\\lambda_{\\max} \\approx 2"}</InlineMath> and <InlineMath>{"\\Delta = \\mathbf{I} - \\Theta"}</InlineMath></li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Simplified with single parameter (Eq. 9):"
              math="\mathbf{g} \star \mathbf{x} \approx \theta \mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{H}^{\top} \mathbf{D}_v^{-1/2} \mathbf{x}"
              description=""
              id="firstOrderEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\theta \\in \\mathbb{R}"}</InlineMath>: single learnable parameter</li>
              <li>This simplification leads directly to the HGNN layer formula</li>
            </ul>
            </div>
          </>
        }
        leftTitle="Mathematical Theory"
        code={chebyshevApproxCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="spectralTheoryEq"
        rightId="chebyshevApproxCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="spectralTheory"
      />

      {/* Section 6: HGNN Layer */}
      <ComparisonSection
        title="6. HGNN Layer (Equation 11)"
        leftContent={
          <>
            <EquationBlock 
              math="\mathbf{X}^{(l+1)} = \sigma\left(\mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{H}^{\top} \mathbf{D}_v^{-1/2} \mathbf{X}^{(l)} \Theta^{(l)}\right)"
              description=""
              id="hgnnLayerEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{X}^{(l)} \\in \\mathbb{R}^{n \\times d_l}"}</InlineMath>: node feature matrix at layer l</li>
              <li><InlineMath>{"\\mathbf{X}^{(l+1)} \\in \\mathbb{R}^{n \\times d_{l+1}}"}</InlineMath>: updated node features after layer l</li>
              <li><InlineMath>{"\\Theta^{(l)} \\in \\mathbb{R}^{d_l \\times d_{l+1}}"}</InlineMath>: learnable weight matrix for layer l</li>
              <li><InlineMath>{"\\sigma"}</InlineMath>: activation function (e.g., ReLU)</li>
              <li><InlineMath>{"d_l"}</InlineMath>: feature dimension at layer l</li>
              <li><InlineMath>{"d_{l+1}"}</InlineMath>: feature dimension at layer l+1</li>
              <li>All other matrices (<InlineMath>{"\\mathbf{D}_v, \\mathbf{H}, \\mathbf{W}, \\mathbf{D}_e"}</InlineMath>) are from previous definitions</li>
            </ul>
            <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Computation flow:</strong> Features are first transformed by Θ, then propagated through the 
                normalized hypergraph structure (node→edge→node), and finally passed through activation.
              </p>
            </div>
          </>
        }
        leftTitle="Mathematical Formula"
        code={hgnnLayerCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="hgnnLayerCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="hgnnLayerCode"
        editorPlaceholder="Try implementing the HGNN layer yourself..."
      />

      {/* Section 7: Session Construction */}
      <ComparisonSection
        title="7. Session-as-Hyperedge Construction"
        leftContent={
          <>
            <p className="mb-4">Key Insight: Sessions naturally form hyperedges in the item hypergraph!</p>
            
            <FormulaWithDescription
              title="Session to hyperedge mapping:"
              math="\text{Session } s = \{i_1, i_2, ..., i_k\} \rightarrow \text{Hyperedge } e_s"
              id="sessionHyperedgeEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <FormulaWithDescription
              title="Incidence matrix entry:"
              math="h(i, s) = \begin{cases} 1, & \text{if item } i \in \text{session } s \\ 0, & \text{otherwise} \end{cases}"
              id="sessionIncidenceEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <FormulaWithDescription
              title="Edge weight (normalization):"
              math="\omega(e_s) = \frac{1}{\sqrt{|s|}}"
              id="edgeWeightNormEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"s = \\{i_1, i_2, ..., i_k\\}"}</InlineMath>: a session containing k items</li>
              <li><InlineMath>{"i_j \\in \\{1, 2, ..., n\\}"}</InlineMath>: item indices</li>
              <li><InlineMath>{"e_s \\in \\mathcal{E}"}</InlineMath>: hyperedge corresponding to session s</li>
              <li><InlineMath>{"h(i, s)"}</InlineMath>: incidence value between item i and session s</li>
              <li><InlineMath>{"\\omega(e_s) \\in \\mathbb{R}^+"}</InlineMath>: weight for session hyperedge</li>
              <li><InlineMath>"|s|"}</InlineMath>: cardinality of session s (number of items)</li>
              <li><InlineMath>{"k \\in \\mathbb{N}"}</InlineMath>: number of items in the session</li>
            </ul>
            
            <div className="mt-4 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Why this works:</strong> Sessions naturally capture co-occurrence patterns. 
                Unlike pairwise graphs, hyperedges preserve the full context of which items appeared together in a session.
              </p>
            </div>
          </>
        }
        leftTitle="Conceptual Framework"
        code={sessionConstructCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="sessionTheoryEq"
        rightId="sessionConstructCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="sessionConstruction"
      />

      {/* Section 8: Attention */}
      <ComparisonSection
        title="8. Attention Mechanisms for Session Representation"
        leftContent={
          <>
            <p className="mb-4">After HGNN propagation, we need to aggregate item embeddings into a session representation:</p>
            
            <FormulaWithDescription
              title="Standard attention scores:"
              math="\alpha_i = \mathbf{v}^{\top} \tanh(\mathbf{W} \cdot \mathbf{h}_i)"
              id="standardAttentionScoreEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\alpha_i \\in \\mathbb{R}"}</InlineMath>: unnormalized attention score for item i</li>
              <li><InlineMath>{"\\mathbf{v} \\in \\mathbb{R}^d"}</InlineMath>: learnable attention vector</li>
              <li><InlineMath>{"\\mathbf{W} \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: attention weight matrix</li>
              <li><InlineMath>{"\\mathbf{h}_i \\in \\mathbb{R}^d"}</InlineMath>: embedding of item i after HGNN layers</li>
              <li><InlineMath>{"\\tanh"}</InlineMath>: hyperbolic tangent activation</li>
              <li><InlineMath>{"d"}</InlineMath>: hidden dimension</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Normalized attention weights:"
              math="\alpha_i = \frac{\exp(\alpha_i)}{\sum_{j \in \text{session}} \exp(\alpha_j)}"
              id="normalizedAttentionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\alpha_i \\in [0,1]"}</InlineMath>: normalized attention weight (after softmax)</li>
              <li><InlineMath>{"\\exp"}</InlineMath>: exponential function</li>
              <li><InlineMath>{"j \\in \\text{session}"}</InlineMath>: all items in the current session</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Session representation:"
              math="\mathbf{s} = \sum_{i \in \text{session}} \alpha_i \cdot \mathbf{h}_i"
              id="sessionRepresentationAttentionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{s} \\in \\mathbb{R}^d"}</InlineMath>: final session representation</li>
              <li><InlineMath>{"\\alpha_i"}</InlineMath>: normalized attention weight for item i</li>
              <li><InlineMath>{"\\mathbf{h}_i"}</InlineMath>: item i embedding after HGNN</li>
            </ul>
            </div>
            
            <SubHeader>Last-click attention (SR-GNN style):</SubHeader>
            <FormulaWithDescription
              title="Query-based attention:"
              math="\alpha_i = \mathbf{v}^{\top} \tanh(\mathbf{W}_{last} \cdot \mathbf{h}_n + \mathbf{W}_{item} \cdot \mathbf{h}_i)"
              id="lastClickAttentionScoreEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{h}_n \\in \\mathbb{R}^d"}</InlineMath>: last item embedding (used as query)</li>
              <li><InlineMath>{"\\mathbf{W}_{last} \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: query transformation matrix</li>
              <li><InlineMath>{"\\mathbf{W}_{item} \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: key transformation matrix</li>
              <li>All other symbols as defined above</li>
            </ul>
            
            <div className="mt-4 p-4 bg-blue-50 border-l-4 border-blue-400">
              <p className="text-sm">
                <strong>Key insight:</strong> Last-click attention uses the most recent item as a query to weight 
                the importance of all items in the session, capturing recency bias in user behavior.
              </p>
            </div>
          </>
        }
        leftTitle="Mathematical Framework"
        code={attentionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="attentionTheoryEq"
        rightId="attentionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="attention"
      />

      {/* Section 9: Session Representation and Prediction */}
      <ComparisonSection
        title="9. Session Representation and Prediction"
        leftContent={
          <>
            <p className="mb-4">After propagating through HGNN layers, we compute session representations and make predictions:</p>
            
            <FormulaWithDescription
              title="Session representation computation:"
              math="\mathbf{s} = \text{Aggregation}(\{\mathbf{x}_i^{(T)} : i \in \text{session}\})"
              description=""
              id="sessionRepresentationEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{x}_i^{(T)} \\in \\mathbb{R}^d"}</InlineMath>: embedding of item i after T HGNN layers</li>
              <li><InlineMath>{"T \\in \\mathbb{N}"}</InlineMath>: total number of HGNN layers</li>
              <li><InlineMath>{"\\text{session}"}</InlineMath>: set of items in the current session</li>
              <li><InlineMath>{"\\mathbf{s} \\in \\mathbb{R}^d"}</InlineMath>: aggregated session representation</li>
            </ul>
            <p className="mt-2">Aggregation options:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li>Mean pooling: <InlineMath>{"\\mathbf{s} = \\frac{1}{|\\text{session}|} \\sum_{i \\in \\text{session}} \\mathbf{x}_i^{(T)}"}</InlineMath></li>
              <li>Last-click attention: <InlineMath>{"\\mathbf{s} = \\sum_{i} \\alpha_i \\mathbf{x}_i^{(T)}"}</InlineMath> with <InlineMath>{"\\alpha_i"}</InlineMath> based on last item</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Transformed session embedding:"
              math="\mathbf{s}_h = \mathbf{W}_{\text{out}} \mathbf{s}"
              description=""
              id="transformedSessionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{s}_h \\in \\mathbb{R}^d"}</InlineMath>: transformed session embedding</li>
              <li><InlineMath>{"\\mathbf{W}_{\\text{out}} \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: output transformation matrix</li>
              <li><InlineMath>{"\\mathbf{s}"}</InlineMath>: aggregated session representation from above</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Prediction scores for all items:"
              math="\hat{\mathbf{z}} = \mathbf{s}_h^{\top} \mathbf{E}^{\top}"
              description=""
              id="predictionScoresEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\hat{\\mathbf{z}} \\in \\mathbb{R}^n"}</InlineMath>: unnormalized scores for all n items</li>
              <li><InlineMath>{"\\mathbf{s}_h^{\\top} \\in \\mathbb{R}^{1 \\times d}"}</InlineMath>: transposed session embedding</li>
              <li><InlineMath>{"\\mathbf{E} \\in \\mathbb{R}^{n \\times d}"}</InlineMath>: item embedding matrix (all n items)</li>
              <li><InlineMath>{"\\mathbf{E}^{\\top} \\in \\mathbb{R}^{d \\times n}"}</InlineMath>: transposed item embeddings</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Probability distribution over items:"
              math="\hat{\mathbf{y}} = \text{softmax}(\hat{\mathbf{z}}) = \frac{\exp(\hat{\mathbf{z}})}{\sum_{j=1}^{n} \exp(\hat{z}_j)}"
              description=""
              id="softmaxPredictionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\hat{\\mathbf{y}} \\in [0,1]^n"}</InlineMath>: predicted probability distribution</li>
              <li><InlineMath>{"\\hat{z}_j"}</InlineMath>: score for item j</li>
              <li><InlineMath>{"\\sum_{j=1}^{n} \\hat{y}_j = 1"}</InlineMath>: probabilities sum to 1</li>
            </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Cross-entropy loss:"
              math="\mathcal{L} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)"
              description=""
              id="crossEntropyLossEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{L} \\in \\mathbb{R}"}</InlineMath>: loss value</li>
              <li><InlineMath>{"\\mathbf{y} \\in \\{0,1\\}^n"}</InlineMath>: one-hot encoded ground truth</li>
              <li><InlineMath>{"y_i = 1"}</InlineMath> for the correct next item, 0 otherwise</li>
              <li><InlineMath>{"\\log"}</InlineMath>: natural logarithm</li>
            </ul>
            </div>
          </>
        }
        leftTitle="Prediction Pipeline"
        rightTitle="Full Implementation"
        code={completeModelCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="sessionRepresentationEq"
        rightId="completeModelCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="completeModelCode"
        editorPlaceholder="Try implementing the complete HGNN model..."
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="completeModel"
      />

      {/* Section 10: Complete Pipeline */}
      <ComparisonSection
        title="10. Complete HGNN Pipeline"
        leftContent={
          <>
            <p className="font-semibold mb-4">End-to-End HGNN Workflow:</p>
            
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded">
                <h4 className="font-semibold mb-2">1. Session to Hypergraph Construction</h4>
                <BlockMath math="\text{Session } s = \{i_1, i_2, ..., i_k\} \rightarrow \text{Hyperedge } e_s" />
                <p className="text-sm mt-2">Each session becomes a hyperedge connecting all its items</p>
              </div>
              
              <div className="bg-green-50 p-4 rounded">
                <h4 className="font-semibold mb-2">2. Initial Item Embeddings</h4>
                <BlockMath math="\mathbf{X}^{(0)} = \text{Embedding}(\text{items}) \in \mathbb{R}^{(n+1) \times d}" />
                <p className="text-sm mt-2">Initialize embeddings for all n items plus padding token</p>
              </div>
              
              <div className="bg-yellow-50 p-4 rounded">
                <h4 className="font-semibold mb-2">3. HGNN Propagation (T layers)</h4>
                <BlockMath math="\mathbf{X}^{(t+1)} = \sigma\left(\mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W} \mathbf{D}_e^{-1} \mathbf{H}^{\top} \mathbf{D}_v^{-1/2} \mathbf{X}^{(t)} \Theta^{(t)}\right)" />
                <p className="text-sm mt-2">Message passing: node → edge → node with normalization</p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded">
                <h4 className="font-semibold mb-2">4. Session Representation</h4>
                <BlockMath math="\mathbf{s} = \text{Attention}(\{\mathbf{x}_i^{(T)} : i \in \text{session}\})" />
                <p className="text-sm mt-2">Aggregate item embeddings using attention mechanism</p>
              </div>
              
              <div className="bg-red-50 p-4 rounded">
                <h4 className="font-semibold mb-2">5. Next-Item Prediction</h4>
                <BlockMath math="\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_{\text{out}}\mathbf{s} \cdot \mathbf{E}^{\top})" />
                <p className="text-sm mt-2">Score all items and apply softmax for probabilities</p>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-gray-100 rounded">
              <h4 className="font-semibold mb-2">Key Design Choices:</h4>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>Sessions naturally map to hyperedges (many-to-many relations)</li>
                <li>Normalized message passing ensures stable gradients</li>
                <li>Attention aggregation captures session context</li>
                <li>Shared item embeddings between HGNN and prediction</li>
                <li>Cross-entropy loss for multi-class classification</li>
              </ul>
            </div>
          </>
        }
        leftTitle="Complete Architecture"
        fullWidth={true}
      />

      {/* Section 11: Training */}
      <ComparisonSection
        title="11. Training HGNN"
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
        editorPlaceholder="Implement the training loop for HGNN..."
        fullWidth={true}
      />

      {/* Summary */}
      <div className="mt-12 p-6 bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">HGNN vs SR-GNN: Key Differences</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">HGNN Advantages</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li><strong>Natural modeling:</strong> Sessions as hyperedges preserve co-occurrence</li>
              <li><strong>Spectral foundation:</strong> Solid theoretical basis from spectral graph theory</li>
              <li><strong>Flexible aggregation:</strong> Can model arbitrary set relationships</li>
              <li><strong>Normalized propagation:</strong> Better gradient flow with proper normalization</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">SR-GNN Advantages</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li><strong>Sequential modeling:</strong> Captures order through directed edges</li>
              <li><strong>GRU-based updates:</strong> Sophisticated gating mechanisms</li>
              <li><strong>Local+Global fusion:</strong> Explicit modeling of immediate and general interest</li>
              <li><strong>Proven performance:</strong> Strong empirical results on benchmarks</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="font-semibold mb-2">When to Use Which?</h3>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li><strong>Use HGNN when:</strong> Sessions have strong co-occurrence patterns, order is less critical, need theoretical guarantees</li>
            <li><strong>Use SR-GNN when:</strong> Sequential patterns are important, proven baseline needed, computational efficiency matters</li>
            <li><strong>Hybrid approach:</strong> Could combine hypergraph structure with sequential gating for best of both worlds</li>
          </ul>
        </div>
      </div>

      <div className="mb-8 bg-green-50 border-l-4 border-green-400 p-4">
        <h3 className="text-lg font-bold text-green-800 mb-2">Key Takeaways</h3>
        <ul className="list-disc list-inside space-y-1 text-green-700">
          <li>HGNN models high-order relationships using hyperedges (sessions)</li>
          <li>The core operation is node→edge→node message passing (Equation 11)</li>
          <li>Proper normalization (D_v^(-1/2) and D_e^(-1)) ensures stable learning</li>
          <li>Spectral theory provides foundation; first-order approximation is practical</li>
          <li>Attention mechanisms enhance session representation quality</li>
          <li>The framework naturally extends GNNs to handle multi-way relationships</li>
        </ul>
      </div>

      <div className="mb-8 bg-yellow-50 border-l-4 border-yellow-400 p-4">
        <h3 className="text-lg font-bold text-yellow-800 mb-2">Production Considerations</h3>
        <ul className="list-disc list-inside space-y-1 text-yellow-700">
          <li>Use sparse operations for large-scale hypergraphs</li>
          <li>Batch processing requires careful hypergraph construction</li>
          <li>Edge weights can encode session importance or recency</li>
          <li>Residual connections help with deeper models</li>
          <li>Dropout and proper initialization prevent overfitting</li>
        </ul>
      </div>
    </ComparisonLayout>
  );
}