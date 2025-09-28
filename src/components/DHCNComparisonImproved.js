import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription } from './ComparisonLayout';

export default function DHCNComparisonImproved() {
  // Initialize visibility state - organized by equation-code pairs
  const [visibility, setVisibility] = useState({
    // Section 1: Hypergraph Convolution
    hypergraphConvEq: true,
    hypergraphConvCode: true,
    // Section 2: Matrix Form
    matrixFormEq: true,
    matrixFormCode: true,
    // Section 3: Layer Aggregation
    layerAggregationEq: true,
    layerAggregationCode: true,
    // Section 4: Position Encoding
    positionEncodingEq: true,
    positionEncodingCode: true,
    // Section 5: Soft Attention
    softAttentionEq: true,
    softAttentionCode: true,
    // Section 6: Score Computation
    scoreComputationEq: true,
    scoreComputationCode: true,
    // Section 7: Line Graph
    lineGraphEq: true,
    lineGraphConvEq: true,
    lineGraphCode: true,
    // Section 8: Self-Supervised Learning
    contrastiveLossEq: true,
    selfSupervisedCode: true,
    // Section 9: Complete Architecture
    completeArchitectureEq: true,
    completeModelCode: true
  });

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const [userCode, setUserCode] = useState({
    hypergraphConvCode: '',
    matrixFormCode: '',
    layerAggregationCode: '',
    positionEncodingCode: '',
    softAttentionCode: '',
    scoreComputationCode: '',
    lineGraphCode: '',
    selfSupervisedCode: '',
    completeModelCode: ''
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
  const hypergraphConvCode = `# Equation 1: Hypergraph Convolution
def hypergraph_convolution(x_j, H, W):
    """
    x_i^{l+1} = Σ_j Σ_ε H_{iε} H_{jε} W_{εε} x_j^{l}
    
    Aggregate features from nodes to hyperedges, then back to nodes
    """
    # H: incidence matrix [n_nodes, n_edges]
    # W: hyperedge weights (diagonal matrix)
    # x_j: node features at layer l
    
    x_next = torch.zeros_like(x_j)
    n_nodes, n_edges = H.shape
    
    # For each target node i
    for i in range(n_nodes):
        # For each source node j
        for j in range(n_nodes):
            # Sum over all hyperedges
            for e in range(n_edges):
                if H[i, e] == 1 and H[j, e] == 1:
                    # Both nodes in same hyperedge
                    x_next[i] += W[e, e] * x_j[j]
    
    return x_next`;

  const matrixFormCode = `# Equation 2: Matrix Form with Row Normalization
class HypergraphConvLayer(nn.Module):
    """
    X_h^{l+1} = D^{-1} H W B^{-1} H^T X_h^{l}
    
    Where:
    - D: node degree matrix
    - H: incidence matrix  
    - W: edge weight matrix (identity in DHCN)
    - B: edge degree matrix
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, X, H):
        # All hyperedges have weight 1
        W = torch.eye(H.size(1))
        
        # Compute degree matrices
        D = torch.diag(torch.sum(H, dim=1))  # Node degrees
        B = torch.diag(torch.sum(H, dim=0))  # Edge degrees
        
        # Normalize
        D_inv = torch.diag(1.0 / (torch.diag(D) + 1e-8))
        B_inv = torch.diag(1.0 / (torch.diag(B) + 1e-8))
        
        # Two-stage refinement: node -> edge -> node
        # Step 1: H^T @ X - aggregate from nodes to hyperedges
        X_edge = H.t() @ X
        
        # Step 2: Apply edge normalization
        X_edge = B_inv @ X_edge
        
        # Step 3: H @ X_edge - propagate from edges back to nodes
        X_next = H @ X_edge
        
        # Step 4: Apply node normalization
        X_next = D_inv @ X_next
        
        return X_next`;

  const layerAggregationCode = `# Layer Aggregation (mentioned in paper but not formally defined)
def aggregate_layers(X_layers):
    """
    X_h = (1/L) Σ_{l=0}^{L} X_h^{(l)}
    
    Average embeddings from all layers to capture multi-scale features
    """
    # X_layers: list of [X^{(0)}, X^{(1)}, ..., X^{(L)}]
    # Each X^{(l)} has shape [n_nodes, d_model]
    
    # Stack all layers: [L+1, n_nodes, d_model]
    X_stacked = torch.stack(X_layers, dim=0)
    
    # Average across layers: [n_nodes, d_model]
    X_final = torch.mean(X_stacked, dim=0)
    
    return X_final

# Apply L layers and aggregate
def multi_layer_hypergraph_conv(X_0, H, n_layers=3):
    """
    Apply multiple layers of hypergraph convolution
    and aggregate the results
    """
    conv_layer = HypergraphConvLayer()
    
    # Collect embeddings from all layers
    X_layers = [X_0]  # Include input layer
    
    X = X_0
    for l in range(n_layers):
        X = conv_layer(X, H)
        X_layers.append(X)
    
    # Average all layers
    X_final = aggregate_layers(X_layers)
    
    return X_final`;

  const positionEncodingCode = `# Equation 3: Reversed Position Embeddings
class PositionEncoding(nn.Module):
    """
    x_t^* = tanh(W_1 [x_t || p_{m-t+1}] + b)
    
    Integrate reversed position embeddings with item representations
    """
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Learnable position matrix
        self.P_r = nn.Embedding(max_len, d_model)
        
        # Transform after concatenation
        self.W_1 = nn.Linear(2 * d_model, d_model)
        self.b = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, item_embeds, session_length):
        """
        Args:
            item_embeds: [seq_len, d_model] - from hypergraph conv output
            session_length: length of current session
        """
        seq_len = item_embeds.size(0)
        
        # Reversed positions: last item gets position 1
        reversed_positions = []
        for t in range(seq_len):
            pos = session_length - t  # m - t + 1 (1-indexed)
            reversed_positions.append(pos - 1)  # 0-indexed for embedding
        
        reversed_positions = torch.tensor(reversed_positions)
        position_embeds = self.P_r(reversed_positions)
        
        # Concatenate item and position embeddings
        combined = torch.cat([item_embeds, position_embeds], dim=-1)
        
        # Transform with activation
        x_star = torch.tanh(self.W_1(combined) + self.b)
        
        return x_star`;

  const softAttentionCode = `# Equation 4: Soft Attention Mechanism
class SoftAttention(nn.Module):
    """
    α_t = f^T σ(W_2 x_s^* + W_3 x_t^* + c)
    θ_h = Σ_t α_t x_t
    
    Aggregate items with attention weights for session representation
    Note: θ_h is the session embedding for hypergraph channel (θ_i^h in Eq. 9)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c = nn.Parameter(torch.zeros(hidden_dim))
        self.f = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, x_items_with_pos, x_items_original):
        """
        Args:
            x_items_with_pos: position-enhanced embeddings [seq_len, hidden_dim]
            x_items_original: original item embeddings (without position) [seq_len, hidden_dim]
        Returns:
            theta_h: session representation [hidden_dim] (this becomes θ_i^h for session i)
        """
        seq_len = x_items_with_pos.size(0)
        
        # Session embedding: average of position-enhanced items
        x_s = torch.mean(x_items_with_pos, dim=0)
        
        # Compute attention scores for each item
        attention_scores = []
        for t in range(seq_len):
            x_t = x_items_with_pos[t]
            
            # Attention computation
            score = torch.dot(
                self.f, 
                torch.sigmoid(self.W_2(x_s) + self.W_3(x_t) + self.c)
            )
            attention_scores.append(score)
        
        # Normalize attention weights
        alpha = F.softmax(torch.stack(attention_scores), dim=0)
        
        # Weighted aggregation using original embeddings
        theta_h = torch.sum(alpha.unsqueeze(1) * x_items_original, dim=0)
        
        return theta_h, alpha`;

  const scoreComputationCode = `# Equations 5-7: Score Computation and Prediction
def compute_scores_and_loss(theta_h, X_h, labels):
    """
    Score computation:
    z_i = θ_h^T x_i  (Eq. 5)
    y_hat = softmax(z)  (Eq. 6)
    
    Loss:
    L_r = -Σ y_i log(y_hat_i) + (1-y_i)log(1-y_hat_i)  (Eq. 7)
    
    Args:
        theta_h: session representation from attention [hidden_dim]
        X_h: all item embeddings after hypergraph conv [n_items, hidden_dim]
        labels: one-hot ground truth [n_items]
    """
    # Compute scores for all items
    z = torch.matmul(theta_h, X_h.t())  # [n_items]
    
    # Apply softmax for probabilities
    y_hat = F.softmax(z, dim=0)
    
    # Cross entropy loss
    loss = -torch.sum(
        labels * torch.log(y_hat + 1e-8) + 
        (1 - labels) * torch.log(1 - y_hat + 1e-8)
    )
    
    return y_hat, loss

# Full forward pass for recommendation
def forward_recommendation(session_items, hypergraph_conv, attention, all_items):
    """
    Complete forward pass for session-based recommendation
    """
    # 1. Construct hypergraph from session
    H, W = construct_session_hypergraph(session_items)
    
    # 2. Apply L layers of hypergraph convolution
    X_h = multi_layer_hypergraph_conv(all_items, H, n_layers=3)
    
    # 3. Get session item embeddings
    session_embeds = X_h[session_items]
    
    # 4. Add position embeddings to session items
    session_embeds_with_pos = add_position_encoding(session_embeds)
    
    # 5. Apply attention to get session representation
    # This theta_h is θ_i^h for session i in the self-supervised loss
    theta_h, _ = attention(session_embeds_with_pos, session_embeds)
    
    # 6. Compute scores
    scores = torch.matmul(theta_h, X_h.t())
    
    return scores, theta_h  # Return both for self-supervised learning`;

  const lineGraphCode = `# Equation 8: Line Graph Convolution
class LineGraphConvolution(nn.Module):
    """
    Line graph of hypergraph: sessions become nodes, 
    shared items create edges between sessions
    
    Θ_l^{l+1} = D̂^{-1} Â Θ_l^{l}
    
    Where:
    - Â = A + I (adjacency matrix with self-loops)
    - D̂: degree matrix of Â
    - Θ_l: session embeddings (initialized from hypergraph channel)
    """
    def __init__(self):
        super().__init__()
        
    def construct_line_graph(self, hypergraph_incidence):
        """
        Transform hypergraph to its line graph
        Sessions (hyperedges) -> nodes
        Shared items -> edges between sessions
        """
        H = hypergraph_incidence  # [n_items, n_sessions]
        n_sessions = H.size(1)
        
        # Adjacency matrix for line graph
        A = torch.zeros(n_sessions, n_sessions)
        
        # Two sessions are connected if they share items
        for i in range(n_sessions):
            for j in range(i+1, n_sessions):
                # Intersection: items in both sessions
                intersection = torch.sum(H[:, i] * H[:, j])
                # Union: items in either session
                union = torch.sum((H[:, i] + H[:, j]) > 0)
                
                # Edge weight: Jaccard similarity
                if union > 0:
                    weight = intersection / union
                    A[i, j] = weight
                    A[j, i] = weight
        
        return A
    
    def forward(self, Theta_l, A):
        """
        Line graph convolution
        Theta_l: session embeddings [n_sessions, hidden_dim]
        """
        # Add self-loops
        A_hat = A + torch.eye(A.size(0))
        
        # Degree matrix
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv = torch.diag(1.0 / (torch.diag(D_hat) + 1e-8))
        
        # Graph convolution
        Theta_next = D_hat_inv @ A_hat @ Theta_l
        
        return Theta_next
    
    def init_session_embeds(self, H, item_embeds):
        """
        Initialize session embeddings from hypergraph channel output
        Θ_l^{(0)} = average pooling of items in each session
        """
        n_sessions = H.size(1)
        hidden_dim = item_embeds.size(1)
        
        Theta_0 = torch.zeros(n_sessions, hidden_dim)
        
        for s in range(n_sessions):
            # Items in session s
            items_mask = H[:, s] > 0
            session_items = item_embeds[items_mask]
            
            # Average pooling
            if session_items.size(0) > 0:
                Theta_0[s] = torch.mean(session_items, dim=0)
        
        return Theta_0

# Multi-layer line graph convolution
def line_graph_channel(H, X_h, n_layers=3):
    """
    Apply line graph convolution to get session embeddings
    
    Returns:
        Theta_l: session embeddings from line graph [n_sessions, hidden_dim]
                 (these become θ_i^l for each session i in Eq. 9)
    """
    line_conv = LineGraphConvolution()
    
    # Construct line graph adjacency
    A = line_conv.construct_line_graph(H)
    
    # Initialize from hypergraph embeddings
    Theta_0 = line_conv.init_session_embeds(H, X_h)
    
    # Apply L layers
    Theta_layers = [Theta_0]
    Theta = Theta_0
    
    for l in range(n_layers):
        Theta = line_conv(Theta, A)
        Theta_layers.append(Theta)
    
    # Average all layers
    Theta_final = torch.mean(torch.stack(Theta_layers), dim=0)
    
    return Theta_final`;

  const selfSupervisedCode = `# Equation 9: Self-Supervised Contrastive Learning
class ContrastiveLearning(nn.Module):
    """
    Maximize mutual information between hypergraph and line graph views
    
    L_s = -log σ(f_D(θ_i^h, θ_i^l)) - log σ(1 - f_D(θ̃_i^h, θ_i^l))
    
    Where:
    - θ_i^h: session i embedding from hypergraph channel (from soft attention)
    - θ_i^l: session i embedding from line graph channel
    - θ̃_i^h: corrupted (negative) samples
    """
    def __init__(self):
        super().__init__()
        
    def discriminator(self, h1, h2):
        """
        Simple dot product discriminator
        Scores agreement between two representations
        """
        return torch.sum(h1 * h2, dim=-1)
    
    def corrupt_embeddings(self, embeddings):
        """
        Generate negative samples by shuffling
        """
        # Row-wise shuffle
        row_idx = torch.randperm(embeddings.size(0))
        corrupted = embeddings[row_idx]
        
        return corrupted
    
    def forward(self, theta_h, theta_l):
        """
        Contrastive loss between two views
        theta_h: hypergraph channel embeddings [batch_size, hidden_dim]
                 (collected from soft attention outputs)
        theta_l: line graph channel embeddings [batch_size, hidden_dim]
                 (from line graph convolution)
        """
        batch_size = theta_h.size(0)
        
        # Positive pairs: matching sessions from both views
        pos_scores = self.discriminator(theta_h, theta_l)
        
        # Negative pairs: corrupted hypergraph with original line graph
        theta_h_corrupted = self.corrupt_embeddings(theta_h)
        neg_scores = self.discriminator(theta_h_corrupted, theta_l)
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8)
        
        loss = torch.mean(pos_loss + neg_loss)
        
        return loss

# Equation 10: Joint Learning Objective
def joint_training_objective(L_r, L_s, beta=0.01):
    """
    L = L_r + β L_s
    
    Combine recommendation loss with self-supervised loss
    
    Args:
        L_r: recommendation loss from Eq. 7
        L_s: self-supervised loss from Eq. 9
        beta: weight for auxiliary task
    """
    return L_r + beta * L_s`;

  const completeModelCode = `# Complete DHCN Model Implementation
class DHCN(nn.Module):
    """
    Dual Channel Hypergraph Convolutional Networks
    for Session-based Recommendation
    """
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100, 
                 n_layers=3, max_session_len=50, beta=0.01):
        super().__init__()
        
        # Item embeddings
        self.embedding = nn.Embedding(n_items + 1, embedding_dim, 
                                    padding_idx=0)
        
        # Hypergraph channel layers
        self.hypergraph_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hypergraph_layers.append(HypergraphConvLayer())
        
        # Line graph channel layers
        self.line_graph_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.line_graph_layers.append(LineGraphConvolution())
        
        # Position encoding
        self.position_encoding = PositionEncoding(
            embedding_dim, max_session_len
        )
        
        # Soft attention
        self.attention = SoftAttention(embedding_dim)
        
        # Self-supervised learning
        self.contrastive = ContrastiveLearning()
        
        # Hyperparameters
        self.n_layers = n_layers
        self.beta = beta
        
    def construct_hypergraph(self, sessions):
        """
        Build hypergraph where each session is a hyperedge
        """
        batch_size = len(sessions)
        n_items = self.embedding.num_embeddings - 1
        
        # Incidence matrix
        H = torch.zeros(n_items + 1, batch_size)
        
        for idx, session in enumerate(sessions):
            items = session[session > 0]  # Remove padding
            if len(items) > 0:
                H[items, idx] = 1
        
        return H
    
    def forward(self, sessions, lengths, training=True):
        """
        Forward pass through dual channels
        
        Args:
            sessions: batch of session sequences [batch_size, max_len]
            lengths: actual lengths of sessions [batch_size]
            training: whether in training mode
        """
        batch_size = sessions.size(0)
        
        # 1. Construct hypergraph from sessions
        H = self.construct_hypergraph(sessions)
        
        # 2. Get all item embeddings
        X_0 = self.embedding.weight  # [n_items+1, embed_dim]
        
        # === Hypergraph Channel ===
        X_h_layers = [X_0]
        X_h = X_0
        
        # Apply L layers of hypergraph convolution
        for layer in self.hypergraph_layers:
            X_h = layer(X_h, H)
            X_h_layers.append(X_h)
        
        # Average all layers (including input)
        X_h_final = torch.mean(torch.stack(X_h_layers), dim=0)
        
        # === Line Graph Channel (for self-supervised learning) ===
        theta_h_list = []  # Collect θ_i^h from each session
        theta_l_list = []  # Collect θ_i^l from each session
        
        if training:
            # Construct line graph
            line_graph_conv = self.line_graph_layers[0]
            A = line_graph_conv.construct_line_graph(H)
            
            # Initialize session embeddings from hypergraph output
            Theta_0 = line_graph_conv.init_session_embeds(H, X_h_final)
            
            # Line graph convolution
            Theta_l_layers = [Theta_0]
            Theta_l = Theta_0
            
            for layer in self.line_graph_layers:
                Theta_l = layer(Theta_l, A)
                Theta_l_layers.append(Theta_l)
            
            # Average all layers
            Theta_l_final = torch.mean(torch.stack(Theta_l_layers), dim=0)
        
        # 3. Process each session for recommendation
        all_scores = []
        
        for idx in range(batch_size):
            seq_len = lengths[idx].item()
            if seq_len == 0:
                # Empty session
                scores = torch.zeros(self.embedding.num_embeddings - 1)
                all_scores.append(scores)
                continue
            
            # Get session items
            session_items = sessions[idx, :seq_len]
            session_items = session_items[session_items > 0]
            
            # Get item embeddings for this session from hypergraph output
            item_embeds = X_h_final[session_items]
            
            # Add position encoding (Eq. 3)
            item_embeds_with_pos = self.position_encoding(
                item_embeds, len(session_items)
            )
            
            # Apply soft attention (Eq. 4)
            # theta_h here is θ_i^h for session i
            theta_h, _ = self.attention(item_embeds_with_pos, item_embeds)
            
            # Store for contrastive learning
            if training:
                theta_h_list.append(theta_h)  # θ_i^h from hypergraph
                theta_l_list.append(Theta_l_final[idx])  # θ_i^l from line graph
            
            # Compute scores (Eq. 5)
            item_embeds_all = X_h_final[1:]  # Skip padding token
            scores = torch.matmul(theta_h, item_embeds_all.t())
            all_scores.append(scores)
        
        # Stack scores
        scores = torch.stack(all_scores)
        
        # Compute self-supervised loss if training
        if training and len(theta_h_list) > 0:
            # Stack embeddings from both channels
            theta_h_batch = torch.stack(theta_h_list)  # θ^h for all sessions
            theta_l_batch = torch.stack(theta_l_list)  # θ^l for all sessions
            
            # Contrastive loss (Eq. 9)
            L_s = self.contrastive(theta_h_batch, theta_l_batch)
            
            return scores, L_s
        
        return scores
    
    def compute_loss(self, scores, labels, L_s=None):
        """
        Compute joint loss (Eq. 10)
        """
        # Recommendation loss (Eq. 7)
        L_r = F.cross_entropy(scores, labels)
        
        # Total loss
        if L_s is not None:
            loss = L_r + self.beta * L_s
            return loss, L_r, L_s
        else:
            return L_r

# Training loop example
def train_dhcn(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        sessions = batch['sessions'].to(device)
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        scores, L_s = model(sessions, lengths, training=True)
        
        # Compute joint loss
        loss, L_r, L_s = model.compute_loss(scores, labels, L_s)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)`;

  return (
    <ComparisonLayout
      title="DHCN: Dual Channel Hypergraph Convolutional Networks (Improved)"
      description="Enhanced version with explicit dependencies and clarified connections between equations. Based on 'Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation' by Xia et al. (AAAI 2021)."
    >
      {/* Section 1: Hypergraph Convolution */}
      <ComparisonSection
        title="1. Hypergraph Convolution Foundation"
        leftContent={
          <>
            <p className="mb-4">Spectral hypergraph convolution propagates embeddings through hyperedges:</p>
            <EquationBlock 
              math="\mathbf{x}_{i}^{(l+1)}=\sum_{j=1}^{N} \sum_{\epsilon=1}^{M} H_{i \epsilon} H_{j \epsilon} W_{\epsilon \epsilon} \mathbf{x}_{j}^{(l)}"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{x}_i^{(l)}"}</InlineMath>: node <InlineMath>{"i"}</InlineMath> embedding at layer <InlineMath>{"l"}</InlineMath></li>
              <li><InlineMath>{"H_{i\\epsilon}"}</InlineMath>: incidence matrix entry (1 if node <InlineMath>{"i"}</InlineMath> in hyperedge <InlineMath>{"\\epsilon"}</InlineMath>)</li>
              <li><InlineMath>{"W_{\\epsilon\\epsilon}"}</InlineMath>: weight of hyperedge <InlineMath>{"\\epsilon"}</InlineMath> (set to 1 in DHCN)</li>
              <li><InlineMath>{"N"}</InlineMath>: number of nodes (items)</li>
              <li><InlineMath>{"M"}</InlineMath>: number of hyperedges (sessions)</li>
            </ul>
            <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-400">
              <p className="text-sm">
                <strong>Foundation:</strong> This is the core operation that all subsequent computations build upon.
              </p>
            </div>
          </>
        }
        leftTitle="Mathematical Formulation"
        code={hypergraphConvCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="hypergraphConvEq"
        rightId="hypergraphConvCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="hypergraphConv"
      />

      {/* Section 2: Matrix Form */}
      <ComparisonSection
        title="2. Matrix Form Implementation"
        leftContent={
          <>
            <p className="mb-4">Efficient matrix form of hypergraph convolution:</p>
            <EquationBlock 
              math="\mathbf{X}_{h}^{(l+1)}=\mathbf{D}^{-1} \mathbf{H} \mathbf{W} \mathbf{B}^{-1} \mathbf{H}^{\mathrm{T}} \mathbf{X}_{h}^{(l)}"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{X}_h^{(l)} \\in \\mathbb{R}^{N \\times d}"}</InlineMath>: node feature matrix at layer <InlineMath>{"l"}</InlineMath></li>
              <li><InlineMath>{"\\mathbf{D} \\in \\mathbb{R}^{N \\times N}"}</InlineMath>: diagonal node degree matrix</li>
              <li><InlineMath>{"\\mathbf{H} \\in \\mathbb{R}^{N \\times M}"}</InlineMath>: incidence matrix</li>
              <li><InlineMath>{"\\mathbf{W} \\in \\mathbb{R}^{M \\times M}"}</InlineMath>: diagonal edge weight matrix (identity in DHCN)</li>
              <li><InlineMath>{"\\mathbf{B} \\in \\mathbb{R}^{M \\times M}"}</InlineMath>: diagonal edge degree matrix</li>
            </ul>
            <div className="mt-6 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Dependency:</strong> Uses <InlineMath>{"\\mathbf{X}_h^{(l)}"}</InlineMath> from previous layer (or initial embeddings).
              </p>
            </div>
          </>
        }
        leftTitle="Mathematical Formulation"
        code={matrixFormCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="matrixFormEq"
        rightId="matrixFormCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="matrixFormCode"
        editorPlaceholder="Try implementing the matrix form of hypergraph convolution..."
      />

      {/* Section 3: Layer Aggregation */}
      <ComparisonSection
        title="3. Multi-Layer Aggregation"
        leftContent={
          <>
            <p className="mb-4">Average embeddings from all layers to capture multi-scale features:</p>
            <EquationBlock 
              math="\mathbf{X}_h = \frac{1}{L+1}\sum_{l=0}^{L} \mathbf{X}_h^{(l)}"
              description="(Implicit in paper implementation)"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{X}_h^{(0)}"}</InlineMath>: initial item embeddings</li>
              <li><InlineMath>{"\\mathbf{X}_h^{(l)}"}</InlineMath>: embeddings after <InlineMath>{"l"}</InlineMath> layers of convolution</li>
              <li><InlineMath>{"L"}</InlineMath>: number of convolution layers (typically 3)</li>
              <li><InlineMath>{"\\mathbf{X}_h"}</InlineMath>: final averaged embeddings used for downstream tasks</li>
            </ul>
            <div className="mt-6 p-4 bg-purple-50 border-l-4 border-purple-400">
              <p className="text-sm">
                <strong>Output:</strong> <InlineMath>{"\\mathbf{X}_h"}</InlineMath> contains the final item representations that will be used in position encoding and scoring.
              </p>
            </div>
          </>
        }
        leftTitle="Layer Aggregation"
        code={layerAggregationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="layerAggregationEq"
        rightId="layerAggregationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="layerAggregation"
      />

      {/* Section 4: Position Encoding */}
      <ComparisonSection
        title="4. Reversed Position Embeddings"
        leftContent={
          <>
            <p className="mb-4">Integrate reversed position embeddings with item representations:</p>
            <EquationBlock 
              math="\mathbf{x}^{*}_{t}=\tanh \left(\mathbf{W}_{1}\left[\mathbf{x}_{t} \| \mathbf{p}_{m-t+1}\right]+\mathbf{b}\right)"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{x}_t"}</InlineMath>: embedding of <InlineMath>{"t"}</InlineMath>-th item from <InlineMath>{"\\mathbf{X}_h"}</InlineMath></li>
              <li><InlineMath>{"\\mathbf{p}_{m-t+1}"}</InlineMath>: reversed position embedding</li>
              <li><InlineMath>{"\\|"}</InlineMath>: concatenation operation</li>
              <li><InlineMath>{"\\mathbf{W}_1 \\in \\mathbb{R}^{d \\times 2d}"}</InlineMath>: transformation matrix</li>
              <li><InlineMath>{"m"}</InlineMath>: session length</li>
            </ul>
            <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Input:</strong> Uses item embeddings <InlineMath>{"\\mathbf{x}_t"}</InlineMath> from the aggregated hypergraph convolution output <InlineMath>{"\\mathbf{X}_h"}</InlineMath>.
              </p>
            </div>
          </>
        }
        leftTitle="Position Encoding"
        code={positionEncodingCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="positionEncodingEq"
        rightId="positionEncodingCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="positionEncoding"
      />

      {/* Section 5: Soft Attention */}
      <ComparisonSection
        title="5. Soft Attention Mechanism"
        leftContent={
          <>
            <FormulaWithDescription
              title="Attention weights:"
              math="\alpha_{t}=\mathbf{f}^{\top} \sigma\left(\mathbf{W}_{2} \mathbf{x}^{*}_{s}+\mathbf{W}_{3} \mathbf{x}^{*}_{t}+\mathbf{c}\right)"
            />
            
            <FormulaWithDescription
              title="Session representation (becomes θᵢʰ for session i):"
              math="\mathbf{\theta}_{h}=\sum_{t=1}^{m} \alpha_{t} \mathbf{x}_{t}"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{x}^*_s = \\frac{1}{m}\\sum_{t=1}^{m}\\mathbf{x}^*_t"}</InlineMath>: average of position-enhanced embeddings</li>
              <li><InlineMath>{"\\mathbf{x}^*_t"}</InlineMath>: position-enhanced item embedding from Eq. 3</li>
              <li><InlineMath>{"\\mathbf{x}_t"}</InlineMath>: original item embedding (without position)</li>
            </ul>
            <div className="mt-6 p-4 bg-indigo-50 border-l-4 border-indigo-400">
              <p className="text-sm">
                <strong>Output:</strong> <InlineMath>{"\\mathbf{\\theta}_h"}</InlineMath> is the session representation that becomes <InlineMath>{"\\theta_i^h"}</InlineMath> in the self-supervised loss (Eq. 9).
              </p>
            </div>
          </>
        }
        leftTitle="Attention Formulation"
        code={softAttentionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="softAttentionEq"
        rightId="softAttentionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="softAttention"
      />

      {/* Section 6: Score Computation */}
      <ComparisonSection
        title="6. Score Computation and Recommendation Loss"
        leftContent={
          <>
            <FormulaWithDescription
              title="Score computation (Eq. 5):"
              math="\hat{\mathbf{z}}_{i}=\mathbf{\theta}_{h}^{T}\mathbf{x}_{i}"
            />
            <FormulaWithDescription
              title="Probability distribution (Eq. 6):"
              math="\hat{\mathbf{y}}=\operatorname{softmax}(\hat{\mathbf{z}})"
            />
            <FormulaWithDescription
              title="Cross-entropy loss (Eq. 7):"
              math="\mathcal{L}_{r}=-\sum_{i=1}^{N} \mathbf{y}_{i} \log \left(\hat{\mathbf{y}}_{i}\right)+\left(1-\mathbf{y}_{i}\right) \log \left(1-\hat{\mathbf{y}}_{i}\right)"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{\\theta}_h"}</InlineMath>: session representation from attention (Eq. 4)</li>
              <li><InlineMath>{"\\mathbf{x}_i"}</InlineMath>: item <InlineMath>{"i"}</InlineMath> embedding from <InlineMath>{"\\mathbf{X}_h"}</InlineMath></li>
              <li><InlineMath>{"\\mathbf{y}"}</InlineMath>: one-hot ground truth vector</li>
            </ul>
            <div className="mt-6 p-4 bg-orange-50 border-l-4 border-orange-400">
              <p className="text-sm">
                <strong>Output:</strong> <InlineMath>{"\\mathcal{L}_r"}</InlineMath> is used in the joint objective (Eq. 10).
              </p>
            </div>
          </>
        }
        leftTitle="Prediction Formulation"
        code={scoreComputationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="scoreComputationEq"
        rightId="scoreComputationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="scoreComputation"
      />

      {/* Section 7: Line Graph */}
      <ComparisonSection
        title="7. Line Graph Channel"
        leftContent={
          <>
            <p className="mb-4">Line Graph Construction:</p>
            <ul className="list-disc pl-5 space-y-1 mb-4">
              <li>Each hyperedge (session) → node in line graph</li>
              <li>Edge weight: <InlineMath>{"W_{pq} = |e_p \\cap e_q| / |e_p \\cup e_q|"}</InlineMath></li>
              <li>Initialize: <InlineMath>{"\\mathbf{\\Theta}_l^{(0)}"}</InlineMath> from averaged item embeddings in each session</li>
            </ul>
            
            <p className="mb-4">Line graph convolution:</p>
            <EquationBlock 
              math="\mathbf{\Theta}^{(l+1)}_{l}=\mathbf{\hat{D}}^{-1} \mathbf{\hat{A}} \mathbf{\Theta}^{(l)}_{l}"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathbf{\\Theta}_l^{(l)} \\in \\mathbb{R}^{M \\times d}"}</InlineMath>: session embeddings</li>
              <li><InlineMath>{"\\mathbf{\\hat{A}} = \\mathbf{A} + \\mathbf{I}"}</InlineMath>: adjacency with self-loops</li>
            </ul>
            <div className="mt-6 p-4 bg-teal-50 border-l-4 border-teal-400">
              <p className="text-sm">
                <strong>Output:</strong> Final <InlineMath>{"\\mathbf{\\Theta}_l"}</InlineMath> provides <InlineMath>{"\\theta_i^l"}</InlineMath> for each session <InlineMath>{"i"}</InlineMath> used in Eq. 9.
              </p>
            </div>
          </>
        }
        leftTitle="Line Graph Theory"
        code={lineGraphCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="lineGraphEq"
        rightId="lineGraphCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="lineGraph"
      />

      {/* Section 8: Self-Supervised Learning */}
      <ComparisonSection
        title="8. Self-Supervised Contrastive Learning"
        leftContent={
          <>
            <FormulaWithDescription
              title="Contrastive loss (Eq. 9):"
              math="\mathcal{L}_{s}=-\log\sigma(f_{\mathrm{D}}(\theta^{h}_{i}, \theta^{l}_{i}))-\log\sigma(1- f_{\mathrm{D}}(\tilde\theta^{h}_{i}, \theta^{l}_{i}))"
            />
            <FormulaWithDescription
              title="Joint objective (Eq. 10):"
              math="\mathcal{L}=\mathcal{L}_{r}+\beta\mathcal{L}_{s}"
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\theta_i^h"}</InlineMath>: session <InlineMath>{"i"}</InlineMath> embedding from hypergraph (output of Eq. 4)</li>
              <li><InlineMath>{"\\theta_i^l"}</InlineMath>: session <InlineMath>{"i"}</InlineMath> embedding from line graph (output of Eq. 8)</li>
              <li><InlineMath>{"\\mathcal{L}_r"}</InlineMath>: recommendation loss from Eq. 7</li>
              <li><InlineMath>{"\\beta"}</InlineMath>: weight for self-supervised task</li>
            </ul>
            <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-400">
              <p className="text-sm">
                <strong>Dependencies:</strong> Combines outputs from both channels and the recommendation loss to form the final training objective.
              </p>
            </div>
          </>
        }
        leftTitle="Self-Supervised Learning"
        code={selfSupervisedCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="selfSupLossEq"
        rightId="selfSupervisedCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="selfSupervisedCode"
        editorPlaceholder="Try implementing the contrastive learning objective..."
      />

      {/* Section 9: Complete Architecture */}
      <ComparisonSection
        title="9. Complete DHCN Architecture"
        leftTitle="Model Overview"
        code={completeModelCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="completeModelCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="completeModelCode"
        editorPlaceholder="Try implementing the complete DHCN model..."
        fullWidth={true}
      />

      {/* Dependency Flow Diagram */}
      <div className="mt-12 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Dependency Flow</h2>
        
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">X⁽⁰⁾</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Hypergraph Conv (Eq. 1-2)</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Layer Avg</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Xₕ</span>
          </div>
          
          <div className="ml-8 space-y-4">
            <div className="flex items-center space-x-2">
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">Xₕ</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">Position Encoding (Eq. 3)</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">x*ₜ</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">x*ₜ</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">Soft Attention (Eq. 4)</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">θₕ (= θᵢʰ)</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">θₕ, Xₕ</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">Score Comp (Eq. 5-7)</span>
              <span>→</span>
              <span className="font-mono bg-gray-100 px-2 py-1 rounded">Lᵣ</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Xₕ</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Line Graph Conv (Eq. 8)</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Θₗ (contains θᵢˡ)</span>
          </div>
          
          <div className="flex items-center space-x-2 mt-4 pt-4 border-t">
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">θᵢʰ, θᵢˡ, Lᵣ</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">Joint Loss (Eq. 9-10)</span>
            <span>→</span>
            <span className="font-mono bg-gray-100 px-2 py-1 rounded">L</span>
          </div>
        </div>
      </div>
    </ComparisonLayout>
  );
}