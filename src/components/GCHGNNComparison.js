import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription, SubHeader } from './ComparisonLayout';
import '../styles/ComparisonLayout.css';

export default function GCHGNNComparison() {
  // Initialize visibility state - organized by equation-code pairs
  const [visibility, setVisibility] = useState({
    // Section 1: Basic Definitions
    hypergraphDefEq: true,
    sessionHyperedgeEq: true,
    // Section 2: Input Embeddings
    itemEmbedEq: true,
    inputEmbedCode: true,
    // Section 3: Global Hypergraph Construction
    incidenceMatrixEq: true,
    degreeMatricesEq: true,
    hypergraphConstructCode: true,
    // Section 4: Global Hypergraph Convolution
    normalizedAdjacencyEq: true,
    globalPropagationEq: true,
    globalAverageEq: true,
    hypergraphPropagationCode: true,
    // Section 5: Local Graph Construction
    sessionGraphEq: true,
    edgeCreationEq: true,
    localGraphCode: true,
    // Section 6: Graph Attention Network (GAT)
    gatAttentionEq1: true,
    gatAttentionEq2: true,
    gatAttentionEq3: true,
    gatLocalCode: true,
    // Section 7: Feature Fusion
    fusionEq: true,
    fusionCode: true,
    // Section 8: Position Encoding
    positionConcatEq: true,
    positionTransformEq: true,
    positionCode: true,
    // Section 9: Session Attention
    preliminarySessionEq: true,
    attentionScoreEq: true,
    attentionWeightsEq: true,
    finalSessionEq: true,
    sessionAttentionCode: true,
    // Section 10: Prediction and Loss
    predictionScoreEq: true,
    predictionSoftmaxEq: true,
    lossEq: true,
    predictionCode: true,
    // Section 11: Complete Model
    completeModelCode: true,
    // Section 12: Training Pipeline
    trainingCode: true,
    // Section 13: Making Recommendations
    recommendationCode: true
  });

  const toggleVisibility = (id) => {
    setVisibility(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const [userCode, setUserCode] = useState({
    inputEmbedCode: '',
    hypergraphConstructCode: '',
    hypergraphPropagationCode: '',
    localGraphCode: '',
    gatLocalCode: '',
    fusionCode: '',
    positionCode: '',
    sessionAttentionCode: '',
    predictionCode: '',
    completeModelCode: '',
    trainingCode: '',
    recommendationCode: ''
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
  const inputEmbedCode = `# Equation 1: Item Embeddings
class GCHGNN(nn.Module):
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100, 
                 n_layers=3, n_heads=4, dropout=0.25, alpha=0.2):
        super().__init__()
        
        # Item embedding matrix E_ID ∈ R^{(n+1) × d}
        # +1 for padding token at index 0
        self.item_embedding = nn.Embedding(
            n_items + 1, 
            embedding_dim, 
            padding_idx=0
        )
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.item_embedding.weight[1:], 0, 0.1)
        
    def get_item_embedding(self, item_id):
        """
        x_i = E_ID(i) ∈ R^d
        
        Args:
            item_id: Item index (1-based)
        Returns:
            d-dimensional embedding vector
        """
        return self.item_embedding(item_id)`;

  const hypergraphConstructCode = `# Equations for Global Hypergraph Construction
def build_global_hypergraph(self, sessions):
    """
    Build global hypergraph where each session forms a hyperedge
    H[i,j] = 1 if item i ∈ session j, else 0
    """
    print("Building global hypergraph...")
    
    # Each session is a hyperedge connecting all its items
    n_hyperedges = len(sessions)
    rows, cols = [], []
    
    for edge_idx, session in enumerate(sessions):
        # Remove padding (item 0)
        items = [item for item in session if item > 0]
        if len(items) < 2:  # Skip sessions with less than 2 items
            continue
            
        # Add all items in this session to the hyperedge
        for item in items:
            rows.append(item)      # item index (1-based)
            cols.append(edge_idx)  # hyperedge (session) index
    
    # Create incidence matrix H ∈ {0,1}^{(n+1) × m}
    # n+1 items (including padding), m hyperedges
    data = np.ones(len(rows))
    self.H = sp.coo_matrix(
        (data, (rows, cols)), 
        shape=(self.n_items + 1, n_hyperedges)
    )
    
    # Compute degree matrices
    # D_v[i] = Σ_j H[i,j] - degree of vertex i
    self.D_v = np.array(self.H.sum(axis=1)).flatten()
    
    # D_e[j] = Σ_i H[i,j] - degree of hyperedge j  
    self.D_e = np.array(self.H.sum(axis=0)).flatten()
    
    # Avoid division by zero
    self.D_v[self.D_v == 0] = 1
    self.D_e[self.D_e == 0] = 1
    
    print(f"Global hypergraph: {self.n_items} items, {n_hyperedges} hyperedges")`;

  const hypergraphPropagationCode = `# Equations 2-3: Global Hypergraph Convolution
def hypergraph_propagation(self, x):
    """
    Multi-layer hypergraph convolution for global context
    
    Layer propagation (Eq. 2):
    h^{(l+1)} = σ(S^{(G)} h^{(l)} W^{(l)})
    
    Where S^{(G)} = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
    
    Final representation (Eq. 3):
    h^{(G)} = (1/L) Σ_{l=1}^L h^{(l)}
    """
    device = x.device
    
    # Compute normalization factors
    D_v_inv = torch.from_numpy(1.0 / self.D_v).float().to(device)
    D_e_inv = torch.from_numpy(1.0 / self.D_e).float().to(device)
    D_v_sqrt_inv = torch.sqrt(D_v_inv)
    
    # Convert sparse H to torch sparse tensor for efficient computation
    H_coo = self.H.tocoo()
    indices = torch.LongTensor([H_coo.row, H_coo.col]).to(device)
    values = torch.FloatTensor(H_coo.data).to(device)
    H_sparse = torch.sparse.FloatTensor(indices, values, self.H.shape)
    
    # Apply L layers of hypergraph convolution
    h = x  # Initial features h^{(0)} = x
    layer_outputs = []
    
    for layer in self.hypergraph_layers:
        # Compute S^{(G)} h = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} h
        
        # Step 1: D_v^{-1/2} * h
        h_scaled = h * D_v_sqrt_inv.unsqueeze(-1)
        
        # Step 2: H^T * (D_v^{-1/2} * h) - aggregate from vertices to hyperedges
        h_hyper = torch.sparse.mm(H_sparse.t(), h_scaled)
        
        # Step 3: D_e^{-1} * aggregated features
        h_hyper = h_hyper * D_e_inv.unsqueeze(-1)
        
        # Step 4: H * (D_e^{-1} * features) - broadcast from hyperedges to vertices
        h_prop = torch.sparse.mm(H_sparse, h_hyper)
        
        # Step 5: D_v^{-1/2} * propagated features
        h_prop = h_prop * D_v_sqrt_inv.unsqueeze(-1)
        
        # Apply linear transformation W^{(l)} and activation σ
        h = layer(h_prop)  # layer contains W^{(l)} transformation
        h = self.leaky_relu(h)  # σ = LeakyReLU
        h = self.dropout_layer(h)
        
        layer_outputs.append(h)
    
    # Average all L layer outputs (Equation 3)
    h_global = torch.stack(layer_outputs).mean(dim=0)
    
    return h_global

# Hypergraph Convolution Layer
class HypergraphConv(nn.Module):
    """Single hypergraph convolution layer with learnable W"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        # Apply linear transformation: x W^{(l)}
        return torch.matmul(x, self.weight)`;

  const localGraphCode = `# Session Graph Construction for Local Context
def construct_session_graph(self, sequences, lengths):
    """
    Construct local directed graphs from session sequences
    Each session creates a separate graph with sequential edges
    """
    device = sequences.device
    batch_size = sequences.size(0)
    
    all_edge_indices = []
    all_items = []
    all_alias = []  # Maps sequence positions to graph node indices
    node_offset = 0
    
    for i in range(batch_size):
        seq_len = lengths[i].item()
        if seq_len == 0:
            all_alias.append(
                torch.zeros(sequences.size(1), dtype=torch.long, device=device)
            )
            continue
            
        # Get items in this session (remove padding)
        seq = sequences[i, :seq_len]
        seq = seq[seq > 0]
        
        if len(seq) == 0:
            all_alias.append(
                torch.zeros(sequences.size(1), dtype=torch.long, device=device)
            )
            continue
        
        # Get unique items and their mapping
        unique_items, inverse_indices = torch.unique(seq, return_inverse=True)
        num_nodes = len(unique_items)
        
        # Create edges for sequential patterns
        edges = []
        for j in range(len(seq) - 1):
            # Edge from item at position j to item at position j+1
            u = inverse_indices[j].item() + node_offset
            v = inverse_indices[j + 1].item() + node_offset
            edges.extend([[u, v], [v, u]])  # Bidirectional edges
        
        # Add self-loops for all nodes
        for j in range(num_nodes):
            u = j + node_offset
            edges.append([u, u])
        
        if edges:
            edge_index = torch.tensor(edges, device=device).t()
            all_edge_indices.append(edge_index)
        
        all_items.append(unique_items)
        
        # Create alias mapping: sequence position -> graph node index
        alias = torch.zeros(sequences.size(1), dtype=torch.long, device=device)
        alias[:len(seq)] = inverse_indices + node_offset + 1  # +1 for padding
        all_alias.append(alias)
        
        node_offset += num_nodes
    
    # Combine all graphs
    if not all_items:
        return None, None, torch.stack(all_alias)
    
    items = torch.cat(all_items)
    if all_edge_indices:
        edge_index = torch.cat(all_edge_indices, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    alias_inputs = torch.stack(all_alias)
    
    return edge_index, items, alias_inputs`;

  const gatLocalCode = `# Equations 4-6: Graph Attention Network for Local Context
def apply_gat_layers(self, h_local, edge_index):
    """
    Multi-head Graph Attention Network layers
    
    Attention coefficients (Eq. 4):
    e_{ij} = LeakyReLU(a^T [W h_i || W h_j])
    
    Normalized attention weights (Eq. 5):
    α_{ij} = exp(e_{ij}) / Σ_{k∈N(i)} exp(e_{ik})
    
    Node update (Eq. 6):
    h_i^{(L)} = σ(Σ_{j∈N(i)} α_{ij} W h_j)
    """
    # Initialize GAT layers in __init__
    self.gat_layers = nn.ModuleList()
    for i in range(self.n_layers):
        in_dim = self.embedding_dim if i == 0 else self.hidden_dim
        # Multi-head attention except last layer
        heads = 1 if i == self.n_layers - 1 else self.n_heads
        
        self.gat_layers.append(
            GATConv(
                in_dim, 
                self.hidden_dim, 
                heads=heads,
                dropout=self.dropout,
                negative_slope=self.alpha,  # For LeakyReLU
                concat=(i < self.n_layers - 1)  # Concat heads except last
            )
        )
    
    # Apply GAT layers
    h = h_local
    for i, gat_layer in enumerate(self.gat_layers):
        # GATConv internally computes:
        # 1. Linear transformations: W h_i, W h_j
        # 2. Attention: e_ij = LeakyReLU(a^T [W h_i || W h_j])
        # 3. Softmax: α_ij = softmax(e_ij)
        # 4. Aggregation: h_i' = Σ α_ij W h_j
        
        h = gat_layer(h, edge_index)
        
        # Apply activation and dropout (except last layer)
        if i < len(self.gat_layers) - 1:
            h = F.elu(h)  # σ = ELU activation
            h = self.dropout_layer(h)
    
    return h

# In forward pass:
# Get embeddings for items in local graph
local_embeds = self.item_embedding(local_items)  # [num_local_items, d]

# Apply GAT layers for local context
h_local = self.apply_gat_layers(local_embeds, edge_index)  # [num_local_items, d']`;

  const fusionCode = `# Equation 7: Fusion of Global and Local Features
def fuse_features(self, h_global, h_local, local_items):
    """
    h_i* = Aggregate(h_i^{(G)}, h_i^{(L)})
    
    The paper uses sum-pooling as the aggregation function.
    This combines global hypergraph features with local GAT features.
    """
    # Start with global features for all items
    h_fused = h_global.clone()  # [n_items+1, hidden_dim]
    
    # Add local features where available (sum pooling)
    # local_items contains the actual item IDs present in current batch
    h_fused[local_items] = h_fused[local_items] + h_local
    
    return h_fused

# In forward pass:
# Global context via hypergraph convolution
h_global = self.hypergraph_propagation(all_item_embeds)

# Local context via GAT  
h_local = self.apply_gat_layers(local_embeds, edge_index)

# Fusion: combine global and local features
h_fused = h_global.clone()
h_fused[local_items] = h_fused[local_items] + h_local`;

  const positionCode = `# Equations 8-9: Positional Encoding
def add_position_encoding(self, seq_h, batch_size, max_len):
    """
    Incorporate position information into item representations
    
    Position concatenation (Eq. 8):
    x_i* = [h_i* ; p_i]
    
    Transformation (Eq. 9):
    x_i* = tanh(W_1 x_i* + b)
    """
    device = seq_h.device
    
    # Initialize position embedding in __init__
    self.position_embedding = nn.Embedding(200, self.embedding_dim)
    self.W1 = nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim)
    
    # Generate position indices [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    positions = positions.expand(batch_size, -1)  # [batch_size, max_len]
    
    # Get position embeddings p_i
    pos_embeds = self.position_embedding(positions)  # [batch_size, max_len, d]
    
    # Concatenate item features with position embeddings
    # seq_h: [batch_size, max_len, hidden_dim] 
    # pos_embeds: [batch_size, max_len, embedding_dim]
    seq_h_with_pos = torch.cat([seq_h, pos_embeds], dim=-1)
    
    # Apply transformation W_1 and activation
    x_star = self.W1(seq_h_with_pos)  # [batch_size, max_len, hidden_dim]
    x_star = torch.tanh(x_star)
    
    return x_star

# In forward pass:
# Map fused features to sequences using alias
padded_h = torch.cat([
    torch.zeros(1, h_fused.size(-1), device=device),  # Padding embedding
    h_fused
], dim=0)
seq_h = padded_h[alias_inputs]  # [batch_size, max_len, hidden_dim]

# Add position encodings
x_star = self.add_position_encoding(seq_h, batch_size, max_len)`;

  const sessionAttentionCode = `# Equations 10-13: Session Representation with Attention
def compute_session_representation(self, seq_h, x_star, alias_inputs):
    """
    Compute final session representation using attention mechanism
    
    Preliminary session vector (Eq. 10):
    s* = (1/|S|) Σ_{i∈S} h_i*
    
    Attention scores (Eq. 11):
    a_i = q^T (W_2 x_i* + W_3 s* + c)
    
    Attention weights (Eq. 12):
    α_i = exp(a_i) / Σ_{j∈S} exp(a_j)
    
    Final session representation (Eq. 13):
    S = Σ_{i∈S} α_i h_i*
    """
    # Initialize parameters in __init__
    self.W2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.W3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    self.q = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
    self.W_f = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    # Create mask for valid positions (non-padding)
    mask = alias_inputs.gt(0).float().unsqueeze(-1)  # [batch_size, max_len, 1]
    
    # Equation 10: Preliminary session vector (average pooling)
    s_star = (seq_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    # s_star: [batch_size, hidden_dim]
    
    # Equation 11: Compute attention scores
    # a_i = q^T (W_2 x_i* + W_3 s*)
    attention_scores = torch.matmul(
        self.W2(x_star) + self.W3(s_star).unsqueeze(1),  # [batch_size, max_len, hidden_dim]
        self.q  # [hidden_dim, 1]
    ).squeeze(-1)  # [batch_size, max_len]
    
    # Mask padding positions before softmax
    attention_scores = attention_scores.masked_fill(~alias_inputs.gt(0), -1e9)
    
    # Equation 12: Normalize with softmax
    attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, max_len]
    
    # Equation 13: Weighted sum using attention
    session_repr = torch.sum(
        attention_weights.unsqueeze(-1) * seq_h,  # [batch_size, max_len, hidden_dim]
        dim=1
    )  # [batch_size, hidden_dim]
    
    # Final transformation
    session_repr = self.W_f(session_repr)
    session_repr = self.dropout_layer(session_repr)
    
    return session_repr`;

  const predictionCode = `# Equations 14-16: Prediction and Loss Computation
def compute_scores(self, session_repr, h_global):
    """
    Score computation (Eq. 14):
    ẑ_i = S^T h_i^{(G)}
    
    Uses global embeddings for scoring all candidate items
    """
    # h_global[1:] excludes padding token at index 0
    scores = torch.matmul(session_repr, h_global[1:].t())
    # scores: [batch_size, n_items]
    
    return scores

def compute_loss(self, scores, targets, n_neg_samples=100):
    """
    Loss computation with negative sampling
    
    Probability (Eq. 15):
    ŷ_i = softmax(ẑ_i)
    
    Cross-entropy loss (Eq. 16):
    L = -Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
    """
    device = scores.device
    batch_size = len(targets)
    
    # Get positive item embeddings
    pos_items = targets  # 1-indexed
    pos_embeds = h_global[pos_items]
    pos_scores = torch.sum(session_repr * pos_embeds, dim=1)
    
    # Sample negative items randomly
    neg_items = torch.randint(
        1, self.n_items + 1, 
        (batch_size, n_neg_samples), 
        device=device
    )
    neg_embeds = h_global[neg_items]  # [batch_size, n_neg_samples, hidden_dim]
    
    # Compute negative scores
    neg_scores = torch.matmul(
        session_repr.unsqueeze(1),  # [batch_size, 1, hidden_dim]
        neg_embeds.transpose(1, 2)   # [batch_size, hidden_dim, n_neg_samples]
    ).squeeze(1)  # [batch_size, n_neg_samples]
    
    # Two loss options:
    if self.loss_type == 'BPR':
        # Bayesian Personalized Ranking loss
        pos_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)
        loss = -torch.log(torch.sigmoid(pos_expanded - neg_scores) + 1e-10).mean()
    else:
        # Cross-entropy loss (Equation 16)
        # Concatenate positive and negative scores
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        # Labels: positive item is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
    
    return loss`;

  const completeModelCode = `# Complete GC-HGNN Model Implementation
class GCHGNN(nn.Module):
    """
    Global-Context Hypergraph Neural Network for Session-based Recommendation
    
    Architecture:
    1. Global hypergraph construction from all sessions
    2. Multi-layer hypergraph convolution for global context
    3. Local graph attention for sequential patterns
    4. Feature fusion combining global and local
    5. Position-aware session encoding
    6. Attention-based session representation
    7. Prediction using global embeddings
    """
    
    def __init__(self, n_items, embedding_dim=100, hidden_dim=100, 
                 n_layers=3, n_heads=4, dropout=0.25, alpha=0.2):
        super(GCHGNN, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Item embedding (Equation 1)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, 
                                         padding_idx=0)
        
        # Global hypergraph convolution layers (Equation 2)
        self.hypergraph_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.hypergraph_layers.append(
                HypergraphConv(in_dim, hidden_dim)
            )
        
        # Local GAT layers (Equations 4-6)
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            heads = 1 if i == n_layers - 1 else n_heads
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim, heads=heads, 
                       dropout=dropout, negative_slope=alpha,
                       concat=(i < n_layers - 1))
            )
        
        # Position embedding (Equation 8)
        self.position_embedding = nn.Embedding(200, embedding_dim)
        
        # Session encoding layers (Equations 9-13)
        self.W1 = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        # Global hypergraph structure
        self.H = None
        self.D_v = None
        self.D_e = None
        
        self._init_weights()
    
    def forward(self, sequences, lengths):
        """
        Forward pass combining all components
        
        Args:
            sequences: [batch_size, max_len] padded item sequences
            lengths: [batch_size] actual lengths of sequences
            
        Returns:
            scores: [batch_size, n_items] prediction scores
        """
        device = sequences.device
        batch_size = sequences.size(0)
        max_len = sequences.size(1)
        
        # 1. Get all item embeddings (Equation 1)
        all_item_embeds = self.item_embedding.weight
        
        # 2. Global context via hypergraph (Equations 2-3)
        h_global = self.hypergraph_propagation(all_item_embeds)
        
        # 3. Local context via GAT (Equations 4-6)
        edge_index, local_items, alias_inputs = self.construct_session_graph(
            sequences, lengths
        )
        
        if local_items is None:
            return torch.zeros(batch_size, self.n_items, device=device)
        
        local_embeds = self.item_embedding(local_items)
        h_local = local_embeds
        for i, gat_layer in enumerate(self.gat_layers):
            h_local = gat_layer(h_local, edge_index)
            if i < len(self.gat_layers) - 1:
                h_local = F.elu(h_local)
                h_local = self.dropout_layer(h_local)
        
        # 4. Fusion (Equation 7)
        h_fused = h_global.clone()
        h_fused[local_items] = h_fused[local_items] + h_local
        
        # 5. Map to sequences and add positions (Equations 8-9)
        padded_h = torch.cat([
            torch.zeros(1, h_fused.size(-1), device=device), 
            h_fused
        ], dim=0)
        seq_h = padded_h[alias_inputs]
        
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        seq_h_with_pos = self.W1(torch.cat([seq_h, pos_embeds], dim=-1))
        x_star = torch.tanh(seq_h_with_pos)
        
        # 6. Session representation with attention (Equations 10-13)
        mask = alias_inputs.gt(0).float().unsqueeze(-1)
        s_star = (seq_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        attention_scores = torch.matmul(
            self.W2(x_star) + self.W3(s_star).unsqueeze(1), 
            self.q
        ).squeeze(-1)
        
        attention_scores = attention_scores.masked_fill(~alias_inputs.gt(0), -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        session_repr = torch.sum(attention_weights.unsqueeze(-1) * seq_h, dim=1)
        session_repr = self.W_f(session_repr)
        session_repr = self.dropout_layer(session_repr)
        
        # 7. Prediction scores (Equation 14)
        scores = torch.matmul(session_repr, h_global[1:].t())
        
        return scores`;

  const trainingCode = `# Training Pipeline for GC-HGNN
def train_gchgnn(model, train_data, val_data, epochs=30, batch_size=100, 
                 lr=0.001, weight_decay=1e-5):
    """
    Train GC-HGNN model on session data
    
    Key steps:
    1. Build global hypergraph from all training sessions
    2. Train with mini-batch gradient descent
    3. Use negative sampling for efficiency
    4. Validate periodically
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Step 1: Build global hypergraph from training data
    all_sessions = []
    for sequences, _, _ in train_data:
        for seq in sequences:
            items = seq[seq > 0].tolist()
            if len(items) >= 2:
                all_sessions.append(items)
    
    model.build_global_hypergraph(all_sessions)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                               weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )
    
    best_mrr = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Training loop
        for sequences, lengths, targets in train_data:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            
            # Forward pass
            scores = model(sequences, lengths)
            
            # Compute loss with negative sampling
            loss = model.compute_loss(sequences, lengths, targets, 
                                    n_neg_samples=100)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # Validation
        if (epoch + 1) % 3 == 0:
            mrr = evaluate_model(model, val_data)
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, MRR@20={mrr:.4f}')
            
            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(model.state_dict(), 'gchgnn_best.pth')
        
        scheduler.step()

def create_mini_batches(sessions, targets, batch_size):
    """Create mini-batches with padding"""
    n_sessions = len(sessions)
    indices = np.random.permutation(n_sessions)
    
    for start in range(0, n_sessions, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_sessions = [sessions[i] for i in batch_idx]
        batch_targets = [targets[i] for i in batch_idx]
        
        # Pad sequences
        lengths = [len(s) for s in batch_sessions]
        max_len = max(lengths)
        
        padded_sequences = torch.zeros(len(batch_sessions), max_len, 
                                      dtype=torch.long)
        for i, seq in enumerate(batch_sessions):
            padded_sequences[i, :len(seq)] = torch.tensor(seq)
        
        lengths = torch.tensor(lengths)
        targets = torch.tensor(batch_targets)
        
        yield padded_sequences, lengths, targets`;

  const recommendationCode = `# Making Recommendations with Trained GC-HGNN
def get_recommendations(model, session_sequence, k=20):
    """
    Generate top-k item recommendations for a given session
    
    Args:
        model: Trained GC-HGNN model
        session_sequence: List of item IDs in the session
        k: Number of recommendations to return
        
    Returns:
        top_k_items: List of top-k recommended item IDs
        scores: Corresponding scores
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        sequences = torch.tensor([session_sequence], device=device)
        lengths = torch.tensor([len(session_sequence)], device=device)
        
        # Get prediction scores
        scores = model(sequences, lengths)  # [1, n_items]
        
        # Apply softmax to get probabilities (Equation 15)
        probs = F.softmax(scores, dim=-1)
        
        # Get top-k items
        top_k_scores, top_k_indices = torch.topk(probs[0], k)
        
        # Convert to item IDs (add 1 because of 0-indexing)
        top_k_items = (top_k_indices + 1).cpu().tolist()
        top_k_scores = top_k_scores.cpu().tolist()
        
    return top_k_items, top_k_scores

def evaluate_model(model, test_data):
    """
    Evaluate model using MRR@20 and P@20 metrics
    """
    model.eval()
    total_mrr = 0.0
    total_p20 = 0.0
    n_sessions = 0
    
    with torch.no_grad():
        for sequences, lengths, targets in test_data:
            scores = model(sequences, lengths)
            
            # Get top-20 predictions for each session
            _, top_20 = torch.topk(scores, 20, dim=1)
            
            for i, target in enumerate(targets):
                # Find position of target in top-20
                positions = (top_20[i] == target - 1).nonzero()
                
                if len(positions) > 0:
                    # MRR@20
                    rank = positions[0].item() + 1
                    total_mrr += 1.0 / rank
                    # P@20
                    total_p20 += 1.0
                
                n_sessions += 1
    
    mrr = total_mrr / n_sessions
    p20 = total_p20 / n_sessions
    
    return mrr, p20

# Example usage
if __name__ == "__main__":
    # Initialize model
    n_items = 43097  # e.g., Diginetica dataset
    model = GCHGNN(
        n_items=n_items,
        embedding_dim=100,
        hidden_dim=100,
        n_layers=3,
        n_heads=4,
        dropout=0.25
    )
    
    # Example session
    session = [125, 214, 875, 125, 589]
    
    # Get recommendations
    recommendations, scores = get_recommendations(model, session, k=10)
    
    print(f"Session: {session}")
    print(f"Top-10 recommendations: {recommendations}")
    print(f"Scores: {scores}")`;

  return (
    <ComparisonLayout
      title="Global-Context Hypergraph Neural Networks (GC-HGNN) - Complete Implementation"
      description="This implementation follows 'GC-HGNN: A Global-Context Supported Hypergraph Neural Network for Enhancing Session-Based Recommendation' (Electronic Commerce Research and Applications, 2022). Each code section directly implements the corresponding mathematical equations from the paper."
    >
      {/* Section 1: Basic Definitions and Architecture Overview */}
      <ComparisonSection
        title="1. Basic Definitions and Architecture Overview"
        leftContent={
          <>
            <p className="mb-4">GC-HGNN combines global and local contexts for session-based recommendation:</p>
            
            <SubHeader>Key Innovation:</SubHeader>
            <p className="mb-4">The model captures both global item co-occurrence patterns (via hypergraph) and local sequential patterns (via GAT) in a unified framework.</p>
            
            <EquationBlock 
              math="\text{GC-HGNN} = \text{Global Hypergraph} + \text{Local GAT} + \text{Feature Fusion}"
              description=""
              id="architectureOverview"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <SubHeader>Session-as-Hyperedge Mapping:</SubHeader>
            <EquationBlock 
              math="\text{Session } s = \{i_1, i_2, ..., i_k\} \rightarrow \text{Hyperedge } e_s"
              description=""
              id="sessionHyperedgeEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"s"}</InlineMath>: a user session containing k items</li>
              <li><InlineMath>{"i_j \\in \\{1, 2, ..., n\\}"}</InlineMath>: item indices in the catalog</li>
              <li><InlineMath>{"e_s"}</InlineMath>: hyperedge representing session s</li>
              <li><InlineMath>{"n"}</InlineMath>: total number of unique items</li>
              <li><InlineMath>{"k"}</InlineMath>: number of items in the session</li>
            </ul>
            
            <div className="mt-4 p-4 bg-blue-50 border-l-4 border-blue-400">
              <p className="text-sm">
                <strong>Why hypergraphs?</strong> Traditional graphs only capture pairwise relations. 
                Hypergraphs naturally model many-to-many relations where a session (hyperedge) connects multiple items (vertices) that co-occurred.
              </p>
            </div>
          </>
        }
        fullWidth={true}
      />

      {/* Section 2: Input Embeddings */}
      <ComparisonSection
        title="2. Input Embeddings (Equation 1)"
        leftContent={
          <>
            <EquationBlock 
              math="x_i = E_{\text{ID}}(i) \in \mathbb{R}^d"
              description=""
              id="itemEmbedEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"x_i \\in \\mathbb{R}^d"}</InlineMath>: d-dimensional embedding vector for item i</li>
              <li><InlineMath>{"E_{\\text{ID}} \\in \\mathbb{R}^{(n+1) \\times d}"}</InlineMath>: learnable embedding matrix</li>
              <li><InlineMath>{"i \\in \\{1, 2, ..., n\\}"}</InlineMath>: item index (1-based)</li>
              <li><InlineMath>{"d"}</InlineMath>: embedding dimension (typically 100)</li>
              <li>Index 0 is reserved for padding</li>
            </ul>
          </>
        }
        code={inputEmbedCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="itemEmbedEq"
        rightId="inputEmbedCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="inputEmbed"
      />

      {/* Section 3: Global Hypergraph Construction */}
      <ComparisonSection
        title="3. Global Hypergraph Construction"
        leftContent={
          <>
            <p className="mb-4">Build a global hypergraph from all training sessions:</p>
            
            <FormulaWithDescription
              title="Incidence matrix:"
              math="H_{ij} = \begin{cases} 1, & \text{if item } i \in \text{session } j \\ 0, & \text{otherwise} \end{cases}"
              description=""
              id="incidenceMatrixEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"H \\in \\{0,1\\}^{(n+1) \\times m}"}</InlineMath>: incidence matrix</li>
              <li><InlineMath>{"n+1"}</InlineMath>: number of items (including padding)</li>
              <li><InlineMath>{"m"}</InlineMath>: number of hyperedges (sessions)</li>
              <li><InlineMath>{"H_{ij} = 1"}</InlineMath>: item i appears in session j</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Degree matrices:"
                math="D_v[i] = \sum_{j=1}^{m} H_{ij}, \quad D_e[j] = \sum_{i=1}^{n+1} H_{ij}"
                description=""
                id="degreeMatricesEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"D_v[i]"}</InlineMath>: degree of vertex i (number of sessions containing item i)</li>
                <li><InlineMath>{"D_e[j]"}</InlineMath>: degree of hyperedge j (number of items in session j)</li>
                <li><InlineMath>{"D_v \\in \\mathbb{R}^{n+1}"}</InlineMath>: vertex degree vector</li>
                <li><InlineMath>{"D_e \\in \\mathbb{R}^m"}</InlineMath>: hyperedge degree vector</li>
              </ul>
            </div>
          </>
        }
        code={hypergraphConstructCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="hypergraphConstructCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="hypergraphConstruct"
      />

      {/* Section 4: Global Hypergraph Convolution */}
      <ComparisonSection
        title="4. Global Hypergraph Convolution (Equations 2-3)"
        leftContent={
          <>
            <p className="mb-4">Multi-layer hypergraph convolution captures global context:</p>
            
            <FormulaWithDescription
              title="Normalized adjacency matrix:"
              math="S^{(G)} = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}"
              description=""
              id="normalizedAdjacencyEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"S^{(G)} \\in \\mathbb{R}^{(n+1) \\times (n+1)}"}</InlineMath>: normalized hypergraph adjacency</li>
              <li><InlineMath>{"D_v^{-1/2}"}</InlineMath>: inverse square root of vertex degrees</li>
              <li><InlineMath>{"D_e^{-1}"}</InlineMath>: inverse of hyperedge degrees</li>
              <li><InlineMath>{"H^T"}</InlineMath>: transpose of incidence matrix</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Layer propagation rule (Equation 2):"
                math="h^{(\ell+1)} = \sigma\left( S^{(G)} h^{(\ell)} W^{(\ell)} \right)"
                description=""
                id="globalPropagationEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"h^{(\\ell)} \\in \\mathbb{R}^{(n+1) \\times d_\\ell}"}</InlineMath>: features at layer ℓ</li>
                <li><InlineMath>{"W^{(\\ell)} \\in \\mathbb{R}^{d_\\ell \\times d_{\\ell+1}}"}</InlineMath>: learnable weight matrix</li>
                <li><InlineMath>{"\\sigma"}</InlineMath>: activation function (LeakyReLU)</li>
                <li><InlineMath>{"\\ell \\in \\{0, 1, ..., L-1\\}"}</InlineMath>: layer index</li>
              </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Final global representation (Equation 3):"
                math="h^{(G)} = \frac{1}{L} \sum_{\ell=1}^L h^{(\ell)}"
                description=""
                id="globalAverageEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"h^{(G)} \\in \\mathbb{R}^{(n+1) \\times d}"}</InlineMath>: final global embeddings</li>
                <li><InlineMath>{"L"}</InlineMath>: total number of layers (typically 3)</li>
                <li>Average pooling aggregates information from all layers</li>
              </ul>
            </div>
          </>
        }
        code={hypergraphPropagationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="hypergraphPropagationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="hypergraphPropagationCode"
        editorPlaceholder="Try implementing the hypergraph convolution..."
      />

      {/* Section 5: Local Graph Construction */}
      <ComparisonSection
        title="5. Local Graph Construction for Sessions"
        leftContent={
          <>
            <p className="mb-4">Construct local directed graphs to capture sequential patterns:</p>
            
            <FormulaWithDescription
              title="Session graph:"
              math="G_s = (V_s, E_s)"
              description=""
              id="sessionGraphEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"V_s"}</InlineMath>: set of unique items in session s</li>
              <li><InlineMath>{"E_s"}</InlineMath>: set of edges representing transitions</li>
              <li><InlineMath>{"G_s"}</InlineMath>: directed graph for session s</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Edge creation:"
                math="(i_j, i_{j+1}) \in E_s \text{ for } j = 1, ..., |s|-1"
                description=""
                id="edgeCreationEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"i_j"}</InlineMath>: item at position j in the session</li>
                <li><InlineMath>{"(i_j, i_{j+1})"}</InlineMath>: directed edge from item j to item j+1</li>
                <li>Bidirectional edges are added in implementation</li>
                <li>Self-loops are added for all nodes</li>
              </ul>
            </div>
          </>
        }
        code={localGraphCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="localGraphCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="localGraph"
      />

      {/* Section 6: Graph Attention Network */}
      <ComparisonSection
        title="6. Graph Attention Network for Local Context (Equations 4-6)"
        leftContent={
          <>
            <p className="mb-4">Apply multi-head graph attention to capture local sequential patterns:</p>
            
            <FormulaWithDescription
              title="Attention coefficient computation (Equation 4):"
              math="e_{ij} = \text{LeakyReLU}\left( a^\top [W h_i \Vert W h_j] \right)"
              description=""
              id="gatAttentionEq1"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"e_{ij}"}</InlineMath>: unnormalized attention score between nodes i and j</li>
              <li><InlineMath>{"a \\in \\mathbb{R}^{2d'}"}</InlineMath>: learnable attention vector</li>
              <li><InlineMath>{"W \\in \\mathbb{R}^{d \\times d'}"}</InlineMath>: learnable weight matrix</li>
              <li><InlineMath>{"h_i, h_j \\in \\mathbb{R}^d"}</InlineMath>: node feature vectors</li>
              <li><InlineMath>{"[\\cdot \\Vert \\cdot]"}</InlineMath>: concatenation operation</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Normalized attention weights (Equation 5):"
                math="\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}"
                description=""
                id="gatAttentionEq2"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"\\alpha_{ij}"}</InlineMath>: normalized attention weight</li>
                <li><InlineMath>{"\\mathcal{N}(i)"}</InlineMath>: neighborhood of node i</li>
                <li>Softmax normalization ensures weights sum to 1</li>
              </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Node feature update (Equation 6):"
                math="h_i^{(L)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)"
                description=""
                id="gatAttentionEq3"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"h_i^{(L)} \\in \\mathbb{R}^{d'}"}</InlineMath>: updated features for node i</li>
                <li><InlineMath>{"\\sigma"}</InlineMath>: activation function (ELU)</li>
                <li>Multi-head attention: multiple attention mechanisms in parallel</li>
                <li>Concatenate heads except in the last layer</li>
              </ul>
            </div>
          </>
        }
        code={gatLocalCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="gatLocalCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="gatLocal"
      />

      {/* Section 7: Feature Fusion */}
      <ComparisonSection
        title="7. Fusion of Global and Local Features (Equation 7)"
        leftContent={
          <>
            <EquationBlock 
              math="h_i^* = \text{Aggregate}(h_i^{(G)}, h_i^{(L)})"
              description=""
              id="fusionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"h_i^* \\in \\mathbb{R}^d"}</InlineMath>: fused feature vector for item i</li>
              <li><InlineMath>{"h_i^{(G)} \\in \\mathbb{R}^d"}</InlineMath>: global features from hypergraph</li>
              <li><InlineMath>{"h_i^{(L)} \\in \\mathbb{R}^d"}</InlineMath>: local features from GAT</li>
              <li><InlineMath>{"\\text{Aggregate}"}</InlineMath>: aggregation function (sum in the paper)</li>
            </ul>
            
            <div className="mt-4 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Key insight:</strong> Sum pooling allows both global co-occurrence patterns 
                and local sequential dependencies to contribute to the final representation.
              </p>
            </div>
          </>
        }
        code={fusionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="fusionEq"
        rightId="fusionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="fusion"
      />

      {/* Section 8: Position Encoding */}
      <ComparisonSection
        title="8. Positional Encoding (Equations 8-9)"
        leftContent={
          <>
            <p className="mb-4">Add position information to distinguish items at different positions:</p>
            
            <FormulaWithDescription
              title="Position concatenation (Equation 8):"
              math="x_i^* = [h_i^* ; p_i]"
              description=""
              id="positionConcatEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"x_i^* \\in \\mathbb{R}^{d + d_p}"}</InlineMath>: position-enhanced features</li>
              <li><InlineMath>{"h_i^* \\in \\mathbb{R}^d"}</InlineMath>: fused item features</li>
              <li><InlineMath>{"p_i \\in \\mathbb{R}^{d_p}"}</InlineMath>: position embedding for position i</li>
              <li><InlineMath>{"[\\cdot ; \\cdot]"}</InlineMath>: concatenation operation</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Feature transformation (Equation 9):"
                math="x_i^* = \tanh\left( W_1 x_i^* + b \right)"
                description=""
                id="positionTransformEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"W_1 \\in \\mathbb{R}^{d \\times (d + d_p)}"}</InlineMath>: transformation matrix</li>
                <li><InlineMath>{"b \\in \\mathbb{R}^d"}</InlineMath>: bias vector</li>
                <li><InlineMath>{"\\tanh"}</InlineMath>: hyperbolic tangent activation</li>
                <li>Output dimension matches hidden dimension d</li>
              </ul>
            </div>
          </>
        }
        code={positionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="positionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="position"
      />

      {/* Section 9: Session Attention */}
      <ComparisonSection
        title="9. Session Representation with Attention (Equations 10-13)"
        leftContent={
          <>
            <p className="mb-4">Compute session representation using attention mechanism:</p>
            
            <FormulaWithDescription
              title="Preliminary session vector (Equation 10):"
              math="s^* = \frac{1}{|S|} \sum_{i \in S} h_i^*"
              description=""
              id="preliminarySessionEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"s^* \\in \\mathbb{R}^d"}</InlineMath>: average session representation</li>
              <li><InlineMath>{"|S|"}</InlineMath>: number of items in session S</li>
              <li><InlineMath>{"h_i^*"}</InlineMath>: fused features for item i</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Attention scores (Equation 11):"
                math="a_i = q^\top \left( W_2 x_i^* + W_3 s^* + c \right)"
                description=""
                id="attentionScoreEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"a_i \\in \\mathbb{R}"}</InlineMath>: attention score for item i</li>
                <li><InlineMath>{"q \\in \\mathbb{R}^d"}</InlineMath>: learnable query vector</li>
                <li><InlineMath>{"W_2, W_3 \\in \\mathbb{R}^{d \\times d}"}</InlineMath>: weight matrices</li>
                <li><InlineMath>{"c \\in \\mathbb{R}^d"}</InlineMath>: bias term (often omitted)</li>
              </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Normalized attention weights (Equation 12):"
                math="\alpha_i = \frac{\exp(a_i)}{\sum_{j \in S} \exp(a_j)}"
                description=""
                id="attentionWeightsEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"\\alpha_i \\in [0, 1]"}</InlineMath>: normalized attention weight</li>
                <li>Softmax ensures weights sum to 1 across the session</li>
              </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Final session representation (Equation 13):"
                math="S = \sum_{i \in S} \alpha_i h_i^*"
                description=""
                id="finalSessionEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"S \\in \\mathbb{R}^d"}</InlineMath>: final session representation</li>
                <li>Weighted sum using attention weights</li>
                <li>Additional transformation <InlineMath>{"W_f"}</InlineMath> may be applied</li>
              </ul>
            </div>
          </>
        }
        code={sessionAttentionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="sessionAttentionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="sessionAttentionCode"
        editorPlaceholder="Try implementing the attention mechanism..."
      />

      {/* Section 10: Prediction and Loss */}
      <ComparisonSection
        title="10. Prediction and Loss Computation (Equations 14-16)"
        leftContent={
          <>
            <p className="mb-4">Make predictions and compute training loss:</p>
            
            <FormulaWithDescription
              title="Score computation (Equation 14):"
              math="\hat{z}_i = S^\top h_i^{(G)}"
              description=""
              id="predictionScoreEq"
              visibility={visibility}
              toggleVisibility={toggleVisibility}
              drawingData={drawingData}
              onSaveDrawing={handleSaveDrawing}
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\hat{z}_i \\in \\mathbb{R}"}</InlineMath>: score for item i</li>
              <li><InlineMath>{"S \\in \\mathbb{R}^d"}</InlineMath>: session representation</li>
              <li><InlineMath>{"h_i^{(G)} \\in \\mathbb{R}^d"}</InlineMath>: global embedding for item i</li>
              <li>Dot product measures similarity</li>
            </ul>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Probability distribution (Equation 15):"
                math="\hat{y}_i = \text{softmax}(\hat{z}_i) = \frac{\exp(\hat{z}_i)}{\sum_{j=1}^{n} \exp(\hat{z}_j)}"
                description=""
                id="predictionSoftmaxEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"\\hat{y}_i \\in [0, 1]"}</InlineMath>: predicted probability for item i</li>
                <li><InlineMath>{"\\sum_{i=1}^{n} \\hat{y}_i = 1"}</InlineMath>: probabilities sum to 1</li>
                <li>Softmax converts scores to probabilities</li>
              </ul>
            </div>
            
            <div className="mt-6">
              <FormulaWithDescription
                title="Cross-entropy loss (Equation 16):"
                math="\mathcal{L} = - \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i) \right]"
                description=""
                id="lossEq"
                visibility={visibility}
                toggleVisibility={toggleVisibility}
                drawingData={drawingData}
                onSaveDrawing={handleSaveDrawing}
              />
              
              <p className="mt-4">Where:</p>
              <ul className="list-disc pl-5 space-y-1 mt-2">
                <li><InlineMath>{"\\mathcal{L} \\in \\mathbb{R}"}</InlineMath>: loss value</li>
                <li><InlineMath>{"y_i \\in \\{0, 1\\}"}</InlineMath>: ground truth (1 for target item)</li>
                <li>Negative sampling is used for efficiency</li>
                <li>Alternative: BPR loss for ranking</li>
              </ul>
            </div>
          </>
        }
        code={predictionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="predictionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="prediction"
      />

      {/* Section 11: Complete Model */}
      <ComparisonSection
        title="11. Complete GC-HGNN Model Implementation"
        fullWidth={true}
        code={completeModelCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="completeModelCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="completeModelCode"
        editorPlaceholder="Try implementing the complete GC-HGNN model..."
      />

      {/* Section 12: Training Pipeline */}
      <ComparisonSection
        title="12. Training Pipeline"
        fullWidth={true}
        code={trainingCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="trainingCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="trainingCode"
        editorPlaceholder="Implement the training loop..."
      />

      {/* Section 13: Making Recommendations */}
      <ComparisonSection
        title="13. Making Recommendations with Trained Model"
        fullWidth={true}
        code={recommendationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        rightId="recommendationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="recommendationCode"
        editorPlaceholder="Implement the recommendation function..."
      />

      {/* Key Insights */}
      <div className="mb-8 bg-blue-50 border-l-4 border-blue-400 p-4">
        <h3 className="text-lg font-bold text-blue-800 mb-2">Key GC-HGNN Innovations</h3>
        <ul className="list-disc list-inside space-y-1 text-blue-700">
          <li><strong>Dual Context Modeling:</strong> Combines global item co-occurrence (hypergraph) with local sequential patterns (GAT)</li>
          <li><strong>Global Hypergraph:</strong> Each session forms a hyperedge connecting all its items, capturing many-to-many relations</li>
          <li><strong>Local Graph Attention:</strong> Sequential item transitions modeled via graph attention networks</li>
          <li><strong>Feature Fusion:</strong> Sum-pooling effectively combines both contexts</li>
          <li><strong>Position Awareness:</strong> Incorporates item positions within sessions for better understanding</li>
          <li><strong>Attention Aggregation:</strong> Uses learned attention to create final session representations</li>
        </ul>
      </div>

      {/* Architecture Summary */}
      <div className="mb-8 bg-green-50 border-l-4 border-green-400 p-4">
        <h3 className="text-lg font-bold text-green-800 mb-2">End-to-End Architecture Flow</h3>
        <ol className="list-decimal list-inside space-y-1 text-green-700">
          <li><strong>Input:</strong> Session sequences with variable lengths</li>
          <li><strong>Embedding Layer:</strong> Map items to d-dimensional embeddings</li>
          <li><strong>Global Path:</strong> Multi-layer hypergraph convolution → Average pooling</li>
          <li><strong>Local Path:</strong> Session graph construction → Multi-head GAT</li>
          <li><strong>Fusion:</strong> Sum global and local features for each item</li>
          <li><strong>Position Enhancement:</strong> Add position embeddings and transform</li>
          <li><strong>Session Encoding:</strong> Attention-based aggregation using query vector</li>
          <li><strong>Output:</strong> Dot product with global embeddings for scoring</li>
        </ol>
      </div>

      {/* Performance Tips */}
      <div className="mb-8 bg-yellow-50 border-l-4 border-yellow-400 p-4">
        <h3 className="text-lg font-bold text-yellow-800 mb-2">Implementation Tips for Production</h3>
        <ul className="list-disc list-inside space-y-1 text-yellow-700">
          <li><strong>Sparse Operations:</strong> Use sparse matrix operations for hypergraph convolution to handle large catalogs</li>
          <li><strong>Batch Processing:</strong> Construct separate graphs for each session in a batch</li>
          <li><strong>Negative Sampling:</strong> Sample 100-500 negative items per positive for efficiency</li>
          <li><strong>Gradient Clipping:</strong> Clip gradients to prevent exploding gradients</li>
          <li><strong>Learning Rate Schedule:</strong> Decay learning rate every few epochs</li>
          <li><strong>Early Stopping:</strong> Monitor validation MRR@20 to prevent overfitting</li>
        </ul>
      </div>

      {/* Comparison with Other Models */}
      <div className="mb-8 bg-purple-50 border-l-4 border-purple-400 p-4">
        <h3 className="text-lg font-bold text-purple-800 mb-2">GC-HGNN vs Other Session-based Models</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div>
            <h4 className="font-semibold mb-2">vs HGNN</h4>
            <ul className="list-disc list-inside space-y-1 text-purple-700 text-sm">
              <li>HGNN: Only global hypergraph</li>
              <li>GC-HGNN: Adds local GAT for sequences</li>
              <li>Better captures item transitions</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">vs SR-GNN</h4>
            <ul className="list-disc list-inside space-y-1 text-purple-700 text-sm">
              <li>SR-GNN: Only local graph + GRU</li>
              <li>GC-HGNN: Adds global hypergraph</li>
              <li>Better captures co-occurrence patterns</li>
            </ul>
          </div>
        </div>
      </div>
    </ComparisonLayout>
  );
}