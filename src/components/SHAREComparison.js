import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath } from 'react-katex';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import { ComparisonLayout, ComparisonSection, EquationBlock, FormulaWithDescription } from './ComparisonLayout';

export default function SHAREComparison() {
  // Initialize visibility state - organized by equation-code pairs
  const [visibility, setVisibility] = useState({
    // Section 1: Hypergraph Construction
    hypergraphConstructionTheory: true,
    hypergraphConstructionCode: true,
    // Section 2: Scaled Dot-Product Attention
    scaledDotProductEq: true,
    scaledDotProductCode: true,
    // Section 3: Node to Hyperedge Attention
    nodeToHyperedgeEq: true,
    nodeToHyperedgeCode: true,
    // Section 4: Hyperedge to Node Attention
    hyperedgeToNodeEq: true,
    hyperedgeToNodeCode: true,
    // Section 5: High-order Propagation
    highOrderPropagationTheory: true,
    highOrderPropagationCode: true,
    // Section 6: Self-Attention for Next Item
    selfAttentionEq: true,
    selfAttentionCode: true,
    // Section 7: Score and Loss
    scoreLossEq: true,
    scoreLossCode: true,
    // Section 8: Complete SHARE Model
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
    hypergraphConstructionCode: '',
    scaledDotProductCode: '',
    nodeToHyperedgeCode: '',
    hyperedgeToNodeCode: '',
    highOrderPropagationCode: '',
    selfAttentionCode: '',
    scoreLossCode: '',
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
  const hypergraphConstructionCode = `# Hypergraph Construction with Sliding Windows
def construct_session_hypergraph(session_items, max_window_size=5):
    """
    Build hypergraph where hyperedges represent contextual windows.
    Uses sliding windows of varying sizes to capture different contexts.
    
    E_s = E_s^2 ∪ E_s^3 ∪ ... ∪ E_s^W
    """
    # Get unique items in session (nodes)
    unique_items = list(set(session_items))
    n_items = len(unique_items)
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Collect all hyperedges
    hyperedges = []
    
    # Apply sliding windows of different sizes
    for window_size in range(2, min(max_window_size + 1, len(session_items) + 1)):
        # Slide window across session
        for start_idx in range(len(session_items) - window_size + 1):
            # Get items in current window
            window_items = session_items[start_idx:start_idx + window_size]
            
            # Create hyperedge connecting these items
            hyperedge = [item_to_idx[item] for item in window_items]
            hyperedges.append(hyperedge)
    
    # Create incidence matrix
    n_edges = len(hyperedges)
    H = torch.zeros(n_items, n_edges)
    
    for edge_idx, hyperedge in enumerate(hyperedges):
        for node_idx in hyperedge:
            H[node_idx, edge_idx] = 1
    
    return H, unique_items, hyperedges

# Example usage
session = ['item_A', 'item_B', 'item_C', 'item_D']
H, nodes, edges = construct_session_hypergraph(session, max_window_size=3)

# Window size 2: [A,B], [B,C], [C,D]
# Window size 3: [A,B,C], [B,C,D]
# Each creates a hyperedge in the hypergraph`;

  const scaledDotProductCode = `# Equation 3: Scaled Dot-Product Attention
def scaled_dot_product_attention(a, b):
    """
    S(a, b) = (a^T b) / √D
    
    Compute similarity between vectors a and b,
    scaled by square root of dimension for stability.
    """
    D = a.size(-1)  # Dimension size
    score = torch.dot(a, b) / math.sqrt(D)
    return score

# Batched version for efficiency
def batched_scaled_dot_product(Q, K):
    """
    Compute attention scores for multiple queries and keys
    Q: [batch, d_model]
    K: [batch, d_model]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    return scores

# Usage in HGAT layers
class ScaledDotProductSimilarity(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, q, k):
        """
        Args:
            q: query vectors [*, d]
            k: key vectors [*, d]
        """
        d = q.size(-1)
        scores = torch.sum(q * k, dim=-1) / (math.sqrt(d) * self.temperature)
        return scores`;

  const nodeToHyperedgeCode = `# Equation 1: Node to Hyperedge Attention
class NodeToHyperedgeAttention(nn.Module):
    """
    e_j^(l) = Σ_{t ∈ N_j} m_{t~j}^(l)
    m_{t~j}^(l) = α_{jt} W_1^(l) n_t^(l-1)
    
    α_{jt} = softmax(S(Ŵ_1^(l) n_t^(l-1), u^(l)))
    
    Aggregate node information to hyperedges with attention.
    Highlights informative nodes on each hyperedge.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W_hat_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.u = nn.Parameter(torch.randn(out_dim))  # Context vector
        
    def forward(self, node_features, H):
        """
        Args:
            node_features: [n_nodes, in_dim]
            H: incidence matrix [n_nodes, n_edges]
        Returns:
            edge_features: [n_edges, out_dim]
        """
        n_nodes, n_edges = H.shape
        edge_features = []
        
        for j in range(n_edges):
            # Find nodes connected to hyperedge j
            nodes_in_edge = torch.where(H[:, j] > 0)[0]
            
            if len(nodes_in_edge) == 0:
                edge_features.append(torch.zeros(node_features.size(1)))
                continue
            
            # Transform node features for attention
            node_feats = node_features[nodes_in_edge]  # [k, in_dim]
            transformed = self.W_hat_1(node_feats)  # [k, out_dim]
            
            # Compute attention scores
            scores = torch.matmul(transformed, self.u) / math.sqrt(self.u.size(0))
            alpha = F.softmax(scores, dim=0)  # [k]
            
            # Transform for aggregation
            messages = self.W_1(node_feats)  # [k, out_dim]
            
            # Weighted aggregation
            edge_feat = torch.sum(alpha.unsqueeze(1) * messages, dim=0)
            edge_features.append(edge_feat)
        
        return torch.stack(edge_features)`;

  const hyperedgeToNodeCode = `# Equation 2: Hyperedge to Node Attention
class HyperedgeToNodeAttention(nn.Module):
    """
    n_t^(l) = Σ_{j ∈ Y_t} m_{j→t}^(l)
    m_{j→t}^(l) = β_{tj} W_2^(l) e_j^(l)
    
    β_{tj} = softmax(S(Ŵ_2^(l) e_j^(l), W_3^(l) n_t^(l-1)))
    
    Update nodes by aggregating hyperedge features with attention.
    Emphasizes evidence from hyperedges with larger impacts.
    """
    def __init__(self, edge_dim, node_dim, out_dim):
        super().__init__()
        self.W_2 = nn.Linear(edge_dim, out_dim, bias=False)
        self.W_hat_2 = nn.Linear(edge_dim, out_dim, bias=False)
        self.W_3 = nn.Linear(node_dim, out_dim, bias=False)
        
    def forward(self, edge_features, node_features, H):
        """
        Args:
            edge_features: [n_edges, edge_dim]
            node_features: [n_nodes, node_dim] (from previous layer)
            H: incidence matrix [n_nodes, n_edges]
        Returns:
            updated_nodes: [n_nodes, out_dim]
        """
        n_nodes, n_edges = H.shape
        updated_nodes = []
        
        for t in range(n_nodes):
            # Find hyperedges connected to node t
            edges_of_node = torch.where(H[t, :] > 0)[0]
            
            if len(edges_of_node) == 0:
                updated_nodes.append(torch.zeros(edge_features.size(1)))
                continue
            
            # Get relevant edge features
            edge_feats = edge_features[edges_of_node]  # [k, edge_dim]
            
            # Transform for attention
            edge_transformed = self.W_hat_2(edge_feats)  # [k, out_dim]
            node_transformed = self.W_3(node_features[t])  # [out_dim]
            
            # Compute attention scores
            scores = torch.sum(edge_transformed * node_transformed, dim=1)
            scores = scores / math.sqrt(node_transformed.size(0))
            beta = F.softmax(scores, dim=0)  # [k]
            
            # Transform edge features for aggregation
            messages = self.W_2(edge_feats)  # [k, out_dim]
            
            # Weighted aggregation
            node_updated = torch.sum(beta.unsqueeze(1) * messages, dim=0)
            updated_nodes.append(node_updated)
        
        return torch.stack(updated_nodes)`;

  const highOrderPropagationCode = `# High-order Propagation through Stacked HGAT Layers
class HGAT(nn.Module):
    """
    Hypergraph Attention Network with L layers.
    Each layer captures increasingly high-order relations.
    """
    def __init__(self, input_dim, hidden_dims, n_layers):
        super().__init__()
        
        # Stack of HGAT layers
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(n_layers):
            layer = HGATLayer(
                node_in_dim=dims[i],
                node_out_dim=dims[i+1],
                edge_dim=dims[i+1]
            )
            self.layers.append(layer)
        
        self.n_layers = n_layers
        
    def forward(self, node_features, H):
        """
        Multi-hop propagation through hypergraph
        Layer 0: Direct neighbors
        Layer 1: 2-hop neighbors  
        Layer L: (L+1)-hop neighbors
        """
        # Store features from all layers
        all_node_features = [node_features]
        
        x = node_features
        for i, layer in enumerate(self.layers):
            x = layer(x, H)
            all_node_features.append(x)
            
            # Optional: Add residual connections
            if i > 0:
                x = x + all_node_features[-2]
        
        # Final features capture multi-hop information
        return x, all_node_features

class HGATLayer(nn.Module):
    """Single HGAT layer combining node→edge→node attention"""
    
    def __init__(self, node_in_dim, node_out_dim, edge_dim):
        super().__init__()
        
        # Node to hyperedge
        self.node_to_edge = NodeToHyperedgeAttention(
            node_in_dim, edge_dim
        )
        
        # Hyperedge to node
        self.edge_to_node = HyperedgeToNodeAttention(
            edge_dim, node_in_dim, node_out_dim
        )
        
        # Optional: Layer norm and dropout
        self.norm = nn.LayerNorm(node_out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features, H):
        # Step 1: Aggregate node info to hyperedges
        edge_features = self.node_to_edge(node_features, H)
        
        # Step 2: Update nodes from hyperedge info
        updated_nodes = self.edge_to_node(
            edge_features, node_features, H
        )
        
        # Post-processing
        updated_nodes = self.norm(updated_nodes)
        updated_nodes = self.dropout(updated_nodes)
        
        return updated_nodes`;

  const selfAttentionCode = `# Equations 4-5: Self-Attention for Next-Item Prediction
class NextItemAttention(nn.Module):
    """
    Treat last item as query, all items as keys and values.
    Captures both general interest and current need.
    
    h_s = Σ_{i≤t} σ_{ti} W_V n_{s,i}^(L)
    σ_{ti} = softmax(S(W_Q n_{s,t}^(L), W_K n_{s,i}^(L)))
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Query, Key, Value projections
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # For scaled dot-product
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
    def forward(self, item_sequence):
        """
        Args:
            item_sequence: [seq_len, hidden_dim]
                Session-wise item embeddings from HGAT
        Returns:
            h_s: [hidden_dim] - Session representation
        """
        seq_len = item_sequence.size(0)
        
        # Last item as query (current need)
        last_item = item_sequence[-1]  # [hidden_dim]
        query = self.W_Q(last_item).unsqueeze(0)  # [1, hidden_dim]
        
        # All items as keys and values
        keys = self.W_K(item_sequence)  # [seq_len, hidden_dim]
        values = self.W_V(item_sequence)  # [seq_len, hidden_dim]
        
        # Compute attention scores
        # S(W_Q n_t, W_K n_i) for all i
        scores = torch.matmul(query, keys.transpose(0, 1))  # [1, seq_len]
        scores = scores * self.scale
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [1, seq_len]
        
        # Weighted sum of values
        h_s = torch.matmul(attention_weights, values).squeeze(0)  # [hidden_dim]
        
        return h_s, attention_weights
    
    def forward_batch(self, batch_sequences, lengths):
        """
        Batch processing for efficiency
        """
        batch_size = len(batch_sequences)
        hidden_dim = batch_sequences[0].size(1)
        
        session_reprs = []
        attention_maps = []
        
        for i in range(batch_size):
            seq_len = lengths[i]
            if seq_len == 0:
                # Empty session
                session_reprs.append(torch.zeros(hidden_dim))
                attention_maps.append(None)
                continue
                
            # Get sequence for this session
            seq = batch_sequences[i][:seq_len]
            
            # Apply attention
            h_s, attn = self.forward(seq)
            
            session_reprs.append(h_s)
            attention_maps.append(attn)
        
        return torch.stack(session_reprs), attention_maps`;

  const scoreLossCode = `# Score Computation and Training
class SHAREPredictor(nn.Module):
    """
    Compute preference scores and handle training
    
    p_sv = h_s^T i_v
    ŷ_s = softmax(p_s)
    L = -Σ_s Σ_v y_sv log(ŷ_sv)
    """
    def __init__(self, n_items, item_embed_dim):
        super().__init__()
        
        # Item embeddings (shared with HGAT input)
        self.item_embeddings = nn.Embedding(
            n_items, item_embed_dim
        )
        
    def compute_scores(self, session_repr, return_all=True):
        """
        Compute preference scores for all items
        
        Args:
            session_repr: [hidden_dim]
            return_all: whether to score all items
        """
        if return_all:
            # Score against all items
            all_items = self.item_embeddings.weight  # [n_items, hidden_dim]
            scores = torch.matmul(session_repr, all_items.t())  # [n_items]
        else:
            # Can also score specific items for efficiency
            scores = None
            
        return scores
    
    def forward(self, session_repr):
        """
        Generate probability distribution over items
        """
        scores = self.compute_scores(session_repr)
        probs = F.softmax(scores, dim=-1)
        return probs
    
    def compute_loss(self, session_reprs, targets):
        """
        Cross-entropy loss for next-item prediction
        
        Args:
            session_reprs: [batch_size, hidden_dim]
            targets: [batch_size] - ground truth item indices
        """
        batch_size = session_reprs.size(0)
        
        # Compute scores for all items
        all_items = self.item_embeddings.weight
        scores = torch.matmul(session_reprs, all_items.t())  # [batch, n_items]
        
        # Cross-entropy loss
        loss = F.cross_entropy(scores, targets)
        
        return loss
    
    def predict(self, session_reprs, k=20):
        """
        Get top-k recommendations
        """
        scores = torch.matmul(
            session_reprs, 
            self.item_embeddings.weight.t()
        )
        
        # Top-k items
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        return top_indices, top_scores`;

  const completeModelCode = `# Complete SHARE Model Implementation
class SHARE(nn.Module):
    """
    Session-based Recommendation with Hypergraph Attention Networks
    
    Key components:
    1. Sliding window hypergraph construction
    2. HGAT for contextual information propagation
    3. Self-attention for session representation
    4. Dot-product scoring for recommendations
    """
    def __init__(self, n_items, embed_dim=64, hidden_dims=[64, 64], 
                 n_layers=2, max_window_size=5, dropout=0.1):
        super().__init__()
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        
        # HGAT network
        self.hgat = HGAT(
            input_dim=embed_dim,
            hidden_dims=hidden_dims,
            n_layers=n_layers
        )
        
        # Self-attention for session modeling
        self.session_attention = NextItemAttention(hidden_dims[-1])
        
        # Hyperparameters
        self.max_window_size = max_window_size
        self.n_items = n_items
        self.dropout = nn.Dropout(dropout)
        
    def construct_session_hypergraph(self, session_items):
        """
        Build hypergraph with sliding windows
        """
        unique_items = list(set(session_items))
        n_nodes = len(unique_items)
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        hyperedges = []
        
        # Sliding windows of different sizes
        for w in range(2, min(self.max_window_size + 1, len(session_items) + 1)):
            for i in range(len(session_items) - w + 1):
                window = session_items[i:i+w]
                edge = [item_to_idx[item] for item in window]
                hyperedges.append(edge)
        
        # Build incidence matrix
        n_edges = len(hyperedges)
        H = torch.zeros(n_nodes, n_edges)
        
        for e_idx, edge in enumerate(hyperedges):
            for n_idx in edge:
                H[n_idx, e_idx] = 1
                
        return H, unique_items, item_to_idx
    
    def forward(self, session_sequences, lengths):
        """
        Forward pass for batch of sessions
        
        Args:
            session_sequences: [batch_size, max_seq_len]
            lengths: [batch_size] - actual session lengths
        """
        batch_size = session_sequences.size(0)
        device = session_sequences.device
        
        all_scores = []
        
        for b in range(batch_size):
            # Get session items
            seq_len = lengths[b].item()
            if seq_len == 0:
                scores = torch.zeros(self.n_items, device=device)
                all_scores.append(scores)
                continue
                
            session = session_sequences[b, :seq_len].tolist()
            
            # 1. Construct session hypergraph
            H, unique_items, item_to_idx = self.construct_session_hypergraph(session)
            H = H.to(device)
            
            # 2. Get initial embeddings for unique items
            unique_indices = torch.tensor(unique_items, device=device)
            init_embeds = self.item_embeddings(unique_indices)
            
            # 3. Apply HGAT layers
            node_embeds, _ = self.hgat(init_embeds, H)
            
            # 4. Get embeddings for items in sequence order
            seq_embeds = []
            for item in session:
                idx = item_to_idx[item]
                seq_embeds.append(node_embeds[idx])
            seq_embeds = torch.stack(seq_embeds)
            
            # 5. Apply self-attention
            session_repr, _ = self.session_attention(seq_embeds)
            
            # 6. Score all items
            all_item_embeds = self.item_embeddings.weight
            scores = torch.matmul(session_repr, all_item_embeds.t())
            
            all_scores.append(scores)
        
        # Stack batch scores
        batch_scores = torch.stack(all_scores)
        
        return batch_scores
    
    def training_step(self, batch):
        """
        Single training step
        """
        sessions = batch['sessions']
        lengths = batch['lengths'] 
        targets = batch['targets']
        
        # Forward pass
        scores = self.forward(sessions, lengths)
        
        # Compute loss
        loss = F.cross_entropy(scores, targets)
        
        return loss
    
    def predict(self, sessions, lengths, k=20):
        """
        Get top-k recommendations for sessions
        """
        scores = self.forward(sessions, lengths)
        
        # Get top-k items for each session
        top_scores, top_items = torch.topk(scores, k, dim=-1)
        
        return top_items, top_scores

# Training example
def train_share(model, train_loader, optimizer, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            sessions = batch['sessions'].to(device)
            lengths = batch['lengths'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            scores = model(sessions, lengths)
            
            # Compute loss
            loss = F.cross_entropy(scores, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

# Evaluation example  
@torch.no_grad()
def evaluate_share(model, test_loader, device, k=20):
    model.eval()
    
    total_hit = 0
    total_mrr = 0
    n_samples = 0
    
    for batch in test_loader:
        sessions = batch['sessions'].to(device)
        lengths = batch['lengths'].to(device)
        targets = batch['targets'].to(device)
        
        # Get predictions
        top_items, _ = model.predict(sessions, lengths, k=k)
        
        # Compute metrics
        for i in range(len(targets)):
            target = targets[i].item()
            predictions = top_items[i].tolist()
            
            if target in predictions:
                total_hit += 1
                rank = predictions.index(target) + 1
                total_mrr += 1.0 / rank
                
            n_samples += 1
    
    hit_rate = total_hit / n_samples
    mrr = total_mrr / n_samples
    
    return hit_rate, mrr`;

  return (
    <ComparisonLayout
      title="SHARE: Session-based Recommendation with Hypergraph Attention Networks"
      description="Based on 'Session-based Recommendation with Hypergraph Attention Networks' by Wang et al. (2021). This component shows the exact equations from the paper alongside their implementations."
    >
      {/* Section 1: Hypergraph Construction */}
      <ComparisonSection
        title="Session Hypergraph Construction"
        leftContent={
          <>
            <p className="mb-4">Hypergraph construction using sliding windows of varying sizes:</p>
            <EquationBlock 
              math="\mathcal{E}_s = \mathcal{E}_s^2 \cup \mathcal{E}_s^3 \cup ... \cup \mathcal{E}_s^W"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{E}_s"}</InlineMath>: set of all hyperedges for session <InlineMath>{"s"}</InlineMath></li>
              <li><InlineMath>{"\\mathcal{E}_s^w"}</InlineMath>: hyperedges created with sliding window of size <InlineMath>{"w"}</InlineMath></li>
              <li><InlineMath>{"W"}</InlineMath>: maximum window size considered</li>
              <li>Each window creates a hyperedge connecting items within that context</li>
            </ul>
            
            <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Example:</strong> Session [A, B, C, D] with W=3 creates:
                <ul className="list-disc pl-5 mt-2">
                  <li>Window size 2: hyperedges (A,B), (B,C), (C,D)</li>
                  <li>Window size 3: hyperedges (A,B,C), (B,C,D)</li>
                </ul>
              </p>
            </div>
            
            <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-400">
              <p className="text-sm">
                <strong>Key insight:</strong> Different window sizes capture various contextual scopes. 
                Small windows capture local item transitions, larger windows capture broader contexts.
              </p>
            </div>
          </>
        }
        leftTitle="Sliding Window Construction"
        code={hypergraphConstructionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="hypergraphConstructionTheory"
        rightId="hypergraphConstructionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="hypergraphConstruction"
      />

      {/* Section 2: Scaled Dot-Product Attention */}
      <ComparisonSection
        title="Scaled Dot-Product Attention"
        leftContent={
          <>
            <p className="mb-4">Similarity function used throughout HGAT:</p>
            <EquationBlock 
              math="S(\textbf{a}, \textbf{b}) = \frac{\textbf{a}^T\textbf{b}}{\sqrt{D}}"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\textbf{a}, \\textbf{b}"}</InlineMath>: vectors to compare</li>
              <li><InlineMath>{"D"}</InlineMath>: dimension of the vectors</li>
              <li>Scaling by <InlineMath>{"\\sqrt{D}"}</InlineMath> prevents gradient issues in softmax</li>
            </ul>
            
            <p className="mt-6 mb-2 font-semibold">Usage in SHARE:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li><strong>Node→Hyperedge:</strong> Compare node features with context vector</li>
              <li><strong>Hyperedge→Node:</strong> Compare hyperedge features with node features</li>
              <li><strong>Session attention:</strong> Compare last item (query) with all items (keys)</li>
            </ul>
            
            <div className="mt-6 p-4 bg-orange-50 border-l-4 border-orange-400">
              <p className="text-sm">
                <strong>Technical note:</strong> Following Transformer architecture, scaling prevents 
                large dot products from saturating the softmax function, ensuring stable gradients.
              </p>
            </div>
          </>
        }
        leftTitle="Attention Similarity"
        code={scaledDotProductCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="scaledDotProductEq"
        rightId="scaledDotProductCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="scaledDotProduct"
      />

      {/* Section 3: Node to Hyperedge Attention */}
      <ComparisonSection
        title="Node to Hyperedge Attention"
        leftContent={
          <>
            <FormulaWithDescription
              title="Hyperedge representation (Equation 1):"
              math="\textbf{e}_j^{(l)} = \sum_{t \in \mathcal{N}_j} \textbf{m}_{t \sim j}^{(l)}"
            />
            
            <FormulaWithDescription
              title="Message from node t to edge j:"
              math="\textbf{m}_{t \sim j}^{(l)} = \alpha_{jt} \textbf{W}_1^{(l)} \textbf{n}_t^{(l-1)}"
            />
            
            <FormulaWithDescription
              title="Attention weights:"
              math="\alpha_{jt} = \frac{S(\hat{\textbf{W}}_1^{(l)}\textbf{n}_t^{(l-1)}, \textbf{u}^{(l)})}{\sum_{f \in \mathcal{N}_j} S( \hat{\textbf{W}}_1^{(l)}\textbf{n}_f^{(l-1)}, \textbf{u}^{(l)})}"
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{N}_j"}</InlineMath>: set of nodes connected by hyperedge <InlineMath>{"j"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{n}_t^{(l-1)}"}</InlineMath>: node <InlineMath>{"t"}</InlineMath> features at layer <InlineMath>{"l-1"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{u}^{(l)}"}</InlineMath>: trainable context vector for layer <InlineMath>{"l"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{W}_1^{(l)}, \\hat{\\textbf{W}}_1^{(l)}"}</InlineMath>: transformation matrices</li>
              <li><InlineMath>{"S(\\cdot,\\cdot)"}</InlineMath>: scaled dot-product similarity function</li>
            </ul>
            
            <div className="mt-6 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Design principle:</strong> Attention highlights informative items within each 
                contextual window (hyperedge), focusing on items most relevant to user intent.
              </p>
            </div>
          </>
        }
        leftTitle="Node→Hyperedge Aggregation"
        code={nodeToHyperedgeCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="nodeToHyperedgeEq"
        rightId="nodeToHyperedgeCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="nodeToHyperedgeCode"
        editorPlaceholder="Try implementing node-to-hyperedge attention..."
      />

      {/* Section 4: Hyperedge to Node Attention */}
      <ComparisonSection
        title="Hyperedge to Node Attention"
        leftContent={
          <>
            <FormulaWithDescription
              title="Node update (Equation 2):"
              math="\textbf{n}_t^{(l)} = \sum_{j \in \mathcal{Y}_t} \textbf{m}_{j \rightarrow t}^{(l)}"
            />
            
            <FormulaWithDescription
              title="Message from edge j to node t:"
              math="\textbf{m}_{j \rightarrow t}^{(l)} = \beta_{tj} \textbf{W}_2^{(l)} \textbf{e}_j^{(l)}"
            />
            
            <FormulaWithDescription
              title="Attention weights:"
              math="\beta_{tj} = \frac{S(\hat{\textbf{W}}_2^{(l)}\textbf{e}_j^{(l)}, \textbf{W}_3^{(l)}\textbf{n}_t^{(l-1)})}{\sum_{f \in \mathcal{Y}_t} S(\hat{\textbf{W}}_2^{(l)}\textbf{e}_f^{(l)}, \textbf{W}_3^{(l)}\textbf{n}_t^{(l-1)})}"
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\mathcal{Y}_t"}</InlineMath>: set of hyperedges containing node <InlineMath>{"t"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{e}_j^{(l)}"}</InlineMath>: hyperedge <InlineMath>{"j"}</InlineMath> representation from previous step</li>
              <li><InlineMath>{"\\beta_{tj}"}</InlineMath>: attention weight for hyperedge <InlineMath>{"j"}</InlineMath>'s impact on node <InlineMath>{"t"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{W}_2^{(l)}, \\hat{\\textbf{W}}_2^{(l)}, \\textbf{W}_3^{(l)}"}</InlineMath>: learnable matrices</li>
            </ul>
            
            <div className="mt-6 p-4 bg-purple-50 border-l-4 border-purple-400">
              <p className="text-sm">
                <strong>Key contribution:</strong> Different contextual windows (hyperedges) have 
                varying importance. Attention mechanism emphasizes evidence from more impactful contexts.
              </p>
            </div>
          </>
        }
        leftTitle="Hyperedge→Node Update"
        code={hyperedgeToNodeCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="hyperedgeToNodeEq"
        rightId="hyperedgeToNodeCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="hyperedgeToNode"
      />

      {/* Section 5: High-order Propagation */}
      <ComparisonSection
        title="High-order Propagation"
        leftContent={
          <>
            <p className="mb-4">Multi-layer HGAT captures increasingly complex relationships:</p>
            
            <div className="mb-4">
              <p className="font-semibold mb-2">Information flow through layers:</p>
              <ul className="list-disc pl-5 space-y-1">
                <li><strong>Layer 0:</strong> Initial item embeddings</li>
                <li><strong>Layer 1:</strong> Direct contextual windows</li>
                <li><strong>Layer 2:</strong> 2-hop: contexts of contexts</li>
                <li><strong>Layer L:</strong> L-hop propagation range</li>
              </ul>
            </div>
            
            <p className="mb-4">Stacked HGAT layers:</p>
            <EquationBlock 
              math="[\textbf{n}_1^{(L)}, \textbf{n}_2^{(L)},..., \textbf{n}_{p}^{(L)}] = \text{HGAT}^{L}(...\text{HGAT}^{1}([\textbf{n}_1^{(0)}, \textbf{n}_2^{(0)},..., \textbf{n}_{p}^{(0)}]))"
              description=""
            />
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\textbf{n}_i^{(0)}"}</InlineMath>: initial embedding for item <InlineMath>{"i"}</InlineMath></li>
              <li><InlineMath>{"\\text{HGAT}^{l}"}</InlineMath>: <InlineMath>{"l"}</InlineMath>-th HGAT layer</li>
              <li><InlineMath>{"L"}</InlineMath>: total number of layers</li>
              <li><InlineMath>{"p"}</InlineMath>: number of unique items in session</li>
            </ul>
            
            <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-400">
              <p className="text-sm">
                <strong>Implementation insight:</strong> Residual connections help with deeper networks. 
                Too many layers can lead to over-smoothing, so 2-3 layers are typically optimal.
              </p>
            </div>
          </>
        }
        leftTitle="Multi-Layer Architecture"
        code={highOrderPropagationCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="highOrderPropagationTheory"
        rightId="highOrderPropagationCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        showEditor={true}
        editorKey="highOrderPropagationCode"
        editorPlaceholder="Try implementing the multi-layer HGAT..."
      />

      {/* Section 6: Self-Attention for Next Item */}
      <ComparisonSection
        title="Self-Attention for Next-Item Prediction"
        leftContent={
          <>
            <p className="mb-4">Last item as query, all items as keys and values:</p>
            
            <FormulaWithDescription
              title="Session representation (Equation 4):"
              math="\textbf{h}_s = \sum_{i \leq t} \sigma_{ti} \textbf{W}_V\textbf{n}_{s,i}^{(L)}"
            />
            
            <FormulaWithDescription
              title="Attention weights (Equation 5):"
              math="\sigma_{ti} = \frac{S(\textbf{W}_Q\textbf{n}_{s,t}^{(L)}, \textbf{W}_K\textbf{n}_{s,i}^{(L)})}{\sum_{j \leq t} S(\textbf{W}_Q\textbf{n}_{s,t}^{(L)}, \textbf{W}_K\textbf{n}_{s,j}^{(L)})}"
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\textbf{n}_{s,t}^{(L)}"}</InlineMath>: last item in session (current need)</li>
              <li><InlineMath>{"\\textbf{n}_{s,i}^{(L)}"}</InlineMath>: <InlineMath>{"i"}</InlineMath>-th item embeddings from HGAT</li>
              <li><InlineMath>{"\\textbf{W}_Q, \\textbf{W}_K, \\textbf{W}_V"}</InlineMath>: query, key, value projection matrices</li>
              <li><InlineMath>{"t"}</InlineMath>: length of the session</li>
              <li><InlineMath>{"S(\\cdot,\\cdot)"}</InlineMath>: scaled dot-product from Equation 3</li>
            </ul>
            
            <div className="mt-6 p-4 bg-green-50 border-l-4 border-green-400">
              <p className="text-sm">
                <strong>Design rationale:</strong> Combines general interest (all items) with current 
                need (last item focus). No positional encoding as order is less important in short sessions.
              </p>
            </div>
          </>
        }
        leftTitle="Session Representation"
        code={selfAttentionCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="selfAttentionEq"
        rightId="selfAttentionCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="selfAttention"
      />

      {/* Section 7: Score and Loss */}
      <ComparisonSection
        title="Preference Scoring and Training"
        leftContent={
          <>
            <FormulaWithDescription
              title="Preference score:"
              math="p_{sv} = \textbf{h}_s^T\textbf{i}_v"
            />
            
            <FormulaWithDescription
              title="Probability distribution:"
              math="\hat{\textbf{y}}_s = \text{softmax}(\textbf{p}_s)"
            />
            
            <FormulaWithDescription
              title="Cross-entropy loss:"
              math="\mathcal{L} = - \sum_{s \in S_{train}} \sum_{v=1}^{N} y_{sv}\log{\hat{y}_{sv}}"
            />
            
            <p className="mt-4">Where:</p>
            <ul className="list-disc pl-5 space-y-1 mt-2">
              <li><InlineMath>{"\\textbf{h}_s"}</InlineMath>: session representation from self-attention</li>
              <li><InlineMath>{"\\textbf{i}_v"}</InlineMath>: embedding of item <InlineMath>{"v"}</InlineMath></li>
              <li><InlineMath>{"\\textbf{p}_s = [p_{s1}, p_{s2}, ..., p_{sN}]"}</InlineMath>: scores for all items</li>
              <li><InlineMath>{"\\textbf{y}_s"}</InlineMath>: one-hot ground truth vector</li>
              <li><InlineMath>{"S_{train}"}</InlineMath>: set of training sessions</li>
              <li><InlineMath>{"N"}</InlineMath>: total number of items</li>
            </ul>
            
            <div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <p className="text-sm">
                <strong>Evaluation metrics:</strong> Paper reports P@20 (Precision) and MRR@20 (Mean 
                Reciprocal Rank) on Yoochoose and Diginetica datasets, showing significant improvements 
                over baselines like SR-GNN and STAMP.
              </p>
            </div>
          </>
        }
        leftTitle="Scoring and Loss"
        code={scoreLossCode}
        visibility={visibility}
        toggleVisibility={toggleVisibility}
        leftId="scoreLossEq"
        rightId="scoreLossCode"
        userCode={userCode}
        setUserCode={setUserCode}
        highlightCode={highlightCode}
        drawingData={drawingData}
        onSaveDrawing={handleSaveDrawing}
        canvasId="scoreLoss"
      />

      {/* Section 8: Complete SHARE Model */}
      <ComparisonSection
        title="Complete SHARE Architecture"
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
        editorPlaceholder="Try implementing the complete SHARE model..."
        fullWidth={true}
      />

      {/* Summary */}
      <div className="mt-12 p-6 bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Key Contributions from the Paper</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">1. Model Architecture</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Sliding windows capture various contextual scopes simultaneously</li>
              <li>Dual attention: node→edge and edge→node with different focuses</li>
              <li>Self-attention uses last item as query for current need modeling</li>
              <li>No explicit sequential modeling - items in windows are unordered</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">2. Experimental Results</h3>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Yoochoose 1/64: P@20=72.25%, MRR@20=32.11%</li>
              <li>Yoochoose 1/4: P@20=72.25%, MRR@20=31.45%</li>
              <li>Diginetica: P@20=52.73%, MRR@20=18.05%</li>
              <li>Outperforms SR-GNN, STAMP, and other baselines</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="font-semibold mb-2">3. Key Insights</h3>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>Different contextual windows reveal different aspects of user intent</li>
            <li>Attention mechanisms at both levels improve representation quality</li>
            <li>Hypergraph structure naturally captures set-like item relationships</li>
            <li>Flexible framework adapts window sizes to session characteristics</li>
          </ul>
        </div>
        
        <p className="text-sm mt-4 text-center italic">
          "SHARE outperforms all of the baselines under each of the metrics for session-based recommendation."
        </p>
      </div>
    </ComparisonLayout>
  );
}