import React, { useState } from 'react';
import { BlockMath } from 'react-katex';
import { Card } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';
import EquationCanvas from './EquationCanvas';


export default function PPOBreakdown() {
  const [selectedTab, setSelectedTab] = useState('actor');
  
  // Initialize visibility states for each component's code and equation
  const [visibility, setVisibility] = useState({
    actor: { code: true, equation: true },
    critic: { code: true, equation: true },
    gae: { code: true, equation: true },
    ppoLoss: { code: true, equation: true }
  });

  // Initialize user code state for memorization mode
  const [userCode, setUserCode] = useState({
    actor: '',
    critic: '',
    gae: '',
    ppoLoss: ''
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

  // Toggle function for visibility
  const toggleVisibility = (component, type) => {
    setVisibility(prev => ({
      ...prev,
      [component]: {
        ...prev[component],
        [type]: !prev[component][type]
      }
    }));
  };

  const components = {
    actor: {
      title: "Policy Network (Actor)",
      description: "The actor network outputs action probabilities given a state",
      code: `class MLPGaussianActor(nn.Module):
    def __init__(self, state_shape, action_shape, net_arch, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        prev_size = state_shape[0]
        
        for size in net_arch:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            prev_size = size
            
        self.shared_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(prev_size, action_shape[0])
        
    def forward(self, state):
        features = self.shared_net(state)
        action_logits = self.action_net(features)
        return th.distributions.Categorical(logits=action_logits)`,
      equation: "\\pi_\\theta(a|s) = P(a|s; \\theta)"
    },
    critic: {
      title: "Value Network (Critic)",
      description: "The critic estimates the value function for a given state",
      code: `class MLPCritic(nn.Module):
    def __init__(self, input_shape, output_shape, net_arch, activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        prev_size = input_shape[0]
        
        for size in net_arch:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_shape[0]))
        self.net = nn.Sequential(*layers)`,
      equation: "V_\\theta(s_t) \\approx \\mathbb{E}[R_t]"
    },
    gae: {
      title: "Generalized Advantage Estimation",
      description: "GAE computes advantage estimates for more stable training",
      code: `def __call__(self, rewards, values, dones, next_value):
    advantages = th.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[-1]
            next_val = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_val = values[t + 1]
            
        delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
        advantages[t] = last_gae = delta + self.gamma * self.lambda_ * next_nonterminal * last_gae`,
      equation: "A_t = \\delta_t + (\\gamma \\lambda)\\delta_{t+1} + ... + (\\gamma \\lambda)^{T-t+1}\\delta_{T-1}"
    },
    ppoLoss: {
      title: "PPO Loss Computation",
      description: "The clipped objective function that defines PPO's policy update",
      code: `# Calculate policy loss
ratio = th.exp(log_probs - batch_old_log_probs)
surr1 = batch_advantages * ratio
surr2 = batch_advantages * th.clamp(
    ratio, 
    1.0 - self.epsilon, 
    1.0 + self.epsilon
)
policy_loss = -th.min(surr1, surr2).mean()

# Calculate value loss
value_loss = ((batch_returns - values) ** 2).mean()

# Calculate total loss
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy`,
      equation: "L^{CLIP}(\\theta) = \\mathbb{E}_t[\\min(r_t(\\theta)A_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)A_t)]"
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 space-y-6">
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          {Object.keys(components).map((key) => (
            <TabsTrigger key={key} value={key} className="text-sm">
              {components[key].title}
            </TabsTrigger>
          ))}
        </TabsList>
        
        {Object.entries(components).map(([key, component]) => (
          <TabsContent key={key} value={key}>
            <Card className="p-6">
              <h2 className="text-xl font-bold mb-4">{component.title}</h2>
              <p className="mb-4 text-gray-700">{component.description}</p>
              
              <div className="bg-white p-4 rounded-lg mb-4">
                <h3 className="text-lg font-semibold mb-2">Implementation:</h3>
                <div className="relative">
                  <button
                    onClick={() => toggleVisibility(key, 'code')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                    aria-label={visibility[key].code ? "Hide code" : "Show code"}
                  >
                    {visibility[key].code ? 'Hide' : 'Show'}
                  </button>
                  {visibility[key].code ? (
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
                      {component.code}
                    </SyntaxHighlighter>
                  ) : (
                    <Editor
                      value={userCode[key]}
                      onValueChange={code => setUserCode(prev => ({ ...prev, [key]: code }))}
                      highlight={highlightCode}
                      padding={16}
                      style={{
                        fontFamily: '"Fira code", "Fira Mono", monospace',
                        fontSize: 14,
                        backgroundColor: '#f6f8fa',
                        borderRadius: '6px',
                        border: '1px solid #e1e4e8',
                        minHeight: '200px'
                      }}
                      textareaClassName="font-mono"
                      preClassName="font-mono"
                    />
                  )}
                </div>
              </div>
              
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-2">Key Equation:</h3>
                <div className="bg-white p-4 rounded-lg relative">
                  <button
                    onClick={() => toggleVisibility(key, 'equation')}
                    className="absolute top-2 right-2 z-10 text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                    aria-label={visibility[key].equation ? "Hide equation" : "Show equation"}
                  >
                    {visibility[key].equation ? 'Hide' : 'Show'}
                  </button>
                  <div className="flex justify-center">
                    {visibility[key].equation ? (
                      <BlockMath>{component.equation}</BlockMath>
                    ) : (
                      <EquationCanvas 
                        canvasId={`${key}-equation`} 
                        savedData={drawingData[`${key}-equation`]}
                        onSaveData={handleSaveDrawing}
                      />
                    )}
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}