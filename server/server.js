const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');
require('dotenv').config({ path: '../.env' });

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const MODEL = 'claude-sonnet-4-5-20250929';

/**
 * POST /api/ask
 * Ask Claude a question about selected content
 *
 * Request body:
 * {
 *   selectedText: string,
 *   question: string,
 *   derivationContext: object,  // Full derivation JSON for context
 *   conversationHistory: array  // Previous messages in conversation
 * }
 */
app.post('/api/ask', async (req, res) => {
  try {
    const { selectedText, question, derivationContext, conversationHistory = [], attachedImages = [] } = req.body;

    if (!question && attachedImages.length === 0) {
      return res.status(400).json({ error: 'Question or image is required' });
    }

    // Build context for Claude with full derivation structure
    const contextInfo = {
      derivationTitle: derivationContext?.title || 'Unknown',
      derivationDescription: derivationContext?.description || '',
      derivationSubtitle: derivationContext?.subtitle || '',
      selectedContent: selectedText || 'No specific content selected',
    };

    // Build comprehensive context string including equations
    let derivationStructure = '';
    if (derivationContext?.sections && derivationContext.sections.length > 0) {
      derivationStructure = '\n\n**Full Derivation Structure:**\n\n';
      derivationContext.sections.forEach((section, idx) => {
        derivationStructure += `\n**${section.title}**\n`;
        if (section.steps && section.steps.length > 0) {
          section.steps.forEach(step => {
            derivationStructure += `- ${step.title}: ${step.explanation}\n`;
            if (step.equation) {
              const equations = Array.isArray(step.equation) ? step.equation : [step.equation];
              equations.forEach(eq => {
                derivationStructure += `  → $$${eq}$$\n`;
              });
            }
          });
        }
      });
    }

    // Build system prompt
    const systemPrompt = `You are an expert mathematics and physics tutor helping students understand mathematical derivations, particularly in areas like quantum mechanics, general relativity, and advanced physics.

You can analyze images, handwritten notes, diagrams, and equations that students share with you.

You have access to the following context from an interactive learning platform:

**Derivation**: ${contextInfo.derivationTitle}
${contextInfo.derivationSubtitle ? `**Subtitle**: ${contextInfo.derivationSubtitle}` : ''}
**Description**: ${contextInfo.derivationDescription}

${selectedText ? `**Student selected text**: ${selectedText}` : ''}

${derivationStructure}

Your role is to:
1. Explain mathematical concepts clearly and rigorously, referencing specific steps from the derivation above when relevant
2. Break down complex equations step-by-step
3. Connect concepts to physical intuition where applicable
4. Reference LaTeX equations using proper formatting
5. Be encouraging and pedagogical
6. When the student asks about a specific equation, you can reference other related steps in the derivation to provide context

When explaining equations, you can use LaTeX notation inline like $equation$ or in display mode like $$equation$$.`;

    // Build conversation messages
    const messages = [
      ...conversationHistory.map(msg => ({
        role: msg.role,
        content: msg.content
      }))
    ];

    // Build current user message with images if provided
    const currentMessageContent = [];

    // Add images first
    if (attachedImages && attachedImages.length > 0) {
      attachedImages.forEach(img => {
        currentMessageContent.push({
          type: 'image',
          source: img.source
        });
      });
    }

    // Add text question
    if (question) {
      currentMessageContent.push({
        type: 'text',
        text: question
      });
    }

    messages.push({
      role: 'user',
      content: currentMessageContent.length === 1 && currentMessageContent[0].type === 'text'
        ? currentMessageContent[0].text  // If only text, use string format
        : currentMessageContent           // If images or mixed, use array format
    });

    // Call Claude API
    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      messages: messages
    });

    const answer = response.content[0].text;

    res.json({
      answer,
      usage: response.usage
    });

  } catch (error) {
    console.error('Error in /api/ask:', error);
    res.status(500).json({
      error: 'Failed to get response from Claude',
      details: error.message
    });
  }
});

/**
 * POST /api/improve-answer
 * Get Claude to improve/enhance a derivation step
 *
 * Request body:
 * {
 *   currentStep: object,        // The step to improve
 *   derivationContext: object,  // Full derivation for context
 *   improvementRequest: string  // What to improve (optional)
 * }
 */
app.post('/api/improve-answer', async (req, res) => {
  try {
    const { currentStep, derivationContext, improvementRequest } = req.body;

    if (!currentStep) {
      return res.status(400).json({ error: 'Current step is required' });
    }

    // Build prompt for improvement
    const systemPrompt = `You are an expert mathematics educator helping to improve educational content for an interactive learning platform.

The platform displays mathematical derivations in JSON format with the following structure:
{
  "id": "step-id",
  "title": "Step Title",
  "explanation": "Clear explanation of what's happening in this step",
  "equation": ["LaTeX equation 1", "LaTeX equation 2", ...]
}

Your task is to improve the explanation and ensure equations are clear and pedagogically sound.`;

    const userPrompt = `Here is a step from the derivation "${derivationContext?.title || 'Mathematical Derivation'}":

**Current Step**:
- Title: ${currentStep.title}
- Explanation: ${currentStep.explanation}
- Equations: ${JSON.stringify(currentStep.equation, null, 2)}

${improvementRequest ? `**Specific improvement requested**: ${improvementRequest}` : '**Task**: Improve this step to be clearer, more pedagogical, and mathematically rigorous.'}

Please return an improved version of this step as valid JSON with the same structure. Make the explanation clearer and more detailed. Ensure all LaTeX is correct. Return ONLY the JSON object, no other text.`;

    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: systemPrompt,
      messages: [{
        role: 'user',
        content: userPrompt
      }]
    });

    const answer = response.content[0].text;

    // Try to extract JSON from the response
    let improvedStep;
    try {
      // Try to parse directly
      improvedStep = JSON.parse(answer);
    } catch (e) {
      // Try to extract JSON from markdown code blocks
      const jsonMatch = answer.match(/```json\n([\s\S]*?)\n```/) || answer.match(/```\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        improvedStep = JSON.parse(jsonMatch[1]);
      } else {
        throw new Error('Could not parse JSON from response');
      }
    }

    res.json({
      improvedStep,
      rawResponse: answer,
      usage: response.usage
    });

  } catch (error) {
    console.error('Error in /api/improve-answer:', error);
    res.status(500).json({
      error: 'Failed to improve answer',
      details: error.message
    });
  }
});

/**
 * POST /api/save-derivation
 * Save edited derivation JSON to file system
 *
 * Request body:
 * {
 *   derivationId: string,  // e.g., 'general-relativity'
 *   data: object           // Complete derivation JSON
 * }
 */
app.post('/api/save-derivation', async (req, res) => {
  try {
    const { derivationId, data } = req.body;

    if (!derivationId || !data) {
      return res.status(400).json({ error: 'derivationId and data are required' });
    }

    // Map derivation IDs to their file paths
    const derivationFiles = {
      'diffusion-models': 'diffusion.json',
      'schrodinger-equation': 'schrodinger.json',
      'euler-lagrange': 'euler-lagrange.json',
      'general-relativity': 'generalrelativity.json',
    };

    const fileName = derivationFiles[derivationId];
    if (!fileName) {
      return res.status(400).json({ error: `Unknown derivation ID: ${derivationId}` });
    }

    const fs = require('fs');
    const path = require('path');
    const filePath = path.join(__dirname, '..', 'src', 'data', 'derivations', fileName);

    // Pretty print JSON with 2-space indentation
    const jsonString = JSON.stringify(data, null, 2);

    // Write to file
    fs.writeFileSync(filePath, jsonString, 'utf8');

    console.log(`✅ Saved ${fileName}`);

    res.json({
      success: true,
      filePath: fileName,
      message: 'Derivation saved successfully'
    });

  } catch (error) {
    console.error('Error in /api/save-derivation:', error);
    res.status(500).json({
      error: 'Failed to save derivation',
      details: error.message
    });
  }
});

/**
 * POST /api/edit-derivation
 * Use Claude to edit a derivation based on natural language commands
 *
 * Request body:
 * {
 *   command: string,              // e.g., "Fix equation in step 3.2"
 *   currentData: object,          // Current derivation JSON
 *   conversationHistory: array    // Previous edits in this session
 * }
 */
app.post('/api/edit-derivation', async (req, res) => {
  try {
    const { command, currentData, conversationHistory = [], attachedImages = [] } = req.body;

    if ((!command && attachedImages.length === 0) || !currentData) {
      return res.status(400).json({ error: 'command or image and currentData are required' });
    }

    // Build system prompt for editing with targeted approach
    const systemPrompt = `You are an expert AI assistant helping to edit mathematical derivations.

You can analyze images, handwritten notes, diagrams, and screenshots that show corrections or new content.

**Your task:** Analyze the user's command (and any images provided) and return a JSON patch describing what to change.

**Return format:**

If editing ONE step:
{
  "editType": "modify_step" | "add_step" | "delete_step" | "modify_metadata" | "add_section",
  "target": {
    "sectionIndex": number (0-indexed),
    "stepIndex": number (0-indexed, for step operations)
  },
  "changes": {
    "title": "new title" (optional),
    "explanation": "new explanation" (optional),
    "equation": ["new equations"] or "new equation" (optional),
    "canvasHeight": number (optional)
  }
}

If editing MULTIPLE steps, return an array:
[
  {
    "editType": "modify_step",
    "target": { "sectionIndex": 2, "stepIndex": 0 },
    "changes": { ... }
  },
  {
    "editType": "modify_step",
    "target": { "sectionIndex": 2, "stepIndex": 1 },
    "changes": { ... }
  }
]

**IMPORTANT:**
1. Return ONLY the patch JSON - no markdown, no extra text
2. If the user asks to edit multiple steps (e.g., "fix 3.1 and 3.2"), return an ARRAY of patches
3. Be precise about which section/step to edit (use 0-based indices)
4. Only include changed fields in "changes"
5. For equations, use LaTeX without $$ delimiters

**Current derivation structure:**
${JSON.stringify({
  title: currentData.title,
  subtitle: currentData.subtitle,
  sections: currentData.sections.map((sec, si) => ({
    index: si,
    title: sec.title,
    steps: sec.steps.map((step, sti) => ({
      index: sti,
      id: step.id,
      title: step.title
    }))
  }))
}, null, 2)}

**User's command:**
"${command}"

Return the patch JSON:`;

    // Build conversation messages
    const messages = [
      ...conversationHistory.map(msg => ({
        role: msg.role,
        content: msg.content
      }))
    ];

    // Build current user message with images if provided
    const currentMessageContent = [];

    // Add images first
    if (attachedImages && attachedImages.length > 0) {
      attachedImages.forEach(img => {
        currentMessageContent.push({
          type: 'image',
          source: img.source
        });
      });
    }

    // Add text command
    if (command) {
      currentMessageContent.push({
        type: 'text',
        text: command
      });
    }

    messages.push({
      role: 'user',
      content: currentMessageContent.length === 1 && currentMessageContent[0].type === 'text'
        ? currentMessageContent[0].text  // If only text, use string format
        : currentMessageContent           // If images or mixed, use array format
    });

    // Call Claude API
    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 16384, // Increased for large derivations
      system: systemPrompt,
      messages: messages,
      temperature: 0 // More deterministic for JSON editing
    });

    let modifiedJSON = response.content[0].text;

    // Try to extract JSON if Claude wrapped it in markdown
    const jsonMatch = modifiedJSON.match(/```json\n([\s\S]*?)\n```/) ||
                      modifiedJSON.match(/```\n([\s\S]*?)\n```/);

    if (jsonMatch) {
      modifiedJSON = jsonMatch[1];
    }

    // Parse the patch JSON
    let patches;
    try {
      const parsed = JSON.parse(modifiedJSON);
      // Handle both single patch and array of patches
      patches = Array.isArray(parsed) ? parsed : [parsed];
      console.log('📝 Patch(es) received from Claude:', JSON.stringify(patches, null, 2));
    } catch (parseError) {
      // Check if response was truncated
      if (response.stop_reason === 'max_tokens') {
        return res.status(500).json({
          error: 'Response too large',
          details: 'The AI response was too long. Please try a more specific command.',
          suggestion: 'Target a specific step, e.g., "Fix equation in step 2.5"'
        });
      }

      console.error('Failed to parse Claude response as JSON:', modifiedJSON.substring(0, 500));
      return res.status(500).json({
        error: 'Invalid response from AI',
        details: parseError.message,
        rawResponse: modifiedJSON.substring(0, 500)
      });
    }

    // Validate all patches
    for (const patch of patches) {
      if (!patch.editType) {
        return res.status(500).json({
          error: 'Invalid patch format',
          details: 'Missing editType field',
          patch: patch
        });
      }
    }

    // Apply all patches to currentData
    const modifiedData = JSON.parse(JSON.stringify(currentData)); // Deep clone

    try {
      for (const patch of patches) {
        switch (patch.editType) {
        case 'modify_step':
          if (!patch.target || patch.target.sectionIndex === undefined || patch.target.stepIndex === undefined) {
            throw new Error('modify_step requires target.sectionIndex and target.stepIndex');
          }

          const section = modifiedData.sections[patch.target.sectionIndex];
          if (!section) {
            throw new Error(`Section ${patch.target.sectionIndex} not found`);
          }

          const step = section.steps[patch.target.stepIndex];
          if (!step) {
            throw new Error(`Step ${patch.target.stepIndex} not found in section ${patch.target.sectionIndex}`);
          }

          // Apply changes to the step
          if (patch.changes) {
            if (patch.changes.title !== undefined) step.title = patch.changes.title;
            if (patch.changes.explanation !== undefined) step.explanation = patch.changes.explanation;
            if (patch.changes.equation !== undefined) step.equation = patch.changes.equation;
            if (patch.changes.canvasHeight !== undefined) step.canvasHeight = patch.changes.canvasHeight;
          }
          break;

        case 'add_step':
          if (!patch.target || patch.target.sectionIndex === undefined) {
            throw new Error('add_step requires target.sectionIndex');
          }
          if (!patch.newStep) {
            throw new Error('add_step requires newStep object');
          }

          const targetSection = modifiedData.sections[patch.target.sectionIndex];
          if (!targetSection) {
            throw new Error(`Section ${patch.target.sectionIndex} not found`);
          }

          // Insert at specific position or append
          if (patch.target.stepIndex !== undefined) {
            targetSection.steps.splice(patch.target.stepIndex, 0, patch.newStep);
          } else {
            targetSection.steps.push(patch.newStep);
          }
          break;

        case 'delete_step':
          if (!patch.target || patch.target.sectionIndex === undefined || patch.target.stepIndex === undefined) {
            throw new Error('delete_step requires target.sectionIndex and target.stepIndex');
          }

          const delSection = modifiedData.sections[patch.target.sectionIndex];
          if (!delSection) {
            throw new Error(`Section ${patch.target.sectionIndex} not found`);
          }

          if (patch.target.stepIndex >= delSection.steps.length) {
            throw new Error(`Step ${patch.target.stepIndex} not found in section ${patch.target.sectionIndex}`);
          }

          delSection.steps.splice(patch.target.stepIndex, 1);
          break;

        case 'add_section':
          if (!patch.newSection) {
            throw new Error('add_section requires newSection object');
          }

          // Insert at specific position or append
          if (patch.target && patch.target.sectionIndex !== undefined) {
            modifiedData.sections.splice(patch.target.sectionIndex, 0, patch.newSection);
          } else {
            modifiedData.sections.push(patch.newSection);
          }
          break;

        case 'modify_metadata':
          if (!patch.metadata) {
            throw new Error('modify_metadata requires metadata object');
          }

          if (patch.metadata.title !== undefined) modifiedData.title = patch.metadata.title;
          if (patch.metadata.subtitle !== undefined) modifiedData.subtitle = patch.metadata.subtitle;
          if (patch.metadata.description !== undefined) modifiedData.description = patch.metadata.description;
          break;

        default:
          throw new Error(`Unknown editType: ${patch.editType}`);
        }
      }
    } catch (patchError) {
      console.error('Error applying patch:', patchError);
      return res.status(500).json({
        error: 'Failed to apply edit',
        details: patchError.message,
        patches: patches
      });
    }

    res.json({
      modifiedData: modifiedData,
      explanation: `Applied ${patches.length} edit(s): "${command}"`,
      usage: response.usage
    });

  } catch (error) {
    console.error('Error in /api/edit-derivation:', error);
    res.status(500).json({
      error: 'Failed to edit derivation',
      details: error.message
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    model: MODEL,
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Paper-viz AI server running on http://localhost:${PORT}`);
  console.log(`📖 Model: ${MODEL}`);
  console.log(`🔑 API Key: ${process.env.ANTHROPIC_API_KEY ? '✓ Loaded' : '✗ Missing'}`);
});
