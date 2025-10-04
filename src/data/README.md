# Paper Visualization Data Directory

This directory contains all the data for the paper visualization application, organized into three main categories: **derivations**, **comparisons**, and **models**.

## Directory Structure

```
src/data/
├── index.json                  # Manifest file listing all available content
├── derivations/                # Mathematical derivations with interactive practice
│   └── diffusion.json
├── comparisons/                # Side-by-side equation and code comparisons
│   └── dhcn.json
└── models/                     # Interactive architecture diagrams
    ├── bert/
    │   ├── model.json
    │   └── diagram.svg (optional)
    ├── llmhg/
    │   ├── model.json
    │   └── model_alt.json
    └── lrm/
        └── model.json
```

## How to Use This Data

### For This Application

The React components automatically read from these directories:
- `MathDerivation.js` → reads from `derivations/`
- `BertModel.js`, `LLMHGModel.js`, `LRMComplete.js` → read from `models/`
- Comparison components → currently use hardcoded data (future: will read from `comparisons/`)

### For External Applications

Other apps can consume this data by:

1. **Read the manifest**: Start with `index.json` to discover all available content
2. **Load specific files**: Use the paths provided in the manifest
3. **Parse the JSON**: Follow the data structure specifications in `index.json`

#### Example: Reading a Derivation

```javascript
// 1. Load the manifest
const manifest = await fetch('/src/data/index.json').then(r => r.json());

// 2. Find the derivation you want
const diffusion = manifest.categories.derivations.items.find(
  item => item.id === 'diffusion-models'
);

// 3. Load the derivation data
const data = await fetch(`/src/data/${diffusion.path}`).then(r => r.json());

// 4. Render it using your preferred LaTeX renderer
renderDerivation(data);
```

#### Example: Reading a Model Architecture

```javascript
// 1. Load the manifest
const manifest = await fetch('/src/data/index.json').then(r => r.json());

// 2. Find the model
const bert = manifest.categories.models.items.find(
  item => item.id === 'bert'
);

// 3. Load the model data
const modelData = await fetch(
  `/src/data/${bert.directory}${bert.files.model}`
).then(r => r.json());

// 4. Optionally load the SVG diagram
if (bert.hasSVG) {
  const svg = await fetch(
    `/src/data/${bert.directory}${bert.files.diagram}`
  ).then(r => r.text());
}
```

## Data Format Specifications

### Derivation Format

```json
{
  "title": "string",
  "subtitle": "string",
  "description": "string",
  "sections": [
    {
      "title": "string",
      "steps": [
        {
          "id": "string",
          "title": "string",
          "explanation": "string",
          "equation": "string (LaTeX)",
          "canvasHeight": "number"
        }
      ]
    }
  ]
}
```

### Comparison Format

```json
{
  "id": "string",
  "title": "string",
  "description": "string",
  "sections": [
    {
      "id": "string",
      "title": "string",
      "leftContent": {
        "description": "string",
        "equation": "string (LaTeX)",
        "variables": [...]
      },
      "code": "string (Python code)"
    }
  ]
}
```

### Model Format

```json
{
  "svgDiagram": "string (SVG markup)",
  "componentQuizzes": {
    "componentId": {
      "title": "string",
      "equations": [...],
      "initialization": {...},
      "forward": {...}
    }
  },
  "componentExplanations": {
    "componentId": {
      "title": "string",
      "explanation": "string"
    }
  }
}
```

## Adding New Content

### Adding a New Derivation

1. Create a new JSON file in `derivations/` following the derivation format
2. Update `index.json` to include the new derivation
3. Update the component's derivation mapping if needed

### Adding a New Model

1. Create a new directory in `models/` (e.g., `models/gpt/`)
2. Add `model.json` with the model data
3. Optionally add `diagram.svg` with the architecture diagram
4. Update `index.json` to include the new model

### Adding a New Comparison

1. Create a new JSON file in `comparisons/` following the comparison format
2. Update `index.json` to include the new comparison
3. Create or update the corresponding React component to read from this file

## Benefits of This Structure

✅ **Separation of concerns**: Data is separate from presentation logic
✅ **Reusability**: Other apps can consume the same data
✅ **Maintainability**: Easy to update content without touching code
✅ **Discoverability**: Manifest file provides a single source of truth
✅ **Extensibility**: Easy to add new derivations, models, or comparisons
✅ **Type safety**: Clear data format specifications

## Next Steps

- [ ] Migrate remaining comparison components to read from JSON files
- [ ] Add SVG diagrams to models that need them
- [ ] Create more derivations for other topics
- [ ] Add validation schemas for each data type
- [ ] Create a CLI tool to generate new content templates
