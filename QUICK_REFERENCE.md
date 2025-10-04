# Quick Reference Guide

## For External Apps: How to Read This Data

### 🚀 3-Step Integration

```javascript
// 1️⃣ Load manifest
const manifest = await fetch('/src/data/index.json').then(r => r.json());

// 2️⃣ Find what you need
const bert = manifest.categories.models.items.find(m => m.id === 'bert');

// 3️⃣ Load the files
const data = await fetch(`/src/data/${bert.directory}${bert.files.model}`)
  .then(r => r.json());
const svg = await fetch(`/src/data/${bert.directory}${bert.files.diagram}`)
  .then(r => r.text());
```

## 📁 File Locations

| Type | Location | Format |
|------|----------|--------|
| **Manifest** | `/src/data/index.json` | JSON |
| **Derivations** | `/src/data/derivations/*.json` | JSON |
| **Comparisons** | `/src/data/comparisons/*.json` | JSON |
| **Models** | `/src/data/models/*/model.json` | JSON |
| **SVG Diagrams** | `/src/data/models/*/diagram.svg` | SVG |

## 🎯 What Changed

| Before | After |
|--------|-------|
| `data/diffusion.json` | `data/derivations/diffusion.json` |
| `data/bertmodel_updated.json` | `data/models/bert/model.json` |
| SVG embedded in JSON | `data/models/bert/diagram.svg` |

## 💡 Common Use Cases

### Just need the SVG?
```html
<img src="/src/data/models/bert/diagram.svg" alt="BERT" />
```

### Need model data only?
```javascript
const model = await fetch('/src/data/models/bert/model.json')
  .then(r => r.json());
```

### Need both?
```javascript
const [model, svg] = await Promise.all([
  fetch('/src/data/models/bert/model.json').then(r => r.json()),
  fetch('/src/data/models/bert/diagram.svg').then(r => r.text())
]);
```

## 📖 Documentation

- **Full Guide**: `/src/data/README.md`
- **Manifest**: `/src/data/index.json`
- **Data Formats**: See `dataStructure` section in `index.json`

## ✅ Verification

The app still works! Test it:
```bash
npm start
```

Navigate to:
- BERT Model → SVG loads from separate file ✅
- Diffusion Models → JSON from derivations/ ✅
