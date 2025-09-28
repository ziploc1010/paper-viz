# Adding New Mathematical Derivations

## Quick Start

1. **Create a JSON file** in the `src/data/` directory (e.g., `fourier-transform.json`)
2. **Follow the structure** from `derivation-template.json`
3. **Add the mapping** in `MathDerivation.js`:
   ```javascript
   const derivationFiles = {
     'diffusion-models': 'diffusion.json',
     'fourier-transform': 'fourier-transform.json', // Add your new file here
   };
   ```
4. **Add to Sidebar** in `Sidebar.js`:
   ```javascript
   { id: 'FourierTransform', name: 'Fourier Transform', category: 'Mathematical Proofs' },
   ```
5. **Add routing** in `App.js`:
   ```javascript
   {view === "FourierTransform" && <MathDerivation derivationId="fourier-transform" />}
   ```

## JSON Structure

### Basic Structure
```json
{
  "id": "unique-id",
  "title": "Main Title (supports LaTeX)",
  "subtitle": "Subtitle",
  "description": "Brief description",
  "sections": [...]
}
```

### Step Structure
```json
{
  "id": "unique-step-id",
  "title": "Step Title",
  "explanation": "What the student should do",
  "equation": "Single equation or array of equations",
  "canvasHeight": 120
}
```

### Canvas Height Guidelines
- Single line equation: 100-120px
- Two lines: 140-160px
- Three lines: 180-200px
- Complex multi-line: 220px+

### LaTeX Tips
- Use double backslashes in JSON: `\\frac{1}{2}` not `\frac{1}{2}`
- For multiple equations, use an array:
  ```json
  "equation": [
    "equation 1",
    "equation 2"
  ]
  ```

## Example Derivations You Could Add

1. **Fourier Transform**
   - Derivation from Fourier series
   - Properties and theorems
   - Discrete vs continuous

2. **Schrödinger Equation**
   - Time-dependent form
   - Time-independent form
   - Particle in a box

3. **Navier-Stokes Equations**
   - Conservation of mass
   - Conservation of momentum
   - Simplifications

4. **Black-Scholes Equation**
   - From random walk
   - Risk-neutral pricing
   - Solution methods

5. **Einstein Field Equations**
   - From equivalence principle
   - Metric tensor
   - Energy-momentum tensor

6. **Maxwell's Equations**
   - From experimental laws
   - Differential forms
   - Wave equation derivation