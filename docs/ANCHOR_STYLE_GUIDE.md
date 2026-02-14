# Anchor Style Guide
**Version 1.0 | TreeHacks 2026**

---

## Logo Analysis

The mark is a **padlock-anchor hybrid**—security (protection from scams) + stability (grounding for vulnerable users). This dual-meaning should inform all design decisions.

---

## 1. Color Palette

| Role | Name | Hex | RGB | Usage |
|------|------|-----|-----|-------|
| **Primary** | Anchor Teal | `#5BBFB3` | 91, 191, 179 | Icon, CTAs, accent links, positive states |
| **Text** | Charcoal | `#2D3436` | 45, 52, 54 | Body text, headings, wordmark |
| **Background** | Warm White | `#F9F7F5` | 249, 247, 245 | Page backgrounds, cards |
| **Alert** | Soft Red | `#E17055` | 225, 112, 85 | Scam detected, warnings |
| **Success** | Mint | `#00B894` | 0, 184, 148 | Safe call confirmed |
| **Neutral** | Cool Gray | `#B2BEC3` | 178, 190, 195 | Borders, disabled states, secondary text |

### Rationale

- Teal communicates trust/calm (healthcare/fintech conventions) without being clinical blue
- Warm white avoids sterile feel—important for elderly-facing product
- Alert red is softened (coral vs. crimson) to inform without panic

---

## 2. Typography

| Use Case | Font | Weight | Size (web) |
|----------|------|--------|------------|
| **Headings** | Inter or Plus Jakarta Sans | SemiBold (600) | 32–48px |
| **Subheadings** | Inter | Medium (500) | 20–24px |
| **Body** | Inter | Regular (400) | 16–18px |
| **Captions/Labels** | Inter | Regular (400) | 12–14px |
| **Code/Data** | JetBrains Mono | Regular | 14px |

### Rules

- **Lowercase preference** for product name ("anchor" not "Anchor" in logos/headers)
- Sentence case for UI elements
- Minimum body text: **16px** (accessibility for older users)
- Line height: 1.5–1.6 for body

---

## 3. Logo Usage

| Variant | Background | Use |
|---------|------------|-----|
| Full color (teal icon + charcoal text) | Light backgrounds | Primary |
| Monochrome charcoal | Light backgrounds | Documents, grayscale print |
| White knockout | Dark/teal backgrounds | Hero sections, slides |
| Icon only | Any | Favicon, app icon, small spaces |

### Clear Space

Minimum padding = height of the "a" in wordmark on all sides

### Don'ts

- Don't rotate
- Don't change icon/text color relationship
- Don't stretch
- Don't add effects (shadows, gradients)

---

## 4. Visual Language

### Shape Principles

| Element | Treatment |
|---------|-----------|
| Corners | Rounded (8–16px radius) |
| Icons | Line style, 2px stroke, rounded caps |
| Cards | Subtle shadow (`0 2px 8px rgba(0,0,0,0.08)`) |
| Buttons | Pill-shaped or 8px radius |

### Data Visualization (Judge's Window)

| Data Type | Color |
|-----------|-------|
| Victim stress level | Teal → Coral gradient |
| Scammer tactic nodes | Charcoal with teal accents |
| Safe baseline | Mint |
| Uncertainty/inference | Gray with dashed borders |
| Time series | Teal line, coral highlights for peaks |

### Graph Network Styling

- Node default: `#B2BEC3` (neutral)
- Active/inferred tactic: `#5BBFB3` (teal)
- High-confidence edge: Solid, 2px
- Low-confidence edge: Dashed, 1px
- Background: `#F9F7F5`

---

## 5. Application Guidelines

### Website

```
Background: #F9F7F5
Nav: White bar, charcoal text, teal hover states
Hero: Teal gradient overlay on imagery, white text
Cards: White, rounded, subtle shadow
CTAs: Teal fill, white text, pill shape
```

### Slide Decks

| Slide Type | Background | Text |
|------------|------------|------|
| Title | Teal | White |
| Content | Warm White | Charcoal |
| Data/Demo | White | Charcoal + teal accents |
| Key Insight | Charcoal | White + teal highlight |

### Judge's Dashboard

- Dark mode option: `#1E272E` background, teal accents
- Real-time indicators: Pulsing teal dot
- Alert state: Coral border glow, not full-screen red

---

## 6. Tone Alignment

The visual system supports the messaging:

| Message | Visual Expression |
|---------|-------------------|
| "We protect without surveilling" | Open, breathable layouts; no harsh borders |
| "Sophisticated inference" | Clean data viz; precise typography |
| "Calm in crisis" | Muted alert colors; no panic-inducing red |
| "Stability for families" | Anchor motif; grounded compositions |

---

## 7. Quick Reference

```
Primary:    #5BBFB3
Text:       #2D3436
Background: #F9F7F5
Alert:      #E17055
Success:    #00B894
Gray:       #B2BEC3

Font:       Inter (fallback: system-ui)
Radius:     8px default
Shadow:     0 2px 8px rgba(0,0,0,0.08)
```

---

## 8. CSS Variables (apps/web)

Defined in `apps/web/src/app/globals.css` and used by Tailwind theme:

- `--color-primary`, `--color-text`, `--color-background`, `--color-alert`, `--color-success`, `--color-gray`, `--color-dark`
- `--font-family`, `--font-mono`, `--font-size-*`
- `--radius-sm`, `--radius-md`, `--radius-lg`, `--radius-full`
- `--shadow-sm`, `--shadow-md`, `--shadow-lg`

Semantic HSL vars (`--background`, `--foreground`, `--primary`, etc.) are mapped to the palette above for components.

---

## 9. Tailwind (apps/web)

In `tailwind.config.ts`:

- **Colors:** `anchor.teal`, `anchor.charcoal`, `anchor.warm`, `anchor.coral`, `anchor.mint`, `anchor.gray`, `anchor.dark`
- **Font:** `font-sans` uses Inter via `next/font`
- **Shadow:** `shadow-anchor` = `0 2px 8px rgba(0,0,0,0.08)`
- **Radius:** `rounded-anchor` = 8px
