# Design System Specification

## 1. Overview & Creative North Star: "The Modern Agrarian"

This design system is built to bridge the gap between ancestral agricultural wisdom and modern technological precision. Our Creative North Star is **"The Modern Agrarian"**—an editorial-inspired aesthetic that treats agricultural data with the same prestige as a high-end lifestyle journal. 

Unlike generic "dashboard" apps, this system avoids the cluttered, boxy layouts typical of agri-tech. Instead, we use **intentional asymmetry**, vast **breathing room**, and **tonal depth** to create a sense of calm and authority. We want the Indian farmer to feel empowered, not overwhelmed; the UI should feel like a premium tool that respects their profession.

---

## 2. Color Philosophy: Organic Depth

Our palette moves away from "digital neon greens" toward earth-inspired, muted tones that feel stable and trustworthy.

### The "No-Line" Rule
To achieve a premium editorial feel, **1px solid borders are strictly prohibited for sectioning.** Boundaries must be defined solely through background color shifts or subtle tonal transitions. For example, a `surface-container-low` section should sit directly on a `surface` background to create a soft, edge-less division.

### Surface Hierarchy & Nesting
We treat the interface as a series of physical layers, like stacked sheets of fine handmade paper.
- **Base Layer:** `surface` (#fbf9f3) — The canvas.
- **Primary Content Area:** `surface-container-low` (#f5f3ee) — For subtle grouping.
- **Interactive Cards:** `surface-container-lowest` (#ffffff) — To create a "lifted" effect.
- **Accents:** Use `primary` (#0f5238) for high-authority elements and `tertiary` (#683d00) for highlights.

### The "Glass & Gradient" Rule
For floating elements or hero sections, use semi-transparent surface colors with a `backdrop-blur`. When a flat color feels stagnant, apply a subtle linear gradient from `primary` to `primary_container` (angled at 135°) to provide a "soul" and professional polish.

---

## 3. Typography: The Editorial Voice

The typography strategy relies on high-contrast scales to establish a clear information hierarchy.

- **The Authority (Headlines):** We use **Newsreader** (or Playfair Display) for all `display` and `headline` levels. The serif denotes tradition, trust, and the "editorial" voice. Large-scale headings (Display-LG at 3.5rem) should be used with generous leading to let the words breathe.
- **The Utility (Body):** We use **Plus Jakarta Sans** (or DM Sans) for all `title`, `body`, and `label` levels. This modern sans-serif ensures maximum legibility for crop data, weather reports, and financial figures.
- **Hierarchy through Contrast:** By pairing a bold, serif headline with a clean, medium-weight sans-serif body, we create a visual rhythm that feels both heritage-rich and future-proof.

---

## 4. Elevation & Depth: Tonal Layering

We reject the "drop-shadow everything" approach. Hierarchy is achieved through **Tonal Layering**.

- **The Layering Principle:** Place a `surface-container-lowest` card on a `surface-container-low` section. This creates a soft, natural lift without the need for high-contrast shadows.
- **Ambient Shadows:** If a floating action button or a modal requires a shadow, it must be an "Ambient Shadow": 
    - **Color:** A tinted version of the surface, e.g., `rgba(15, 82, 56, 0.08)`.
    - **Style:** Extra-diffused (Blur: 24px–40px, Spread: -4px).
- **Ghost Borders:** If an input or container requires a boundary for accessibility, use a "Ghost Border"—the `outline-variant` token at **20% opacity**. Never use 100% opaque borders.

---

## 5. Components: Precision Built for the Field

### Buttons (The Interaction Pillars)
- **Primary:** Pill-shaped (`rounded-full`), using the `primary` (#0f5238) fill. Use `on-primary` (#ffffff) for text.
- **Secondary:** Pill-shaped, using `secondary-container` fill. No border.
- **Tertiary:** Text-only with an icon, using the `primary` color. No container.

### Cards & Data Lists
- **The Card Rule:** Cards use `rounded-md` (1.5rem) and **must not have dividers.**
- **Separation:** Use vertical white space from our spacing scale (e.g., `spacing-6` or 2rem) to separate list items. If separation is visually required, use a subtle background shift rather than a line.
- **Visual Rhythm:** In lists, use `surface-container-high` for the "active" or "selected" state.

### Input Fields
- **Style:** Filled style only (using `surface-container-highest`). No outlined boxes.
- **Indicator:** Use a 2px bottom-bar in `primary` only when the field is focused.
- **Radius:** `rounded-sm` (0.5rem) to provide a slight architectural structure compared to the pill-shaped buttons.

### Specialized Agri-Components
- **Crop Health Badges:** Use `primary-fixed` for positive status and `tertiary-fixed` (Amber) for warnings.
- **Data Visualization:** Use a mix of `primary` and `secondary` for charts. Ensure all charts sit on a `surface-container-lowest` background to pop.

---

## 6. Do’s and Don’ts

### Do:
- **Use "Aggressive" White Space:** Use `spacing-24` (8.5rem) or `spacing-20` (7rem) for section padding. Farmers deal with vast landscapes; the UI should reflect that openness.
- **Embrace Asymmetry:** Align headlines to the left while keeping data points in a structured grid to create a sophisticated, non-template look.
- **Prioritize Tonal Shifts:** Always ask, "Can I define this area with a background color instead of a line?"

### Don’t:
- **Don’t Use Pure Black:** Use `on-surface` (#1b1c19) for text to maintain a soft, premium feel.
- **Don’t Overuse the Amber (Tertiary):** Reserve `tertiary` for critical warnings or small callouts. Overuse breaks the "trustworthy" green-centric calm.
- **Don’t Use Standard Grid Dividers:** Horizontal lines are the enemy of this design system’s editorial flow. Use padding and color blocks instead.