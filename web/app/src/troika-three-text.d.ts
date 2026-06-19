// Minimal ambient types for the bits of troika-three-text we call directly.
// The package ships no declarations of its own; drei bundles its own copy for
// the <Text> component but does not re-export these globals.
declare module "troika-three-text" {
  export function configureTextBuilder(config: {
    defaultFontURL?: string | null
    unicodeFontsURL?: string | null
    sdfGlyphSize?: number
  }): void

  export function preloadFont(
    options: { font?: string; characters?: string | string[]; sdfGlyphSize?: number },
    callback: () => void,
  ): void
}
