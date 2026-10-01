import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Vite's default treats colocated hooks/variants and entry-point-local
      // components as errors even though they are safe; keep the diagnostics
      // visible without making those established patterns fail CI.
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true, allowExportNames: ['useToast'] },
      ],
      // Clearing a stream immediately when its session id changes is the
      // intended external-subscription reset, not derived-state mirroring.
      'react-hooks/set-state-in-effect': 'warn',
    },
  },
])
