/**
 * This file includes `eslint` settings that are distributed via the
 * `@allenai/eslint-config-varnish` package and shared amongst many AI2
 * projects.
 *
 * @see https://eslint.org/docs/user-guide/configuring
 */
 
// Note that these settings are a modified version of the above, without
// typescript linting.

module.exports = {
    extends: [ "standard", "plugin:prettier/recommended" ],
    env: {
        browser: true,
        es6: true,
        node: true,
        mocha: true
    },
    globals: {
        Atomics: "readonly",
        SharedArrayBuffer: "readonly"
    },
    plugins: [
        "react",
        "prettier"
    ],
    rules: {
        "prettier/prettier": [
            "error",
            {
                printWidth: 100,
                tabWidth: 4,
                singleQuote: true,
                semi: true,
                trailingComma: "none",
                jsxBracketSameLine: true,
                jsxSingleQuote: false
            }
        ],
        "react/jsx-uses-react": 1,
        "react/jsx-uses-vars": 1,
        "no-unused-vars": "off",
        "no-useless-constructor": "off"
    }
}
