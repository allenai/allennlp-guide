import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';

import Head from './Head';
import { Link } from './link';
import { Container, Content } from './Container';

const Layout = ({ isHome, title, description, children }) => {
    return (
        <StaticQuery
            query={graphql`
                {
                    site {
                        siteMetadata {
                            title
                            description
                            headerLinks {
                                text
                                url
                            }
                        }
                    }
                }
            `}
            render={data => {
                const { title, description, headerLinks } = data.site.siteMetadata;
                return (
                    <ThemeProvider>
                        <Head title={title} description={description} />
                        <GlobalStyle />
                        <Header />
                        <Main>
                            {children}
                        </Main>
                        <Footer />
                    </ThemeProvider>
                );
            }}
        />
    );
};

export default Layout;

const Main = styled.main`
`;

// Resetting root layout
const GlobalStyle = createGlobalStyle`
    html,
    body {
        width: 100%;
        height: 100%;
    }

    #___gatsby,
    #___gatsby > div,
    main {
        height: 100%;
    }
    
    main {
        display: flex;
        flex-direction: column;
    }
    
    // footer {
    //     margin-top: auto !important;
    // }
    // 
    // 
    // 
    // *, *:before, *:after {
    //     box-sizing: border-box;
    //     padding: 0;
    //     margin: 0;
    //     border: 0;
    //     outline: 0;
    // }
    // 
    // .textblock {
    //     width: 800px;
    //     max-width: 100%;
    //     margin: auto;
    // }
    // 
    // html {
    //     font-family: sans-serif;
    //     -ms-text-size-adjust: 100%;
    //     -webkit-text-size-adjust: 100%;
    // }
    // 
    // body {
    //     margin: 0;
    // }
    // 
    // article, aside, details, figcaption, figure, footer, header, main, menu, nav,
    // section, summary, progress {
    //     display: block;
    // }
    // 
    // a {
    //     background-color: transparent;
    //     color: inherit;
    //     text-decoration: none;
    // 
    //     &:active,
    //     &:hover {
    //         outline: 0;
    //     }
    // }
    // 
    // abbr[title] {
    //     border-bottom: none;
    //     text-decoration: underline;
    //     text-decoration: underline dotted;
    // }
    // 
    // b, strong {
    //     font-weight: inherit;
    //     font-weight: bolder;
    // }
    // 
    // small {
    //     font-size: 80%;
    // }
    // 
    // sub, sup {
    //     position: relative;
    //     font-size: 65%;
    //     line-height: 0;
    //     vertical-align: baseline;
    // }
    // 
    // sup {
    //     top: -0.5em;
    // }
    // 
    // sub {
    //     bottom: -0.15em;
    // }
    // 
    // img {
    //     border: 0;
    //     height: auto;
    //     max-width: 100%;
    // }
    // 
    // svg {
    //     max-width: 100%;
    //     color-interpolation-filters: sRGB;
    //     fill: currentColor;
    // 
    //     &:not(:root) {
    //         overflow: hidden;
    //     }
    // }
    // 
    // hr {
    //     box-sizing: content-box;
    //     overflow: visible;
    //     height: 0;
    // }
    // 
    // pre {
    //     overflow: auto;
    // }
    // 
    // code, pre {
    //     font-family: monospace, monospace;
    //     font-size: 1em;
    // }
    // 
    // table {
    //     text-align: left;
    //     width: 100%;
    //     max-width: 100%;
    //     border-collapse: collapse;
    //     margin-bottom: 2rem;
    // 
    //     td, th {
    //         vertical-align: top;
    //         padding: 0.5rem;
    //         border-bottom: 1px solid #eee;
    //     }
    // 
    //     code {
    //         white-space: nowrap;
    //     }
    // }
    // 
    // button {
    //     appearance: none;
    //     background: transparent;
    //     cursor: pointer;
    // }
    // 
    // progress {
    //     appearance: none;
    // }
    // 
    // /* Layout */
    // 
    // html {
    //     font-size: 11px;
    // }
    // 
    // @media(max-width: 767px) {
    //     html {
    //         font-size: 10px;
    //     }
    // }
    // 
    // p {
    //     margin-bottom: 3rem;
    // }
    // 
    // /* Code */
    // 
    // pre, code {
    //     font-family: 'Roboto Mono', monospace;
    // }
    // 
    // pre {
    //     margin-bottom: 3rem;
    // }
    // 
    // pre code {
    //     display: block;
    //     padding: 2rem !important;
    // }
    // 
    // /* Syntax highlighting */
    // 
    // .CodeMirror.cm-s-default {
    //     font-family: 'Roboto Mono', monospace;
    //     background: #f7f7f7;
    //     color: #403f53;
    //     word-wrap: break-word;
    // 
    //     .CodeMirror-line {
    //         padding: 0;
    //     }
    // 
    //     .CodeMirror-selected {
    //         background: #7a81812b;
    //     }
    // 
    //     .CodeMirror-cursor {
    //         border-left-color: currentColor;
    //     }
    // 
    //     .cm-variable-2 {
    //         color: inherit;
    //         font-style: italic;
    //     }
    // 
    //     .cm-comment {
    //         color: #989fb1;
    //     }
    // 
    //     .cm-keyword, .cm-builtin {
    //         color: #994cc3;
    //     }
    // 
    //     .cm-operator {
    //         color: #994cc3;
    //     }
    // 
    //     .cm-string {
    //         color: #994cc3;
    //     }
    // 
    //     .cm-number {
    //         color: #aa0982;
    //     }
    // 
    //     .cm-def {
    //         color: #4876d6;
    //     }
    // }
    // 
    // .jp-RenderedText pre {
    //     .ansi-cyan-fg.ansi-cyan-fg {
    //         color: #00d8ff;
    //     }
    // 
    //     .ansi-green-fg.ansi-green-fg {
    //         color: #12dc55;
    //     }
    // 
    //     .ansi-red-fg.ansi-red-fg {
    //         color: #f76464;
    //     }
    // }
    // 
    // /* Gatsby Images */
    // 
    // .gatsby-resp-image-link {
    //     border: 0;
    // }
    // 
    // .gatsby-resp-image-figure {
    //     margin-bottom: 4rem;
    // }
    // 
    // .gatsby-resp-image-figcaption {
    //     padding-top: 1rem;
    //     text-align: center;
    // 
    //     code {
    //         color: inherit;
    //     }
    // }

`;
