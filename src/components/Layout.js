import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';
import { Header } from '@allenai/varnish/components/Header';
import { HeaderColumns } from '@allenai/varnish/components/Header';

import Head from './Head';
import { Link } from './Link';
import { AllenNLPLogo } from './inlineSVG/AllenNLPLogo';

const Layout = ({ title, description, children }) => {
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
                        <Header alwaysVisible={true}>
                            <HeaderColumnsWithSpace gridTemplateColumns="18rem auto">
                                <LogoContainer>
                                    <Link to="/">
                                        <AllenNLPLogo />
                                        <span>Course</span>
                                    </Link>
                                </LogoContainer>
                                <nav>
                                    <ul>
                                        {headerLinks.map((headerLink) => (
                                            <li key={headerLink.url}>
                                                <Link to={headerLink.url}>
                                                    {headerLink.text}
                                                </Link>
                                            </li>
                                        ))}
                                    </ul>
                                </nav>
                            </HeaderColumnsWithSpace>
                        </Header>
                        <Main>
                            {children}
                        </Main>
                    </ThemeProvider>
                );
            }}
        />
    );
};

export default Layout;

const HeaderColumnsWithSpace = styled(HeaderColumns)`
    padding: 9px 0;
    align-items: center;
    
    nav ul {
      display: flex;
      justify-content: flex-end;
      
      li + li {
        margin-left: 40px;
      }
    }
`;

const LogoContainer = styled.div`
    a {
      display: flex;
      align-items: center;

      svg {
        display: block;
        transition: fill 0.2s ease;
      }

      span {
        display: block;
        font-size: 34px;
        padding-left: 14px;
        transition: color 0.2s ease;
        color: ${({ theme }) => theme.color.N10};
      }

      &:hover {
        text-decoration: none !important;

        svg {
          fill: ${({ theme }) => theme.color.B6};
        }

        span {
          color: ${({ theme }) => theme.color.B6};
        }
      }
    }
`;

const Main = styled.div`
    display: flex;
    flex-direction: column;
    flex: 1;
`;

// Resetting root layout
const GlobalStyle = createGlobalStyle`
    html,
    body {
        width: 100%;
        height: 100%;
        background: ${({ theme }) => theme.color.N1} !important;
    }

    body {
        display: flex;
        flex-direction: column;
    }

    #___gatsby {
        flex: 1;
    }

    #___gatsby > div[role="group"],
    #gatsby-focus-wrapper {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    // Reset styles ported from SASS
    // TODO(aarons): Whittle these styles down to essential. There is likely
    // stuff in here that we don't want to keep.

    article, aside, details, figcaption, figure, footer, header, main, menu, nav,
    section, summary, progress {
        display: block;
    }

    abbr[title] {
        border-bottom: none;
        text-decoration: underline;
        text-decoration: underline dotted;
    }

    small {
        font-size: 80%;
    }

    sub, sup {
        position: relative;
        font-size: 65%;
        line-height: 0;
        vertical-align: baseline;
    }
    
    sup {
        top: -0.5em;
    }
    
    sub {
        bottom: -0.15em;
    }

    img {
        border: 0;
        height: auto;
        max-width: 100%;
    }
    
    svg {
        max-width: 100%;
        color-interpolation-filters: sRGB;
        fill: currentColor;
    
        &:not(:root) {
            overflow: hidden;
        }
    }

    hr {
        box-sizing: content-box;
        overflow: visible;
        height: 0;
    }
    
    table {
        text-align: left;
        width: 100%;
        max-width: 100%;
        border-collapse: collapse;
        margin-bottom: 2rem;
    
        td, th {
            vertical-align: top;
            padding: 0.5rem;
            border-bottom: 1px solid #eee;
        }
    
        code {
            white-space: nowrap;
        }
    }

    progress {
        appearance: none;
    }


    // Code styles

    &&& {
        pre,
        code,
        code[class*="language-"],
        pre[class*="language-"],
        p > code,
        li > code {
            font-family: 'Roboto Mono', Courier, monospace;
            font-size: 0.87rem;
            -webkit-font-smoothing: subpixel-antialiased;
        }

        pre > code {
            background: transparent;
        }

        pre,
        p > code,
        li > code,
        [class^="CodeBlock__StyledCodeBlock"] {
            // Halfway between Varnish N2 and N3
            background: #f4f6f8 !important;
        }

        p > code,
        li > code {
           padding-left: 0.3rem;
           padding-right: 0.3rem;
        }

        pre {
            overflow: auto;
        }
    }
    
    // Prism highlighting styles

    /* PrismJS 1.16.0
    https://prismjs.com/download.html#themes=prism&languages=python */
    /**
     * prism.js default theme for JavaScript, CSS and HTML
     * Based on dabblet (http://dabblet.com)
     * @author Lea Verou
     */

    code[class*="language-"],
    pre[class*="language-"] {
        color: black;
        background: none;
        text-align: left;
        white-space: pre;
        word-spacing: normal;
        word-break: normal;
        word-wrap: normal;
        line-height: 1.5;
        -moz-tab-size: 4;
        -o-tab-size: 4;
        tab-size: 4;
        -webkit-hyphens: none;
        -moz-hyphens: none;
        -ms-hyphens: none;
        hyphens: none;
    }

    /* Code blocks */
    [class*="MarkdownContainer"] > pre,
    .gatsby-highlight > pre {
        padding: 1rem 1.25rem 1rem;
        margin: 1.75rem 0;
        overflow: auto;
    }

    // TODO(aarons): try to merge prism and CodeMirror syntax highlighting styles

    /* Inline code */
    :not(pre) > code[class*="language-"] {
        padding: .1em;
        border-radius: .3em;
        white-space: normal;
    }

    .token.comment,
    .token.prolog,
    .token.doctype,
    .token.cdata {
        color: slategray;
    }

    .token.punctuation {
        color: #999;
    }

    .namespace {
        opacity: .7;
    }

    .token.property,
    .token.tag,
    .token.boolean,
    .token.number,
    .token.constant,
    .token.symbol,
    .token.deleted {
        color: #905;
    }

    .token.selector,
    .token.attr-name,
    .token.string,
    .token.char,
    .token.builtin,
    .token.inserted {
        color: #690;
    }

    .token.operator,
    .token.entity,
    .token.url,
    .language-css .token.string,
    .style .token.string {
        color: #9a6e3a;
        // background: hsla(0, 0%, 100%, .5);
    }

    .token.atrule,
    .token.attr-value,
    .token.keyword {
        color: #07a;
    }

    .token.function,
    .token.class-name {
        color: #DD4A68;
    }

    .token.regex,
    .token.important,
    .token.variable {
        color: #e90;
    }

    .token.important,
    .token.bold {
        font-weight: bold;
    }
    .token.italic {
        font-style: italic;
    }

    .token.entity {
        cursor: help;
    }

    .line-highlight {
        background: ${({ theme }) => theme.color.N4} !important;
        outline: 1px solid ${({ theme }) => theme.color.N5};
        margin-top: 17px;
    }

    .line-numbers-rows {
        z-index: 1;
        border-right-color: ${({ theme }) => theme.color.N5} !important;

        & > span:before {
            color: ${({ theme }) => theme.color.N6};
        }
    }

    pre[class*="language-"].line-numbers > code {
        z-index: 2;
    }

    .line-numbers .line-numbers-rows {
        top: -1px;
    }

    // CodeMirror Styles

    .CodeMirror.cm-s-default {
        font-family: 'Roboto Mono', monospace;
        background: #f7f7f7;
        color: #403f53;
        word-wrap: break-word;
    
        .CodeMirror-line {
            padding: 0;
        }
    
        .CodeMirror-selected {
            background: #7a81812b;
        }
    
        .CodeMirror-cursor {
            border-left-color: currentColor;
        }
    
        .cm-variable-2 {
            color: inherit;
            font-style: italic;
        }
    
        .cm-comment {
            color: #989fb1;
        }
    
        .cm-keyword, .cm-builtin {
            color: #994cc3;
        }
    
        .cm-operator {
            color: #994cc3;
        }
    
        .cm-string {
            color: #994cc3;
        }
    
        .cm-number {
            color: #aa0982;
        }
    
        .cm-def {
            color: #4876d6;
        }
    }
    
    .jp-RenderedText pre {
        .ansi-cyan-fg.ansi-cyan-fg {
            color: #00d8ff;
        }
    
        .ansi-green-fg.ansi-green-fg {
            color: #12dc55;
        }
    
        .ansi-red-fg.ansi-red-fg {
            color: #f76464;
        }
    }
    
    // Gatsby image styles
    
    .gatsby-resp-image-link {
        border: 0;
    }
    
    .gatsby-resp-image-figure {
        margin-bottom: 4rem;
    }
    
    .gatsby-resp-image-figcaption {
        padding-top: 1rem;
        text-align: center;
    
        code {
            color: inherit;
        }
    }
`;
