import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';
import { Header } from '@allenai/varnish/components/Header';
import { HeaderColumns } from '@allenai/varnish/components/Header';

import Head from '../components/Head';
import { LinkComponent } from '../components/LinkComponent';
import { AllenNLPLogo } from '../components/inlineSVG/AllenNLPLogo';

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
                                    <LinkComponent to="/">
                                        <AllenNLPLogo />
                                        <span>Course</span>
                                    </LinkComponent>
                                </LogoContainer>
                                <nav>
                                    <ul>
                                        {headerLinks.map((headerLink) => (
                                            <li key={headerLink.url}>
                                                <LinkComponent to={headerLink.url}>
                                                    {headerLink.text}
                                                </LinkComponent>
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
        background: ${({ theme }) => theme.color.N1};
    }

    body {
        display: flex;
        flex-direction: column;
    }

    #___gatsby {
        flex: 1;
    }

    #___gatsby > div[role="group"] {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    // Reset styles ported from SASS
    // TODO(aarons): Whittle these styles down to essential. There is likely
    // stuff in here that we don't want to keep.

    *, *:before, *:after {
        box-sizing: border-box;
        padding: 0;
        margin: 0;
        border: 0;
        outline: 0;
    }
    
    .textblock {
        width: 800px;
        max-width: 100%;
        margin: auto;
    }

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

    pre, code {
        font-family: 'Roboto Mono', monospace;
    }
    
    pre {
        margin-bottom: 3rem;
        overflow: auto;
    }
    
    pre code {
        display: block;
        padding: 2rem !important;
    }
    
    // Syntax highlighting styles
    
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
