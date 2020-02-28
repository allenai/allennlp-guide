import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';
import { Header } from '@allenai/varnish/components/Header';

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
                            headerLinks {
                                text
                                url
                            }
                        }
                    }
                }
            `}
            render={data => {
                const { headerLinks } = data.site.siteMetadata;
                return (
                    <ThemeProvider>
                        <Head title={title} description={description} />
                        <GlobalStyle />
                        <Header alwaysVisible={true}>
                            <HeaderColumns>
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
                            </HeaderColumns>
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

const HeaderColumns = styled.div`
    display: grid;
    grid-template-columns: 18rem auto;
    grid-gap: 1rem;
    width: 100%;
    padding: 9px 0;
    align-items: center;

    nav ul {
      display: flex;
      justify-content: flex-end;

      li + li {
        margin-left: 40px;
      }
    }

    @media (max-width: 1024px) {
        nav {
            display: none;
        }
    }

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        padding: 0;
        line-height: 1.5 !important;
        min-height: 50px;
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

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        height: 100%;

        a {
            height: 100%;

            svg {
                width: 116px;
                height: 20px;
                fill: ${({ theme }) => theme.color.B6};
            }

            span {
                font-size: 27px;
                padding-left: 11px;
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
        font-size: 100% !important;
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

        & > header {
            main {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                max-width: 1252px !important;

                @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
                    padding: 0 18px !important;
                }
            }
        }
    }

    article, aside, details, figcaption, figure, footer, header, main, menu, nav,
    section, summary, progress {
        display: block;
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
