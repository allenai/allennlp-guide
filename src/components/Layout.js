import React, { useState } from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';
import { Header } from '@allenai/varnish/components/Header';
import { Menu } from 'antd';

import Head from './Head';
import { Link } from './Link';
import { Navigation, mobileNavEntrance } from './Navigation';
import { AllenNLPLogo, MenuIcon } from './inlineSVG';
import { toRem } from '../utils';

const Layout = ({
    title,
    description,
    groupedChapters,
    defaultSelectedKeys = [],
    defaultOpenKeys = [],
    children
}) => {
    const [mobileNavIsActive, setMobileNav] = useState(false);

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
                        <GlobalStyle mobileNavIsActive={mobileNavIsActive} />
                        <HeaderContainer mobileNavIsActive={mobileNavIsActive}>
                            <VarnishHeader alwaysVisible={true}>
                                <HeaderContent>
                                    <LogoContainer>
                                        <a href="/">
                                            <AllenNLPLogo />
                                            <span>Course</span>
                                        </a>
                                    </LogoContainer>
                                    <nav>
                                        <ul>
                                            {headerLinks.map(headerLink => (
                                                <li key={headerLink.url}>
                                                    <Link to={headerLink.url}>
                                                        {headerLink.text}
                                                    </Link>
                                                </li>
                                            ))}
                                        </ul>
                                    </nav>
                                    <MobileNavTrigger
                                        onClick={() => setMobileNav(!mobileNavIsActive)}
                                        aria-label="Toggle navigation">
                                        <MenuIcon mobileNavIsActive={mobileNavIsActive} />
                                    </MobileNavTrigger>
                                </HeaderContent>
                            </VarnishHeader>
                            <MobileNavContainer mobileNavIsActive={mobileNavIsActive}>
                                <MobileNavContent>
                                    <Navigation
                                        isMobile={true}
                                        headerLinks={headerLinks}
                                        groupedChapters={groupedChapters}
                                        defaultSelectedKeys={defaultSelectedKeys}
                                        defaultOpenKeys={defaultOpenKeys}
                                    />
                                </MobileNavContent>
                            </MobileNavContainer>
                        </HeaderContainer>
                        <Main>{children}</Main>
                    </ThemeProvider>
                );
            }}
        />
    );
};

export default Layout;

const HeaderContainer = styled(({ mobileNavIsActive, ...props }) => <div {...props} />)`
    width: 100%;
    top: 0;
    position: sticky;
    z-index: ${({ theme }) => theme.zIndex.header};

    @media (max-width: 1024px) {
        min-height: ${({ mobileNavIsActive }) => (mobileNavIsActive ? '100%' : '0')};
    }
`;

const VarnishHeader = styled(Header)``;

const HeaderContent = styled.div`
    display: flex;
    position: relative;
    width: 100%;
    padding: 9px 0;
    align-items: center;

    nav {
        margin-left: auto;

        ul {
            display: flex;
            justify-content: flex-end;

            li + li {
                margin-left: 40px;
            }
        }
    }

    @media (max-width: 1024px) {
        nav {
            display: none;
        }
    }

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        display: flex;
        justify-content: center;
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

// This is the menu trigger element that includes hamburger menu icon
const MobileNavTrigger = styled.button`
    display: none;

    // Show mobile menu below tablet portrait
    @media (max-width: 1024px) {
        display: grid;
        border: none;
        outline: none;
        background: transparent;
        cursor: pointer;
        position: relative;
        right: -16px;
        margin-left: auto;
        top: 0;
        padding: 13px;
    }
`;

// Panel that contains the mobile menu, takes up entire
// mobile screen, but sits behind header bar
const MobileNavContainer = styled(({ mobileNavIsActive, ...props }) => <div {...props} />)`
    width: 100%;
    height: 100%;
    background: ${({ theme }) => theme.color.N1};
    position: absolute;
    top: 0;
    left: 0;
    overflow-x: hidden;
    overflow-y: auto;
    display: none;

    // Only show MobileNavContainer if mobileNavIsActive is true
    // AND screen width is tablet portrait or below
    ${({ mobileNavIsActive, theme }) =>
        mobileNavIsActive
            ? `
        @media (max-width: 1024px) {
            display: block;
        }
    `
            : ''}
`;

const MobileNavContent = styled.div`
    padding-top: ${toRem(116)};
    animation: ${({ theme }) => mobileNavEntrance(theme.spacing.xxl)} 0.66s ease forwards;

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        padding-top: ${toRem(82)};
    }
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

        & > ${HeaderContainer} > header {
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

    // This enables use of min-height: 100% by HeaderContainer and
    // height: 100% by MobileNavContainer while limiting content height to
    // height of viewport so user cannot scroll past the menu and wind up
    // in a confusing scroll state -- but it only does this if mobile
    // nav is open and screen width is below tablet portrait
    ${({ mobileNavIsActive, theme }) =>
        mobileNavIsActive
            ? `
        @media (max-width: 1024px) {
            html,
            body,
            #___gatsby,
            #___gatsby > div[role="group"],
            #gatsby-focus-wrapper {
                height: 100%;
                overflow: hidden;
            }
        }
    `
            : ''}

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
