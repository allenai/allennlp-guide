// Prism Code formatting modules
import Prism from 'prismjs';
import 'prismjs/plugins/line-highlight/prism-line-highlight.js';
import 'prismjs/plugins/line-numbers/prism-line-numbers.js';
import 'prismjs/plugins/normalize-whitespace/prism-normalize-whitespace.js';
// Prism CSS
import 'prismjs/plugins/line-highlight/prism-line-highlight.css';
import 'prismjs/plugins/line-numbers/prism-line-numbers.css';

import React, { useState, useEffect } from 'react';
import { graphql, navigate } from 'gatsby';
import { Menu, Icon } from 'antd';
import useLocalStorage from '@illinois/react-use-local-storage';
import styled, { createGlobalStyle } from 'styled-components';
import { Button } from '@allenai/varnish/components/button';

import { renderAst } from '../markdown';
import { ChapterContext } from '../context';
import Layout from '../components/Layout';
import { Footer } from '../components/Footer';
import { IconBox } from '../components/IconBox';
import { Link } from '../components/Link';
import { outline } from '../outline';
import { getGroupedChapters, getIcon } from '../utils';

const Template = ({ data, location }) => {
    const { allMarkdownRemark, markdownRemark, site } = data;
    const { courseId } = site.siteMetadata;
    const { frontmatter, fields, htmlAst } = markdownRemark;
    const { title, description } = frontmatter;
    const { slug } = fields;
    const [activeExc, setActiveExc] = useState(null);
    const [completed, setCompleted] = useLocalStorage(`${courseId}-completed-${slug.substring(1)}`, []);
    const html = renderAst(htmlAst);
    import(`prismjs/components/prism-python`).then(() => Prism.highlightAll());

    const handleSetActiveExc = (id) => {
        const loc = window.location;
        if (id !== null) {
            loc.hash = `${id}`;
        } else {
            // Prevent #null from showing up in the URL
            loc.replace('#');
            if (typeof window.history.replaceState === 'function') {
                window.history.replaceState({}, '', loc.href.slice(0, -1));
            }
        }
        setActiveExc(id);
    };

    useEffect(() => {
        if (location.hash) {
            setActiveExc(parseInt(location.hash.split('#')[1]));
        }
    }, [location.hash]);

    // Build flat list of outline slugs that the prev/next navigation buttons can easily step through
    let slugList = {}
    slugList[`${outline.overview.slug}`] = "";
    outline.parts.forEach((part) => {
      if (part.chapterSlugs) {
        part.chapterSlugs.forEach((slug) => {
          slugList[`${slug}`] = part.title;
        });
      }
    });

    // Util consts for slugs and outline data
    const groupedChapters = getGroupedChapters(allMarkdownRemark);
    const links = Object.keys(slugList);
    const thisPart = outline.parts.find(part => part.title === slugList[slug]);
    const isOverview = slug === outline.overview.slug;
    const getProp = (prop) => isOverview ? outline.overview[prop] : thisPart[prop];

    const getMenuIcon = (icon) => icon === 'tools' ? (
        <Icon type="setting" />
    ) : (
        <CustomIcon component={() => getIcon(icon)} />
    );

    return (
        <ChapterContext.Provider
            value={{ activeExc, setActiveExc: handleSetActiveExc, completed, setCompleted }}
        >
            <Layout title={title} description={description}>
                <GlobalStyle />
                <Wrapper>
                    <LeftContainer>
                        <LeftContent>
                            <SideNav>
                                <Menu
                                    defaultSelectedKeys={[slug]}
                                    defaultOpenKeys={[!isOverview ? thisPart.title : null ]}
                                    mode="inline">
                                    <Menu.Item key={outline.overview.slug}>
                                        <Link to={outline.overview.slug}>
                                            {getMenuIcon(outline.overview.icon)}
                                            <span>{groupedChapters[outline.overview.slug].node.frontmatter.title}</span>
                                        </Link>
                                    </Menu.Item>
                                    {outline.parts.map((part) => part.chapterSlugs && (
                                        <Menu.SubMenu
                                            key={part.title}
                                            title={
                                                <span>
                                                    {getMenuIcon(part.icon)}
                                                    <span>{part.title}</span>
                                                </span>
                                            }>
                                            {part.chapterSlugs.map((chapterSlug) => (
                                                <Menu.Item key={chapterSlug}>
                                                    <Link to={chapterSlug}>{groupedChapters[chapterSlug].node.frontmatter.title}</Link>
                                                </Menu.Item>
                                            ))}
                                        </Menu.SubMenu>
                                    ))}
                                </Menu>
                            </SideNav>
                        </LeftContent>
                    </LeftContainer>
                    <RightContainer>
                        <RightContent>
                            <ChapterIntro>
                                <div>
                                    <StyledIconBox
                                        color={getProp('color')}
                                        icon={getProp('icon')}
                                    />
                                </div>
                                <ChapterIntroText>
                                    {title && <h1>{title}</h1>}
                                    {description && (
                                        <p>{description}</p>
                                    )}
                                </ChapterIntroText>
                            </ChapterIntro>
                            {html}
                            <Pagination>
                              <div>
                                {links.indexOf(slug) !== 0 && (
                                    <Button variant="primary" onClick={() => navigate(links[links.indexOf(slug) - 1])}>« Previous Chapter</Button>
                                )}
                              </div>
                              <div>
                                {links.indexOf(slug) !== links.length - 1 && (
                                    <Button variant="primary" onClick={() => navigate(links[links.indexOf(slug) + 1])}>Next Chapter »</Button>
                                )}
                              </div>
                            </Pagination>
                            <ChapterFooter />
                        </RightContent>
                    </RightContainer>
                </Wrapper>
            </Layout>
        </ChapterContext.Provider>
    );
};

export default Template;

// GraphQL Query
export const pageQuery = graphql`
    query($slug: String!) {
        site {
            siteMetadata {
                courseId
                title
            }
        }
        allMarkdownRemark {
            edges {
                node {
                    fields {
                        slug
                    }
                    frontmatter {
                        title
                        description
                    }
                }
            }
        }
        markdownRemark(fields: { slug: { eq: $slug } }) {
            htmlAst
            fields {
                slug
            }
            frontmatter {
                title
                description
            }
        }
    }
`;

const CustomIcon = styled(Icon)``;

// Resetting Ant Menu Styles
const GlobalStyle = createGlobalStyle`
    &&& {
        .ant-menu {
            border: none !important;

            svg {
                color: ${({ theme }) => theme.color.N8};
            }
            
            ${CustomIcon} {
                svg {
                    width: 17px;
                    height: 17px;
                    margin-right: -4px;
                    transform: translate(-2px, 1.5px);
                    stroke: ${({ theme }) => theme.color.N8};
                }
            }
            
            .ant-menu-submenu {
                border-top: 1px solid ${({ theme }) => theme.color.N4} !important;
                
                &.ant-menu-submenu-selected {
                    span,
                    i,
                    svg {
                        color: ${({ theme }) => theme.color.B5} !important;
                        stroke: ${({ theme }) => theme.color.B5};
                    }
                }

                .ant-menu-submenu-title:hover {
                    span,
                    i,
                    svg,
                    i:before,
                    i:after {
                        color: ${({ theme }) => theme.color.B5} !important;
                        stroke: ${({ theme }) => theme.color.B5};
                    }
                }
            }

            .ant-menu-submenu-title {
                &:hover {
                    .ant-menu-submenu-arrow {
                        &:before,
                        &:after {
                            background: linear-gradient(90deg, ${({ theme }) => `${theme.color.B5}, ${theme.color.B5}`}) !important;
                        }
                    }
                }
            }

            // Support multi-line items without truncation
            .ant-menu-submenu-title,
            .ant-menu-item {
                overflow: visible !important;
                white-space: normal !important;
                height: auto !important;
                line-height: 1.5 !important;
                padding-top: 9px !important;
                padding-bottom 10px !important;
            }

            .ant-menu-item {
                a {
                    color: ${({ theme }) => theme.color.N10};
                    
                    &:hover {
                        &,
                        svg {
                            color: ${({ theme }) => theme.color.B5};
                            stroke: ${({ theme }) => theme.color.B5};
                        }
                        text-decoration: none;
                    }
                }
                
                &.ant-menu-item-selected {
                    background: ${({ theme }) => theme.color.B1} !important;
                    
                    &:after {
                        border-color: ${({ theme }) => theme.color.B5} !important;
                    }
                    
                    a {
                        color: ${({ theme }) => theme.color.B5} !important;
                    }
                }
            }
        }
    }
`;

// Everything below the header, container for left and right containers with distinct backgrounds
const Wrapper = styled.div`
    display: flex;
    width: 100%;
    height: 100%;
    background: ${({ theme }) => theme.color.N3};
`;

// Left-aligned container with white background
const LeftContainer = styled.div`
    background: ${({ theme }) => theme.color.N1};
    width: calc(${({ theme }) => `324px + ((100vw - (${theme.breakpoints.xl} + ${theme.spacing.xxl}) - ${theme.spacing.xxl}) / 2) + ${theme.spacing.xxl}`});
    height: 100%;
    display: flex;
    position: relative;
    z-index: 3;
`;

// Constrained content descendent of LeftContainer (holds sidenav)
const LeftContent = styled.div`
    width: 350px;
    height: 100%;
    margin-left: auto;
`;

// Sticky Outline navigation
const SideNav = styled.nav`
    position: relative;
    z-index: 3;
    box-sizing: content-box;
    padding-top: 30px;
    position: sticky;
    top: 115px;
    height: calc(100vh - 175px);
    overflow: auto;
`;

const RightContainer = styled.div`
    position: relative;
    flex: 1;
    max-width: ${({ theme }) => (theme.breakpoints.xl.getRemValue() + theme.spacing.xxl.getRemValue()) - theme.spacing.xxl.getRemValue()}rem;
    height: 100%;

    &:before {
        position: fixed;
        top: 65px;
        display: block;
        content: "";
        width: 100%;
        height: 50px;
        z-index: 2;
        margin-left: -30px;
        box-shadow: 0 -${({ theme }) => `${theme.spacing.md} ${theme.spacing.xl} ${theme.spacing.lg} ${theme.color.N3}`};
    }
`;

const RightContent = styled.div`
    max-width: ${({ theme }) => (theme.breakpoints.xl.getRemValue() + theme.spacing.xxl.getRemValue()) - theme.spacing.xxl.getRemValue() - (324 / 16)}rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: ${({ theme }) => `${theme.spacing.xxl} 0 0 ${theme.spacing.xxl}`};
    box-sizing: border-box;
    margin-right: ${({ theme }) => theme.spacing.xxl};
`;

// Intro content rendered from markdown frontmatter and outline data
const ChapterIntro = styled.div`
    display: grid;
    grid-template-columns: 75px auto;
    grid-gap: ${({ theme }) => theme.spacing.xl};
`;

// Colored box with icon that denotes Part
const StyledIconBox = styled(IconBox)`
    width: 75px;
`;

// Text displayed in chapter intro next to icon
const ChapterIntroText = styled.div`
    h1 {
        ${({ theme }) => theme.typography.h2}
        margin: ${({ theme }) => `-${theme.spacing.xxs} 0 ${theme.spacing.md} 0`};
        color: ${({ theme }) => theme.color.B6};
    }
    
    p {
        ${({ theme }) => theme.typography.bodyBig}
    }
`;

// Previous / Next chapter buttons
const Pagination = styled.div`
    padding: 54px 0;
    width: 100%;
    display: flex;
    
    div:last-child {
        margin-left: auto;
    }
`;

// Special chapter template instance of global footer
const ChapterFooter = styled(Footer)`
    &&& {
        padding: ${({ theme }) => theme.spacing.xl} 0;
        background: transparent;
        text-align: left;
    }
`;
