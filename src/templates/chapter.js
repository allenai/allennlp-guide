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
import styled, { createGlobalStyle, css } from 'styled-components';
import { Button } from '@allenai/varnish/components/button';

import { renderAst } from '../markdown';
import { ChapterContext } from '../context';
import Layout from '../components/Layout';
import { Footer } from '../components/Footer';
import { IconBox } from '../components/IconBox';
import { Link } from '../components/Link';
import { codeBlockTextStyles, codeBlockWrappingStyles } from '../components/code/CodeBlock';

import { outline } from '../outline';
import { getGroupedChapters, getIcon } from '../utils';

const Template = ({ data, location }) => {
    const { allMarkdownRemark, markdownRemark, site } = data;
    const { courseId } = site.siteMetadata;
    const { frontmatter, fields, htmlAst } = markdownRemark;
    const { title, description } = frontmatter;
    const { slug } = fields;

    // Build flat list of outline slugs that the prev/next navigation buttons can easily step through
    const slugList = {};
    slugList[`${outline.overview.slug}`] = '';
    outline.parts.forEach(part => {
        if (part.chapterSlugs) {
            part.chapterSlugs.forEach(slug => {
                slugList[`${slug}`] = part.title;
            });
        }
    });

    // Util consts for slugs and outline data
    const groupedChapters = getGroupedChapters(allMarkdownRemark);
    const links = Object.keys(slugList);
    const thisPart = outline.parts.find(part => part.title === slugList[slug]);
    const isOverview = slug === outline.overview.slug;
    const getProp = prop => (isOverview ? outline.overview[prop] : thisPart[prop]);

    const [activeExc, setActiveExc] = useState(null);
    const [completed, setCompleted] = useLocalStorage(
        `${courseId}-completed-${slug.substring(1)}`,
        []
    );
    const [storedUserExpandedGroups, setUserExpandedGroups] = useLocalStorage('expandedGroups');

    // User-defined nav group expand/collapse state
    const userExpandedGroups = [].concat(storedUserExpandedGroups);
    if (!isOverview && !userExpandedGroups.includes(thisPart.title)) {
        userExpandedGroups.push(thisPart.title);
    }
    const toggleMenuKey = key => {
        const index = userExpandedGroups.indexOf(key);
        if (index > -1) {
            userExpandedGroups.splice(index, 1);
        } else {
            userExpandedGroups.push(key);
        }
        setUserExpandedGroups(userExpandedGroups);
    };

    const html = renderAst(htmlAst);
    import(`prismjs/components/prism-python`).then(() => Prism.highlightAll());

    const handleSetActiveExc = (id) => {
        let scrollV, scrollH, loc = window.location;
        if (id !== null) {
            loc.hash = `${id}`;
        } else {
            if ('pushState' in history) {
                history.pushState('', document.title, loc.pathname + loc.search);
            } else {
                // Prevent scrolling by storing the page's current scroll offset
                scrollV = document.body.scrollTop;
                scrollH = document.body.scrollLeft;
                loc.hash = '';
                // Restore the scroll offset, should be flicker free
                document.body.scrollTop = scrollV;
                document.body.scrollLeft = scrollH;
            }

        }
        setActiveExc(id);
    };

    useEffect(() => {
        if (location.hash) {
            setActiveExc(parseInt(location.hash.split('#')[1]));
        }
    }, [location.hash]);

    const getMenuIcon = obj =>
        obj.antMenuIcon ? (
            <Icon type={obj.antMenuIcon} />
        ) : (
            obj.icon && <CustomIcon component={() => getIcon(obj.icon, 17)} />
        );

    return (
        <ChapterContext.Provider
            value={{
                activeExc,
                setActiveExc: handleSetActiveExc,
                completed,
                setCompleted,
                userExpandedGroups,
                setUserExpandedGroups
            }}>
            <Layout title={title} description={description}>
                <GlobalStyle />
                <Wrapper>
                    <LeftContainer>
                        <LeftContent>
                            <SideNav>
                                <Menu
                                    defaultSelectedKeys={[slug]}
                                    defaultOpenKeys={userExpandedGroups}
                                    mode="inline">
                                    <Menu.Item key={outline.overview.slug}>
                                        <Link to={outline.overview.slug}>
                                            {getMenuIcon(outline.overview)}
                                            <span>
                                                {
                                                    groupedChapters[outline.overview.slug].node
                                                        .frontmatter.title
                                                }
                                            </span>
                                        </Link>
                                    </Menu.Item>
                                    {outline.parts.map(
                                        part =>
                                            part.chapterSlugs && (
                                                <Menu.SubMenu
                                                    key={part.title}
                                                    onTitleClick={() => toggleMenuKey(part.title)}
                                                    title={
                                                        <span>
                                                            {getMenuIcon(part)}
                                                            <span>{part.title}</span>
                                                        </span>
                                                    }>
                                                    {part.chapterSlugs.map(chapterSlug => (
                                                        <Menu.Item key={chapterSlug}>
                                                            <Link to={chapterSlug}>
                                                                {
                                                                    groupedChapters[chapterSlug]
                                                                        .node.frontmatter.title
                                                                }
                                                            </Link>
                                                        </Menu.Item>
                                                    ))}
                                                </Menu.SubMenu>
                                            )
                                    )}
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
                                    {!isOverview && (
                                        <PartTitle>
                                            <span>{thisPart.title}</span>
                                        </PartTitle>
                                    )}
                                    {title && (
                                        <h1>
                                            <span>{title}</span>
                                        </h1>
                                    )}
                                    {description && <p>{description}</p>}
                                </ChapterIntroText>
                            </ChapterIntro>
                            {html}
                            <Pagination>
                                <div>
                                    {links.indexOf(slug) !== 0 && (
                                        <Button
                                            variant="primary"
                                            onClick={() =>
                                                navigate(links[links.indexOf(slug) - 1])
                                            }>
                                            « Previous Chapter
                                        </Button>
                                    )}
                                </div>
                                <div>
                                    {links.indexOf(slug) !== links.length - 1 && (
                                        <Button
                                            variant="primary"
                                            onClick={() =>
                                                navigate(links[links.indexOf(slug) + 1])
                                            }>
                                            Next Chapter »
                                        </Button>
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

const codeBgStyles = css`
    // Halfway between Varnish N2 and N3
    background: #f4f6f8 !important;
`;

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
                            background: linear-gradient(90deg, ${({ theme }) =>
                                `${theme.color.B5}, ${theme.color.B5}`}) !important;
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

        // Generic Elements

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

        table {
            text-align: left;
            width: 100%;
            max-width: 100%;
            border-collapse: collapse;
            margin-bottom: 1.5rem;

            td, th {
                ${({ theme }) => theme.typography.body}
                vertical-align: top;
                padding: 0.5rem 0.75rem;
                border: 1px solid ${({ theme }) => theme.color.N4};
            }

            th {
                color: ${({ theme }) => theme.color.N10};
            }

            tbody {
                tr:nth-child(even) {
                    background: ${({ theme }) => theme.color.N2};
                }
            }

            code {
                white-space: nowrap;
            }
        }

        progress {
            appearance: none;
        }

        // Inline Code styles

        pre,
        code,
        code[class*="language-"],
        pre[class*="language-"],
        p > code,
        li > code,
        th > code,
        td > code,
        a > code {
            ${codeBlockTextStyles}
        }

        pre {
            ${codeBgStyles}
            overflow: visible;
        }

        pre > code {
            background: transparent;
        }

        p,
        li,
        th,
        td,
        a {
            & > code {
                ${codeBgStyles}
                padding-left: 0.3rem;
                padding-right: 0.3rem;
            }
        }

        /* Code blocks */
        [class*="MarkdownContainer"] > pre,
        .gatsby-highlight > pre {
            padding: 1rem 1.25rem 1rem;
            margin: 1.75rem 0;
            overflow: visible;
        }

        /* Inline code */
        :not(pre) > code[class*="language-"] {
            padding: .1em;
            border-radius: .3em;
            white-space: normal;
        }

        pre[data-line] {
            overflow-x: hidden !important;
        }

        code[class*="language-"],
        pre[class*="language-"] {
            color: black;
            ${codeBlockWrappingStyles}

            // Prism highlighting styles

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
            .token.inserted {
                color: #690;
            }

            .token.operator,
            .token.entity,
            .token.url,
            .language-css .token.string,
            .style .token.string {
                color: #9a6e3a;
            }

            .token.atrule,
            .token.attr-value,
            .token.keyword,
            .token.builtin {
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
                margin-top: 15px;
            }

            .line-numbers-rows {
                z-index: 1;
                border-right-color: ${({ theme }) => theme.color.N5} !important;

                & > span:before {
                    color: ${({ theme }) => theme.color.N6};
                }
            }

            &.line-numbers {
                & > code {
                    z-index: 2;
                }

                .line-numbers-rows {
                    top: -1px;
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

    @media (max-width: 1024px) {
        display: block;
    }
`;

// Left-aligned container with white background
const LeftContainer = styled.div`
    background: ${({ theme }) => theme.color.N1};
    width: calc(
        ${({ theme }) =>
            `324px + ((100vw - (${theme.breakpoints.xl} + ${theme.spacing.xxl}) - ${theme.spacing.xxl}) / 2) + ${theme.spacing.xxl}`}
    );
    height: 100%;
    display: flex;
    position: relative;
    z-index: 3;

    @media (max-width: 1024px) {
        display: none;
    }
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
    overflow-x: hidden;
    overflow-y: auto;
`;

const RightContainer = styled.div`
    position: relative;
    flex: 1;
    max-width: ${({ theme }) =>
        theme.breakpoints.xl.getRemValue() +
        theme.spacing.xxl.getRemValue() -
        theme.spacing.xxl.getRemValue()}rem;
    height: 100%;

    &:before {
        position: fixed;
        top: 65px;
        display: block;
        content: '';
        width: 100%;
        height: 50px;
        z-index: 2;
        margin-left: -30px;
        box-shadow: 0 -${({ theme }) => `${theme.spacing.md} ${theme.spacing.xl} ${theme.spacing.lg} ${theme.color.N3}`};

        @media (max-width: 1024px) {
            display: none;
        }
    }
`;

const RightContent = styled.div`
    max-width: ${({ theme }) =>
        theme.breakpoints.xl.getRemValue() +
        theme.spacing.xxl.getRemValue() -
        theme.spacing.xxl.getRemValue() -
        324 / 16}rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: ${({ theme }) => `${theme.spacing.xxl} 0 0 ${theme.spacing.xxl}`};
    box-sizing: border-box;
    margin-right: ${({ theme }) => theme.spacing.xxl};

    @media (max-width: 1024px) {
        max-width: 100%;
        padding-right: ${({ theme }) => theme.spacing.xxl};
        margin-right: 0;
    }

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        padding: ${({ theme }) => theme.spacing.lg};
    }
`;

// Intro content rendered from markdown frontmatter and outline data
const ChapterIntro = styled.div`
    display: grid;
    grid-template-columns: 75px auto;
    grid-gap: ${({ theme }) => theme.spacing.xl};

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        display: block;
    }
`;

// Colored box with icon that denotes Part
const StyledIconBox = styled(IconBox)`
    width: 75px;

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        width: 50px;
    }
`;

const mobileChapterTitleStyles = css`
    margin: -50px 0 24px 74px;
    min-height: 50px;
    display: flex;
    align-items: center;
`;

// Text displayed in chapter intro next to icon
const ChapterIntroText = styled.div`
    h1 {
        ${({ theme }) => theme.typography.h2}
        margin: ${({ theme }) => `-${theme.spacing.xxs} 0 ${theme.spacing.md} 0`};
        color: ${({ theme }) => theme.color.B6};

        @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
            ${({ theme }) => theme.typography.h3}

            &:first-child {
                ${mobileChapterTitleStyles}
            }
        }
    }

    p {
        ${({ theme }) => theme.typography.bodyBig}
        color: ${({ theme }) => theme.color.N10};
    }
`;

const PartTitle = styled.strong`
    ${({ theme }) => theme.typography.h6}
    display: block;
    text-transform: uppercase;
    margin: 0 0 23px 2px;

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        ${mobileChapterTitleStyles}
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

        @media (max-width: 1024px) {
            text-align: center;
            padding: ${({ theme }) => `${theme.spacing.sm} ${theme.spacing.lg}`};
        }
    }
`;
