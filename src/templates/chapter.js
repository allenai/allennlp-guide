// Prism Code formatting modules
import Prism from 'prismjs';
import 'prismjs/plugins/line-highlight/prism-line-highlight.js';
import 'prismjs/plugins/line-numbers/prism-line-numbers.js';
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
import Layout from './Layout';
import { Footer } from '../components/Footer';
import { IconBox } from '../components/IconBox';
import { LinkComponent } from '../components/LinkComponent';
import { outline } from '../outline';
import { getGroupedChapters } from '../utils';

const Template = ({ data, location }) => {
    const { allMarkdownRemark, markdownRemark, site } = data;
    const { courseId } = site.siteMetadata;
    const { frontmatter, fields, htmlAst } = markdownRemark;
    const { title, description } = frontmatter;
    const { slug } = fields;
    const groupedChapters = getGroupedChapters(allMarkdownRemark);
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
            loc.hash = '';
            if (typeof window.history.replaceState === 'function') {
                window.history.replaceState({}, '', window.location.href.slice(0, -1));
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
//outline.parts slugList[slug]
//slugList[slug]
    const links = Object.keys(slugList);
    const thisPart = outline.parts.find(part => part.title === slugList[slug]);
    const isOverview = slug === outline.overview.slug;
    const getProp = (prop) => isOverview ? outline.overview[prop] : thisPart[prop];

    return (
        <ChapterContext.Provider
            value={{ activeExc, setActiveExc: handleSetActiveExc, completed, setCompleted }}
        >
            <Layout title={title} description={description}>
                <GlobalStyle />
                <Wrapper>
                    <Left>
                        <LeftContent>
                            <SideNav>
                              <NavContent>
                                  <Menu
                                      defaultSelectedKeys={[slug]}
                                      defaultOpenKeys={[!isOverview ? thisPart.title : null ]}
                                      mode="inline"
                                  >
                                      <Menu.Item key={outline.overview.slug}>
                                          <LinkComponent to={outline.overview.slug}>
                                              <Icon type="setting" />
                                              <span>{groupedChapters[outline.overview.slug].node.frontmatter.title}</span>
                                          </LinkComponent>
                                      </Menu.Item>
                                      {outline.parts.map((part) => part.chapterSlugs && (
                                          <Menu.SubMenu
                                              key={part.title}
                                              title={
                                                  <span>
                                                      <Icon type="setting" />
                                                      <span>{part.title}</span>
                                                  </span>
                                              }
                                          >
                                              {part.chapterSlugs.map((chapterSlug) => (
                                                  <Menu.Item key={chapterSlug}><LinkComponent to={chapterSlug}>{groupedChapters[chapterSlug].node.frontmatter.title}</LinkComponent></Menu.Item>
                                              ))}
                                          </Menu.SubMenu>
                                      ))}
                                  </Menu>
                              </NavContent>
                            </SideNav>
                        </LeftContent>
                    </Left>
                    <Right>
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
                    </Right>
                </Wrapper>
            </Layout>
        </ChapterContext.Provider>
    );
};

export default Template;

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

// Resetting root layout
const GlobalStyle = createGlobalStyle`
    &&& {
        .ant-menu {
            border: none !important;
            
            .ant-menu-submenu {
                border-top: 1px solid ${({ theme }) => theme.color.N4} !important;
                
                &.ant-menu-submenu-selected {
                    span,
                    i {
                        color: ${({ theme }) => theme.color.B5} !important;
                    }
                }

                .ant-menu-submenu-title:hover {
                    span,
                    i,
                    i:before,
                    i:after {
                        color: ${({ theme }) => theme.color.B5} !important;
                    }
                }
            }

            .ant-menu-item {
                overflow: visible !important;
                white-space: normal !important;
                height: auto !important;
                line-height: 1.5 !important;
                padding-top: 9px !important;
                padding-bottom 10px !important;
                
                a {
                    color: ${({ theme }) => theme.color.N10};
                    
                    &:hover {
                        color: ${({ theme }) => theme.color.B5};
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

const Wrapper = styled.div`
    display: flex;
    width: 100%;
    height: 100%;
    background: ${({ theme }) => theme.color.N3};
`;

const Left = styled.div`
    background: ${({ theme }) => theme.color.N1};
    width: calc(${({ theme }) => `300px + ((100vw - ${theme.breakpoints.xl} - ${theme.spacing.xxl}) / 2) + ${theme.spacing.xxl}`});
    height: 100%;
    display: flex;
    position: relative;
    z-index: 3;
`;

const LeftContent = styled.div`
    width: 326px;
    height: 100%;
    margin-left: auto;
`;

// Nav

const SideNav = styled.nav`
    position: relative;
    height: 100%;
    z-index: 3;
    font-size: 14px;
    box-sizing: content-box;;

    h1 {
        margin-bottom: 20px;
    }

    ol {
        display: none;
        list-style: none;
        padding-left: 0;
        
        strong {
          display: block;
          color: ${({ theme }) => theme.color.N10};
          padding-top: 15px;
          border-top: 1px solid #ddd;
          margin-top: 15px;
          padding-bottom: 5px;
        }
    }
`;

const NavContent = styled.div`
    padding-top: 30px;
    position: sticky;
    top: 115px;
`;

const Right = styled.div`
    position: relative;
    flex: 1;
    max-width: ${({ theme }) => theme.breakpoints.xl.getRemValue() - theme.spacing.xxl.getRemValue()}rem;
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
        box-shadow: 0 -15px 30px 30px ${({ theme }) => theme.color.N3};
    }
`;

const RightContent = styled.div`
    width: 100%;
    max-width: ${({ theme }) => theme.breakpoints.xl.getRemValue() - theme.spacing.xxl.getRemValue() - (300 / 16)}rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: ${({ theme }) => `${theme.spacing.xxl} 0 0 ${theme.spacing.xxl}`};
    box-sizing: border-box;
`;

const ChapterIntro = styled.div`
    display: grid;
    grid-template-columns: 75px auto;
    grid-gap: ${({ theme }) => theme.spacing.xl};
`;

const StyledIconBox = styled(IconBox)`
    width: 75px;
`;

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

const Pagination = styled.div`
    padding: 54px 0;
    width: 100%;
    display: flex;
    
    div:last-child {
        margin-left: auto;
    }
`;

const ChapterFooter = styled(Footer)`
    &&& {
        padding: ${({ theme }) => theme.spacing.xl} 0;
        background: transparent;
        text-align: left;
    }
`;
