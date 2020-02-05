import Prism from 'prismjs'
import 'prismjs/plugins/line-highlight/prism-line-highlight.js'
import 'prismjs/plugins/line-highlight/prism-line-highlight.css'
import React, { useState, useEffect } from 'react'
import { graphql, navigate } from 'gatsby'
import useLocalStorage from '@illinois/react-use-local-storage'
import styled from 'styled-components';

import { renderAst } from '../markdown'
import { ChapterContext } from '../context'
import Layout from './Layout';
import { Footer } from '../components/Footer';
import { Button } from '../components/button'
import { LinkComponent } from '../components/LinkComponent';
import { outline } from '../outline';
import { getGroupedChapters } from '../utils';

const Template = ({ data, location }) => {
    const { allMarkdownRemark, markdownRemark, site } = data
    const { courseId } = site.siteMetadata
    const { frontmatter, fields, htmlAst } = markdownRemark
    const { title, description } = frontmatter
    const { slug } = fields
    const groupedChapters = getGroupedChapters(allMarkdownRemark);
    const [activeExc, setActiveExc] = useState(null)
    const [completed, setCompleted] = useLocalStorage(`${courseId}-completed-${slug.substring(1)}`, [])
    const html = renderAst(htmlAst)
    import(`prismjs/components/prism-python`).then(() => Prism.highlightAll())
    const handleSetActiveExc = id => {
        window.location.hash = `${id}`
        setActiveExc(id)
    }
    useEffect(() => {
        if (location.hash) {
            setActiveExc(parseInt(location.hash.split('#')[1]))
        }
    }, [location.hash])

    // Build flat list of outline slugs that the prev/next navigation buttons can easily step through
    let linkList = [];
    outline.forEach((node) => {
      if (node.slug) {
        linkList.push(node.slug);
      }
      if (node.chapterSlugs) {
        linkList = linkList.concat(node.chapterSlugs);
      }
    });

    return (
        <ChapterContext.Provider
            value={{ activeExc, setActiveExc: handleSetActiveExc, completed, setCompleted }}
        >
            <Layout title={title} description={description}>
                <ContentContainer>
                    <SideNav>
                      <NavContent>
                        <ol>
                          {outline.map((outlineNode) => !outlineNode.chapterSlugs ? (
                              <NavItem key={outlineNode.slug} isActive={outlineNode.slug === slug}>
                                <LinkComponent to={outlineNode.slug}>{groupedChapters[outlineNode.slug].node.frontmatter.title}</LinkComponent>
                              </NavItem>
                            ) : (
                              <li key={outlineNode.title}>
                                <strong>{outlineNode.title}</strong>
                                <ol>
                                  {outlineNode.chapterSlugs.map((chapterSlug) => (
                                      <NavItem key={chapterSlug} isActive={chapterSlug === slug}>
                                        <LinkComponent to={chapterSlug}>{groupedChapters[chapterSlug].node.frontmatter.title}</LinkComponent>
                                      </NavItem>
                                  ))}
                                </ol>
                              </li>
                            )
                          )}
                        </ol>
                      </NavContent>
                    </SideNav>
                    <BodyContent>
                        <div>
                            {title && <h1>{title}</h1>}
                            {description && (
                                <p>{description}</p>
                            )}
                        </div>
                        {html}
                        <Pagination>
                          <div>
                            {linkList.indexOf(slug) !== 0 && (
                                <Button variant="secondary" small onClick={() => navigate(linkList[linkList.indexOf(slug) - 1])}>« Previous Chapter</Button>
                            )}
                          </div>
                          <div>
                            {linkList.indexOf(slug) !== linkList.length - 1 && (
                                <Button variant="secondary" small onClick={() => navigate(linkList[linkList.indexOf(slug) + 1])}>Next Chapter »</Button>
                            )}
                          </div>
                        </Pagination>
                        <ChapterFooter />
                    </BodyContent>
                </ContentContainer>
            </Layout>
        </ChapterContext.Provider>
    )
}

export default Template

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
`

const ChapterFooter = styled(Footer)`
    &&& {
        padding: ${({ theme }) => theme.spacing.xl} 0;
        background: transparent;
        text-align: left;
    }
`;

// The following is placeholder style
// TODO(aarons): Rework these styles when there is an approved design
// and Varnish is integrated.

const ContentContainer = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  background: #fff;
`;

const SideNav = styled.nav`
  max-width: 300px;
  padding-right: 40px;
  font-size: 14px;

  h1 {
    margin-bottom: 20px;
  }

  ol {
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
  position: sticky;
  top: 115px;
  padding-top: 30px;
`;

const NavItem = styled(({ isActive, ...props }) =>
    <li {...props} />
)`
  position: relative;

  && {
    a {
      display: block;
      line-height: 16px;
      padding: 5px 0;
      color: ${({ isActive, theme }) => isActive ? theme.color.B6 : theme.color.N10};

      &:hover {
        color: #2a79e2;
        text-decoration: underline;
      }
    }
  }

  ${({ isActive }) => isActive ? `
    &:before {
      display: block;
      content: "▸";
      color: #2a79e2;
      font-size: 24px;
      position: absolute;
      left: -20px;
      top: 0;
    }
  ` : null}
`;

const BodyContent = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  border-left: 1px solid #ddd;
  max-width: 800px;
  padding: ${({ theme }) => `${theme.spacing.xl} 0 0 ${theme.spacing.xl}`};
  padding-right: 0;
`;

const Pagination = styled.div`
  padding-top: 20px;
  width: 100%;
  display: flex;
  
  div:last-child {
    margin-left: auto;
  }
`;
