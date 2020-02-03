import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { outline } from '../outline';
import { getGroupedChapters } from '../utils';
import Layout from '../components/layout';
import { Link } from '../components/link';
import Logo from '../../static/logo.svg';

import classes from '../styles/index.module.sass';

export default ({ data }) => {
    const siteMetadata = data.site.siteMetadata;
    const groupedChapters = getGroupedChapters(data.allMarkdownRemark);

    return (
        <Layout isHome>
            <Logo className={classes.logo} aria-label={siteMetadata.title} />
            {outline.map((outlineNode) => !outlineNode.chapterSlugs ? (
                <StandaloneChapter key={outlineNode.slug}>
                  <InteractiveLink hidden to={outlineNode.slug}>
                      <section className={classes.chapter}>
                          <h2 className={classes.chapterTitle}>
                              {groupedChapters[outlineNode.slug].node.frontmatter.title}
                          </h2>
                          <p className={classes.chapterDesc}>
                              {groupedChapters[outlineNode.slug].node.frontmatter.description}
                          </p>
                      </section>
                  </InteractiveLink>
                </StandaloneChapter>
              ) : (
                <PartContainer key={outlineNode.title}>
                  <PartHeading>{outlineNode.title}</PartHeading>
                  {outlineNode.chapterSlugs.map((chapterSlug) => (
                      <InteractiveLink key={chapterSlug} hidden to={chapterSlug}>
                          <section className={classes.chapter}>
                            <h2 className={classes.chapterTitle}>
                                {groupedChapters[chapterSlug].node.frontmatter.title}
                            </h2>
                            <p className={classes.chapterDesc}>
                                {groupedChapters[chapterSlug].node.frontmatter.description}
                            </p>
                          </section>
                      </InteractiveLink>
                  ))}
                </PartContainer>
              )
            )}
        </Layout>
    );
};

export const pageQuery = graphql`
    {
        site {
            siteMetadata {
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
    }
`;

// The following is placeholder style
// TODO(aarons): Rework these styles when there is an approved design
// and Varnish is integrated.

const StandaloneChapter = styled.div`
    max-width: 800px;
    margin: auto;
`;

const PartContainer = styled.div`
    // outline: 1px solid black;
    box-shadow: 0 5px 30px rgba(10,20,30,0.125);
    max-width: 800px;
    padding: 15px 30px 0 30px;
    margin: auto;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.05);

    & + & {
        margin-top: 30px;
    }
`;

const PartHeading = styled.h2`
    ${({ theme }) => theme.typography.h4};
    padding: 10px 0;
    color: ${({ theme }) => theme.color.B6};
`;

const InteractiveLink = styled(Link)`
    &&:hover {
        text-decoration: none;

        section {
          border-color: #2a79e2;
        }
    }
`;
