import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { outline } from '../outline';
import Layout from '../components/layout';
import { Link } from '../components/link';
import Logo from '../../static/logo.svg';

import classes from '../styles/index.module.sass';

export default ({ data }) => {
    const siteMetadata = data.site.siteMetadata;
    // Create a lookup table of chapters by slug value
    const chapters = data.allMarkdownRemark.edges.reduce((acc, obj) => {
      const key = obj.node.fields.slug;
      if (!acc[key]) {
        acc[key] = {};
      }
      acc[key] = obj;
      return acc;
    }, {});

    return (
        <Layout isHome>
            <Logo className={classes.logo} aria-label={siteMetadata.title} />
            {outline.map((node) => node.type === 'chapter' ? (
                <StandaloneChapter key={node.slug}>
                  <section className={classes.chapter}>
                      <Link hidden to={node.slug}>
                          <h2 className={classes.chapterTitle}>
                              {chapters[node.slug].node.frontmatter.title}
                          </h2>
                          <p className={classes.chapterDesc}>
                              {chapters[node.slug].node.frontmatter.description}
                          </p>
                      </Link>
                  </section>
                </StandaloneChapter>
              ) : (
                <PartContainer key={node.title}>
                  <PartHeading>{node.title}</PartHeading>
                  {node.chapterSlugs.map((chapterSlug) => (
                    <section key={chapterSlug} className={classes.chapter}>
                        <Link hidden to={chapterSlug}>
                            <h2 className={classes.chapterTitle}>
                                {chapters[chapterSlug].node.frontmatter.title}
                            </h2>
                            <p className={classes.chapterDesc}>
                                {chapters[chapterSlug].node.frontmatter.description}
                            </p>
                        </Link>
                    </section>
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

const StandaloneChapter = styled.div``;

const PartContainer = styled.div`
    // outline: 1px solid black;
    box-shadow: 0 5px 30px rgba(10,20,30,0.125);
    max-width: 800px;
    padding: 15px 30px 10px 30px;
    margin: auto;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.05);

    & + & {
        margin-top: 30px;
    }
`;

const PartHeading = styled.h2`
    padding-bottom: 15px;
    color: #2a79e2;
`;
