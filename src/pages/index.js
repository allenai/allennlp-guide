import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { parts } from '../parts';
import Layout from '../components/layout';
import { Link } from '../components/link';
import Logo from '../../static/logo.svg';

import classes from '../styles/index.module.sass';

export default ({ data }) => {
    const siteMetadata = data.site.siteMetadata;
    // Convert array of chapter objects to new array of objects grouped by part
    const groupedChapters = Object.values(
        data.allMarkdownRemark.edges.reduce((accumulator, chapter) => {
          // Set partId to first digit of chapter id
          const partId = chapter.node.frontmatter.id.toString().charAt(0);
          if (!accumulator[partId]) {
            accumulator[partId] = {
                partId,
                chapters: []
              };
          }
          accumulator[partId].chapters.push(chapter);
          return accumulator;
        }, {})
    );

    return (
        <Layout isHome>
            <Logo className={classes.logo} aria-label={siteMetadata.title} />
            {groupedChapters.map((part) => (
                <PartContainer key={part.partId}>
                    <PartHeading>{part.partId in parts ? parts[part.partId] : `Part ${part.partId}`}</PartHeading>
                    {part.chapters.map((chapter) => {
                        const { fields, frontmatter } = chapter.node;
                        return (
                            <section key={fields.slug} className={classes.chapter}>
                                <Link hidden to={fields.slug}>
                                    <h2 className={classes.chapterTitle}>
                                        {frontmatter.title}
                                    </h2>
                                    <p className={classes.chapterDesc}>
                                        {frontmatter.description}
                                    </p>
                                </Link>
                            </section>
                        );
                    })}
                </PartContainer>
            ))}
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
        allMarkdownRemark(
            sort: { fields: [frontmatter___id], order: ASC }
            filter: { frontmatter: { type: { eq: "chapter" } } }
        ) {
            edges {
                node {
                    fields {
                        slug
                    }
                    frontmatter {
                        title
                        description
                        id
                    }
                }
            }
        }
    }
`;

// The following is placeholder style
// TODO(aarons): Rework these styles when there is an approved design
// and Varnish is integrated.

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
