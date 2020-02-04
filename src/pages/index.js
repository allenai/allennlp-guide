import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { outline } from '../outline';
import { getGroupedChapters } from '../utils';
import Layout from '../templates/Layout';
import { LinkComponent } from '../components/LinkComponent';
import { Container } from '../components/Container';
import { Card } from '../components/Card';

export default ({ data }) => {
    const groupedChapters = getGroupedChapters(data.allMarkdownRemark);

    return (
        <Layout isHome>
            <Banner>
                <h1>Diving Into Natural Language Processing With AllenNLP</h1>
            </Banner>
            <Parts>
                {outline.map((outlineNode) => !outlineNode.chapterSlugs ? (
                    <StandaloneChapter key={outlineNode.slug}>
                      <ChapterLink to={outlineNode.slug}>
                          <Chapter>
                              <h4>
                                  {groupedChapters[outlineNode.slug].node.frontmatter.title}
                              </h4>
                              <p>
                                  {groupedChapters[outlineNode.slug].node.frontmatter.description}
                              </p>
                          </Chapter>
                      </ChapterLink>
                    </StandaloneChapter>
                  ) : (
                    <PartContainer key={outlineNode.title}>
                      <PartHeading>{outlineNode.title}</PartHeading>
                      {outlineNode.chapterSlugs.map((chapterSlug) => (
                          <ChapterLink key={chapterSlug} to={chapterSlug}>
                              <Chapter>
                                <h4>
                                    {groupedChapters[chapterSlug].node.frontmatter.title}
                                </h4>
                                <p>
                                    {groupedChapters[chapterSlug].node.frontmatter.description}
                                </p>
                              </Chapter>
                          </ChapterLink>
                      ))}
                    </PartContainer>
                  )
                )}
            </Parts>
            <Credits>
                Written by the <LinkComponent to={data.site.siteMetadata.siteUrl}>AllenNLP</LinkComponent> team at the <LinkComponent to="https://allenai.org/">Allen Institute for AI</LinkComponent>.<br />
                This course was built with <LinkComponent to="https://github.com/ines/course-starter-python">Online Course Starter</LinkComponent>.
            </Credits>
        </Layout>
    );
};

export const pageQuery = graphql`
    {
        site {
            siteMetadata {
                siteUrl
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

const Banner = styled(Container)`
    background: url('/ui/bannerDotsLeft.svg') left center / auto 100% no-repeat,
                url('/ui/bannerDotsRight.svg') right center / auto 100% no-repeat,
                linear-gradient(168.81deg, #1B4596 27.29%, #1052D2 82.34%);

    h1 {
        font-size: ${({ theme }) => theme.spacing.xl};
        line-height: ${({ theme }) => theme.spacing.xxl};
        font-weight: ${({ theme }) => theme.typography.fontWeightBold};
        color: ${({ theme }) => theme.color.N1};
        text-align: center;
        margin: 0;
    }
`;

const Parts = styled(Container)`
    background: ${({ theme }) => theme.color.N4};
`;

const Credits = styled(Container)`
    background: ${({ theme }) => theme.color.N2};
    border-bottom: 1px solid ${({ theme }) => theme.color.N4};
    padding: ${({ theme }) => `${theme.spacing.xl} ${theme.spacing.xxl}`};
    text-align: center;
`;

// The following is placeholder style
// TODO(aarons): Rework these styles when there is an approved design
// and Varnish is integrated.

const StandaloneChapter = styled.div`
    max-width: 800px;
    margin: auto;
`;

const PartContainer = styled(Card)`
    padding: 15px 32px 32px 32px;
`;

const PartHeading = styled.h3`
    ${({ theme }) => theme.typography.h4};
    padding: 10px 0;
    color: ${({ theme }) => theme.color.B6};
`;

const Chapter = styled.div`
    padding: 32px;
    border: 1px solid ${({ theme }) => theme.color.N4};
    border-radius: ${({ theme }) => theme.spacing.xxs};

    h4 {
      ${({ theme }) => theme.typography.bodyBig}
      margin: 0;
    }

    p {
      margin: 0;
      color: ${({ theme }) => theme.color.N10};
    }
`;

const ChapterLink = styled(LinkComponent)`
    && {
        display: block;

        :hover {
            text-decoration: none;
        }
    }

    && + && {
        margin-top: 32px;
    }
`;
