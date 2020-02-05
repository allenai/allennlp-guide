import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { outline } from '../outline';
import { getGroupedChapters } from '../utils';
import Layout from '../templates/Layout';
import { LinkComponent } from '../components/LinkComponent';
import { Container } from '../components/Container';
import { Card, CardContent } from '../components/Card';
import { Footer } from '../components/Footer';

export default ({ data }) => {
    const groupedChapters = getGroupedChapters(data.allMarkdownRemark);

    return (
        <Layout>
            <Banner>
                <h1>Diving Into Natural Language Processing With AllenNLP</h1>
            </Banner>
            <Parts>
                {outline.map((outlineNode) => !outlineNode.chapterSlugs ? (
                    <StandaloneChapter key={outlineNode.slug}>
                      <ChapterLink to={outlineNode.slug}>
                          <h4>
                              {groupedChapters[outlineNode.slug].node.frontmatter.title}
                          </h4>
                          <p>
                              {groupedChapters[outlineNode.slug].node.frontmatter.description}
                          </p>
                      </ChapterLink>
                    </StandaloneChapter>
                  ) : (
                    <PartContainer key={outlineNode.title}>
                        <PartContent>
                            <PartHeading>{outlineNode.title}</PartHeading>
                            {outlineNode.chapterSlugs.map((chapterSlug) => (
                                <ChapterLink key={chapterSlug} to={chapterSlug}>
                                    <h4>
                                        {groupedChapters[chapterSlug].node.frontmatter.title}
                                    </h4>
                                    <p>
                                        {groupedChapters[chapterSlug].node.frontmatter.description}
                                    </p>
                                </ChapterLink>
                            ))}
                        </PartContent>
                    </PartContainer>
                  )
                )}
            </Parts>
            <Credits>
                Written by the <LinkComponent to={data.site.siteMetadata.siteUrl}>AllenNLP</LinkComponent> team at the <LinkComponent to="https://allenai.org/">Allen Institute for AI</LinkComponent>.<br />
                This course was built with <LinkComponent to="https://github.com/ines/course-starter-python">Online Course Starter</LinkComponent>.
            </Credits>
            <Footer />
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
        margin: 0 auto;
        max-width: 720px;
    }
`;

const Parts = styled(Container)`
    background: ${({ theme }) => theme.color.N4};
`;

const PartHeading = styled.h3`
    ${({ theme }) => theme.typography.h4};
    padding-bottom: ${({ theme }) => theme.spacing.md};
    color: ${({ theme }) => theme.color.B6};
`;

const ChapterLink = styled(LinkComponent)`
    && {
        display: block;
        background: ${({ theme }) => theme.color.N1};
        border: 1px solid ${({ theme }) => theme.color.N6};
        border-radius: ${({ theme }) => theme.spacing.xxs};
        padding: ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md.getRemValue() * 2}rem`};

        h4 {
          ${({ theme }) => theme.typography.bodyBig}
          margin: 0;
        }

        p {
          margin: 0;
          color: ${({ theme }) => theme.color.N10};
        }

        :hover {
            text-decoration: none;
            border-color: ${({ theme }) => theme.color.B6};
        }
    }

    && + && {
        margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    }
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

const PartContainer = styled(Card)``;

const PartContent = styled(CardContent)`
    padding-bottom: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
`;

const StandaloneChapter = styled.div`
    max-width: 800px;
    margin: auto;
    
    & + ${PartContainer} {
      margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    }
`;
