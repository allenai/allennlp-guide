import React from 'react';
import { graphql } from 'gatsby';
import styled from 'styled-components';

import { outline } from '../outline';
import { getGroupedChapters } from '../utils';
import Layout from '../components/Layout';
import { Link } from '../components/Link';
import { Container } from '../components/Container';
import { Card, CardContent } from '../components/Card';
import { Footer } from '../components/Footer';
import { IconBox } from '../components/IconBox';
import { ArrowRightIcon } from '../components/inlineSVG';

// Home Page Export
export default ({ data }) => {
    const groupedChapters = getGroupedChapters(data.allMarkdownRemark);

    return (
        <Layout>
            <Banner>
                <h1>{data.site.siteMetadata.title}</h1>
            </Banner>
            <About>
                <SectionIntro>
                    <h2>About this course</h2>
                    <p>{data.site.siteMetadata.description}</p>
                </SectionIntro>
                <PartContainer>
                    <StandaloneChapterLink to={outline.overview.slug}>
                        <PartHeader
                            color={outline.overview.color}
                            icon={outline.overview.icon}
                            title={groupedChapters[outline.overview.slug].node.frontmatter.title}
                            description={groupedChapters[outline.overview.slug].node.frontmatter.description}
                            slug={outline.overview.slug}
                        />
                    </StandaloneChapterLink>
                </PartContainer>
            </About>
            <Parts>
                <SectionIntro>
                    <h2>Explore the course material</h2>
                </SectionIntro>
                {outline.parts.map((part) => part.chapterSlugs && (
                    <Part data={part} groupedChapters={groupedChapters} key={part.title} />
                ))}
            </Parts>
            <Credits>
                Written by the <Link to={data.site.siteMetadata.siteUrl}>AllenNLP</Link> team at the <Link to="https://allenai.org/">Allen Institute for AI</Link>.<br />
              This course was inspired by <Link to="https://github.com/ines/course-starter-python">Online Course Starter</Link>.
            </Credits>
            <Footer />
        </Layout>
    );
};

// GraphQL Query
export const pageQuery = graphql`
    {
        site {
            siteMetadata {
                siteUrl,
                title,
                description
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

// Hero Banner
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

// Intro Content

const About = styled(Container)`
    background: ${({ theme }) => theme.color.N1};
`;

const Parts = styled(Container)`
    background: ${({ theme }) => theme.color.N4};
`;

const SectionIntro = styled.div`
    margin-bottom: ${({ theme }) => theme.spacing.xl};

    h2 {
        ${({ theme }) => theme.typography.h4};
    }
    
    p {
        margin: 0;
        padding-top: ${({ theme }) => theme.spacing.xxs};
        padding-bottom: ${({ theme }) => theme.spacing.xs};
    }
`;

// Part UI

// Container for colored icon, title and description
const PartHeader = ({ color, icon, title, description, slug }) => (
    <PartHeaderContainer>
        <IconBox color={color} icon={icon} />
        <PartHeaderText>
            {title && (
                <PartTitle>{title}</PartTitle>
            )}
            {description && (
                <p>{description}</p>
            )}
            {slug && (
                <BeginLink>Begin Chapter <ArrowRightIcon /></BeginLink>
            )}
        </PartHeaderText>
    </PartHeaderContainer>
);

// Styled wrapper for `PartHeader` component
const PartHeaderContainer = styled.div`
    display: flex;
`;

// Container for part title and description
const PartHeaderText = styled.div`
    padding: ${({ theme }) => `${(theme.spacing.md.getRemValue() * 2) - theme.spacing.xxs.getRemValue()}rem ${(theme.spacing.md.getRemValue() * 2)}rem`};
    padding-bottom: ${({ theme }) => (theme.spacing.md.getRemValue() * 2) + theme.spacing.xxl.getRemValue() - theme.spacing.xxs.getRemValue()}rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    
    & > :last-child {
        margin-bottom: 0;
    }
`;

const PartTitle = styled.h3`
    ${({ theme }) => theme.typography.h4};
    padding-bottom: 0;
    color: ${({ theme }) => theme.color.B6};
`;

// Begin Chapter link for Overview
const BeginLink = styled.div`
    ${({ theme }) => theme.typography.bodySmall};
    display: flex;
    align-items: center;
    margin-top: auto;

    svg {
        margin-left: ${({ theme }) => theme.spacing.xs};
    }
`;

// Clickable wrapper for standalone chapter
const StandaloneChapterLink = styled(Link)`
    && {
        &,
        &:hover {
            text-decoration: none;

            ${PartTitle} {
                color: ${({ theme }) => theme.color.B6};
            }
            
            p {
                color: ${({ theme }) => theme.palette.text.primary};
            }
        }

        &:hover {
            ${BeginLink} {
                text-decoration: underline;
            }
        }

        ${PartHeaderText} {
            padding-bottom: ${({ theme }) => (theme.spacing.md.getRemValue() * 2) - theme.spacing.xxs.getRemValue()}rem;
        }
    }
`;

// Container for PartHeader and chapter list
const Part = ({ data, groupedChapters }) => {
    const { color, icon, title, description, chapterSlugs } = data;

    return (
        <PartContainer>
            <PartHeader color={color} icon={icon} title={title} description={description} />
            <div>
                <ChapterListTrigger />
                <ChapterList>
                    {chapterSlugs.map((chapterSlug) => (
                        <ChapterLink key={chapterSlug} to={chapterSlug}>
                            <h4>
                                {groupedChapters[chapterSlug].node.frontmatter.title}
                            </h4>
                            <p>
                                {groupedChapters[chapterSlug].node.frontmatter.description}
                            </p>
                        </ChapterLink>
                    ))}
                </ChapterList>
            </div>
        </PartContainer>
    );
};

// Styled wrapper for `Part` component
const PartContainer = styled(Card)`
    overflow: hidden;
`;

// Clickable bar that triggers expand/collapse of chapter list
// TODO(aarons): implement expand/collapse functionality
const ChapterListTrigger = styled.div`
    background: ${({ theme }) => theme.color.N2};
    outline: 1px solid black;
    height: ${({ theme }) => theme.spacing.xxl};
    margin-top: -${({ theme }) => theme.spacing.xxl};
`;

// Container for list of chapters for a given part
const ChapterList = styled(CardContent)`
    background: ${({ theme }) => theme.color.N2};
    padding-bottom: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
`;

// Clickable item in a chapter list
const ChapterLink = styled(Link)`
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
        margin-top: ${({ theme }) => theme.spacing.md};
    }
`;

// Footer Site Credits
const Credits = styled(Container)`
    background: ${({ theme }) => theme.color.N2};
    border-bottom: 1px solid ${({ theme }) => theme.color.N4};
    padding: ${({ theme }) => `${theme.spacing.xl} ${theme.spacing.xxl}`};
    text-align: center;
`;
