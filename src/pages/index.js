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
    // Convert array of chapter objects to new array of objects grouped by part
    // const groupedChapters = Object.values(
    //     data.allMarkdownRemark.edges.reduce((accumulator, chapter) => {
    //       // Set partId to first digit of chapter id
    //       const partId = chapter.node.frontmatter.id.toString().charAt(0);
    //       if (!accumulator[partId]) {
    //         accumulator[partId] = {
    //             partId,
    //             chapters: []
    //           };
    //       }
    //       accumulator[partId].chapters.push(chapter);
    //       return accumulator;
    //     }, {})
    // );

            // {groupedChapters.map((part) => (
            //     <PartContainer key={part.partId}>
            //         <PartHeading>{part.partId in parts ? parts[part.partId] : `Part ${part.partId}`}</PartHeading>
            //         {part.chapters.map((chapter) => {
            //             const { fields, frontmatter } = chapter.node;
            //             return (
            //                 <section key={fields.slug} className={classes.chapter}>
            //                     <Link hidden to={fields.slug}>
            //                         <h2 className={classes.chapterTitle}>
            //                             {frontmatter.title}
            //                         </h2>
            //                         <p className={classes.chapterDesc}>
            //                             {frontmatter.description}
            //                         </p>
            //                     </Link>
            //                 </section>
            //             );
            //         })}
            //     </PartContainer>
            // ))}



    const fakeData = [
      {
        "node": {
          "fields": {
            "slug": "/introduction"
          },
          "frontmatter": {
            "title": "Introduction",
            "description": "This chapter will give an introduction to the task we'll be using throughout Part 2 (text classification) and how to use AllenNLP to solve it"
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/overview"
          },
          "frontmatter": {
            "title": "Course overview",
            "description": "This chapter will give an overview of AllenNLP, and will outline the main chapters of this course"
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/next-steps"
          },
          "frontmatter": {
            "title": "Next steps",
            "description": "Now that you have a working model, here are some things you can try with AllenNLP!"
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/reading-textual-data"
          },
          "frontmatter": {
            "title": "Reading textual data",
            "description": "This chapter provides a deep dive into AllenNLP abstractions that are essential for reading textual data, including fields and instances, dataset readers, vocabulary, and how batching is handled in AllenNLP"
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/representing-text-as-features"
          },
          "frontmatter": {
            "title": "Chapter 3: Representing text as features: Tokenizers, TextFields, and TextFieldEmbedders",
            "description": "A deep dive into AllenNLP's core abstraction: how exactly we represent textual inputs, both on the data side and the model side."
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/semantic-parsing"
          },
          "frontmatter": {
            "title": "Semantic Parsing",
            "description": "We will look at how you can implement executable semantic parsers in AllenNLP."
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/training-and-prediction"
          },
          "frontmatter": {
            "title": "Training and prediction",
            "description": "This chapter will outline how to train your model and run prediction on new data"
          }
        }
      },
      {
        "node": {
          "fields": {
            "slug": "/your-first-model"
          },
          "frontmatter": {
            "title": "Your first model",
            "description": "In this chapter you are going to build your first text classification model using AllenNLP."
          }
        }
      }
    ];


    // Convert array of chapter objects to new array of objects grouped by part
    // const groupedChapters = Object.values(
    //     data.allMarkdownRemark.edges.reduce((accumulator, chapter) => {
    //       // Set partId to first digit of chapter id
    //       const partId = chapter.node.frontmatter.id.toString().charAt(0);
    //       if (!accumulator[partId]) {
    //         accumulator[partId] = {
    //             partId,
    //             chapters: []
    //           };
    //       }
    //       accumulator[partId].chapters.push(chapter);
    //       return accumulator;
    //     }, {})
    // );


    const table = fakeData.reduce((acc, obj) => {
      const key = obj.node.fields.slug;
      delete obj.node.fields;
      if (!acc[key]) {
        acc[key] = {
          title: obj.node.frontmatter.title,
          description: obj.node.frontmatter.description
        };
      }
      acc[key] = obj;
      return acc;
    }, {});

    // const chapters = fakeData.map(({ node }) => ({
    //     slug: node.fields.slug,
    //     title: node.frontmatter.title,
    //     description: node.frontmatter.description,
    // }));

    return (
        <Layout isHome>
            <pre><code>{JSON.stringify(table, null, 2)}</code></pre>
            <Logo className={classes.logo} aria-label={siteMetadata.title} />
            {outline.map((item) => item.type === 'chapter' ? (
                <StandaloneChapter key={item.slug}>
                  <section className={classes.chapter}>
                      <Link hidden to={item.slug}>
                          <h2 className={classes.chapterTitle}>
                              {/*frontmatter.title*/}
                          </h2>
                          <p className={classes.chapterDesc}>
                              {/*frontmatter.description*/}
                          </p>
                      </Link>
                  </section>
                </StandaloneChapter>
              ) : (
                <PartContainer key={item.title}>
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
