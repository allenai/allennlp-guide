import React, { useState, useEffect } from 'react'
import { graphql, navigate } from 'gatsby'
import useLocalStorage from '@illinois/react-use-local-storage'

import { renderAst } from '../markdown'
import { ChapterContext } from '../context'
import Layout from '../components/layout'
import { Button } from '../components/button'

import classes from '../styles/chapter.module.sass'

const Template = ({ data, location }) => {
    const { markdownRemark, site } = data
    const { courseId } = site.siteMetadata
    const { frontmatter, htmlAst } = markdownRemark
    const { title, description, prev, next, id } = frontmatter
    const [activeExc, setActiveExc] = useState(null)
    const [completed, setCompleted] = useLocalStorage(`${courseId}-completed-${id}`, [])
    const html = renderAst(htmlAst)
    const buttons = [
        { slug: prev, text: '« Previous Chapter' },
        { slug: next, text: 'Next Chapter »' },
    ]
    const handleSetActiveExc = id => {
        window.location.hash = `${id}`
        setActiveExc(id)
    }
    useEffect(() => {
        if (location.hash) {
            setActiveExc(parseInt(location.hash.split('#')[1]))
        }
    }, [location.hash])

    return (
        <ChapterContext.Provider
            value={{ activeExc, setActiveExc: handleSetActiveExc, completed, setCompleted }}
        >
            <Layout title={title} description={description}>
                {html}

                <section className={classes.pagination}>
                    {buttons.map(({ slug, text }) => (
                        <div key={slug}>
                            {slug && (
                                <Button variant="secondary" small onClick={() => navigate(slug)}>
                                    {text}
                                </Button>
                            )}
                        </div>
                    ))}
                </section>
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
            }
        }
        markdownRemark(fields: { slug: { eq: $slug } }) {
            htmlAst
            frontmatter {
                id
                title
                description
                next
                prev
            }
        }
    }
`
