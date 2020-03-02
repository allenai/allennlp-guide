const meta = require('./meta.json');
const autoprefixer = require('autoprefixer');

module.exports = {
    siteMetadata: meta,
    plugins: [
        `gatsby-plugin-react-helmet`,
        {
            resolve: `gatsby-source-filesystem`,
            options: {
                name: `chapters`,
                path: `${__dirname}/chapters`
            }
        },
        {
            resolve: `gatsby-source-filesystem`,
            options: {
                name: `exercises`,
                path: `${__dirname}/exercises`
            }
        },
        {
            resolve: 'gatsby-plugin-react-svg',
            options: {
                rule: {
                    include: /static/
                }
            }
        },
        {
            resolve: `gatsby-plugin-styled-components`
        },
        {
            resolve: 'gatsby-plugin-antd'
        },
        {
            resolve: `gatsby-transformer-remark`,
            options: {
                plugins: [
                    `gatsby-remark-copy-linked-files`,
                    {
                        resolve: `gatsby-remark-prismjs`,
                        options: {
                            noInlineHighlight: true
                        }
                    },
                    {
                        resolve: `gatsby-remark-smartypants`,
                        options: {
                            dashes: 'oldschool'
                        }
                    },
                    {
                        resolve: `gatsby-remark-images`,
                        options: {
                            maxWidth: 790,
                            linkImagesToOriginal: true,
                            sizeByPixelDensity: false,
                            showCaptions: true,
                            quality: 80,
                            withWebp: { quality: 80 }
                        }
                    },
                    `gatsby-remark-unwrap-images`
                ]
            }
        },
        `gatsby-transformer-sharp`,
        `gatsby-plugin-sharp`,
        `gatsby-plugin-sitemap`,
        {
            resolve: `gatsby-plugin-manifest`,
            options: {
                name: meta.title,
                short_name: meta.title,
                start_url: `/`,
                background_color: meta.theme,
                theme_color: meta.theme,
                display: `minimal-ui`,
                legacy: false
            }
        },
        `gatsby-plugin-offline`
    ]
};
