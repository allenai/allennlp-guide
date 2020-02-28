const path = require('path');
const { createFilePath } = require('gatsby-source-filesystem');

const chapterTemplate = path.resolve('src/templates/chapter.js');

function replacePath(pagePath) {
    return pagePath === `/` ? pagePath : pagePath.replace(/\/$/, ``);
}

async function onCreateNode({
    node,
    actions,
    getNode,
    loadNodeContent,
    createNodeId,
    createContentDigest
}) {
    const { createNodeField, createNode, createParentChildLink } = actions;
    if (node.internal.type === 'MarkdownRemark') {
        const slug = createFilePath({ node, getNode, basePath: 'chapters', trailingSlash: false });
        // Use regex to remove any subdirectory strings preceeding the target md file in the path
        // eslint-disable-next-line no-useless-escape
        const formattedSlug = `/${slug.match(/([^\/]+$)/gm)}`;
        createNodeField({ name: 'slug', node, value: formattedSlug });
    } else if (node.extension === 'py') {
        // Load the contents of the Python file and make it available via GraphQL
        // https://www.gatsbyjs.org/docs/creating-a-transformer-plugin/
        const content = await loadNodeContent(node);
        const contentDigest = createContentDigest(content);
        const id = createNodeId(`${node.id}-code`);
        const internal = { type: 'Code', contentDigest };
        const codeNode = {
            id,
            parent: node.id,
            children: [],
            code: content,
            name: node.name,
            dir: node.dir,
            internal
        };
        createNode(codeNode);
        createParentChildLink({ parent: node, child: codeNode });
    }
}

exports.onCreateNode = onCreateNode;

exports.createPages = ({ actions, graphql }) => {
    const { createPage } = actions;
    return graphql(`
        {
            allMarkdownRemark {
                edges {
                    node {
                        frontmatter {
                            title
                            type
                        }
                        fields {
                            slug
                        }
                    }
                }
            }
        }
    `).then(result => {
        if (result.errors) {
            return Promise.reject(result.errors);
        }
        const posts = result.data.allMarkdownRemark.edges.filter(
            ({ node }) => node.frontmatter.type === 'chapter'
        );
        posts.forEach(({ node }) => {
            createPage({
                path: replacePath(node.fields.slug),
                component: chapterTemplate,
                context: { slug: node.fields.slug }
            });
        });
    });
};
