import React from 'react';
import Layout from '../components/layout';
import Logo from '../../static/logo.svg';
import classes from '../styles/index.module.sass';

export default ({ data }) => {
    const siteMetadata = data.site.siteMetadata;
    return (
        <Layout>
            <Logo className={classes.logo} aria-label={siteMetadata.title} />
            <h1>Error 404</h1>
            <p>Page not found.</p>
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
    }
`;
