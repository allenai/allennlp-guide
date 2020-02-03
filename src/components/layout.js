import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import { createGlobalStyle } from 'styled-components';
import { ThemeProvider } from '@allenai/varnish/theme';

import Head from './Head';
import { Link } from './link';
import { H3 } from './typography';

import '../styles/index.sass';
import classes from '../styles/layout.module.sass';

const Layout = ({ isHome, title, description, children }) => {
    return (
        <StaticQuery
            query={graphql`
                {
                    site {
                        siteMetadata {
                            title
                            description
                            bio
                            showProfileImage
                            footerLinks {
                                text
                                url
                            }
                        }
                    }
                }
            `}
            render={data => {
                const meta = data.site.siteMetadata;
                return (
                    <ThemeProvider>
                        <Head title={title} description={description} />
                        <GlobalStyle />
                        <main className={classes.root}>
                            <div className={classes.content}>
                                {children}
                            </div>

                            <footer className={classes.footer}>
                                <div className={classes.footerContent}>
                                    <section className={classes.footerSection}>
                                        <H3>About this course</H3>
                                        <p>{meta.description}</p>
                                    </section>

                                    <section className={classes.footerSection}>
                                        <H3>About me</H3>
                                        {meta.showProfileImage && (
                                            <img
                                                src="/profile.jpg"
                                                alt=""
                                                className={classes.profile}
                                            />
                                        )}
                                        <p>{meta.bio}</p>
                                    </section>

                                    {meta.footerLinks && (
                                        <ul className={classes.footerLinks}>
                                            {meta.footerLinks.map(({ text, url }, i) => (
                                                <li key={i} className={classes.footerLink}>
                                                    <Link variant="secondary" to={url}>
                                                        {text}
                                                    </Link>
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            </footer>
                        </main>
                    </ThemeProvider>
                );
            }}
        />
    );
};

export default Layout;

// Resetting root layout
const GlobalStyle = createGlobalStyle`
  html,
  body {
    width: 100%;
    height: 100%;
  }

  #___gatsby,
  #___gatsby > div,
  main {
    height: 100%;
  }
  
  main {
    display: flex;
    flex-direction: column;
  }
  
  footer {
    margin-top: auto !important;
  }
`;
