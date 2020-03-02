// This component is used for all internal and external anchor links.
// External links are determined by the existence of `http` in the URL.

import React from 'react';
import { Link as GatsbyLink } from 'gatsby';

export const Link = ({ children, to, href, onClick, className, ...other }) => {
    const dest = to || href;
    const external = /(http(s?)):\/\//gi.test(dest);

    if (!external) {
        if ((dest && /^#/.test(dest)) || onClick) {
            return (
                <a href={dest} onClick={onClick} className={className}>
                    {children}
                </a>
            );
        }

        return (
            <GatsbyLink to={dest} className={className} {...other}>
                {children}
            </GatsbyLink>
        );
    }

    return (
        <a
            href={dest}
            className={className}
            target="_blank"
            rel="noopener nofollow noreferrer"
            {...other}>
            {children}
        </a>
    );
};
