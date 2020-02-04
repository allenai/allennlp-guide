import React from 'react';
import PropTypes from 'prop-types';
import { Link as GatsbyLink } from 'gatsby';

export const LinkComponent = ({ children, to, href, onClick, className, ...other }) => {
    const dest = to || href;
    const external = /(http(s?)):\/\//gi.test(dest);

    if (!external) {
        if ((dest && /^#/.test(dest)) || onClick) {
            return (
                <a href={dest} onClick={onClick} className={className}>
                    {children}
                </a>
            )
        }
        return (
            <GatsbyLink to={dest} className={className} {...other}>
                {children}
            </GatsbyLink>
        )
    }
    return (
        <a
            href={dest}
            className={className}
            target="_blank"
            rel="noopener nofollow noreferrer"
            {...other}
        >
            {children}
        </a>
    );
};

LinkComponent.propTypes = {
    children: PropTypes.node.isRequired,
    to: PropTypes.string,
    href: PropTypes.string,
    onClick: PropTypes.func,
    className: PropTypes.string,
};
