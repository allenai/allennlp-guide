// This component is an SVG boilerplate wrapper for part icons

import React from 'react';

export const PartIcon = ({ className, size, children }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 108 108"
        className={className}
        xmlns="http://www.w3.org/2000/svg"
        children={children}
    />
);
