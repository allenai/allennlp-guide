// This SVG asset displays a mobile disclosure indicator

import React from 'react';
import styled from 'styled-components';

export const Disclosure = ({ className }) => (
    <svg
        width={11}
        height={18}
        className={className}
        viewBox="0 0 11 18"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true">
        <rect
            width="1.5"
            height={12}
            transform="matrix(0.707107 0.707107 0.707107 -0.707107 0.712158 16.1978)"
        />
        <rect
            width="1.5"
            height={12}
            transform="matrix(-0.707107 0.707107 0.707107 0.707107 1.77295 0.287842)"
        />
    </svg>
);
