// This SVG asset displays a dual-purpose expand/collapse icon that can transition between the two
// Meant for expandable part containers on home page and expandable chapter cards on chapter pages

import React from 'react';
import styled from 'styled-components';

export const ExpandCollapseIcon = ({ className }) => (
    <SVGContainer className={className}>
        <svg
            width={18}
            height={11}
            viewBox="0 0 18 11"
            xmlns="http://www.w3.org/2000/svg"
        >
          <rect
              width="1.5"
              height={12}
              transform="matrix(-0.707107 -0.707107 -0.707107 0.707107 17.4854 2)"
          />
          <rect
              width="1.5"
              height={12}
              transform="matrix(-0.707107 0.707107 0.707107 0.707107 1.57544 0.939209)"
          />
        </svg>
    </SVGContainer>
);

const SVGContainer = styled.div`
    width: ${({ theme }) => theme.spacing.lg};
    height: ${({ theme }) => theme.spacing.lg};
    display: flex;
    align-items: center;
    justify-content: center;

    rect {
        fill: ${({ theme }) => theme.color.N8};
        transition: transform 0.2s ease, fill 0.2s ease;
    }

    &:hover {
        rect {
            fill: ${({ theme }) => theme.color.B6};
        }
    }
`;
