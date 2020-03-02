/**
 * This component renders part icons in colored gradient boxes.
 *
 * Supported `color` values:
 *     'aqua',
 *     'blue',
 *     'green',
 *     'orange',
 *     'purpe',
 *     'red',
 *     'teal',
 *     'default' (generic neutral color)
 */

import React from 'react';
import styled from 'styled-components';
import { getIcon } from '../utils';

export const IconBox = ({ className, icon, color }) => (
    <StyledIconBox className={className} color={color}>
        <OuterWrapper>
            <InnerWrapper>{getIcon(icon)}</InnerWrapper>
        </OuterWrapper>
    </StyledIconBox>
);

// Colored square that contains part icon
const StyledIconBox = styled(({ color, ...props }) => <div {...props} />)`
    background: linear-gradient(
        151.76deg,
        ${({ color }) => {
                if (color === 'aqua') {
                    return '#00C1E8 17.77%, #0278A7';
                } else if (color === 'blue') {
                    return '#6192fb 17.77%, #295ece';
                } else if (color === 'green') {
                    return '#32c694 17.77%, #099d6b';
                } else if (color === 'orange') {
                    return '#FFC72E 17.77%, #FF9100';
                } else if (color === 'purple') {
                    return '#D864C8 17.77%, #A44397';
                } else if (color === 'red') {
                    return '#fb6769 17.77%, #d23e40';
                } else if (color === 'teal') {
                    return '#1EC2CC 17.77%, #0191A7';
                } else {
                    // 'default'
                    return '#a3b0be 17.77%, #79899c';
                }
            }}
            95.72%
    );
`;

const OuterWrapper = styled.div`
    position: relative;
    width: 100%;
    padding-bottom: 100%;
    height: 0;
`;

const InnerWrapper = styled.div`
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;

    svg {
        width: 56.25%;
        height: 56.25%;
        fill: #fff;
    }
`;
