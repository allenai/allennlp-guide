import React from 'react';
import styled from 'styled-components';

/**
 * The <MenuIcon /> component is a dual-purpose "hamburger" menu and
 * "X" close icon that can transition between the two configurations via
 * the value of its mobileNavIsActive prop
 */
export const MenuIcon = ({ mobileNavIsActive = false }) => {
    return (
        <Svg mobileNavIsActive={mobileNavIsActive} aria-hidden="true">
            <TopRect />
            <MiddleRect1 />
            <MiddleRect2 />
            <BottomRect />
        </Svg>
    );
};

// General rect element of hamburger/close icon
const Rect = styled.rect.attrs({
    x: '1',
    width: '22',
    height: '2'
})`
    transform-origin: 50% 50%;
    transition: opacity 0.3s ease, transform 0.2s ease;
`;

// Top rectangle of hamburger
// Fades out when transitioned into Close icon
const TopRect = styled(Rect).attrs({
    y: '1'
})``;

// Middle rectangle (1 of 2) of hamburger
// Forms half of the "X" when transitioned into Close icon
const MiddleRect1 = styled(Rect).attrs({
    y: '9'
})``;

// Middle rectangle (2 of 2) of hamburger
// Identical to MiddleRect1 except it rotates
// in the opposite direction in order to form
// the other half of the "X" when transitioned into Close icon
const MiddleRect2 = styled(MiddleRect1)``;

// Bottom rectangle of hamburger
// Fades out when transitioned into Close icon
const BottomRect = styled(Rect).attrs({
    y: '17'
})``;

// Default configuration of Rects into hamburger icon
const Svg = styled.svg.attrs({
    viewBox: '0 0 24 20'
})`
    fill: ${({ theme }) => theme.color.B6};
    width: ${({ theme }) => theme.spacing.lg};
    height: ${({ theme }) => theme.spacing.lg};

    // The following styles are for configuring Rects into "X" (close) icon
    // which is used when mobile nav is visible. The conditional logic inside
    // the styled block was necessary for transition to work. Setting styled
    // component conditionally outside the return statement was preventing
    // transition when props changed, presumably because entire svg html
    // structure was being replaced.
    ${({ mobileNavIsActive }) =>
        mobileNavIsActive
            ? `
        ${Rect} {
            transition: opacity 0.1s ease, transform 0.2s ease;
        }

        ${TopRect},
        ${BottomRect} {
          opacity: 0;
        }

        ${MiddleRect1} {
          transform: rotate(45deg);
        }

        ${MiddleRect2} {
          transform: rotate(-45deg);
        }
    `
            : ''}
`;
