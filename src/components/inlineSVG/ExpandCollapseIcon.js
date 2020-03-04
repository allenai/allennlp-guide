// This asset displays a dual-purpose expand/collapse icon that can transition between the two
// Meant for expandable part containers on home page and expandable chapter cards on chapter pages

import React from 'react';
import styled, { css } from 'styled-components';
import { above } from '@allenai/varnish/theme/breakpoints';
import { toRem } from '../../utils';

export const ExpandCollapseIcon = ({ className, isExpanded = false }) => (
    <PaddedContainer
        className={className}
        isExpanded={isExpanded}
        title={`${isExpanded ? 'Collapse' : 'Expand'} panel`}>
        <IconContainer>
            <LeftOuter>
                <LeftInner className="rect" />
            </LeftOuter>
            <RightOuter>
                <RightInner className="rect" />
            </RightOuter>
        </IconContainer>
    </PaddedContainer>
);

const IconContainer = styled.span`
    display: block;
    width: 18px;
    height: 11px;
    position: relative;
`;

const outerStyles = css`
    transition: transform 0.2s ease;
    transform-origin: 50% 50%;
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    top: 50%;
`;

const LeftOuter = styled.span`
    ${outerStyles}
    left: -0.1rem;
`;

const RightOuter = styled.span`
    ${outerStyles}
    right: 0;
`;

const strokeWeight = 1.5;

const innerStyles = css`
    display: block;
    transform-origin: 50% 50%;
    width: ${({ theme }) => theme.spacing.sm};
    height: ${toRem(strokeWeight)};
    transition: background-color 0.2s ease;
`;

const LeftInner = styled.span`
    ${innerStyles}
    transform: rotate(45deg) translate(${toRem(strokeWeight / 4)}, 0);
`;

const RightInner = styled.span`
    ${innerStyles}
    transform: rotate(-45deg) translate(-${toRem(strokeWeight / 4)}, 0);
`;

const PaddedContainer = styled(({ isExpanded, ...props }) => <span {...props} />)`
    width: ${({ theme }) => theme.spacing.lg};
    height: ${({ theme }) => theme.spacing.lg};
    display: flex;
    align-items: center;
    justify-content: center;

    ${LeftInner},
    ${RightInner} {
        background: ${({ theme }) => theme.color.N6};
    }

    ${({ isExpanded, theme }) =>
        isExpanded
            ? `
        ${LeftOuter} {
            transform: rotate(-90deg);
        }

        ${RightOuter} {
            transform: rotate(90deg);
        }
    `
            : null}

    @media ${({ theme }) => above(theme.breakpoints.md)} {
        &:hover {
            ${LeftInner},
            ${RightInner} {
                background: ${({ theme }) => theme.color.B6};
            }
        }
    }
`;
