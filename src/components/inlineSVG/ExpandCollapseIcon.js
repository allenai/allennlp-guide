// This asset displays a dual-purpose expand/collapse icon that can transition between the two
// Meant for expandable part containers on home page and expandable chapter cards on chapter pages

import React from 'react';
import styled, { css } from 'styled-components';

export const ExpandCollapseIcon = ({ className, isExpanded = false }) => (
    <PaddedContainer className={className} isExpanded={isExpanded} title={`${isExpanded ? 'Collapse' : 'Expand'} panel`}>
        <IconContainer>
            <LeftOuter>
                <LeftInner />
            </LeftOuter>
            <RightOuter>
                <RightInner />
            </RightOuter>
        </IconContainer>
    </PaddedContainer>
);

const IconContainer = styled.div`
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

const LeftOuter = styled.div`
    ${outerStyles}
    left: -0.1rem;
`;

const RightOuter = styled.div`
    ${outerStyles}
    right: 0;
`;

const innerStyles = css`
    display: block;
    transform-origin: 50% 50%;
    width: 0.75rem;
    height: 0.09375rem;
    transition: background-color 0.2s ease;
`;

const LeftInner = styled.span`
    ${innerStyles}
    transform: rotate(45deg);
`;

const RightInner = styled.span`
    ${innerStyles}
    transform: rotate(-45deg);
`;

const PaddedContainer = styled(({ isExpanded, ...props }) => <div {...props} />)`
    width: ${({ theme }) => theme.spacing.lg};
    height: ${({ theme }) => theme.spacing.lg};
    display: flex;
    align-items: center;
    justify-content: center;

    ${LeftInner},
    ${RightInner} {
        background: ${({ theme, isExpanded }) => isExpanded ? theme.color.B6 : theme.color.N6};
    }

    ${({ isExpanded, theme }) => isExpanded ? `
        ${LeftOuter} {
            transform: rotate(-90deg);
        }

        ${RightOuter} {
            transform: rotate(90deg);
        }
    ` : null}

    &:hover {
        ${LeftInner},
        ${RightInner} {
            background: ${({ theme }) => theme.color.B6};
        }
    }
`;
