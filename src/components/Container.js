// This component offers a common layout structure utilizing a
// constrained-width content container nested inside of a full-width
// parent container that can be given a custom background color
// via Styled when it's instanced.

import React from 'react';
import styled from 'styled-components';

export const Container = ({ children, className }) => (
    <StyledContainer className={className}>
        <Content>{children}</Content>
    </StyledContainer>
);

const StyledContainer = styled.div`
    width: 100%;
    padding: 72px 32px;

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        padding: ${({ theme }) => theme.spacing.lg};
    }
`;

const Content = styled.div`
    margin: auto;
    max-width: 804px;
`;
