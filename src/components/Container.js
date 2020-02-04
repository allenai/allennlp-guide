import React from 'react';
import styled from 'styled-components';

export const Container = ({ children, className }) => (
    <StyledContainer className={className}>
        <Content>
            {children}
        </Content>
    </StyledContainer>
);

const StyledContainer = styled.div`
    width: 100%;
    padding: 72px 32px;
`;

const Content = styled.div`
    margin: auto;
    max-width: 800px;
`;
