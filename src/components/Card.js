import styled from 'styled-components';

export const Card = styled.div`
    background: ${({ theme }) => theme.color.N1};
    box-shadow: 0 ${({ theme }) => `${theme.spacing.xxs} ${theme.spacing.sm}`} rgba(10, 41, 57, 0.2);
    border-radius: ${({ theme }) => theme.spacing.xxs};

    & + & {
        margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    }
`;
