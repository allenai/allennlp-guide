// This component offers a common stylistic element of a medium-sized
// content container with rounded corners and a drop-shadow that is
// reminiscent of a playing card look and feel.

import styled from 'styled-components';

export const Card = styled.div`
    position: relative;
    background: ${({ theme }) => theme.color.N1};
    box-shadow: 0 ${({ theme }) => `${theme.spacing.xxs} ${theme.spacing.sm}`} rgba(10, 41, 57, 0.2);
    border-radius: ${({ theme }) => theme.spacing.xxs};
    transition: box-shadow 0.2s ease, transform 0.2s ease;

    & + & {
        margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    }

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        margin: 0 -6px;

        & + & {
            margin-top: ${({ theme }) => theme.spacing.lg};
        }
    }
`;

export const CardContent = styled.div`
    padding: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        padding: ${({ theme }) => theme.spacing.md};
    }
`;
