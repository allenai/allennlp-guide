// This component can be used inside Markdown files and is typically used
// as a chapter introduction before any excercise modules.

import styled from 'styled-components';

export const TextBlock = styled.div`
    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
        margin: ${({ theme }) => `-${theme.spacing.sm} 0 ${theme.spacing.lg} 0`};
    }

    & {
        *:last-child {
            margin-bottom: 0;
        }
    }
`;
