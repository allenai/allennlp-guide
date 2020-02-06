// This component can be used inside Markdown files and is typically used
// as a chapter introduction before any excercise modules.

import styled from 'styled-components';

export const TextBlockComponent = styled.div`
    padding-bottom: ${({ theme }) => theme.spacing.lg};
`;
