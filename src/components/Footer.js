import styled from 'styled-components';
import { Footer as VarnishFooter } from '@allenai/varnish/components/Footer';

export const Footer = styled(VarnishFooter)`
    &&& {
      padding: ${({ theme }) => `${theme.spacing.xl} ${theme.spacing.xxl}`};
      margin-top: auto;
    }
`;
