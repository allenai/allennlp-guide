import React, { useRef, useCallback, useContext, useEffect } from 'react';
import styled from 'styled-components';
import { Button } from '@allenai/varnish/components/button';

import { ChapterContext } from '../context';
import { Card, CardContent } from './Card';
import { ExpandCollapseIcon } from './inlineSVG';

const Exercise = ({ id, title, type, children }) => {
    const excRef = useRef();
    const excId = parseInt(id);
    const { activeExc, setActiveExc } = useContext(ChapterContext);
    const isExpanded = activeExc === excId;

    const handleEscape = (e) => {
        if (e.keyCode === 27) {  // 27 is ESC
            setActiveExc(null);
        }
    };

    useEffect(() => {
        document.addEventListener('keyup', handleEscape, false);
        return () => {
            document.removeEventListener('keyup', handleEscape, false);
        }
    });

    const handleExpand = useCallback(() => setActiveExc(isExpanded ? null : excId), [
        isExpanded,
        setActiveExc,
        excId
    ]);

    const handleNext = useCallback(() => setActiveExc(excId + 1), [
        setActiveExc,
        excId
    ]);

    return (
        <StyledCard isExpanded={isExpanded}>
            <Anchor id={id} ref={excRef} />
            <SectionTitle onClick={handleExpand}>
                <SectionId>
                    {excId}
                </SectionId>
                {title}
                <TriggerIcon isExpanded={isExpanded} />
            </SectionTitle>
            {isExpanded && (
                <StyledCardContent>
                    <MarkdownContainer>
                        {children}
                        <Toolbar>
                            <Button onClick={handleNext}>Next</Button>
                        </Toolbar>
                    </MarkdownContainer>
                </StyledCardContent>
            )}
        </StyledCard>
    );
};

export default Exercise;

const TriggerIcon = styled(ExpandCollapseIcon)`
    margin: 2px -${({ theme }) => theme.spacing.sm} 0 auto;
`;

const SectionTitle = styled.h2`
    ${({ theme }) => theme.typography.bodyBig}
    display: flex;
    margin: 0;
    padding: ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md.getRemValue() * 2}rem`};
    cursor: pointer;

    svg {
        margin-left: auto;
        margin-right: -6px;
        opacity: 0;
    }

    &:hover {
        ${TriggerIcon} {
            span {
                background: ${({ theme }) => theme.color.B6};
            }
        }
    }
`;

const Toolbar = styled.div`
    display: flex;

    &:not(:only-child) {
        border-top: 1px solid ${({ theme }) => theme.color.N4};
        padding-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
        margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    }

    button:last-child {
        margin-left: auto;
    }
`;

const StyledCard = styled(({ isExpanded, ...props }) => <Card {...props} />)`
    border: 1px solid transparent;
    transition: border-color 0.1s ease;

    &:hover {
        ${({ isExpanded, theme }) => !isExpanded ? `
            border-color: ${theme.color.B6};
        ` : null}
    }

    // This is a hack to hide the last section "next" button.
    // I couldn't figure out how to get an array of Excercise components from
    // markdown. Excercise is not a child of chapter template, but part of
    // the markdown data that gets injected. I don't know of a way for an Excercise
    // component to self-identify as "last" without explicitly setting a "last" flag
    // in the markdown data itself. This solution hides the button from the user but does
    // not actually prevent the button from being rendered to the DOM which
    // would be prefarable. This is addressing an issue that was inherited from
    // https://github.com/ines/course-starter-python.
    // TODO: investigate possible ways of hiding this programmatically.

    &:last-child {
        ${Toolbar} {
            display: none;
        }
    }
`;

// Workaround to position anchored content below sticky header
const Anchor = styled.div`
    width: 1px;
    height: 164px;
    margin-top: -164px;
    transform: translateX(-10px);
`;

const SectionId = styled.span`
   font-weight: normal;
   color: ${({ theme }) => theme.color.B6};
   font-size: 19px;
   padding-right: 20px;
`;

const StyledCardContent = styled(CardContent)`
    border-top: 1px solid ${({ theme }) => theme.color.N4};
`;

const MarkdownContainer = styled.div`
    h1,
    h2,
    h3,
    h4 {
      ${({ theme }) => theme.typography.h4}
    }

    &&& {
        ol,
        ul {
            padding-left: 20px;
            margin: 1.5rem 0;

            &:first-child {
                margin-top: 0;
            }

            li {
                margin: 2px 0 2px 6px;
                padding-left: 2px;
            }

            li > p,
            ol,
            ul {
                margin: 0;
            }
        }

        img {
            margin-bottom: 1.5rem;
        }
    }

    ul {
        list-style: disc;

        ul {
            list-style: circle;

            ul {
                list-style-type: square;
            }
        }
    }

    ol {
        ol {
            list-style-type: lower-roman;

            ol {
                list-style-type: lower-alpha;
            }
        }
    }

    a {
      text-decoration: none;

      &&:hover {
          text-decoration: underline;
      }
    }

    table,
    hr,
    pre,
    div[class^="code-module-root"],
    .gatsby-highlight {
        & + ${Toolbar} {
            border: none;
            padding-top: 0;
        }
    }

    hr {
        display: block;
        border: none;
        height: 1px;
        margin: 2rem 0;
        background: ${({ theme }) => theme.color.N4};
    }
`;
