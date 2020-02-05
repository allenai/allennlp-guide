import React, { useRef, useCallback, useContext, useEffect } from 'react';
import styled, { css } from 'styled-components';

import { Button, CompleteButton } from './button';
import { ChapterContext } from '../context';
import { Card, CardContent } from '../components/Card';

const Exercise = ({ id, title, type, children }) => {
    const excRef = useRef();
    const excId = parseInt(id);
    const { activeExc, setActiveExc, completed, setCompleted } = useContext(ChapterContext);
    const isExpanded = activeExc === excId;
    const isCompleted = completed.includes(excId);
    useEffect(() => {
        if (isExpanded && excRef.current) {
            excRef.current.scrollIntoView();
        }
        document.addEventListener('keyup', handleEscape, false);
        return () => {
          document.removeEventListener('keyup', handleEscape, false);
        }
    }, [isExpanded]);
    const handleEscape = (e) => {
        if (e.keyCode === 27) {  // 27 is ESC
            setActiveExc(null);
        }
    };
    const handleExpand = useCallback(() => setActiveExc(isExpanded ? null : excId), [
        isExpanded,
        excId,
    ]);

    const handleNext = useCallback(() => setActiveExc(excId + 1));
    const handleSetCompleted = useCallback(() => {
        const newCompleted = isCompleted
            ? completed.filter(v => v !== excId)
            : [...completed, excId];
        setCompleted(newCompleted);
    }, [isCompleted, completed, excId]);

    const Title = isCompleted ? CompletedSectionTitle : SectionTitle;

    return (
        <Card ref={excRef} id={id}>
            <Title onClick={handleExpand}>
                <SectionId>
                    {excId}
                </SectionId>
                {title}
            </Title>
            {isExpanded && (
                <CardContent>
                    <MarkdownContainer>
                        {children}
                        <footer>
                            <CompleteButton
                                completed={isCompleted}
                                toggleComplete={handleSetCompleted}
                            />
                            <Button onClick={handleNext} variant="secondary" small>
                                Next
                            </Button>
                        </footer>
                    </MarkdownContainer>
                </CardContent>
            )}
        </Card>
    );
};

export default Exercise;

const titleStyles = css`
    ${({ theme }) => theme.typography.bodyBig}
    margin: 0;
    padding: ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md.getRemValue() * 2}rem`};
`;

const SectionTitle = styled.h2`
    ${titleStyles}
`;

const SectionId = styled.span`
   font-weight: normal;
   color: ${({ theme }) => theme.color.B6};
   font-size: 19px;
   padding-right: 20px;
`;

const MarkdownContainer = styled.div`
    h1,
    h2,
    h3,
    h4 {
      ${({ theme }) => theme.typography.h4}
    }
    
    ul {
      list-style: disc;
      
      ul {
        list-style: circle;
      }
    }
    
    code {
      font-size: 14px;
      -webkit-font-smoothing: subpixel-antialiased;
    }
    
    a {
      text-decoration: none;
      
      &&:hover {
        text-decoration: underline;
      }
    }
`;

// TODO(aarons): Placeholder style
const CompletedSectionTitle = styled.h2`
    ${titleStyles}
    outline: 1px solid green;
`;
