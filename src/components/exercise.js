import React, { useRef, useCallback, useContext, useEffect } from 'react';
import classNames from 'classnames';
import styled from 'styled-components';

import { Button, CompleteButton } from './button';
import { ChapterContext } from '../context';
import { Card } from '../components/Card';
import classes from '../styles/exercise.module.sass';

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
    const rootClassNames = classNames(classes.root, {
        [classes.expanded]: isExpanded,
        [classes.completed]: !isExpanded && isCompleted,
    });
    const titleClassNames = classNames(classes.title, {
        [classes.titleExpanded]: isExpanded,
    });
    return (
        <Card ref={excRef} id={id} className={rootClassNames}>
            <h2 className={titleClassNames} onClick={handleExpand}>
                <span>
                    <span
                        className={classNames(classes.id, { [classes.idCompleted]: isCompleted })}
                    >
                        {excId}
                    </span>
                    {title}
                </span>
            </h2>
            {isExpanded && (
                <MarkdownContainer>
                    {children}
                    <footer className={classes.footer}>
                        <CompleteButton
                            completed={isCompleted}
                            toggleComplete={handleSetCompleted}
                        />
                        <Button onClick={handleNext} variant="secondary" small>
                            Next
                        </Button>
                    </footer>
                </MarkdownContainer>
            )}
        </Card>
    );
};

export default Exercise;

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
