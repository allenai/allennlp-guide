import React from 'react';
import rehypeReact from 'rehype-react';

import Exercise from './components/Exercise';
import { TextBlock } from './components/TextBlock';
import CodeBlock from './components/codeBlock/CodeBlock';
import { Link } from './components/Link';

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        exercise: Exercise,
        textblock: TextBlock,
        codeblock: CodeBlock,
        a: Link,
    },
}).Compiler;
