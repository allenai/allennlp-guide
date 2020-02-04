import React from 'react';
import rehypeReact from 'rehype-react';

import Exercise from './components/exercise';
import TextBlock from './components/textblock';
import CodeBlock from './components/CodeBlock';
import { Link } from './components/link';
import { H3, Hr, Ol, Ul, Li, InlineCode } from './components/typography';

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        exercise: Exercise,
        textblock: TextBlock,
        codeblock: CodeBlock,
        a: Link,
        hr: Hr,
        h3: H3,
        ol: Ol,
        ul: Ul,
        li: Li,
        code: InlineCode,
    },
}).Compiler;
