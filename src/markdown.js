import React from 'react';
import rehypeReact from 'rehype-react';

import Exercise from './components/exercise';
import { TextBlockComponent } from './components/TextBlockComponent';
import CodeBlock from './components/CodeBlock';
import { LinkComponent } from './components/LinkComponent';
import { H3, Hr, Ol, Ul, Li, InlineCode } from './components/typography';

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        exercise: Exercise,
        textblock: TextBlockComponent,
        codeblock: CodeBlock,
        a: LinkComponent,
        hr: Hr,
        h3: H3,
        ol: Ol,
        ul: Ul,
        li: Li,
        code: InlineCode,
    },
}).Compiler;
