import React from 'react';
import rehypeReact from 'rehype-react';

import ExerciseComponent from './components/ExerciseComponent';
import { TextBlockComponent } from './components/TextBlockComponent';
import CodeBlock from './components/CodeBlock';
import { LinkComponent } from './components/LinkComponent';

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        exercise: ExerciseComponent,
        textblock: TextBlockComponent,
        codeblock: CodeBlock,
        a: LinkComponent,
    },
}).Compiler;
