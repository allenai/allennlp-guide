// This set of components wraps and styles the Juniper component and is
// the injection point for instances of <codeblock>

import React, { useState } from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled, { css } from 'styled-components';
import AnimateHeight from 'react-animate-height';
import { Button } from '@allenai/varnish/components/button';

import { Card } from '../Card';
import { ExpandCollapseIcon } from '../inlineSVG';

function getFiles({ allCode }, sourceId, setupId) {
    var files = {};
    allCode.edges.forEach(({ node }) => {
        const filename = node.dir + '/' + node.name;
        if (filename.includes(sourceId)) {
            files['sourceFile'] = node.code;
        }
        if (filename.includes(setupId)) {
            files['setupFile'] = node.code;
        }
    });
    return files;
}

function addSetupCode(code, setup) {
    // Prepend setup code if specified.
    return setup ? setup + code : code;
}

export class CodeBlock extends React.Component {
    state = {
        Juniper: null,
        key: 0,
        outputIsVisible: false,
        focused: false
    };

    handleReset() {
        // Using the key as a hack to force component to rerender
        this.setState({
            key: this.state.key + 1,
            outputIsVisible: false
        });
    }

    updateJuniper() {
        // This type of stuff only really works in class components. I'm not
        // sure why, but I've tried with function components and hooks lots of
        // times and couldn't get it to work. So class component it is.
        if (!this.state.Juniper) {
            // We need a dynamic import here for SSR. Juniper's dependencies
            // include references to the global window object and I haven't
            // managed to fix this using webpack yet. If we imported Juniper
            // at the top level, Gatsby won't build.
            import('./Juniper').then(Juniper => {
                this.setState({ Juniper: Juniper.default });
            });
        }
    }

    componentDidMount() {
        this.updateJuniper();
    }

    componentDidUpdate() {
        this.updateJuniper();
    }

    render() {
        const { Juniper, outputIsVisible, focused } = this.state;
        const { id, source, setup, executable } = this.props;
        const sourceId = source || `${id}_source`;
        const setupId = setup || `${id}_setup`;
        const execute = executable !== "false";

        return (
            <StaticQuery
                query={graphql`
                    {
                        site {
                            siteMetadata {
                                juniper {
                                    repo
                                    branch
                                    kernelType
                                    lang
                                    debug
                                }
                            }
                        }
                        allCode {
                            edges {
                                node {
                                    dir
                                    name
                                    code
                                }
                            }
                        }
                    }
                `}
                render={data => {
                    const { repo, branch, kernelType, debug, lang } = data.site.siteMetadata.juniper;
                    const { sourceFile, setupFile } = getFiles(data, sourceId, setupId);
                    return (
                        <Container className={focused ? 'Container-focused' : ''}>
                            {setupFile && setupFile !== '' && (
                                <CodeSection
                                    title="Setup"
                                    setupFile={setupFile}
                                />
                            )}
                            {Juniper && (
                                <Juniper
                                    key={this.state.key}
                                    repo={repo}
                                    branch={branch}
                                    lang={lang}
                                    kernelType={kernelType}
                                    handleReset={() => this.handleReset()}
                                    outputIsVisible={outputIsVisible}
                                    debug={debug}
                                    setupFile={setupFile}
                                    sourceFile={sourceFile}
                                    actions={({ runCode }) => execute && (
                                        <RunButton onClick={() => {
                                            runCode(code => addSetupCode(code, setupFile));
                                            this.setState({ outputIsVisible: true });
                                        }}>Run Code</RunButton>
                                    )}
                                />
                            )}
                        </Container>
                    )
                }}
            />
        );
    }
}

// Styled wrapper for the CodeBlock exercise structure
const Container = styled(Card)`
    background: ${({ theme }) => theme.color.N9};
    overflow: hidden;
    position: relative;
    z-index: 3;
    margin-bottom: 2rem;
    margin-top: 2rem;
`;

// Text styles used for code exercises and inline prism codeblocks
export const codeBlockTextStyles = css`
    font-family: 'Roboto Mono', Courier, monospace !important;
    font-size: 0.8625rem !important;
    -webkit-font-smoothing: subpixel-antialiased !important;
    line-height: 1.5 !important;
`;

// Wrapping styles used for code exercises and inline prism codeblocks
export const codeBlockWrappingStyles = css`
    background: none;
    text-align: left;
    white-space: pre-wrap;
    word-spacing: normal;
    word-break: normal;
    word-wrap: break-word;
    line-height: 1.5;
    -moz-tab-size: 4;
    -o-tab-size: 4;
    tab-size: 4;
    -webkit-hyphens: none;
    -moz-hyphens: none;
    -ms-hyphens: none;
    hyphens: none;
`;

// Header and Content section within a CodeBlock
export const CodeSection = ({
    actions,
    children,
    className,
    clearFunction,
    setupFile,
    title
}) => {
    const [sectionIsVisible, setSectionVisibility] = useState(false);
    const Header = setupFile ? ToggleableSectionHeader : SectionHeader;
    const Section = title === "Output" ? OutputSection : InputSection;
    return (
        <Section className={className}>
            <Header onClick={setupFile ? () => setSectionVisibility(!sectionIsVisible) : () => {}}>
                <strong>{title}</strong>
                {setupFile && (
                    <span className={`${sectionIsVisible ? 'label-visible' : ''}`}>(Read-only)</span>
                )}
                {clearFunction && (
                    <ClearBtn onClick={clearFunction}>Clear</ClearBtn>
                )}
                {setupFile && (
                    <TriggerIcon isExpanded={sectionIsVisible} />
                )}
            </Header>
            {setupFile ? (
                <AnimateHeight animateOpacity={true} height={sectionIsVisible ? 'auto' : 0}>
                    <SectionContent>
                        <PrismRender>
                            <pre className="language-python line-numbers"><code>{setupFile}</code></pre>
                            <PrismSelect defaultValue={setupFile} readOnly={true} />
                        </PrismRender>
                    </SectionContent>
                </AnimateHeight>
            ) : (
                <SectionContent>
                    {children}
                </SectionContent>
            )}
            {actions && (
                <Toolbar>
                    {actions}
                </Toolbar>
            )}
        </Section>
    );
};

const RunButton = styled(Button)`
    &&& {
        background: ${({ theme }) => theme.color.N1};
        color: ${({ theme }) => theme.color.B6};
        transition: color 0.1s ease, background-color 0.1s ease, border-color 0.1s ease;

        &:hover {
            background: ${({ theme }) => theme.color.B5};
            border-color: ${({ theme }) => theme.color.B5};
            color: ${({ theme }) => theme.color.N1};
        }

        &:focus {
            background: ${({ theme }) => theme.color.N1};
            border-color: ${({ theme }) => theme.color.B5};
            color: ${({ theme }) => theme.color.B6};
        }
    }
`;

const Toolbar = styled.div`
    display: flex;
    padding: 0 ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md}`} 0;

    ${RunButton} {
        margin-left: auto;
    }
`;

const InputSection = styled.div`
    margin-bottom: ${({ theme }) => theme.spacing.xxs};

    & > .rah-static > div {
        display: block !important;
    }
`;

const SectionHeader = styled.div`
    background: ${({ theme }) => theme.color.N8};
    font-weight: ${({ theme }) => theme.typography.fontWeightBold};
    color: ${({ theme }) => theme.color.N5};
    padding: ${({ theme }) => `${theme.spacing.sm} ${theme.spacing.lg}`};
    display: flex;
    transition: background-color 0.2s ease;

    & > span {
        ${({ theme }) => theme.typography.bodySmall};
        font-weight: normal;
        color: ${({ theme }) => theme.color.N6};
        padding-left: ${({ theme }) => theme.spacing.xs};
        opacity: 0;
        transition: opacity 0.2s ease;
        margin-top: 2px;

        &.label-visible {
            opacity: 1;
        }
    }
`;

const ToggleableSectionHeader = styled(SectionHeader)`
    cursor: pointer;

    &:hover {
        background: #6d7784;
    }

    &:active {
        background: ${({ theme }) => theme.color.N8};
        transition-duration: 0s;
    }
`;

// Morphing expand/collapse caret
const TriggerIcon = styled(ExpandCollapseIcon)`
    margin-left: auto;
    margin-right: -${({ theme }) => theme.spacing.xs};

    span {
        background: ${({ theme }) => theme.color.N6} !important;
    }
`;

const ClearBtn = styled.button`
    display: block;
    border: none;
    font-weight: ${({ theme }) => theme.typography.fontWeightBold};
    cursor: pointer;
    background: transparent;
    margin: 0;
    margin-left: auto;
    padding: 0;
    appearance: none;
    ${({ theme }) => theme.typography.bodySmall};
    color: ${({ theme }) => theme.color.N7};
    transition: color 0.2s ease;

    &:hover {
        color: ${({ theme }) => theme.color.N5};
    }

    &:focus {
        outline: none;
    }
`;

const PrismRender = styled.div``;
const PrismSelect = styled.textarea``;
export const CodeMirrorRender = styled.div``;
export const OutputRender = styled.div``;

const SectionContent = styled.div`
    &&& {
        padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.xs} ${theme.spacing.xl} ${theme.spacing.xs}`};

        .CodeMirror.cm-s-default {
          width: 100%;
          height: 100%;
        }

        .CodeMirror-scroll {
          padding-right: 9px;
        }

        ${OutputRender} pre,
        ${PrismRender} pre,
        ${PrismRender} code,
        ${PrismRender} code span,
        ${PrismSelect},
        ${CodeMirrorRender} textarea,
        ${CodeMirrorRender} .CodeMirror.cm-s-default,
        ${CodeMirrorRender} .CodeMirror.cm-s-default * {
          ${codeBlockTextStyles}
          text-shadow: none !important;
          background: none !important;
        }

        ${PrismSelect},
        ${PrismRender} pre {
          margin: 0 !important;
          overflow: visible !important;
          padding: 0 1em 0 3.8em !important;
        }

        ${PrismRender} pre::selection,
        ${PrismRender} pre ::selection,
        ${PrismSelect}::selection,
        ${PrismSelect} ::selection,
        ${CodeMirrorRender} .CodeMirror.cm-s-default .CodeMirror-selected {
            background-color: rgba(222,233,255,0.15) !important;
        }

        ${PrismSelect}::selection,
        ${PrismSelect} ::selection {
            color: transparent;
        }

        ${PrismRender} {
          position: relative;
        }

        ${PrismRender} code[class*="language-"],
        ${PrismRender} pre[class*="language-"],
        ${PrismRender} span,
        ${PrismSelect} {
            ${codeBlockWrappingStyles}
        }

        ${PrismSelect} {
            display: none;
            border: none;
            resize: none;
            appearance: none;
            color: transparent;
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            -webkit-font-smoothing: none;
            outline: none !important;
            overflow: hidden !important;
            z-index: 2;
        }

        @media (min-width: 1200px) {
            ${PrismRender} pre,
            ${PrismRender} pre * {
              user-select: none;
            }

            ${PrismSelect} {
              display: block;
            }
        }

        ${PrismRender} pre,
        ${PrismRender} code,
        .token.string-interpolation,
        .token.interpolation,
        ${CodeMirrorRender} .cm-s-default .cm-variable,
        ${CodeMirrorRender} .cm-s-default .cm-variable-2,
        ${CodeMirrorRender} .cm-s-default .cm-variable-3,
        ${CodeMirrorRender} .cm-s-default .cm-type,
        ${CodeMirrorRender} .cm-s-default .cm-property {
            color: ${({ theme }) => theme.color.N4};
        }

        .token.keyword,
        .cm-s-default .cm-keyword,
        .token.builtin,
        .cm-s-default .cm-builtin,
        .cm-s-default .cm-header {
            color: #55b8e3;
        }

        .token.punctuation,
        .token.decorator.annotation.punctuation,
        ${CodeMirrorRender} .cm-s-default,
        .cm-s-default .cm-punctuation,
        .cm-s-default .cm-bracket,
        .cm-s-default .cm-meta,
        .cm-s-default .cm-meta + .cm-property,
        .cm-s-default .cm-qualifier,
        .cm-s-default .cm-atom,
        .cm-s-default .cm-tag,
        .cm-s-default .cm-attribute,
        .cm-s-default .cm-hr,
        .cm-s-default .cm-link {
            color: ${({ theme }) => theme.color.N6};
        }

        .token.comment,
        .cm-s-default .cm-comment {
            color: ${({ theme }) => theme.color.N7};
        }

        .token.operator,
        .cm-s-default .cm-operator {
            color: #a8937a;
            background: transparent;
        }

        .token.property,
        .token.boolean,
        .token.number,
        .token.constant,
        .token.symbol,
        .token.deleted,
        .cm-s-default .cm-number,
        .cm-s-default .cm-boolean,
        .cm-s-default .cm-operator + .cm-keyword {
            color: #de81c3;
        }

        .token.string,
        .token.char,
        .token.inserted,
        .token.triple-quoted-string,
        .cm-s-default .cm-string,
        .cm-s-default .cm-quote,
        .cm-s-default .cm-string-2,
        .cm-s-default .cm-positive {
            color: #b2d27a;
        }

        .token.function,
        .token.class-name,
        .cm-s-default .cm-def,
        .cm-s-default .cm-negative,
        .cm-s-default .cm-error,
        .cm-invalidchar {
            color: #ea9597;
        }

        /* Line Number Border */

        .CodeMirror-gutters {
            transition: border 0.2s ease;
        }

        .CodeMirror-gutters,
        .line-numbers .line-numbers-rows {
            border-right: 1px solid ${({ theme }) => theme.color.N8} !important;
        }

        .CodeMirror-focused .CodeMirror-gutters {
            border-right-color: ${({ theme }) => theme.color.B3} !important;
        }

        .CodeMirror-gutter.CodeMirror-linenumbers {
          padding: 0 !important;
        }

        .line-numbers .line-numbers-rows {
            top: -2px;
        }

        .CodeMirror pre.CodeMirror-line,
        .CodeMirror pre.CodeMirror-line-like {
            padding: 0 4px 0 11px;
        }

        .CodeMirror-lines {
            padding: 0;
        }

        .CodeMirror-linenumber {
            padding-right: 10px;
            padding-left: 8px;
            letter-spacing: -1px;
        }

        // Line Numbers

        .CodeMirror-linenumber,
        .line-numbers-rows > span:before {
            color: ${({ theme }) => theme.color.N7};
        }

        .CodeMirror-cursor {
            border-left: 2px solid ${({ theme }) => theme.color.B3} !important;
        }

        .ansi-cyan-fg.ansi-cyan-fg {
            color: #00d8ff;
        }

        .ansi-green-fg.ansi-green-fg {
            color: #12dc55;
        }

        .ansi-red-fg.ansi-red-fg {
            color: #f76464;
        }
    }
`;

const OutputSection = styled.div`
    &&& {
        background: ${({ theme }) => theme.color.N10};
        color: #f7f7f7;

        ${SectionContent} {
            padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.lg} ${theme.spacing.xl} ${theme.spacing.lg}`};
            min-height: 150px;
        }
    }
`;
