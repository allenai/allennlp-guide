import React from 'react';
import CodeMirror from 'codemirror';
import { Widget } from '@phosphor/widgets';
import { Kernel, ServerConnection } from '@jupyterlab/services';
import { OutputArea, OutputAreaModel } from '@jupyterlab/outputarea';
import { RenderMimeRegistry, standardRendererFactories } from '@jupyterlab/rendermime';
import { window } from 'browser-monads';
import styled from 'styled-components';
import AnimateHeight from 'react-animate-height';

import { Card } from '../components/Card';
import { ExpandCollapseIcon } from '../components/inlineSVG';

class Juniper extends React.Component {
    outputRef = null;
    inputRef = null;
    state = {
        content: null,
        cm: null,
        kernel: null,
        renderers: null,
        fromStorage: null,
        setupIsVisible: true
    };

    static defaultProps = {
        setupFile: '',
        sourceFile: '',
        branch: 'master',
        url: 'https://mybinder.org',
        serverSettings: {},
        kernelType: 'python3',
        lang: 'python',
        isolateCells: true,
        useBinder: true,
        storageKey: 'juniper',
        useStorage: true,
        storageExpire: 60,
        debug: true,
        msgButton: 'run',
        msgLoading: 'Loading...',
        msgError: 'Connecting failed. Please reload and try again.',
    };

    componentDidMount() {
        this.setState({ content: this.props.sourceFile });
        const renderers = standardRendererFactories.filter(factory =>
            factory.mimeTypes.includes('text/latex') ? window.MathJax : true
        );

        const outputArea = new OutputArea({
            model: new OutputAreaModel({ trusted: true }),
            rendermime: new RenderMimeRegistry({ initialFactories: renderers }),
        });

        const cm = new CodeMirror(this.inputRef, {
            value: this.props.sourceFile.trim(),
            mode: this.props.lang,
            lineNumbers: true,
            lineWrapping: true,
            indentUnit: 4
        });
        this.setState({ cm });

        const runCode = wrapper => {
            const value = cm.getValue();
            this.execute(outputArea, wrapper ? wrapper(value) : value);
        };
        const setValue = value => cm.setValue(value);
        cm.setOption('extraKeys', { 'Shift-Enter': runCode });
        Widget.attach(outputArea, this.outputRef);
        this.setState({ runCode, setValue });
    }

    log(logFunction) {
        if (this.props.debug) {
            logFunction();
        }
    }

    componentWillReceiveProps({ sourceFile }) {
        if (sourceFile !== this.state.content && this.state.cm) {
            this.state.cm.setValue(sourceFile.trim());
        }
    }

    /**
     * Request a binder, e.g. from mybinder.org
     * @param {string} repo - Repository name in the format 'user/repo'.
     * @param {string} branch - The repository branch, e.g. 'master'.
     * @param {string} url - The binder reployment URL, including 'http(s)'.
     * @returns {Promise} - Resolved with Binder settings, rejected with Error.
     */
    requestBinder(repo, branch, url) {
        const binderUrl = `${url}/build/gh/${repo}/${branch}`
        this.log(() => console.info('building', { binderUrl }))
        return new Promise((resolve, reject) => {
            const es = new EventSource(binderUrl)
            es.onerror = err => {
                es.close()
                this.log(() => console.error('failed'))
                reject(new Error(err))
            }
            let phase = null
            es.onmessage = ({ data }) => {
                const msg = JSON.parse(data)
                if (msg.phase && msg.phase !== phase) {
                    phase = msg.phase.toLowerCase()
                    this.log(() => console.info(phase === 'ready' ? 'server-ready' : phase))
                }
                if (msg.phase === 'failed') {
                    es.close()
                    reject(new Error(msg))
                } else if (msg.phase === 'ready') {
                    es.close()
                    const settings = {
                        baseUrl: msg.url,
                        wsUrl: `ws${msg.url.slice(4)}`,
                        token: msg.token,
                    }
                    resolve(settings)
                }
            }
        })
    }

    /**
     * Request kernel and estabish a server connection via the JupyerLab service
     * @param {object} settings - The server settings.
     * @returns {Promise} - A promise that's resolved with the kernel.
     */
    requestKernel(settings) {
        if (this.props.useStorage) {
            const timestamp = new Date().getTime() + this.props.storageExpire * 60 * 1000
            const json = JSON.stringify({ settings, timestamp })
            window.localStorage.setItem(this.props.storageKey, json)
        }
        const serverSettings = ServerConnection.makeSettings(settings)
        return Kernel.startNew({
            type: this.props.kernelType,
            name: this.props.kernelType,
            serverSettings,
        }).then(kernel => {
            this.log(() => console.info('ready'))
            return kernel
        })
    }

    /**
     * Get a kernel by requesting a binder or from localStorage / user settings
     * @returns {Promise}
     */
    getKernel() {
        if (this.props.useStorage) {
            const stored = window.localStorage.getItem(this.props.storageKey)
            if (stored) {
                this.setState({ fromStorage: true })
                const { settings, timestamp } = JSON.parse(stored)
                if (timestamp && new Date().getTime() < timestamp) {
                    return this.requestKernel(settings)
                }
                window.localStorage.removeItem(this.props.storageKey)
            }
        }
        if (this.props.useBinder) {
            return this.requestBinder(this.props.repo, this.props.branch, this.props.url).then(
                settings => this.requestKernel(settings)
            )
        }
        return this.requestKernel(this.props.serverSettings)
    }

    /**
     * Render the kernel response in a JupyterLab output area
     * @param {OutputArea} outputArea - The cell's output area.
     * @param {string} code - The code to execute.
     */
    renderResponse(outputArea, code) {
        outputArea.future = this.state.kernel.requestExecute({ code })
        outputArea.model.add({
            output_type: 'stream',
            name: 'loading',
            text: this.props.msgLoading,
        })
        outputArea.model.clear(true)
    }

    /**
     * Process request to execute the code
     * @param {OutputArea} - outputArea - The cell's output area.
     * @param {string} code - The code to execute.
     */
    execute(outputArea, code) {
        this.log(() => console.info('executing'))
        if (this.state.kernel) {
            if (this.props.isolateCells) {
                this.state.kernel
                    .restart()
                    .then(() => this.renderResponse(outputArea, code))
                    .catch(() => {
                        this.log(() => console.error('failed'))
                        this.setState({ kernel: null })
                        outputArea.model.clear()
                        outputArea.model.add({
                            output_type: 'stream',
                            name: 'failure',
                            text: this.props.msgError,
                        })
                    })
                return
            }
            this.renderResponse(outputArea, code)
            return
        }
        this.log(() => console.info('requesting kernel'))
        const url = this.props.url.split('//')[1]
        const action = !this.state.fromStorage ? 'Launching' : 'Reconnecting to'
        outputArea.model.clear()
        outputArea.model.add({
            output_type: 'stream',
            name: 'stdout',
            text: `${action} Docker container on ${url}...`,
        })
        new Promise((resolve, reject) =>
            this.getKernel()
                .then(resolve)
                .catch(reject)
        )
            .then(kernel => {
                this.setState({ kernel })
                this.renderResponse(outputArea, code)
            })
            .catch(() => {
                this.log(() => console.error('failed'))
                this.setState({ kernel: null })
                if (this.props.useStorage) {
                    this.setState({ fromStorage: false })
                    window.localStorage.removeItem(this.props.storageKey)
                }
                outputArea.model.clear()
                outputArea.model.add({
                    output_type: 'stream',
                    name: 'failure',
                    text: this.props.msgError,
                })
            })
    }

    render() {
        return (
            <CodeBlockContainer>
                {this.props.setupFile !== '' && (
                    <CodeBlockSection>
                        <SetupSectionToggle onClick={() => this.setState({setupIsVisible: !this.state.setupIsVisible})}>
                            <strong>Setup</strong>
                            <span className={`${this.state.setupIsVisible ? 'label-visible' : ''}`}>(Read-only)</span>
                            <TriggerIcon isExpanded={this.state.setupIsVisible} />
                        </SetupSectionToggle>
                        <AnimateHeight animateOpacity={true} height={this.state.setupIsVisible ? 'auto' : 0}>
                            <CodeBlockSectionContent>
                                <PrismRender>
                                    <pre className="language-python line-numbers"><code>{this.props.setupFile}</code></pre>
                                    <PrismSelect defaultValue={this.props.setupFile} readOnly={true} />
                                </PrismRender>
                            </CodeBlockSectionContent>
                        </AnimateHeight>
                    </CodeBlockSection>
                )}
                <CodeBlockSection>
                    <CodeBlockSectionHeader>
                        <strong>Source</strong>
                    </CodeBlockSectionHeader>
                    <CodeBlockSectionContent>
                        <CodeMirrorRender ref={x => {this.inputRef = x}} />
                    </CodeBlockSectionContent>
                    {this.props.msgButton && (
                        <button onClick={this.state.runCode}>
                            {this.props.msgButton}
                        </button>
                    )}
                    {this.props.actions && this.props.actions(this.state)}
                </CodeBlockSection>
                <AnimateHeight animateOpacity={true} height={this.props.outputIsVisible ? 'auto' : 0}>
                    <OutputSection>
                        <CodeBlockSectionHeader>
                            <strong>Output</strong>
                            <ClearBtn onClick={this.props.handleReset}>Clear</ClearBtn>
                        </CodeBlockSectionHeader>
                        <CodeBlockSectionContent>
                            <OutputRender ref={x => {this.outputRef = x}} />
                        </CodeBlockSectionContent>
                    </OutputSection>
                </AnimateHeight>
            </CodeBlockContainer>
        );
    }
}

export default Juniper;

const CodeBlockContainer = styled(Card)`
    background: ${({ theme }) => theme.color.N9};
    overflow: hidden;
    position: relative;
    z-index: 3;
`;

const CodeBlockSection = styled.div`
    margin-bottom: ${({ theme }) => theme.spacing.xxs};
`;

const CodeBlockSectionHeader = styled.div`
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

        &.label-visible {
            opacity: 1;
        }
    }
`;

const SetupSectionToggle = styled(CodeBlockSectionHeader)`
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
const CodeMirrorRender = styled.div``;
const OutputRender = styled.div``;

const CodeBlockSectionContent = styled.div`
    padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.xs} ${theme.spacing.xl} ${theme.spacing.xs}`};

    .CodeMirror.cm-s-default {
      width: 100%;
      height: 100%;
    }

    .CodeMirror-scroll {
      padding-right: 9px;
    }

    ${PrismRender} pre,
    ${PrismRender} code,
    ${PrismRender} code span,
    ${PrismSelect},
    ${CodeMirrorRender} textarea,
    ${CodeMirrorRender} .CodeMirror.cm-s-default,
    ${CodeMirrorRender} .CodeMirror.cm-s-default * {
      font-family: 'Roboto Mono', Courier, monospace !important;
      font-size: 0.8625rem !important;
      -webkit-font-smoothing: subpixel-antialiased !important;
      line-height: 1.5 !important;
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
        background-color: rgba(222,233,255,0.133) !important;
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
    ${CodeMirrorRender} .cm-s-default,
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
    .cm-s-default,
    .cm-s-default .cm-punctuation,
    .cm-s-default .cm-bracket,
    .cm-s-default .cm-meta,
    .cm-meta + .cm-property,
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

    .CodeMirror-gutters,
    .line-numbers .line-numbers-rows {
        border-right: 1px solid ${({ theme }) => theme.color.N8} !important;
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
        border-left: 2px solid ${({ theme }) => theme.color.B4} !important;
    }


    ${OutputRender} pre {
        padding: 0;
        overflow-y: hidden;
        overflow-x: auto;
        margin: 0;
        background: transparent !important;
    }


`;

const OutputSection = styled.div`
    background: ${({ theme }) => theme.color.N10};

    color: #f7f7f7;
    font-family: 'Roboto Mono', Courier, monospace;
    font-size: 0.8625rem;
    -webkit-font-smoothing: subpixel-antialiased;

    ${CodeBlockSectionContent} {
        padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.lg}`};
        min-height: 150px;
    }
`;
