import React from 'react';
import PropTypes from 'prop-types';
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
        setupIsVisible: false
    };

    static defaultProps = {
        setupFile: '',
        sourceFile: '',
        branch: 'master',
        url: 'https://mybinder.org',
        serverSettings: {},
        kernelType: 'python3',
        lang: 'python',
        theme: 'default',
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

    static propTypes = {
        setupFile: PropTypes.string,
        sourceFile: PropTypes.string,
        repo: PropTypes.string.isRequired,
        branch: PropTypes.string,
        url: PropTypes.string,
        serverSettings: PropTypes.object,
        kernelType: PropTypes.string,
        lang: PropTypes.string,
        theme: PropTypes.string,
        isolateCells: PropTypes.bool,
        useBinder: PropTypes.bool,
        useStorage: PropTypes.bool,
        storageExpire: PropTypes.number,
        msgButton: PropTypes.string,
        msgLoading: PropTypes.string,
        msgError: PropTypes.string,
        actions: PropTypes.func,
        handleReset: PropTypes.func
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
            theme: this.props.theme,
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
                            <SetupSectionContent>
                                <pre className="language-python line-numbers"><code>{this.props.setupFile}</code></pre>
                            </SetupSectionContent>
                        </AnimateHeight>
                    </CodeBlockSection>
                )}
                <CodeBlockSection>
                    <CodeBlockSectionHeader>
                        <strong>Source</strong>
                    </CodeBlockSectionHeader>
                    <CodeBlockSectionContent>
                        <div ref={x => {this.inputRef = x}} />
                    </CodeBlockSectionContent>
                    {this.props.msgButton && (
                        <button onClick={this.state.runCode}>
                            {this.props.msgButton}
                        </button>
                    )}
                    {this.props.actions && this.props.actions(this.state)}
                </CodeBlockSection>
                <AnimateHeight animateOpacity={true} height={this.state.fromStorage !== null ? 'auto' : 0}>
                    <OutputSection>
                        <CodeBlockSectionHeader>
                            <strong>Output</strong>
                            <ClearBtn onClick={this.props.handleReset}>Clear</ClearBtn>
                        </CodeBlockSectionHeader>
                        <CodeBlockSectionContent>
                            <div ref={x => {this.outputRef = x}} />
                        </CodeBlockSectionContent>
                    </OutputSection>
                </AnimateHeight>
            </CodeBlockContainer>
        );
    }
}

export default Juniper;

// CSS ported from SASS
// TODO(aarons): Revisit these styles

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

const CodeBlockSectionContent = styled.div`
    padding: ${({ theme }) => theme.spacing.md};
    padding-bottom: ${({ theme }) => theme.spacing.xl};

    pre {
        padding: 0;
        overflow-y: hidden;
        overflow-x: auto;
        margin: 0;
        background: transparent !important;
    }
`;

const SetupSectionContent = styled(CodeBlockSectionContent)`
    padding-left: 0;
`;

const OutputSection = styled.div`
    background: ${({ theme }) => theme.color.N10};

    color: #f7f7f7;
    font-family: 'Roboto Mono', Courier, monospace;
    font-size: 0.8625rem;
    -webkit-font-smoothing: subpixel-antialiased;

    ${CodeBlockSectionContent} {
        padding: ${({ theme }) => `${theme.spacing.md} ${theme.spacing.lg}`};
    }
`;
