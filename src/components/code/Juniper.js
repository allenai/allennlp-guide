// This component is a wrapper for Juniper and CodeMirror. It's used for Source and Output
// rendering in conjunction with <codeblock> exercises.

import React from 'react';
import CodeMirror from 'codemirror';
import { Widget } from '@phosphor/widgets';
import { Kernel, ServerConnection } from '@jupyterlab/services';
import { OutputArea, OutputAreaModel } from '@jupyterlab/outputarea';
import { RenderMimeRegistry, standardRendererFactories } from '@jupyterlab/rendermime';
import { window } from 'browser-monads';
import AnimateHeight from 'react-animate-height';

import { CodeSection, CodeMirrorRender, OutputRender } from './CodeBlock';

export default class Juniper extends React.Component {
    constructor(props) {
        super(props);
        this.inputRef = React.createRef();
        this.outputRef = React.createRef();

        this.state = {
            content: null,
            cm: null,
            kernel: null,
            renderers: null,
            fromStorage: null
        };
    }

    componentDidMount() {
        this.setState({ content: this.props.sourceFile });
        const renderers = standardRendererFactories.filter(factory =>
            factory.mimeTypes.includes('text/latex') ? window.MathJax : true
        );

        const outputArea = new OutputArea({
            model: new OutputAreaModel({ trusted: true }),
            rendermime: new RenderMimeRegistry({ initialFactories: renderers })
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
        cm.on('change', () => {
            if (!this.props.resetButtonIsVisible) {
                this.props.showResetButton();
            }
        });
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
        const binderUrl = `${url}/build/gh/${repo}/${branch}`;
        this.log(() => console.info('building', { binderUrl }));
        return new Promise((resolve, reject) => {
            const es = new EventSource(binderUrl);
            es.onerror = err => {
                es.close();
                this.log(() => console.error('failed'));
                reject(new Error(err));
            };
            let phase = null;
            es.onmessage = ({ data }) => {
                const msg = JSON.parse(data);
                if (msg.phase && msg.phase !== phase) {
                    phase = msg.phase.toLowerCase();
                    this.log(() => console.info(phase === 'ready' ? 'server-ready' : phase));
                }
                if (msg.phase === 'failed') {
                    es.close();
                    reject(new Error(msg));
                } else if (msg.phase === 'ready') {
                    es.close();
                    const settings = {
                        baseUrl: msg.url,
                        wsUrl: `ws${msg.url.slice(4)}`,
                        token: msg.token
                    };
                    resolve(settings);
                }
            };
        });
    }

    /**
     * Request kernel and estabish a server connection via the JupyerLab service
     * @param {object} settings - The server settings.
     * @returns {Promise} - A promise that's resolved with the kernel.
     */
    requestKernel(settings) {
        if (this.props.useStorage) {
            const timestamp = new Date().getTime() + this.props.storageExpire * 60 * 1000;
            const json = JSON.stringify({ settings, timestamp });
            window.localStorage.setItem(this.props.storageKey, json);
        }
        const serverSettings = ServerConnection.makeSettings(settings);
        return Kernel.startNew({
            type: this.props.kernelType,
            name: this.props.kernelType,
            serverSettings
        }).then(kernel => {
            this.log(() => console.info('ready'));
            return kernel;
        });
    }

    /**
     * Get a kernel by requesting a binder or from localStorage / user settings
     * @returns {Promise}
     */
    getKernel() {
        if (this.props.useStorage) {
            const stored = window.localStorage.getItem(this.props.storageKey);
            if (stored) {
                this.setState({ fromStorage: true });
                const { settings, timestamp } = JSON.parse(stored);
                if (timestamp && new Date().getTime() < timestamp) {
                    return this.requestKernel(settings);
                }
                window.localStorage.removeItem(this.props.storageKey);
            }
        }
        if (this.props.useBinder) {
            return this.requestBinder(
                this.props.repo,
                this.props.branch,
                this.props.url
            ).then(settings => this.requestKernel(settings));
        }
        return this.requestKernel(this.props.serverSettings);
    }

    /**
     * Render the kernel response in a JupyterLab output area
     * @param {OutputArea} outputArea - The cell's output area.
     * @param {string} code - The code to execute.
     */
    renderResponse(outputArea, code) {
        outputArea.future = this.state.kernel.requestExecute({ code });
        outputArea.model.add({
            output_type: 'stream',
            name: 'loading',
            text: this.props.msgLoading
        });
        outputArea.model.clear(true);
    }

    /**
     * Process request to execute the code
     * @param {OutputArea} - outputArea - The cell's output area.
     * @param {string} code - The code to execute.
     */
    execute(outputArea, code) {
        this.log(() => console.info('executing'));
        const url = this.props.url.split('//')[1];
        const action = !this.state.fromStorage ? 'Launching' : 'Reconnecting to';
        outputArea.model.clear();
        outputArea.model.add({
            output_type: 'stream',
            name: 'stdout',
            text: `${action} Docker container on ${url}...`
        });
        this.log(() => console.info('requesting kernel'));
        if (this.state.kernel) {
            if (this.props.isolateCells) {
                this.state.kernel
                    .restart()
                    .then(() => this.renderResponse(outputArea, code))
                    .catch(() => {
                        this.log(() => console.error('failed'));
                        this.setState({ kernel: null });
                        outputArea.model.clear();
                        outputArea.model.add({
                            output_type: 'stream',
                            name: 'failure',
                            text: this.props.msgError
                        });
                    });
                return;
            }
            this.renderResponse(outputArea, code);
            return;
        }
        new Promise((resolve, reject) =>
            this.getKernel()
                .then(resolve)
                .catch(reject)
        )
            .then(kernel => {
                this.setState({ kernel });
                this.renderResponse(outputArea, code);
            })
            .catch(() => {
                this.log(() => console.error('failed'));
                this.setState({ kernel: null });
                if (this.props.useStorage) {
                    this.setState({ fromStorage: false });
                    window.localStorage.removeItem(this.props.storageKey);
                }
                outputArea.model.clear();
                outputArea.model.add({
                    output_type: 'stream',
                    name: 'failure',
                    text: this.props.msgError
                });
            });
    }

    render() {
        return (
            <React.Fragment>
                <CodeSection
                    title="Source"
                    actions={this.props.actions && this.props.actions(this.state)}>
                    <CodeMirrorRender
                        ref={x => {
                            this.inputRef = x;
                        }}
                    />
                </CodeSection>
                <AnimateHeight
                    animateOpacity={true}
                    height={this.props.outputIsVisible ? 'auto' : 0}>
                    <CodeSection title="Output" clearFunction={() => this.props.hideOutput()}>
                        <OutputRender
                            ref={x => {
                                this.outputRef = x;
                            }}
                        />
                    </CodeSection>
                </AnimateHeight>
            </React.Fragment>
        );
    }
}

Juniper.defaultProps = {
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
    msgLoading: 'Loading...',
    msgError: 'Connecting failed. Please reload and try again.'
};
