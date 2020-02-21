import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled from 'styled-components';

import { Button } from '@allenai/varnish/components/button';

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

class CodeBlock extends React.Component {
    state = { Juniper: null, key: 0 };

    handleReset() {
        // Using the key as a hack to force component to rerender
        this.setState({ key: this.state.key + 1 });
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
        const { Juniper } = this.state;
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
                        <StyledCodeBlock key={this.state.key}>
                            {Juniper && (
                                <Juniper
                                    msgButton={null}
                                    repo={repo}
                                    branch={branch}
                                    lang={lang}
                                    kernelType={kernelType}
                                    handleReset={() => this.handleReset()}
                                    debug={debug}
                                    setupFile={setupFile}
                                    sourceFile={sourceFile}
                                    actions={({ runCode }) => execute && (
                                        <CodeBlockToolbar>
                                            <RunButton onClick={() => runCode(code => addSetupCode(code, setupFile))}>Run Code</RunButton>
                                        </CodeBlockToolbar>
                                    )}
                                />
                            )}
                        </StyledCodeBlock>
                    )
                }}
            />
        );
    }
}

export default CodeBlock;

// CSS ported from SASS
// TODO(aarons): Revisit these styles
const StyledCodeBlock = styled.div`
    margin-bottom: 2rem;
    margin-top: 2rem;
    font-size: 0.8625rem;
`;

const RunButton = styled(Button)`
    &&& {
        background: ${({ theme }) => theme.color.N1};
        color: ${({ theme }) => theme.color.B6};
    }
`;

const CodeBlockToolbar = styled.div`
    display: flex;
    padding: 0 ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md}`} 0;

    ${RunButton} {
        margin-left: auto;
    }
`;
