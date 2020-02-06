import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import styled from 'styled-components';

import { Button } from '@allenai/varnish/components/button';

function getFiles({ allCode }, sourceId, solutionId, testId, setupId) {
    var files = {};
    allCode.edges.forEach(({ node }) => {
        const filename = node.dir + '/' + node.name;
        if (filename.includes(sourceId)) {
            files['sourceFile'] = node.code;
        }
        if (filename.includes(solutionId)) {
            files['solutionFile'] = node.code;
        }
        if (filename.includes(testId)) {
            files['testFile'] = node.code;
        }
        if (filename.includes(setupId)) {
            files['setupFile'] = node.code;
        }
    });
    return files;
}

function makeTest(template, testFile, solution) {
    // Escape quotation marks in the solution code, for cases where we
    // can only place the solution in regular quotes.
    const solutionEscaped = solution.replace(/"/g, '\\"');
    return template
        .replace(/\${solutionEscaped}/g, solutionEscaped)
        .replace(/\${solution}/g, solution)
        .replace(/\${test}/g, testFile);
}

function addSetupCode(code, setup) {
    // Prepend setup code if specified.
    return setup ? setup + code : code;
}

class CodeBlock extends React.Component {
    state = { Juniper: null, showSolution: false, key: 0 };

    handleShowSolution() {
        this.setState({ showSolution: true });
    }

    handleReset() {
        // Using the key as a hack to force component to rerender
        this.setState({ showSolution: false, key: this.state.key + 1 });
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
        const { Juniper, showSolution } = this.state;
        const { id, source, solution, test, setup, executable } = this.props;
        const sourceId = source || `${id}_source`;
        const solutionId = solution || `${id}_solution`;
        const testId = test || `${id}_test`;
        const setupId = setup || `${id}_setup`;
        const execute = executable !== "false";

        return (
            <StaticQuery
                query={graphql`
                    {
                        site {
                            siteMetadata {
                                testTemplate
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
                    const { testTemplate } = data.site.siteMetadata
                    const { repo, branch, kernelType, debug, lang } = data.site.siteMetadata.juniper
                    const {sourceFile, solutionFile, testFile, setupFile} = getFiles(data, sourceId, solutionId, testId, setupId)
                    return (
                        <StyledCodeBlock key={this.state.key}>
                            {Juniper && (
                                <Juniper
                                    msgButton={null}
                                    repo={repo}
                                    branch={branch}
                                    lang={lang}
                                    kernelType={kernelType}
                                    debug={debug}
                                    actions={({ runCode }) => (
                                        <React.Fragment>
                                            {execute && (
                                            <Button onClick={() =>
                                                runCode(code =>
                                                    addSetupCode(code, setupFile))}>Run Code</Button>
                                            )}
                                            {execute && testFile && (
                                                <Button
                                                    variant="primary"
                                                    onClick={() =>
                                                        runCode(value =>
                                                            makeTest(testTemplate, testFile, value)
                                                        )
                                                    }
                                                >
                                                    Submit
                                                </Button>
                                            )}
                                        </React.Fragment>
                                    )}
                                >
                                    {showSolution ? solutionFile : sourceFile}
                                </Juniper>
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
    margin-left: -2rem;
    width: calc(100% + 4rem);
    margin-bottom: 2rem;
    margin-top: 2rem;
    font-size: 0.87rem;
`;
