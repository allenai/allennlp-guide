# AllenNLP Course

This course app was forked from the
[`course-starter-python`](https://github.com/ines/course-starter-python) template developed by Ines Montani.

The outline for the course that I'm envisioning looks like this.  These are the main sections,
which might themselves be several chapters long.

1. Introduction to the course
2. AllenNLP's core abstractions (at a high level, how they all fit together)
3. Deep dives into several of the core abstractions (there's one there for `TextField` and related
   classes).
4. Using AllenNLP - built-in commands, configuration files, etc.
5. Testing with AllenNLP
6. Task-specific chapters (many of these on different tasks)
7. Building demos with AllenNLP

Some of these could maybe be combined or rearranged, but that's the basic idea.

## Running the app

To start the local development server, install [Gatsby](https://gatsbyjs.org)
and then all other dependencies. This should serve up the app on
`localhost:8000`.

```bash
npm install -g gatsby-cli  # Install Gatsby globally
npm install                # Install dependencies
npm run dev                # Run the development server
```

## Dependencies

### Back-end
This app is deployed via [Skiff](https://github.com/allenai/skiff) and code execution is run via [Binder](https://mybinder.org) and [JupyterLab](https://github.com/jupyterlab/jupyterlab).

### Front-end
Like most AI2 web apps, the front-end is powered by the [Varnish](https://github.com/allenai/varnish) UI component library and its dependencies ([React](https://reactjs.org/), [Ant Design](https://ant.design/), and [Styled Components](https://styled-components.com/)).

Unlike most AI2 web apps, package management is handled via [NPM](https://www.npmjs.com/) instead of Yarn, and the routing and static site generation is driven by [Gatsby](http://gatsbyjs.org/) instead of NextJS. This app also does not use TypeScript, as it was included in the template that this app was forked from.

Read-only code blocks are rendred with [Prism](https://prismjs.com/) and interactive code blocks are rendered with [CodeMirror](https://codemirror.net/).

See [`package.json`](https://github.com/allenai/allennlp-course/blob/master/package.json) for list of all packages used in this app.

## Static assets

All files added to `/static` will become available at the root of the deployed
site. So `/static/image.jpg` can be referenced in your course as `/image.jpg`.

## Chapters

Chapters are placed in [`/chapters`](/chapters) and are Markdown files
consisting of `<exercise>` components. They'll be turned into pages, e.g.
`/chapter1`. In their frontmatter block at the top of the file, they need to
specify `type: chapter`, as well as the following meta:

```yaml
---
title: The chapter title
description: The chapter description
type: chapter # important: this creates a standalone page from the chapter
---
```

### Custom Elements

When using custom elements, make sure to place a newline between the
opening/closing tags and the children. Otherwise, Markdown content may not
render correctly.

#### `<exercise>`

Container of a single exercise.

| Argument     | Type            | Description                                                    |
| ------------ | --------------- | -------------------------------------------------------------- |
| `id`         | number / string | Unique exercise ID within chapter.                             |
| `title`      | string          | Exercise title.                                                |
| `type`       | string          | Optional type. `"slides"` makes container wider and adds icon. |
| **children** | -               | The contents of the exercise.                                  |

```markdown
<exercise id="1" title="Introduction to spaCy">

Content goes here...

</exercise>
```

#### `<codeblock>`

| Argument     | Type            | Description                                                                                  |
| ------------ | --------------- | -------------------------------------------------------------------------------------------- |
| `id`         | number / string | Unique identifier of the code exercise.                                                      |
| `source`     | string          | Name of the source file (without file extension). Defaults to `exc_${id}` if not set.        |
| `solution`   | string          | Name of the solution file (without file extension). Defaults to `solution_${id}` if not set. |
| `test`       | string          | Name of the test file (without file extension). Defaults to `test_${id}` if not set.         |
| **children** | string          | Optional hints displayed when the user clicks "Show hints".                                  |

```markdown
<codeblock id="02_03">

This is a hint!

</codeblock>
```

### Setting up Binder

The [`requirements.txt`](binder/requirements.txt) in the repository defines the
packages that are installed when building it with Binder. You can specify the
binder settings like repo, branch and kernel type in the `"juniper"` section of
the `meta.json`. I'd recommend running the very first build via the interface on
the [Binder website](https://mybinder.org), as this gives you a detailed build
log and feedback on whether everything worked as expected. Enter your repository
URL, click "launch" and wait for it to install the dependencies and build the
image.

![Binder](https://user-images.githubusercontent.com/13643239/39412757-a518d416-4c21-11e8-9dad-8b4cc14737bc.png)
