# AllenNLP Course

## Running the app

To start the local development server, install [Gatsby](https://gatsbyjs.org)
and then all other dependencies. This should serve up the app on
`localhost:8000`.

```bash
npm install -g gatsby-cli  # Install Gatsby globally
npm install                # Install dependencies
npm run dev                # Run the development server
```

### Formatting

This app uses REVIZ-preferred code formatting. To ensure components are formatted correctly, you can run the following command to check and fixing linting issues before committing changes.

```
npm run lint:fix
```

## Dependencies

### Back-end
This app is deployed via [Skiff](https://github.com/allenai/skiff) and code exercises are run via [Binder](https://mybinder.org) and [JupyterLab](https://github.com/jupyterlab/jupyterlab).

### Front-end
Like most AI2 web apps, the front-end is powered by the [Varnish](https://github.com/allenai/varnish) UI component library and its dependencies ([React](https://reactjs.org/), [Ant Design](https://ant.design/), and [Styled Components](https://styled-components.com/)).

Unlike most AI2 web apps, package management is handled via [NPM](https://www.npmjs.com/) instead of Yarn, and the routing and static site generation is driven by [Gatsby](http://gatsbyjs.org/) instead of NextJS. This app also does not use TypeScript, as it was not included in the template that this app was forked from.

Read-only code blocks are rendered with [Prism](https://prismjs.com/) and interactive code blocks are rendered with [CodeMirror](https://codemirror.net/).

See [`package.json`](https://github.com/allenai/allennlp-course/blob/master/package.json) for list of all packages used in this app.

## Static assets

All files added to `/static` will become available at the root of the deployed
site. For example, `/static/diagram.svg` can be referenced in the course as `/diagram.svg`.

## Chapters

Chapters are placed in [`/chapters`](/chapters) and are Markdown files
consisting of `<exercise>` components. They'll be turned into pages, e.g.
`/overview`. In their frontmatter block at the top of the file, they need to
specify `type: chapter`, as well as the following meta:

```yaml
---
title: The chapter title
description: The chapter description
type: chapter # important: this creates a standalone page from the chapter
---
```

### `outline.js` and Chapter Organization

Chapter navigation is rendered programmatically via [`outline.js`](https://github.com/allenai/allennlp-course/blob/master/src/outline.js). Chapters are grouped into "parts," as can be seen in the structure of the outline data object. There is a special chapter called Overview that always appears as the first top-level item in the outline.

Each part supports the following properties:

| Property       | Type             | Description                                                                     |
| -------------- | ---------------- | ------------------------------------------------------------------------------- |
| `title`        | string           | Unique part title.                                                              |
| `description`  | string           | Part description.                                                               |
| `chapterSlugs` | array of strings | List of chapter slugs                                                           |
| `icon`         | string           | Optional icon (defaults to 'default').                                          |
| `antMenuIcon`  | string           | Optional [Ant Icon](https://ant.design/components/icon/) for use in outline nav |
| `color`        | string           | Optional color (defaults to 'default')                                          |

## Custom Elements

When using custom elements, make sure to place a newline between the
opening/closing tags and the children. Otherwise, Markdown content may not
render correctly.

### `<exercise>`

Container of a single exercise.

| Argument     | Type            | Description                                                    |
| ------------ | --------------- | -------------------------------------------------------------- |
| `id`         | number / string | Unique exercise ID within chapter.                             |
| `title`      | string          | Exercise title.                                                |
| **children** | -               | The contents of the exercise.                                  |

```markdown
<exercise id="1" title="Introduction">

Content goes here...

</exercise>
```

### `<codeblock>`

| Argument     | Type            | Description                                                                                  |
| ------------ | --------------- | -------------------------------------------------------------------------------------------- |
| `id`         | number / string | Unique identifier of the code exercise.                                                      |
| `source`     | string          | Name of the source file (without file extension).                                            |
| `setup`      | string          | Name of the setup file (without file extension).                                             |

```markdown
<codeblock source="part1/training-and-prediction/training" setup="part1/setup"></codeblock>
```

## Setting up Binder

The [`requirements.txt`](binder/requirements.txt) in the repository defines the
packages that are installed when building it with Binder. You can specify the
binder settings like repo, branch and kernel type in the `"juniper"` section of
the `meta.json`. If modifying these fields, it is recommended that you run the very
first build via the interface on the [Binder website](https://mybinder.org), as
this gives you a detailed build log and feedback on whether everything worked as expected.
Enter your repository URL, click "launch" and wait for it to install the dependencies
and build the image.

![Binder](https://user-images.githubusercontent.com/13643239/39412757-a518d416-4c21-11e8-9dad-8b4cc14737bc.png)

---

This course was initially forked from the
[`course-starter-python`](https://github.com/ines/course-starter-python) template developed by Ines Montani.
