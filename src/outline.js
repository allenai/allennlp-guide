// This outline builds our chapter navigation structure, grouped by Part.

export const outline = [
  // Standalone Chapter
  {
    slug: '/overview'
  },
  // Group containing chapters
  {
    title: 'Part I: Quick Start',
    chapterSlugs: [
      '/introduction',
      '/your-first-model',
      '/training-and-prediction',
      '/next-steps'
    ]
  },
  {
    title: 'Part II: Abstractions, Design, and Testing',
    chapterSlugs: [
      '/reading-textual-data',
      '/representing-text-as-features'
    ]
  },
  {
    title: 'Part III: Practical Tasks With AllenNLP',
    chapterSlugs: [
      '/semantic-parsing'
    ]
  }
];
