// Create a lookup table of chapters by slug value
export const getGroupedChapters = (data) => {
  return data.edges.reduce((acc, obj) => {
    const key = obj.node.fields.slug;
    if (!acc[key]) {
      acc[key] = {};
    }
    acc[key] = obj;
    return acc;
  }, {});
};
