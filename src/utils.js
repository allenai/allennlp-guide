import React from 'react';
import { CubeIcon, RocketIcon, StackIcon, ToolsIcon, TextIcon } from './components/inlineSVG';

export const getIcon = (icon) => {
    if (icon === 'stack') {
        return <StackIcon />;
    } else if (icon === 'rocket') {
        return <RocketIcon />;
    } else if (icon === 'cube') {
        return <CubeIcon />;
    } else if (icon === 'tools') {
        return <ToolsIcon />;
    } else { // 'default'
        return <TextIcon />;
    }
};

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