import React from 'react';
import { CubeIcon, RocketIcon, StackIcon, ToolsIcon, TextIcon } from './components/inlineSVG';

export const getIcon = (icon, size = 108) => {
    if (icon === 'stack') {
        return <StackIcon size={size} />;
    } else if (icon === 'rocket') {
        return <RocketIcon size={size} />;
    } else if (icon === 'cube') {
        return <CubeIcon size={size} />;
    } else if (icon === 'tools') {
        return <ToolsIcon size={size} />;
    } else { // 'default'
        return <TextIcon size={size} />;
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

// Convert px number value to rem string value
export const toRem = (value) => {
    return `${value / 16}rem`;
};
