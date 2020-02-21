// This SVG asset displays a "rocket" icon intended for Quick Start section

import React from 'react';

export const PartIcon = ({ className, size, children }) => (
    <svg
      width={size}
      height={size}
      viewBox="0 0 108 108"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      children={children} />
);
