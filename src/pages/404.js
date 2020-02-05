import React from 'react';
import Layout from '../templates/Layout';

export default () => {
    return (
        <Layout>
            {/* TODO: Replace inline style with styled component when that package is merged */}
            <div style={{textAlign: 'center'}}>
              <h1>Error 404</h1>
              <p>Page not found.</p>
            </div>
        </Layout>
    );
};
