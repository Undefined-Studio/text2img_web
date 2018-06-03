import React from 'react';
import TopMenu from '../components/TopMenu';

const App = ({ children }) => {
  return (
    <div>
      <TopMenu />
      <div> {children} </div>
    </div>
  );
};

export default App;
