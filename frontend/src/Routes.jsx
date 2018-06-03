import React from 'react';
import { Router, Route, broswerHistory, IndexRoute } from 'react-router';

import App from './pages/App';
import Home from './pages/Home';
import Create from './pages/Create';

const Routes = () => {
  return (
    <Router history={broswerHistory}>
      <Route path="/" component={App}>
        <IndexRoute component={Home} />
        <Route path="home" component={Home} />
        <Route path="create" component={Create} />
      </Route>
    </Router>
  );
};

export default Routes;
