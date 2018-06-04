import React from 'react';
import ReactDOM from 'react-dom';
import { Switch, Route, BrowserRouter } from 'react-router-dom';

import TopMenu from './components/TopMenu';
import Home from './pages/Home';
import Create from './pages/Create';

const Main = () => (
  <main>
    <Switch>
      <Route exact path="/" component={Home} />
      <Route exact path="/create" component={Create} />
    </Switch>
  </main>
);

const App = () => (
  <div>
    <header>
      <TopMenu />
    </header>
    <Main />
    <footer />
  </div>
);

ReactDOM.render(
  <BrowserRouter>
    <App />
  </BrowserRouter>,
  document.getElementById('root')
);
