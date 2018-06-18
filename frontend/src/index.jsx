import React from 'react';
import ReactDOM from 'react-dom';
import { Switch, Route, BrowserRouter } from 'react-router-dom';

import TopMenu from './components/TopMenu';
import Home from './pages/Home';
import Create from './pages/Create';
import Success from './pages/Success';
import Result from './pages/Result';

const Main = () => (
  <main>
    <Switch>
      <Route exact path="/" component={Home} />
      <Route exact path="/create" component={Create} />
      <Route exact path="/success" component={Success} />
      <Route exact path="/result" component={Result} />
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
