import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import FineTune from './pages/FineTune';
import Results from './pages/Results';

function App() {
    return (
        <Router>
            <Navbar />
            <Switch>
                <Route path="/" exact component={Home} />
                <Route path="/dataset" component={Dataset} />
                <Route path="/fine-tune" component={FineTune} />
                <Route path="/results" component={Results} />
            </Switch>
        </Router>
    );
}

export default App;
