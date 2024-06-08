import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
    return (
        <nav>
            <ul>
                <li><Link to="/">Home</Link></li>
                <li><Link to="/dataset">Dataset</Link></li>
                <li><Link to="/fine-tune">Fine-Tune</Link></li>
                <li><Link to="/results">Results</Link></li>
            </ul>
        </nav>
    );
}

export default Navbar;
