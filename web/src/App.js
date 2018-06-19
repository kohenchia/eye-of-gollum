import React, { Component } from 'react';
import './App.css';

class App extends Component {
    constructor(props) {
        super(props);
        this.state = {
            frame: null
        }
        this.ws = new WebSocket('ws://172.16.20.70/eyeofgollum/video/stream');
        this.ws.binaryType = 'arraybuffer';
        this.ws.onopen = (event => this.ws.send('.'));
        this.ws.onmessage = (event => this.handleData(event.data));
    }
    handleData(data) {
        let base64String = btoa(String.fromCharCode(...new Uint8Array(data)));
        this.setState({
            frame: base64String
        })
        this.ws.send('.');
    }
    render() {
        return (
            <div className="App">
                <header className="App-header">
                    <h1 className="App-title">
                        AI2 INCUBATOR LIVE FEED
                    </h1>
                </header>
                <img className="App-eog-video-feed"
                     src={`data:image/jpeg;base64,${this.state.frame}`}
                     alt='AI2 Incubator Live Feed' />
            </div>
        );
    }
}

export default App;
