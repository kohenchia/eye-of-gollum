import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      frame: null
    }
  }
  handleData(data) {
    console.log(typeof(data));
    console.log(data.length);
    // TODO: Serialize data to base-64
    this.setState({
      frame: null
    })
  }
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">AI2 Incubator Camera</h1>
        </header>
        <img src={`data:image/jpeg;base64,${this.state.frame}`}/>
      </div>
    );
  }
}

export default App;
