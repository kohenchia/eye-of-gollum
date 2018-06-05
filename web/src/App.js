import React, { Component } from 'react';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      frame: null
    }
  }
  componentDidMount() {
    fetch('/video/frame')
      .then(ret => ret.json())
      .then((ret) => {
        this.setState({
          frame: ret.frame
        })
      });
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
        <img src={`data:image/jpeg;base64,${this.state.frame}`}
             alt='AI2 Incubator Live Feed'/>
      </div>
    );
  }
}

export default App;
