import React from "react"

class HTML extends React.Component
{
    constructor(props) {
        super(props);
        this.state = {
            frame: null
        }
    }
    componentDidMount() {
        fetch('/eyeofgollum/frame')
            .then(res => res.json())
            .then(
                (result) => {
                    this.setState({
                        frame: result.frame.substring(2, result.frame.length - 1)
                    });
                },
                (error) => {
                    console.log('Error fetching frame.')
                }
            )
    }
    render() {
        return (
            <div style={{ margin: '3rem auto', maxWidth: 600 }}>
                <h1>AI2 Incubator Camera</h1>
                <img src={`data:image/jpeg;base64,${this.state.frame}`}/>
            </div>
        );
    }
}

export default HTML;
