class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = { input: '', results: [], message: null, showDiff: true, showAll: true };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleOptionChange = this.handleOptionChange.bind(this);
    }

    handleChange(event) {
        this.setState({ input: event.target.value });
    }

    handleSubmit(event) {
        event.preventDefault();

        var input = this.state.input;
        if (!input) {
            return;
        }

        if (input.match(/[a-zA-Z]$/)) {
            input += ' .'
            this.setState({ input });
        }
        if (input.match(/ $/)) {
            input += '.'
            this.setState({ input });
        }

        var api = '/api/' + input;
        // var api = 'http://localhost:5000/api/' + input;
        // var api = '/static/' + 'response.json'; // mock

        this.setState({ results: [], message: 'correcting...' });
        fetch(api)
            .then((response) => {
                response.json().then((data) => {
                    this.setState({ results: data, message: null });
                });
            })
            .catch((error) => {
                this.setState({ results: [], message: error.toString() });
            });
    }

    handleOptionChange(event) {
        const target = event.target;
        const value = target.type === 'checkbox' ? target.checked : target.value;
        const name = target.name;
    
        this.setState({ [name]: value });
    }
    
    render() {
        var message = null;
        if (this.state.message) {
            message = <Message message={this.state.message} />;
        }

        return (
            <div>
                <div className="form container">
                    <form onSubmit={this.handleSubmit}>
                        <div className="field">
                            <label htmlFor="input">Input</label>
                            <input type="text" id="input" className="input" name="input" size="50" maxLength="100"
                                value={this.state.input} onChange={this.handleChange}></input>
                        </div>
                        <div>
                            <button type="submit" id="correct-btn" className="btn is-primary">Correct</button>
                            <div className="options">
                                <label>
                                    <input type="checkbox" className="checkbox" checked={this.state.showDiff}
                                        name="showDiff" onChange={this.handleOptionChange} />
                                    <span>Show diff</span>
                                </label>
                                <label>
                                    <input type="checkbox" className="checkbox" checked={this.state.showAll}
                                        name="showAll" onChange={this.handleOptionChange} />
                                    <span>Show all candidates</span>
                                </label>
                            </div>
                        </div>
                    </form>
                </div>
                <Results results={this.state.results} showDiff={this.state.showDiff} showAll={this.state.showAll} />
                {message}
            </div>
        );
    }
}

class Results extends React.Component {
    render() {
        return (
            <div className="results">
                {this.props.results.map((results, index) => (
                    <Iteration key={index} index={index} results={results} showDiff={this.props.showDiff} showAll={this.props.showAll} />
                ))}
            </div>
        );
    }
}

class Iteration extends React.Component {
    // Get entry's color based on index and fluency scores
    getEntryColor(entry, index) {
        if (index == 0) {
            return 'is-primary';
        }
        var maxCorrection = this.props.results.reduce(function(accumulator, currentEntry) {
            return currentEntry.fluency_scores > accumulator.fluency_scores ? currentEntry : accumulator;
        });
        if (entry.fluency_scores === maxCorrection.fluency_scores) {
            return 'is-success';
        }

        if (this.props.showAll) {
            return '';
        } else {
            return 'is-hidden';
        }
    }

    render() {
        // skip first element which is from previous iteration 
        var entries = this.props.results.slice(1).map((entry, index) => (
            <Entry key={index} index={index} entry={entry} color={this.getEntryColor(entry, index)} showDiff={this.props.showDiff} />
        ))

        return (
            <div className="iteration container with-title">
                <h3 className="title">Iteration {this.props.index + 1}</h3>
                { entries }
            </div>
        );
    }
}

class Entry extends React.Component {
    render() {
        var tokens = [];
        if (this.props.showDiff) {
            var diff = JsDiff.diffWords(this.props.entry.src_str, this.props.entry.hypo_str);
            diff.forEach(function(part) {
                var color = part.added ? 'is-success' : part.removed ? 'is-error' : '';
                var value = part.value;
                tokens.push({ color, value });
            });
        } else {
            tokens = this.props.entry.hypo_str.split(' ').map((value) => (
                { color: '', value: value + ' ' }
            ));
        }

        return (
            <div className="entry">
                <div className="containers">
                    <div className={"hypo-str container " + (this.props.color)}>
                        <Tokens tokens={tokens} />
                    </div>
                    <div className={"fluency-scores container " + (this.props.color)}>
                        {this.props.entry.fluency_scores.toFixed(4)}
                    </div>
                </div>
                {this.props.index === 0 &&
                    <div className="arrow-down"></div>
                }
            </div>
        );
    }
}

class Tokens extends React.Component {
    render() {
        return (
            <div className="tokens">
                {this.props.tokens.map((token, index) => (
                    <span key={index} className={token.color}>{token.value}</span>
                ))}
            </div>
        );
    }
}

class Message extends React.Component {
    render() {
        return (
            <div className="balloon container">
                <div className="messages">
                    <div className="message from-left">
                        <i className="bcrikko"></i>
                        <div className="balloon from-left">
                            <p>{this.props.message}</p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

ReactDOM.render(
    <App />,
    document.getElementById('app')
)
