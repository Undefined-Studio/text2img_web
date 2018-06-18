import React from 'react';
import { FormGroup, ControlLabel, FormControl } from 'react-bootstrap';

class DescriptionForm extends React.Component {
  constructor(props, context) {
    super(props, context);

    this.handleChange = this.handleChange.bind(this);
    this.handleInput = this.handleInput.bind(this);

    this.state = {
      value: ''
    };
  }

  getValidationState() {
    const length = this.state.value.length;
    if (length > 2) return 'success';
    return null;
  }

  handleChange(e) {
    this.setState({ value: e.target.value });
  }

  handleInput(input) {
    // let max_idx = this.props.list.length - 1;
    // if (max_idx < this.props.id) {
    //   while (max_idx < this.props.id) {
    //     this.props.list.push(null);
    //     max_idx = this.props.list.length - 1;
    //   }
    //   this.props.list.push(input);
    // } else {
    //   this.props.list[this.props.id] = input;
    // }
    this.props.list.push(input);
  }

  render() {
    return (
      <form>
        <FormGroup
          controlId="formBasicText"
          validationState={this.getValidationState()}
        >
          <ControlLabel>{this.props.label}</ControlLabel>
          <FormControl
            type="text"
            value={this.state.value}
            placeholder="Enter text"
            onChange={this.handleChange}
            inputRef={this.handleInput}
          />
          <FormControl.Feedback />
          {/* <HelpBlock>Validation is based on string length.</HelpBlock> */}
        </FormGroup>
      </form>
    );
  }
}

export default DescriptionForm;
