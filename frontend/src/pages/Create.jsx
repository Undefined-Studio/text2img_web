import React from 'react';
import { Grid, Col, Row, Jumbotron, Button } from 'react-bootstrap';

import DescriptionForm from '../../components/DescriptionForm';
import SubmitAPI from '../../api/SubmitAPI';

class Create extends React.Component {
  constructor(props, context) {
    super(props, context);

    this.submit = this.submit.bind(this);

    this.inputs = [];
  }

  submit() {
    let text = [];
    for (var input of this.inputs) {
      text.push(input.value);
    }
    SubmitAPI.submit('api/text2pic/create', {
      data: text
    });
  }
  render() {
    return (
      <div>
        <Grid>
          <Row className="Home">
            <Col xs={12} xsOffset={0}>
              <Jumbotron>
                <p>请用十句话描述你需要的花！</p>
              </Jumbotron>
            </Col>
          </Row>
          <Row className="description">
            <Col xs={6} xsOffset={3}>
              <DescriptionForm label="第一句话" id={0} list={this.inputs} />
              <DescriptionForm label="第二句话" id={1} list={this.inputs} />
              <DescriptionForm label="第三句话" id={2} list={this.inputs} />
              <DescriptionForm label="第四句话" id={3} list={this.inputs} />
              <DescriptionForm label="第五句话" id={4} list={this.inputs} />
              <DescriptionForm label="第六句话" id={5} list={this.inputs} />
              <DescriptionForm label="第七句话" id={6} list={this.inputs} />
              <DescriptionForm label="第八句话" id={7} list={this.inputs} />
              <DescriptionForm label="第九句话" id={8} list={this.inputs} />
              <DescriptionForm label="第十句话" id={9} list={this.inputs} />
            </Col>
          </Row>
          <Row>
            <Col xs={12} xsOffset={5}>
              <Button bsStyle="primary" onClick={this.submit} bsSize="large">
                生成
              </Button>
            </Col>
          </Row>
        </Grid>
      </div>
    );
  }
}

export default Create;
