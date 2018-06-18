import React from 'react';
import { LinkContainer } from 'react-router-bootstrap';
import { Image, Grid, Col, Jumbotron, Row, Button } from 'react-bootstrap';

import Setting from '../config';

const Result = () => (
  <div>
    <Grid>
      <Row className="Home">
        <Col xs={12} xsOffset={0}>
          <Jumbotron>
            <p>尽情欣赏生成的图片吧！</p>
          </Jumbotron>
        </Col>
      </Row>
      <Row className="result">
        <Col xs={6} xsOffset={3}>
          <Image src={Setting.backEndUrl + '/static/gen.png'} responsive />
        </Col>
      </Row>
      <br />
      <Row>
        <Col xs={12} xsOffset={5}>
          <LinkContainer to="/">
            <Button bsStyle="success" bsSize="large">
                返回首页
            </Button>
          </LinkContainer>
        </Col>
      </Row>
    </Grid>
  </div>
);

export default Result;
