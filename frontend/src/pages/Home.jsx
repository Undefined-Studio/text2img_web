import React from 'react';
import { Jumbotron, Button, Grid, Col, Row } from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';

const Home = () => (
  <Grid>
    <Row className="Home">
      <Col xs={12} xsOffset={0}>
        <Jumbotron>
          <h1>Text2Img</h1>
          <p>
            This is a simple hero unit, a simple jumbotron-style component for
            calling extra attention to featured content or information.
          </p>
          <p>
            <LinkContainer to="/create">
              <Button bsStyle="primary">Have a Try!</Button>
            </LinkContainer>
          </p>
        </Jumbotron>
      </Col>
    </Row>
  </Grid>
);

export default Home;
