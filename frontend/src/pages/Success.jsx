import React from 'react';
import { LinkContainer } from 'react-router-bootstrap';
import { Modal, Button, Grid, Row } from 'react-bootstrap';

const Success = () => (
  <Grid>
    <Row className="text-center">
      <div className="static-modal">
        <Modal.Dialog>
          <Modal.Header>
            <Modal.Title>生成成功！</Modal.Title>
          </Modal.Header>

          <Modal.Body>后台已经成功生成对应图片！</Modal.Body>

          <Modal.Footer>
            <LinkContainer to="/result">
              <Button bsStyle="primary">查看图片</Button>
            </LinkContainer>
          </Modal.Footer>
        </Modal.Dialog>
      </div>
    </Row>
  </Grid>
);

export default Success;
