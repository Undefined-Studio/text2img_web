import React from 'react';
import { Link } from 'react-router-dom';
import { Navbar, Nav, NavItem} from 'react-bootstrap';
import { LinkContainer } from 'react-router-bootstrap';

const TopMenu = () => (
  <Navbar>
    <Navbar.Header>
      <Navbar.Brand>
        <Link to='/'>Text2Img</Link>
      </Navbar.Brand>
    </Navbar.Header>
    <Nav>
      <LinkContainer to='/create'>
        <NavItem eventKey={1}>
        Create
        </NavItem>
      </LinkContainer>
    </Nav>
  </Navbar>
);

export default TopMenu;
