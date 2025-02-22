import React from "react";
import { Navbar, Button } from "react-bootstrap";

const Toolbar = () => {
  return (
    <Navbar bg="dark" variant="dark" className="p-2">
      <Button variant="primary" className="mx-2">New</Button>
      <Button variant="success" className="mx-2">Save</Button>
      <Button variant="danger" className="mx-2">Export</Button>
    </Navbar>
  );
};

export default Toolbar;
